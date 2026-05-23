"""
Response API → Chat API 反向转发服务
将 OpenAI Response 协议请求转发为 Chat 协议请求，并将 Chat 响应封装回 Response 协议

与 main.py 方向相反：
- main.py:      Chat请求 → Response模型 → Chat响应
- 本文件:        Response请求 → Chat模型 → Response响应
"""

import os
import json
import time
import uuid
import asyncio
import logging
import traceback
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "false").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "logs/main_reverse.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 默认 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
CONVERT_MODEL = os.getenv("CONVERT_MODEL","deepseek-v4-pro")
# 确保日志目录存在
log_dir = os.path.dirname(LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# 构建 handlers
_handlers = []

# 文件 handler（始终启用，带轮转）
from logging.handlers import RotatingFileHandler
_file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
_handlers.append(_file_handler)

# 控制台 handler（仅当 LOG_TO_CONSOLE=true 时启用）
if LOG_TO_CONSOLE:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    _handlers.append(_console_handler)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=_handlers,
)
logger = logging.getLogger("chat2response")

# ==================== 配置 ====================
CHAT_API_BASE = os.getenv("CHAT_API_BASE")
if not CHAT_API_BASE:
    raise ValueError("必须配置 CHAT_API_BASE 环境变量，请在 .env 文件中设置 Chat API 的基础 URL")

DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "300"))
POOL_TIMEOUT = float(os.getenv("POOL_TIMEOUT", "10"))
STREAM_READ_TIMEOUT = float(os.getenv("STREAM_READ_TIMEOUT", "120"))
STREAM_MAX_DURATION = int(os.getenv("STREAM_MAX_DURATION", "0"))

MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "30"))
KEEPALIVE_EXPIRY = int(os.getenv("KEEPALIVE_EXPIRY", "60"))

# 默认 instructions 配置（反向代理中作为 system 消息插入）
DEFAULT_INSTRUCTIONS = os.getenv("DEFAULT_INSTRUCTIONS", "").strip()
FORCE_DEFAULT_INSTRUCTIONS = os.getenv("FORCE_DEFAULT_INSTRUCTIONS", "false").lower() == "true"


# ==================== SSE 事件格式化 ====================

def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """格式化为带 event: 行的 SSE 事件"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== ID 生成 ====================

def generate_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def generate_item_id(prefix: str = "msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# ==================== Usage 格式转换 ====================

def convert_chat_usage_to_response_usage(chat_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    将 Chat API 的 usage 格式转换为 Response API 的 usage 格式

    Chat API 格式:
    {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 10, "audio_tokens": 0, ...}
    }

    Response API 格式:
    {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 10}
    }
    """
    if chat_usage is None:
        return None

    response_usage = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }

    prompt_details = chat_usage.get("prompt_tokens_details")
    if prompt_details:
        response_usage["input_tokens_details"] = {
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        }

    completion_details = chat_usage.get("completion_tokens_details")
    if completion_details:
        response_usage["output_tokens_details"] = {
            "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
        }

    return response_usage


# ==================== 请求转换: Response → Chat ====================

def convert_content_response_to_chat(
    content: Union[str, List[Dict[str, Any]], None]
) -> Union[str, List[Dict[str, Any]], None]:
    """
    将 Response API 的内容格式转换为 Chat API 格式

    Response: [{"type": "input_text", "text": "..."}, {"type": "input_image", "image_url": "..."}]
    Chat:     [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    converted = []
    for part in content:
        if not isinstance(part, dict):
            converted.append(part)
            continue

        part_type = part.get("type", "")
        if part_type == "input_text":
            converted.append({
                "type": "text",
                "text": part.get("text", "")
            })
        elif part_type == "input_image":
            image_url = part.get("image_url") or part.get("url") or part.get("source_url") or ""
            converted.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": part.get("detail", "auto")}
            })
        elif part_type == "input_audio":
            # Chat API 使用 audio 类型
            converted.append({
                "type": "input_audio",
                "input_audio": {
                    "data": part.get("data", ""),
                    "format": part.get("format", "wav")
                }
            })
        elif part_type == "input_file":
            converted.append({
                "type": "file",
                "file": {
                    "file_id": part.get("file_id", ""),
                    "filename": part.get("filename", "")
                }
            })
        elif part_type == "output_text":
            # 引用之前输出的文本（conversation 历史中可能出现）
            converted.append({
                "type": "text",
                "text": part.get("text", "")
            })
        else:
            converted.append(part)

    return converted


# 已知的 reasoning / thinking 模型前缀（自动进入思考模式，不依赖 reasoning_effort 参数）
_REASONER_MODEL_PATTERNS = (
    "deepseek-r1", "deepseek-reasoner", "deepseek-v4",
    "o1", "o3", "o4",
)


def _is_reasoner_model(model: str) -> bool:
    """检查模型名是否指示其为 reasoning / thinking 模型"""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _REASONER_MODEL_PATTERNS)


def _ensure_reasoning_consistency(messages: List[Dict[str, Any]], model: str) -> None:
    """确保 reasoning 模型中所有 assistant 消息都有 reasoning_content 字段。

    DeepSeek reasoner 等模型要求全部 assistant 消息都携带 reasoning_content。
    若历史消息缺失该字段（无论是从未有过还是部分丢失），用空字符串补全。
    """
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return

    has_any = any(m.get("reasoning_content") for m in assistant_msgs)
    is_reasoner = _is_reasoner_model(model)

    if not has_any and not is_reasoner:
        return  # 非推理模型且无 reasoning 历史

    missing = [m for m in assistant_msgs if not m.get("reasoning_content")]
    if missing:
        reason = "模型自身为推理模型" if (is_reasoner and not has_any) else "历史消息混合"
        logger.info(
            f"reasoning 一致性补全 ({reason}): {len(missing)}/{len(assistant_msgs)} "
            f"条 assistant 消息缺少 reasoning_content，已补填 '' (model='{model}')"
        )
        for m in missing:
            m["reasoning_content"] = ""


def _inject_schema_instruction(messages: List[Dict[str, Any]], fmt: Dict[str, Any]) -> None:
    """将 json_schema 转为自然语言指令注入到消息列表中，引导模型按 schema 输出 JSON。

    DeepSeek API 不支持 json_schema 类型的 response_format，因此通过消息注入 schema 描述，
    配合 response_format=json_object 来引导模型输出符合预期的 JSON 结构。
    """
    schema = fmt.get("schema", {})
    schema_name = fmt.get("name", "response_schema")

    # 构建简明的自然语言描述
    lines = [f'You must respond with a JSON object that conforms to the following schema ("{schema_name}"):']

    def _describe_schema(s: Dict[str, Any], indent: int = 0) -> List[str]:
        prefix = "  " * indent
        result: List[str] = []
        s_type = s.get("type", "any")
        desc = s.get("description", "")

        if s_type == "object":
            result.append(f"{prefix}- JSON object" + (f": {desc}" if desc else "") + ":")
            props = s.get("properties", {})
            required = s.get("required", []) or []
            for prop_name, prop_schema in (props.items() if isinstance(props, dict) else []):
                req_mark = " [REQUIRED]" if prop_name in required else ""
                prop_desc = prop_schema.get("description", "") if isinstance(prop_schema, dict) else ""
                prop_type = prop_schema.get("type", "any") if isinstance(prop_schema, dict) else "any"
                detail = f": {prop_desc}" if prop_desc else ""
                result.append(f"{prefix}  - `{prop_name}` ({prop_type}){req_mark}{detail}")
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "array":
                    items = prop_schema.get("items", {})
                    if isinstance(items, dict) and items.get("type") == "object":
                        result.extend(_describe_schema(items, indent + 2))
        elif s_type == "array":
            result.append(f"{prefix}- JSON array" + (f": {desc}" if desc else "") + ":")
            items = s.get("items", {})
            if isinstance(items, dict):
                max_items = s.get("maxItems") or s.get("max_items")
                if max_items is not None:
                    result.append(f"{prefix}  - Maximum items: {max_items}")
                result.extend(_describe_schema(items, indent + 1))

        return result

    lines.extend(_describe_schema(schema))
    lines.append("")
    lines.append("Output ONLY the JSON object, no other text, markdown wrapping, or explanation.")

    instruction = "\n".join(lines)
    messages.append({"role": "user", "content": instruction})
    logger.debug(f"已注入 schema 自然语言指令 ({len(instruction)} 字符)")


def convert_response_to_chat_request(response_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Response API 请求转换为 Chat API 请求

    Response API 请求格式:
    {
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi"},
            {"type": "function_call", "call_id": "xxx", "name": "f", "arguments": "{...}"},
            {"type": "function_call_output", "call_id": "xxx", "output": "result"}
        ],
        "instructions": "...",
        "tools": [...],
        "stream": true,
        ...
    }

    Chat API 请求格式:
    {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "tool_calls": [...]},
            {"role": "tool", "tool_call_id": "xxx", "content": "result"}
        ],
        "stream": true,
        ...
    }
    """
    messages: List[Dict[str, Any]] = []

    # 处理 instructions → system 消息
    instructions_text = response_body.get("instructions")
    has_system_in_input = any(
        isinstance(item, dict) and item.get("type") == "message" and item.get("role") in ("system", "developer")
        for item in (response_body.get("input") or [])
    )

    if FORCE_DEFAULT_INSTRUCTIONS and DEFAULT_INSTRUCTIONS:
        messages.append({"role": "system", "content": DEFAULT_INSTRUCTIONS})
    elif instructions_text:
        if not has_system_in_input:
            messages.append({"role": "system", "content": instructions_text})
    elif DEFAULT_INSTRUCTIONS and not has_system_in_input:
        messages.append({"role": "system", "content": DEFAULT_INSTRUCTIONS})

    # 处理 input 数组，转换为 messages
    input_items = response_body.get("input") or []
    current_assistant_msg: Optional[Dict[str, Any]] = None
    pending_reasoning: Optional[str] = None  # Response API 中 reasoning 在 message 之前，需延迟附加

    def _extract_reasoning_text(item: Dict[str, Any]) -> Optional[str]:
        """从 Response API reasoning 项中提取纯文本"""
        summary = item.get("summary") or item.get("content")
        if not summary:
            return None
        if isinstance(summary, str):
            return summary
        if isinstance(summary, list):
            parts = []
            for s in summary:
                if isinstance(s, dict):
                    parts.append(s.get("text", ""))
                elif isinstance(s, str):
                    parts.append(s)
            return "\n".join(parts) or None
        return str(summary)

    def _apply_pending_reasoning():
        """将暂存的 reasoning 附加到 current_assistant_msg，必要时创建占位消息"""
        nonlocal current_assistant_msg, pending_reasoning
        if pending_reasoning is None:
            return
        if current_assistant_msg is None:
            current_assistant_msg = {"role": "assistant", "content": ""}
            logger.debug("为 pending_reasoning 创建占位 assistant 消息 (content='')")
        current_assistant_msg["reasoning_content"] = pending_reasoning
        logger.debug(f"已附加 reasoning_content ({len(pending_reasoning)} 字符) 到 assistant 消息")
        pending_reasoning = None

    def _finalize_assistant():
        nonlocal current_assistant_msg
        if current_assistant_msg is not None:
            if current_assistant_msg.get("tool_calls"):
                current_assistant_msg.setdefault("content", "")
            # DeepSeek 要求 assistant 消息必须有 content 或 tool_calls
            if not current_assistant_msg.get("content") and not current_assistant_msg.get("tool_calls"):
                current_assistant_msg["content"] = ""
            messages.append(current_assistant_msg)
            current_assistant_msg = None

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = convert_content_response_to_chat(item.get("content"))

            if role == "assistant":
                _finalize_assistant()
                current_assistant_msg = {"role": role, "content": content}
                if item.get("name"):
                    current_assistant_msg["name"] = item["name"]
                _apply_pending_reasoning()
            else:
                # 非 assistant 消息：如果之前有未完成的 assistant 消息或 pending reasoning，先清理
                if pending_reasoning:
                    _apply_pending_reasoning()
                _finalize_assistant()
                msg = {"role": role, "content": content}
                if item.get("name"):
                    msg["name"] = item["name"]
                messages.append(msg)

        elif item_type == "function_call":
            if current_assistant_msg is None:
                current_assistant_msg = {"role": "assistant", "content": ""}
            _apply_pending_reasoning()
            if "tool_calls" not in current_assistant_msg:
                current_assistant_msg["tool_calls"] = []
            current_assistant_msg["tool_calls"].append({
                "id": item.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}")
                }
            })

        elif item_type == "function_call_output":
            _apply_pending_reasoning()
            _finalize_assistant()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", "")
            })

        elif item_type == "reasoning":
            pending_reasoning = _extract_reasoning_text(item)
            if pending_reasoning:
                logger.debug(f"从 reasoning 项提取文本 ({len(pending_reasoning)} 字符)，待附加到下一 assistant 消息")
            else:
                logger.debug("reasoning 项中未找到可提取的文本")

        elif item_type == "item_reference":
            logger.warning(f"item_reference 类型无法完美映射到 Chat API，已跳过: {item.get('id')}")

        else:
            logger.debug(f"未知的 input 项类型: {item_type}")

    _apply_pending_reasoning()
    _finalize_assistant()

    # 构建 Chat API 请求
    # 固定模型为 deepseek-v4-pro，因为上游仅支持 deepseek-v4-pro / deepseek-v4-flash
    chat_request: Dict[str, Any] = {
        "model": CONVERT_MODEL, #"deepseek-v4-pro",
        "messages": messages,
        "stream": response_body.get("stream", True),
    }

    # DeepSeek reasoner 模型要求所有 assistant 消息都携带 reasoning_content，
    # 若历史消息缺失该字段，补填空字符串以防止上游拒绝。
    _ensure_reasoning_consistency(messages, chat_request["model"])

    # max_output_tokens → max_tokens
    if "max_output_tokens" in response_body:
        chat_request["max_tokens"] = response_body["max_output_tokens"]

    # temperature, top_p 直接映射
    for param in ("temperature", "top_p", "stop", "seed", "user"):
        if param in response_body and response_body[param] is not None:
            chat_request[param] = response_body[param]

    # reasoning.effort → reasoning_effort
    reasoning = response_body.get("reasoning")
    if reasoning and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            has_reasoning = any(
                msg.get("reasoning_content")
                for msg in messages
                if msg.get("role") == "assistant"
            )
            is_reasoner = _is_reasoner_model(chat_request["model"])
            if has_reasoning or is_reasoner:
                chat_request["reasoning_effort"] = effort
            else:
                logger.warning(
                    f"reasoning_effort='{effort}' 已请求但既非推理模型也无 reasoning 历史，"
                    f"已移除 reasoning_effort 以避免上游拒绝"
                )

    # text.format → response_format
    text_config = response_body.get("text")
    if text_config and isinstance(text_config, dict):
        fmt = text_config.get("format")
        if fmt and isinstance(fmt, dict):
            fmt_type = fmt.get("type")
            if fmt_type == "json_object":
                chat_request["response_format"] = {"type": "json_object"}
            elif fmt_type == "json_schema":
                # DeepSeek API 不支持 json_schema 类型，降级为 json_object，
                # 同时将 schema 转为自然语言指令注入消息，引导模型按预期格式输出
                logger.info(
                    f"response_format json_schema (name='{fmt.get('name', '')}') "
                    f"降级为 json_object，schema 将注入为自然语言指令"
                )
                chat_request["response_format"] = {"type": "json_object"}
                _inject_schema_instruction(messages, fmt)

    # tools 格式转换
    # Response: [{"type": "function", "name": "x", ...}, {"type": "web_search", ...}]
    # Chat:     [{"type": "function", "function": {"name": "x", ...}}]
    #           + web_search → web_search_options (顶层参数)
    #           + file_search → 转为 function 工具或跳过
    tools = response_body.get("tools")
    if tools:
        converted_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type")

            if tool_type == "function" and "function" not in tool:
                converted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                })
            elif tool_type == "function":
                converted_tools.append(tool)

            elif tool_type == "web_search":
                # Response API web_search 工具 → Chat API web_search_options
                web_opts: Dict[str, Any] = {}
                if "user_location" in tool:
                    web_opts["user_location"] = tool["user_location"]
                if "search_context_size" in tool:
                    web_opts["search_context_size"] = tool["search_context_size"]
                if web_opts:
                    chat_request["web_search_options"] = web_opts
                logger.info(f"web_search 工具已转换为 web_search_options: {web_opts}")

            elif tool_type == "file_search":
                # Chat API 支持 file_search 类型工具（部分提供商）
                file_search_tool: Dict[str, Any] = {"type": "file_search"}
                if "vector_store_ids" in tool:
                    file_search_tool["vector_store_ids"] = tool["vector_store_ids"]
                if "max_num_results" in tool:
                    file_search_tool["max_num_results"] = tool["max_num_results"]
                converted_tools.append(file_search_tool)
                logger.info("file_search 工具已转换为 Chat API 格式")

            else:
                logger.warning(f"Chat API 不支持的工具类型 '{tool_type}'，已过滤")
        if converted_tools:
            chat_request["tools"] = converted_tools

    # tool_choice 映射: 过滤 Chat API 不支持的 built-in 工具类型
    tool_choice = response_body.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
            if tool_choice == "web_search" and chat_request.get("web_search_options"):
                # web_search 已通过 web_search_options 启用，"auto" 即可
                chat_request["tool_choice"] = "auto"
                logger.info("tool_choice='web_search' 已通过 web_search_options 处理，设为 'auto'")
            else:
                logger.warning(f"Chat API 不支持 tool_choice='{tool_choice}'，已替换为 'required'")
                chat_request["tool_choice"] = "required"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            # {type: "function", name: "xxx"} → {type: "function", function: {name: "xxx"}}
            if "function" not in tool_choice:
                chat_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")}
                }
            else:
                chat_request["tool_choice"] = tool_choice
        else:
            chat_request["tool_choice"] = tool_choice

    # parallel_tool_calls
    if "parallel_tool_calls" in response_body:
        chat_request["parallel_tool_calls"] = response_body["parallel_tool_calls"]

    # 诊断: reasoning 状态摘要
    _log_reasoning_summary(messages, chat_request)

    return chat_request


def _log_reasoning_summary(messages: List[Dict[str, Any]], chat_request: Dict[str, Any]) -> None:
    """记录 reasoning_content 转换摘要，便于排查"""
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return
    with_rc = [m for m in assistant_msgs if m.get("reasoning_content")]
    model = chat_request.get("model", "")
    effort = chat_request.get("reasoning_effort", "")
    logger.info(
        f"reasoning 转换摘要: model='{model}', reasoning_effort='{effort}', "
        f"assistant 消息 {len(assistant_msgs)} 条, "
        f"携带 reasoning_content {len(with_rc)} 条, "
        f"缺失 {len(assistant_msgs) - len(with_rc)} 条"
    )
    if with_rc:
        for m in with_rc:
            rc = m.get("reasoning_content", "")
            logger.debug(f"  reasoning_content ({len(rc)} 字符): {rc[:100]}...")


# ==================== 响应转换: Chat SSE → Response SSE ====================

class ChatStreamProcessor:
    """处理 Chat API 的 SSE 流并转换为 Response API 的 SSE 事件

    Response API 中 reasoning 是独立的 output item (type="reasoning")，位于 message 之前。
    Chat API 的 reasoning_content delta 被映射为独立的 reasoning 输出项，确保客户端能
    在后续请求中将其作为 reasoning 输入项正确传回。
    """

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.created_at = int(time.time())

        # 全局状态
        self.response_started = False
        self.next_output_index = 0
        self.usage: Optional[Dict[str, Any]] = None

        # 推理输出状态 (Response API 中为独立 output item, type="reasoning")
        self.reasoning_active = False
        self.reasoning_item_id: Optional[str] = None
        self.reasoning_output_index = -1
        self.reasoning_summary_index = -1
        self.reasoning_accumulated = ""

        # 消息输出状态
        self.msg_active = False
        self.msg_item_id: Optional[str] = None
        self.msg_output_index = -1
        self.msg_content_index = -1
        self.msg_accumulated = ""

        # 函数调用状态: index -> {...}
        self.fc_states: Dict[int, Dict[str, Any]] = {}
        self.fc_order: List[int] = []

        # 输出项列表
        self.output_items: List[Dict[str, Any]] = []

    # ---------- 公共方法 ----------

    def process_chat_chunk(self, chunk_data: Dict[str, Any]) -> List[str]:
        """处理单个 Chat API SSE chunk，返回 Response API SSE 事件列表"""
        events: List[str] = []

        choices = chunk_data.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        chunk_usage = chunk_data.get("usage")
        if chunk_usage:
            self.usage = chunk_usage

        # --- 推理内容 ---
        if delta.get("reasoning_content"):
            events.extend(self._handle_reasoning(delta["reasoning_content"]))

        # --- 工具调用 ---
        if delta.get("tool_calls"):
            events.extend(self._handle_tool_calls(delta["tool_calls"]))

        # --- 文本内容 ---
        if delta.get("content"):
            events.extend(self._handle_content(delta["content"]))

        # --- 角色 delta ---
        if delta.get("role") and not delta.get("content") and not delta.get("tool_calls"):
            self._ensure_response_started(events)

        # --- 完成处理 ---
        if finish_reason:
            events.extend(self._handle_finish(finish_reason))

        return events

    # ---------- 内部处理方法 ----------

    def _handle_reasoning(self, text: str) -> List[str]:
        events: List[str] = []
        self._ensure_response_started(events)
        self._ensure_reasoning_active(events)
        self.reasoning_accumulated += text
        events.append(format_sse_event("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_output_index,
            "summary_index": self.reasoning_summary_index,
            "delta": text,
        }))
        return events

    def _handle_content(self, text: str) -> List[str]:
        events: List[str] = []
        self._ensure_response_started(events)
        self._finalize_reasoning(events)
        self._finalize_all_function_calls(events)
        self._ensure_message_active(events)
        self.msg_accumulated += text

        events.append(format_sse_event("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": self.msg_item_id,
            "output_index": self.msg_output_index,
            "content_index": self.msg_content_index,
            "delta": text,
        }))
        return events

    def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        events: List[str] = []
        self._ensure_response_started(events)

        for tc in tool_calls:
            idx = tc.get("index", 0)

            if "id" in tc and tc["id"]:
                # 新的工具调用 — 先关闭推理和消息，再打开 function_call
                self._finalize_reasoning(events)
                self._finalize_message(events)
                events.extend(self._start_function_call(idx, tc))

            elif "function" in tc and "arguments" in tc["function"]:
                delta_args = tc["function"]["arguments"]
                fc = self.fc_states.get(idx)
                if fc:
                    fc["arguments"] += delta_args
                    events.append(format_sse_event("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": fc["item_id"],
                        "output_index": fc["output_index"],
                        "delta": delta_args,
                    }))

        return events

    def _handle_finish(self, finish_reason: Optional[str]) -> List[str]:
        events: List[str] = []

        # 关闭推理、消息、函数调用
        self._finalize_reasoning(events)
        self._finalize_message(events)
        self._finalize_all_function_calls(events)

        # response.completed
        output_for_event = self._build_output_list()
        completed_data: Dict[str, Any] = {
            "type": "response.completed",
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "completed",
                "model": self.model,
                "output": output_for_event,
            }
        }

        response_usage = convert_chat_usage_to_response_usage(self.usage)
        if response_usage:
            completed_data["response"]["usage"] = response_usage

        events.append(format_sse_event("response.completed", completed_data))
        events.append(format_sse_event("response.done", {
            "type": "response.done",
            "response": completed_data["response"],
            **({"usage": response_usage} if response_usage else {}),
        }))
        return events

    # ---------- 状态管理: 推理输出 ----------

    def _ensure_reasoning_active(self, events: List[str]):
        if self.reasoning_active:
            return
        self.reasoning_active = True
        self.reasoning_output_index = self._next_output_index()
        self.reasoning_summary_index = 0
        self.reasoning_item_id = generate_item_id("rs")
        self.reasoning_accumulated = ""

        events.append(format_sse_event("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": self.reasoning_output_index,
            "item": {
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            }
        }))
        events.append(format_sse_event("response.reasoning_summary_part.added", {
            "type": "response.reasoning_summary_part.added",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_output_index,
            "summary_index": self.reasoning_summary_index,
            "part": {"type": "summary_text", "text": ""},
        }))

    def _finalize_reasoning(self, events: List[str]):
        if not self.reasoning_active:
            return
        self.reasoning_active = False

        item = {
            "id": self.reasoning_item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [{
                "type": "summary_text",
                "text": self.reasoning_accumulated,
            }],
        }
        self.output_items.append(item)

        events.append(format_sse_event("response.reasoning_summary_part.done", {
            "type": "response.reasoning_summary_part.done",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_output_index,
            "summary_index": self.reasoning_summary_index,
            "part": {"type": "summary_text", "text": self.reasoning_accumulated},
        }))
        events.append(format_sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": self.reasoning_output_index,
            "item": item,
        }))

    # ---------- 状态管理辅助 ----------

    def _ensure_response_started(self, events: List[str]):
        if not self.response_started:
            self.response_started = True
            events.append(format_sse_event("response.created", {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at,
                    "status": "in_progress",
                    "model": self.model,
                    "output": [],
                }
            }))

    def _ensure_message_active(self, events: List[str]):
        if self.msg_active:
            return
        self.msg_active = True
        self.msg_output_index = self._next_output_index()
        self.msg_content_index = 0
        self.msg_item_id = generate_item_id("msg")
        self.msg_accumulated = ""

        events.append(format_sse_event("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": self.msg_output_index,
            "item": {
                "id": self.msg_item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
        }))
        events.append(format_sse_event("response.content_part.added", {
            "type": "response.content_part.added",
            "item_id": self.msg_item_id,
            "output_index": self.msg_output_index,
            "content_index": self.msg_content_index,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }))

    def _finalize_message(self, events: List[str]):
        if not self.msg_active:
            return
        self.msg_active = False

        item = {
            "id": self.msg_item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": self.msg_accumulated,
                "annotations": [],
            }],
        }
        self.output_items.append(item)

        events.append(format_sse_event("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": self.msg_item_id,
            "output_index": self.msg_output_index,
            "content_index": self.msg_content_index,
            "part": {"type": "output_text", "text": self.msg_accumulated, "annotations": []},
        }))
        events.append(format_sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": self.msg_output_index,
            "item": item,
        }))

    def _start_function_call(self, idx: int, tc: Dict[str, Any]) -> List[str]:
        call_id = tc["id"]
        func = tc.get("function", {})
        func_name = func.get("name", "")
        func_args = func.get("arguments", "")

        fc_item_id = generate_item_id("fc")
        fc_output_index = self._next_output_index()

        self.fc_states[idx] = {
            "id": call_id,
            "name": func_name,
            "arguments": func_args,
            "item_id": fc_item_id,
            "output_index": fc_output_index,
            "done": False,
        }
        self.fc_order.append(idx)

        events = [
            format_sse_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": fc_output_index,
                "item": {
                    "id": fc_item_id,
                    "type": "function_call",
                    "call_id": call_id,
                    "name": func_name,
                    "arguments": "",
                    "status": "in_progress",
                }
            })
        ]

        if func_args:
            events.append(format_sse_event("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": fc_item_id,
                "output_index": fc_output_index,
                "delta": func_args,
            }))

        return events

    def _finalize_function_call(self, idx: int, events: List[str]):
        fc = self.fc_states.get(idx)
        if not fc or fc.get("done"):
            return
        fc["done"] = True

        item = {
            "id": fc["item_id"],
            "type": "function_call",
            "call_id": fc["id"],
            "name": fc["name"],
            "arguments": fc["arguments"],
            "status": "completed",
        }
        self.output_items.append(item)

        events.append(format_sse_event("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": fc["item_id"],
            "output_index": fc["output_index"],
            "arguments": fc["arguments"],
        }))
        events.append(format_sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": fc["output_index"],
            "item": item,
        }))

    def _finalize_all_function_calls(self, events: List[str]):
        for idx in list(self.fc_order):
            self._finalize_function_call(idx, events)

    def _next_output_index(self) -> int:
        idx = self.next_output_index
        self.next_output_index += 1
        return idx

    def _build_output_list(self) -> List[Dict[str, Any]]:
        output = list(self.output_items)

        if self.reasoning_active and self.reasoning_accumulated:
            output.append({
                "id": self.reasoning_item_id or generate_item_id("rs"),
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": self.reasoning_accumulated}],
            })

        if self.msg_active and self.msg_accumulated:
            output.append({
                "id": self.msg_item_id or generate_item_id("msg"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": self.msg_accumulated,
                    "annotations": [],
                }],
            })

        for idx in self.fc_order:
            fc = self.fc_states.get(idx)
            if fc and not fc.get("done"):
                output.append({
                    "id": fc["item_id"],
                    "type": "function_call",
                    "call_id": fc["id"],
                    "name": fc["name"],
                    "arguments": fc["arguments"],
                    "status": "completed",
                })
                break  # 未完成的最多一个

        return output

    def get_accumulated_response(self) -> Dict[str, Any]:
        """获取完整响应（用于非流式模式）"""
        output = self._build_output_list()
        response_usage = convert_chat_usage_to_response_usage(self.usage)

        result: Dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output,
        }
        if response_usage:
            result["usage"] = response_usage

        return result


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=KEEPALIVE_EXPIRY
    )
    timeout = httpx.Timeout(
        connect=30.0,
        read=DEFAULT_TIMEOUT,
        write=30.0,
        pool=POOL_TIMEOUT
    )
    app.state.http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=True
    )
    logger.info(f"HTTP 客户端初始化: max_connections={MAX_CONNECTIONS}, keepalive={MAX_KEEPALIVE_CONNECTIONS}, upstream={CHAT_API_BASE}")
    yield
    await app.state.http_client.aclose()
    logger.info("HTTP 客户端已关闭")


app = FastAPI(
    title="Chat to Response API Reverse Proxy",
    description="将 OpenAI Response 协议请求转发为 Chat 协议请求，响应封装回 Response 协议",
    version="1.0.0",
    lifespan=lifespan
)

# 透传时排除的请求头
PASSTHROUGH_REQUEST_EXCLUDED_HEADERS = {
    "authorization", "connection", "content-length", "host", "transfer-encoding",
}

# 透传时排除的响应头
PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS = {
    "connection", "content-encoding", "content-length", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade",
}


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Extract bearer token from Authorization header, fall back to DEFAULT_API_KEY env var."""
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "", 1)
    if authorization:
        return authorization
    fallback = os.getenv("DEFAULT_API_KEY", "").strip() or os.getenv("CHAT_API_KEY", "").strip()
    if fallback:
        logger.info("未提供 Authorization header，使用 DEFAULT_API_KEY 作为 fallback")
        return fallback
    logger.warning("Missing Authorization header and no DEFAULT_API_KEY configured")
    raise HTTPException(status_code=401, detail="Missing Authorization header")


def build_passthrough_request_headers(request: Request, token: str) -> Dict[str, str]:
    """复制客户端请求头用于上游透传，排除逐跳头"""
    headers: Dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in PASSTHROUGH_REQUEST_EXCLUDED_HEADERS:
            continue
        headers[key] = value
    headers["Authorization"] = f"Bearer {token}"
    return headers


def build_passthrough_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    """过滤上游响应头用于下游响应"""
    filtered: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS:
            continue
        filtered[key] = value
    return filtered


@app.post("/v1/chat/completions")
async def chat_completions_passthrough(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Chat Completions 接口 — 直接透传到上游 Chat API，不做协议转换"""
    token = extract_bearer_token(authorization)
    logger.debug(f"Token: {token[:10]}..." if len(token) > 10 else f"Token: {token}")

    raw_body = await request.body()
    is_stream_request = "text/event-stream" in request.headers.get("accept", "").lower()

    if raw_body:
        try:
            request_json = json.loads(raw_body)
            is_stream_request = bool(request_json.get("stream")) or is_stream_request
            logger.info(f"透传 /v1/chat/completions 请求:\n{json.dumps(request_json, ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            logger.info("透传 /v1/chat/completions 请求: <non-json body>")
    else:
        logger.info("透传 /v1/chat/completions 请求: <empty body>")

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_url = f"{CHAT_API_BASE}/chat/completions"
    upstream_headers = build_passthrough_request_headers(request, token)
    upstream_params = list(request.query_params.multi_items())
    stream_timeout = httpx.Timeout(
        connect=30.0, read=STREAM_READ_TIMEOUT, write=30.0, pool=POOL_TIMEOUT
    )

    logger.info(f"Passthrough /v1/chat/completions -> {upstream_url}, stream={is_stream_request}")

    if is_stream_request:
        stream_context = client.stream(
            "POST",
            upstream_url,
            headers=upstream_headers,
            params=upstream_params,
            content=raw_body,
            timeout=stream_timeout
        )
        upstream_response = None

        try:
            upstream_response = await stream_context.__aenter__()
            logger.info(f"上游 Chat API 流式状态码: {upstream_response.status_code}")
            response_headers = build_passthrough_response_headers(upstream_response.headers)

            async def stream_generator():
                try:
                    async for chunk in upstream_response.aiter_raw():
                        if chunk:
                            yield chunk
                except asyncio.CancelledError:
                    logger.warning("/v1/chat/completions 流被客户端取消")
                    raise
                finally:
                    await stream_context.__aexit__(None, None, None)

            return StreamingResponse(
                stream_generator(),
                status_code=upstream_response.status_code,
                headers=response_headers
            )
        except httpx.TimeoutException:
            if upstream_response is not None:
                await upstream_response.aclose()
            logger.error("/v1/chat/completions 流式请求超时")
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "Request timeout", "type": "timeout_error"}}
            )
        except Exception:
            if upstream_response is not None:
                await upstream_response.aclose()
            raise

    try:
        upstream_response = await client.post(
            upstream_url,
            headers=upstream_headers,
            params=upstream_params,
            content=raw_body,
            timeout=DEFAULT_TIMEOUT
        )
        logger.info(f"上游 Chat API 非流式状态码: {upstream_response.status_code}")
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=build_passthrough_response_headers(upstream_response.headers)
        )
    except httpx.TimeoutException:
        logger.error("/v1/chat/completions 非流式请求超时")
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timeout", "type": "timeout_error"}}
        )
    except Exception as e:
        logger.error(f"/v1/chat/completions 透传失败: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


@app.post("/v1/responses")
async def responses_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Response API 接口 — 将请求转换为 Chat API 格式，调用 Chat API，再将响应封装回 Response API 格式
    """
    token = extract_bearer_token(authorization)
    logger.debug(f"Token: {token[:10]}..." if len(token) > 10 else f"Token: {token}")

    try:
        body = await request.json()
        logger.info(f"收到 Response API 请求:\n{json.dumps(body, ensure_ascii=False, indent=2)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # 转换请求: Response → Chat
    chat_request = convert_response_to_chat_request(body)
    logger.info(f"转换后的 Chat API 请求:\n{json.dumps(chat_request, ensure_ascii=False, indent=2)}")

    response_id = generate_response_id()
    is_stream = bool(body.get("stream", True))
    model = body.get("model", "")

    client: httpx.AsyncClient = request.app.state.http_client
    chat_url = f"{CHAT_API_BASE}/chat/completions"
    logger.info(f"转发到: {chat_url}, stream={is_stream}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    if is_stream:
        return await _handle_stream(client, chat_url, headers, chat_request, response_id, model)
    else:
        return await _handle_non_stream(client, chat_url, headers, chat_request, response_id, model)


async def _handle_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    response_id: str,
    model: str,
) -> StreamingResponse:
    """处理流式响应"""

    async def stream_generator():
        processor = ChatStreamProcessor(response_id, model)
        start_time = time.monotonic()

        try:
            logger.debug(f"开始流式请求到 {url}")
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=request_body,
                timeout=httpx.Timeout(
                    connect=30.0, read=STREAM_READ_TIMEOUT, write=30.0, pool=POOL_TIMEOUT
                )
            ) as response:
                logger.info(f"上游 Chat API 状态码: {response.status_code}")

                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="ignore")
                    logger.error(f"上游错误响应: {error_text}")

                    error_data = {"error": {"message": error_text, "type": "upstream_error"}}
                    try:
                        error_data = json.loads(error_text)
                    except json.JSONDecodeError:
                        pass

                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    return

                async for line in response.aiter_lines():
                    if STREAM_MAX_DURATION > 0 and (time.monotonic() - start_time) > STREAM_MAX_DURATION:
                        logger.error(f"流式请求超时: {STREAM_MAX_DURATION}s")
                        yield format_sse_event("error", {
                            "error": {"message": "Stream max duration exceeded", "type": "timeout_error"}
                        })
                        return

                    line = line.strip()
                    if not line:
                        continue

                    logger.debug(f"收到 Chat API 数据行: {line[:200]}..." if len(line) > 200 else f"收到 Chat API 数据行: {line}")

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            logger.info("收到 [DONE]，流结束")
                            break

                        try:
                            chunk_data = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                            continue

                        # 检查错误
                        if "error" in chunk_data:
                            logger.error(f"Chat API 错误: {json.dumps(chunk_data, ensure_ascii=False)}")
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            return

                        events = processor.process_chat_chunk(chunk_data)
                        for event in events:
                            if event:
                                yield event

                # 流结束但没有 finish_reason，补发完成事件
                accumulated = processor.get_accumulated_response()
                logger.info(f"流完成: {json.dumps(accumulated, ensure_ascii=False)}")

        except httpx.TimeoutException:
            logger.error("请求超时")
            yield format_sse_event("error", {
                "error": {"message": "Request timeout", "type": "timeout_error"}
            })
        except httpx.RemoteProtocolError as e:
            logger.error(f"远程协议错误: {e}")
            yield format_sse_event("error", {
                "error": {"message": f"Connection reset: {e}", "type": "connection_error"}
            })
        except asyncio.CancelledError:
            logger.warning(f"流式请求被取消: response_id={response_id}")
            return
        except GeneratorExit:
            logger.warning(f"生成器退出（客户端断开）: response_id={response_id}")
            return
        except Exception as e:
            logger.error(f"流式处理异常: {e}\n{traceback.format_exc()}")
            yield format_sse_event("error", {
                "error": {"message": str(e), "type": "internal_error"}
            })
        finally:
            logger.debug(f"流式生成器结束: response_id={response_id}")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _handle_non_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    response_id: str,
    model: str,
) -> JSONResponse:
    """处理非流式响应 — 使用流式请求收集完整响应后返回"""
    processor = ChatStreamProcessor(response_id, model)

    try:
        logger.debug(f"开始非流式请求到 {url}")
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=request_body,
            timeout=DEFAULT_TIMEOUT
        ) as response:
            logger.info(f"上游 Chat API 状态码: {response.status_code}")

            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                logger.error(f"上游错误响应: {error_text}")

                try:
                    error_data = json.loads(error_text)
                except json.JSONDecodeError:
                    error_data = {"error": {"message": error_text, "type": "upstream_error"}}

                return JSONResponse(
                    status_code=response.status_code,
                    content=error_data
                )

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "error" in chunk_data:
                        return JSONResponse(status_code=502, content=chunk_data)

                    processor.process_chat_chunk(chunk_data)

        result = processor.get_accumulated_response()
        logger.info(f"返回完整响应: {json.dumps(result, ensure_ascii=False)}")
        return JSONResponse(content=result)

    except httpx.TimeoutException:
        logger.error("请求超时")
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timeout", "type": "timeout_error"}}
        )
    except Exception as e:
        logger.error(f"非流式处理异常: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


@app.get("/health")
async def health_check(request: Request):
    client: httpx.AsyncClient = request.app.state.http_client
    pool_status = {}
    try:
        if hasattr(client, '_transport') and client._transport:
            transport = client._transport
            if hasattr(transport, '_pool'):
                pool = transport._pool
                pool_status = {
                    "connections_in_pool": len(pool._connections) if hasattr(pool, '_connections') else "unknown"
                }
    except Exception as e:
        pool_status = {"error": str(e)}

    return {
        "status": "ok",
        "service": "chat-to-response-reverse-proxy",
        "upstream": CHAT_API_BASE,
        "pool_status": pool_status,
        "config": {
            "max_connections": MAX_CONNECTIONS,
            "max_keepalive_connections": MAX_KEEPALIVE_CONNECTIONS,
            "keepalive_expiry": KEEPALIVE_EXPIRY,
            "default_timeout": DEFAULT_TIMEOUT,
            "stream_read_timeout": STREAM_READ_TIMEOUT,
            "stream_max_duration": STREAM_MAX_DURATION,
        }
    }


@app.get("/v1/models")
async def list_models(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    token = extract_bearer_token(authorization)
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        response = await client.get(
            f"{CHAT_API_BASE}/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e)}}
        )


# 同时注册不含 /v1 前缀的接口
app.add_api_route("/chat/completions", chat_completions_passthrough, methods=["POST"])
app.add_api_route("/responses", responses_endpoint, methods=["POST"])
app.add_api_route("/models", list_models, methods=["GET"])


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("REVERSE_PORT", "8001"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting Chat to Response Reverse Proxy on {host}:{port}")
    print(f"Upstream Chat API: {CHAT_API_BASE}")

    uvicorn.run(app, host=host, port=port)
