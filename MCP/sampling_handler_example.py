import asyncio, warnings, logging, json
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import copy
import uuid
from copy import deepcopy

from google.adk.models import LlmResponse
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StreamableHTTPConnectionParams
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.genai import types

from .adk_agent_setup.adk_agent_config import MCP_SERVER_URL, DESIRED_TOOL_NAMES, GCP_MCP_LOCAL
from .llm.llm import get_headers, get_lite_llm, get_summary_client, get_summary_model
from .resources.prompt import SYSTEM_INSTRUCTIONS, SUMMARY_INSTRUCTIONS
from .resources.data_storing import append_short_term, append_long_term, load_short_term_summaries
from .app.headers import get_current_headers

##for local mcp run in GCP QA
if GCP_MCP_LOCAL:
    from .token_manager import get_auth_header
    local_headers = get_auth_header()

current_file_dir = Path(__file__).parent
project_root = current_file_dir.parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("google_genai.types").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")
warnings.filterwarnings("ignore", message=r"auth_config or auth_config\.auth_scheme is missing")

def get_uid() -> str:
    """Helper to fetch the current user_id safely for logging."""
    try:
        return get_current_headers().get('user_id', '1')
    except Exception:
        return '1'

logger.info(f"[INIT][{time.time():.6f}][User:{get_uid()}] Setting up LLM...")
tenant_fallback = "5dd55a13-1176-49f1-8091-30f722b427ba"
_FALLBACK_MODEL = "openai/gpt-5"
lite_llm_instance = get_lite_llm(model= _FALLBACK_MODEL)
logger.info(f"[INIT][{time.time():.6f}][User:{get_uid()}] ADK LLM + headers are ready.")

from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext

async def my_sampling_handler(messages, params=None, context=None):
    """
    Removed type hints to bypass library validation errors.
    We handle the structure manually inside the function.
    """
    print(f"--- SAMPLING TRIGGERED ---")

    try:
        msg_list = messages if isinstance(messages, list) else []
        prompt_parts = []
        for m in msg_list:
            role = getattr(m, 'role', 'user')
            content_obj = getattr(m, 'content', None)
            text = getattr(content_obj, 'text', str(content_obj))
            prompt_parts.append(f"{role}: {text}")

        prompt = "\n".join(prompt_parts) if prompt_parts else "No context."

        openai_client = get_summary_client()
        summary_model = get_summary_model()

        resp = openai_client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are a reasoning assistant. Provide a good 100 words ackowledgement."},
                {"role": "user", "content": prompt},
            ],
        )

        ai_text = resp.choices[0].message.content.strip()

        # 3. Return the exact JSON structure the MCP spec expects
        return {
            "model": summary_model,
            "role": "assistant",
            "content": {
                "type": "text",
                "text": ai_text
            },
            "stopReason": "endTurn"
        }

    except Exception as e:
        print(f"CRITICAL ERROR in handler: {e}")
        # Return a valid failure structure so the tool doesn't hang
        return {
            "role": "assistant",
            "content": {"type": "text", "text": "Error in sampling logic."},
            "stopReason": "endTurn"
        }

class IsolatedMcpToolset(BaseToolset):
    """
    MCPToolset that creates isolated connections per runner to avoid cancel scope conflicts.
    """
    def __init__(self, user_id: str, headers: dict, desired_tool_names: List[str]):
        self.user_id = user_id
        self.headers = headers

        # [OPTIMIZATION 2] Force Keep-Alive Headers
        # Explicitly signals intention to reuse the TCP/SSL socket.
        self.headers["Connection"] = "keep-alive"

        self.desired_tool_names = set(desired_tool_names)
        self._mcp_toolset: Optional[McpToolset] = None
        self._filtered_tools: Optional[List[BaseTool]] = None
        self._session_id = str(uuid.uuid4())
        self._is_active = False
        super().__init__()
        logger.info(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Created for user {user_id}, session {self._session_id}")

    async def initialize(self) -> None:
        if self._is_active:
            logger.debug(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Toolset active for session {self._session_id}")
            return

        try:
            if GCP_MCP_LOCAL:
                self.headers.update(local_headers)
            self._mcp_toolset = McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=MCP_SERVER_URL,
                    headers=self.headers,
                    timeout=300
                ),
                sampling_callback=my_sampling_handler
            )

            logger.info(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Initializing for user {self.user_id}, session {self._session_id}")
            all_tools = await self._mcp_toolset.get_tools(None)

            self._filtered_tools = [
                t for t in all_tools
                if getattr(t, "name", "") in self.desired_tool_names
            ]

            self._is_active = True
            logger.info(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Initialized {len(self._filtered_tools)} tools for session {self._session_id}")

        except Exception as init_error:
            logger.error(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Init failed for session {self._session_id}: {init_error}")
            self._is_active = False
            self._filtered_tools = []
            self._mcp_toolset = None
            raise

    async def get_tools(self, context: ToolContext | None = None) -> List[BaseTool]:
        if not self._is_active:
            try:
                await self.initialize()
            except Exception as init_error:
                logger.error(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Init failed during get_tools: {init_error}")
                return []
        return self._filtered_tools or []

    async def aclose(self) -> None:
        """Async cleanup - abandon problematic sessions gracefully."""
        if self._is_active and self._mcp_toolset:
            logger.info(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Cleaning session {self._session_id}")
            try:
                if hasattr(self._mcp_toolset, 'aclose'):
                    await self._mcp_toolset.aclose()
                elif hasattr(self._mcp_toolset, 'close'):
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._mcp_toolset.close)
                logger.debug(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Session {self._session_id} closed")
            except Exception as close_error:
                if "cancel scope" in str(close_error).lower():
                    logger.debug(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Expected cancel scope for {self._session_id}: {close_error}")
                else:
                    logger.warning(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Cleanup error {self._session_id}: {close_error}")
            finally:
                self._mcp_toolset = None
                self._filtered_tools = None
                self._is_active = False

    def __del__(self):
        if self._is_active and self._mcp_toolset:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    asyncio.run(self.aclose())
            except:
                pass

# Cache: one toolset per runner
_RUNNER_TOOLSETS: Dict[str, IsolatedMcpToolset] = {}

def get_isolated_toolset_for_runner(runner_id: str, user_id: str, headers: dict) -> IsolatedMcpToolset:
    if runner_id not in _RUNNER_TOOLSETS:
        toolset = IsolatedMcpToolset(user_id, headers, DESIRED_TOOL_NAMES)
        _RUNNER_TOOLSETS[runner_id] = toolset
        logger.info(f"[ISO_MCP][{time.time():.6f}][User:{get_uid()}] Assigned toolset to runner {runner_id}")
    return _RUNNER_TOOLSETS[runner_id]


class GracefulCancelScope:
    """Manages cancellation scope for clean stop button handling."""
    def __init__(self, cancel_event: Optional[asyncio.Event] = None):
        self.cancel_event = cancel_event or asyncio.Event()
        self._cancelled = False
        self._tasks = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._cancelled:
            await self.cancel()

    async def cancel(self):
        if not self._cancelled:
            self._cancelled = True
            self.cancel_event.set()
            await asyncio.sleep(0.1)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

def safe_serialize_result(result):
    # If result looks like SDK object with typical attributes, extract contents
    if hasattr(result, 'structuredContent'):
        try:
            return result.structuredContent
        except Exception:
            pass  # fall through if any error

    if hasattr(result, 'content'):
        try:
            return result.content
        except Exception:
            pass

    if isinstance(result, str):
        return result

    # If dict or list, try JSON dump safely
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result)
        except Exception:
            return str(result)
    try:
        return json.dumps(result)
    except Exception:
        return str(result)

def event_to_dict(event: Event, arrival_ts: float = None) -> Dict[str, Any]:
    ts = arrival_ts if arrival_ts is not None else getattr(event, 'timestamp', time.time())

    if not hasattr(event, 'content') or not event.content or not event.content.parts:
        return {}

    role = getattr(event.content, 'role', 'unknown')
    # Use loop to handle cases where multiple parts (text + tool) exist in one event
    parts = event.content.parts

    for part in parts:
        if hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            name = getattr(fc, 'name', None)
            args = getattr(fc, 'args', None)
            call_id = getattr(fc, 'id', None)
            if name:
                logger.info(f"[EVENT][{ts:.6f}][User:{get_uid()}] Tool call -> {name} (ID: {call_id}) args: {args}")
                return {
                    "type": "tool_call",
                    "tool": name,
                    "arguments": str(args),
                    "call_id": call_id,
                    "timestamp": ts,
                    "_event_id": call_id
                }

        if hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            name = getattr(fr, 'name', None)
            resp_id = getattr(fr, 'id', None)
            response = getattr(fr, 'response', {})

            result = response
            if isinstance(result, dict):
                if 'structuredContent' in result:
                    result = safe_serialize_result(result['structuredContent'])
                elif 'content' in result:
                    result = safe_serialize_result(result['content'])
                else:
                    result = safe_serialize_result(result)
            else:
                result = safe_serialize_result(result)

            if name and result:
                logger.info(f"[EVENT][{ts:.6f}][User:{get_uid()}] Tool response <- {name} (Result: {str(result)[:500]}....)")
                return {
                    "type": "tool_response",
                    "tool": name,
                    "call_id": resp_id,
                    "result": str(result),
                    "timestamp": ts,
                    "_event_id": resp_id
                }

        # 3. Handle TEXT Messages
        if hasattr(part, 'text') and part.text:
            text = str(part.text).strip()
            if text:
                if role == 'user':
                    logger.info(f"[EVENT][{ts:.6f}][User:{get_uid()}] USER -> {text[:300]}....")
                    return {"type": "user_message", "message": text, "timestamp": ts}
                if role == 'model':
                    logger.info(f"[EVENT][{ts:.6f}][User:{get_uid()}] MODEL -> {text[:300]}....")
                    is_partial = not event.is_final_response()
                    return {
                        "type": "ai_message",
                        "message": text,
                        "timestamp": ts,
                        "_partial": is_partial,
                        "_event_id": getattr(event, 'invocation_id', f"model_{ts}")
                    }

        sc = getattr(part, 'structured', getattr(part, 'structuredContent', None))
        if isinstance(sc, dict):
            name = sc.get('name') or sc.get('tool')
            result = sc.get('result') or sc.get('output')
            if name and result is not None:
                logger.info(f"[EVENT][{ts:.6f}][User:{get_uid()}] Tool response (structured) <- {name} -> {str(result)[:500]}....")
                return {
                    "type": "tool_response",
                    "tool": name,
                    "result": str(result),
                    "timestamp": ts
                }

    # Handle System Role
    if role == 'system':
        part = parts[0] if parts else None
        text = str(part.text).strip() if part and hasattr(part, 'text') else ""
        logger.info(f"[EVENT][{ts:.6f}][User:{get_uid()}] SYSTEM -> {text[:200]}....")
        return {"type": "system_event", "message": text, "timestamp": ts}

    return {}

def deduplicate_ai_messages(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate AI messages based on content and timing."""
    processed_events = []
    seen_ai_content = {}

    for i, event in enumerate(events):
        if event.get("type") == "ai_message":
            content = event.get("message", "").strip()
            if not content:
                continue

            content_hash = hash(content[:200])
            ts = event.get("timestamp", 0)
            window_key = (content_hash, int(ts))

            if window_key in seen_ai_content:
                prev_index = seen_ai_content[window_key]
                prev_event = events[prev_index]

                if (prev_event.get("_partial", False) and
                        not event.get("_partial", False) and
                        prev_event.get("message") == content):
                    processed_events[prev_index] = event
                    continue
                elif not prev_event.get("_partial", False) and not event.get("_partial", False):
                    logger.debug(f"[DEDUPE][{time.time():.6f}][User:{get_uid()}] Skipping duplicate AI message: {content[:50]}...")
                    continue

            seen_ai_content[window_key] = i
            clean_event = event.copy()
            clean_event.pop("_partial", None)
            clean_event.pop("_event_id", None)
            processed_events.append(clean_event)

        else:
            processed_events.append(event)

    return processed_events

def has_incomplete_tool_calls(events: List[Dict[str, Any]]) -> bool:
    """Detect unresolved tool calls."""
    tool_calls = [e for e in events if e.get("type") == "tool_call" and e.get("call_id")]
    tool_responses = [e for e in events if e.get("type") == "tool_response" and e.get("call_id")]

    call_ids = {e.get("call_id") for e in tool_calls}
    response_ids = {e.get("call_id") for e in tool_responses}

    unresolved = call_ids - response_ids
    if unresolved:
        logger.warning(f"[INCOMPLETE][{time.time():.6f}][User:{get_uid()}] Found {len(unresolved)} unresolved tool calls: {list(unresolved)}")
    return len(unresolved) > 0

def load_past_context() -> List[Dict[str, Any]]:
    """Load past context using current headers for tenant/user isolation."""
    ctx = get_current_headers()
    summaries = load_short_term_summaries(
        ctx["tenant_id"],
        ctx["user_id"],
        ctx["role_id"],
        ctx["workspace_id"]
    )
    logger.info(f"[MEM][{time.time():.6f}][User:{get_uid()}] Loaded {len(summaries)} summaries for tenant={ctx['tenant_id']}, user={ctx['user_id']}")

    if summaries:
        logger.info(f"[MEM][{time.time():.6f}][User:{get_uid()}] Sample summaries for tenant={ctx['tenant_id']}, user={ctx['user_id']}:")
        for s in summaries[:2]:
            logger.info(f"   [User:{get_uid()}] - {s.get('summary', '')[:300]}... (session={s.get('session_id')})")
    else:
        return []

    recent = summaries[:2]
    return recent


class AIAgent:
    def __init__(self, app_name="PowerAssist"):
        self.app_name = app_name
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._session_service_cache: Dict[str, InMemorySessionService] = {}
        self._active_runners: Dict[str, Runner] = {}
        logger.info(f"[AGENT][{time.time():.6f}][User:{get_uid()}] Initialized with app_name={app_name}")

    def _get_session_service(self, user_id: str) -> InMemorySessionService:
        if user_id not in self._session_service_cache:
            svc = InMemorySessionService()
            self._session_service_cache[user_id] = svc
        return self._session_service_cache[user_id]

    async def _safe_close_runner(self, runner: Optional[Runner], reason: str = "normal") -> None:
        if not runner:
            return

        runner_id = id(runner)
        if runner_id not in self._active_runners:
            return

        try:
            logger.info(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Runner {runner_id}: {reason}")

            if runner_id in _RUNNER_TOOLSETS:
                toolset = _RUNNER_TOOLSETS[str(runner_id)]
                logger.info(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Cleaning MCP session {str(toolset._session_id)} for {reason}")
                toolset._is_active = False
                toolset._mcp_toolset = None
                if hasattr(toolset, 'aclose'):
                    try:
                        await toolset.aclose()
                    except Exception as tool_err:
                        if "cancel scope" not in str(tool_err).lower():
                            logger.warning(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Toolset cleanup error: {tool_err}")
                del _RUNNER_TOOLSETS[str(runner_id)]

            if runner_id in self._active_runners:
                cleanup_tasks = []

                if hasattr(runner, 'aclose'):
                    cleanup_task = asyncio.create_task(runner.aclose())
                    cleanup_tasks.append(cleanup_task)
                elif hasattr(runner, 'close'):
                    close_method = getattr(runner, 'close')
                    if asyncio.iscoroutinefunction(close_method):
                        cleanup_task = asyncio.create_task(close_method())
                        cleanup_tasks.append(cleanup_task)
                    else:
                        loop = asyncio.get_event_loop()
                        cleanup_task = loop.run_in_executor(None, close_method)
                        cleanup_tasks.append(cleanup_task)

                if cleanup_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*cleanup_tasks, return_exceptions=True),
                            timeout=6.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Runner cleanup timed out for {reason}")
                    except Exception as cleanup_err:
                        if "cancel scope" not in str(cleanup_err).lower():
                            logger.warning(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Cleanup error {reason}: {cleanup_err}")

            if runner_id in self._active_runners:
                del self._active_runners[str(runner_id)]
            if runner_id in _RUNNER_TOOLSETS:
                del _RUNNER_TOOLSETS[str(runner_id)]

            logger.debug(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Runner {runner_id} cleaned up successfully")

        except Exception as close_error:
            if "cancel scope" in str(close_error).lower() or "cancellederror" in str(close_error).lower():
                logger.debug(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Expected cancellation during {reason}: {close_error}")
            else:
                logger.warning(f"[SAFE_CLOSE][{time.time():.6f}][User:{get_uid()}] Unexpected error during {reason}: {close_error}")
        finally:
            if runner_id in self._active_runners:
                del self._active_runners[str(runner_id)]
            if runner_id in _RUNNER_TOOLSETS:
                del _RUNNER_TOOLSETS[str(runner_id)]

    async def _create_clean_session(self, session_id: str, user_id: str,
                                    preserved_history: Optional[List[Dict[str, Any]]] = None) -> Runner:
        logger.info(f"[CLEAN_SESSION][{time.time():.6f}][User:{get_uid()}] Creating for session={session_id}, user={user_id}")

        ctx = get_current_headers()
        user_id = ctx["user_id"]
        role_id = ctx["role_id"]
        workspace_id = ctx["workspace_id"]
        tenant_id = ctx["tenant_id"]
        headers = get_headers(tenant_id, user_id, role_id, workspace_id)

        runner_id = f"runner_{int(time.time() * 1000)}_{session_id}"
        isolated_toolset = get_isolated_toolset_for_runner(runner_id, user_id, headers)
        await isolated_toolset.initialize()

        instruction = SYSTEM_INSTRUCTIONS

        agent = LlmAgent(
            model=lite_llm_instance,
            name="PowerAssist",
            description="Your trusted assistant for smarter data discovery",
            instruction=instruction,
            tools=[isolated_toolset]
        )

        svc = self._get_session_service(user_id)
        await svc.delete_session(app_name=self.app_name, user_id=user_id, session_id=session_id)

        session = await svc.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )

        past_summaries = load_past_context()

        if past_summaries:
            logger.info(f"[PAST_SUMMARY][{time.time():.6f}][User:{get_uid()}] Injecting {len(past_summaries)} summaries as events")
            for i, summary_dict in enumerate(past_summaries):
                event_text = f"Past session summary: {summary_dict.get('summary', '')} (Session: {summary_dict.get('session_id', 'unknown')})"
                event = Event(
                    invocation_id=f"past_summary_{i}",
                    author="system",
                    content=types.Content(role="system", parts=[types.Part(text=event_text)]),
                    timestamp=time.time() - 1000 * i
                )
                await svc.append_event(session, event)

        if preserved_history:
            logger.info(f"[CLEAN_SESSION][{time.time():.6f}][User:{get_uid()}] Injecting {len(preserved_history)} events chronologically")
            deduplicated_history = deduplicate_ai_messages(preserved_history)
            for i, event_dict in enumerate(deduplicated_history):
                try:
                    if event_dict.get("type") == "user_message":
                        event = Event(
                            invocation_id=f"rec_user_{i}",
                            author="user",
                            content=types.Content(role="user", parts=[types.Part(text=event_dict["message"])]),
                            timestamp=event_dict.get("timestamp", time.time())
                        )
                    elif event_dict.get("type") == "ai_message":
                        event = Event(
                            invocation_id=f"rec_ai_{i}",
                            author="PowerAssist",
                            content=types.Content(role="model", parts=[types.Part(text=event_dict["message"])]),
                            timestamp=event_dict.get("timestamp", time.time())
                        )
                    elif event_dict.get("type") in ["tool_call", "tool_response"]:
                        continue
                    else:
                        continue

                    await svc.append_event(session, event)

                except Exception as ie:
                    logger.warning(f"[CLEAN_SESSION][{time.time():.6f}][User:{get_uid()}] Failed event {i}: {ie}")
                    continue

        runner = Runner(agent=agent, app_name=self.app_name, session_service=svc)
        self._active_runners[str(id(runner))] = runner

        logger.info(f"[CLEAN_SESSION][{time.time():.6f}][User:{get_uid()}] Runner {id(runner)} created for {session_id}")
        return runner

    async def _new_runner(self, session_id: str, user_id: str) -> None:
        logger.info(f"[NEW_RUNNER][{time.time():.6f}][User:{get_uid()}] Requesting new runner for {session_id}")
        runner = await self._create_clean_session(session_id, user_id)

        runner_id = id(runner)
        toolset = _RUNNER_TOOLSETS.get(str(runner_id))
        mcp_session_id = toolset._session_id if isinstance(toolset, IsolatedMcpToolset) else None

        ctx = get_current_headers()

        self.sessions[session_id] = {
            "runner": runner,
            "events": [],
            "dirty": False,
            "user_id": user_id,
            "session_id": session_id,
            "last_cleanup": None,
            "recovery_count": 0,
            "session_service": runner.session_service,
            "runner_id": runner_id,
            "mcp_session_id": mcp_session_id,
            "context": ctx
        }

        logger.info(f"[NEW_RUNNER][{time.time():.6f}][User:{get_uid()}] Session {session_id} created with context: tenant={ctx['tenant_id']}, user={ctx['user_id']}, ws={ctx['workspace_id']}")

    async def _run_runner_with_cancel(self, runner: Runner, user_id: str,
                                      session_id: str, user_msg: types.Content,
                                      cancel_scope: GracefulCancelScope) -> dict[
                                                                                str, str | list[dict[str, Any]]] | None:
        """Run the runner with cancellation awareness - returns captured events and final text."""
        captured_events = []
        final_text = ""

        perf_tool_starts: Dict[str, float] = {}
        perf_last_action_ts = time.time()

        llm_wait_start = time.time()
        llm_wait_reason = "initial"
        logger.info(f"[LLM_WAIT_START][User:{get_uid()}] reason={llm_wait_reason} session={session_id} ts={llm_wait_start:.4f}")

        max_duration = 0.0
        max_process_name = "None"

        try:
            logger.info(f"[LLM_START][{time.time():.6f}][User:{get_uid()}] Sending request to LLM (Thinking started)...")
            runner_iter = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_msg
            )

            async for event in runner_iter:
                arrival_ts = time.time()
                if cancel_scope.is_cancelled():
                    logger.debug(f"[_RUNNER][{arrival_ts:.6f}][User:{get_uid()}] Cancellation detected...")
                    break

                ev_dict = event_to_dict(event, arrival_ts=arrival_ts)

                now_ts = time.time()

                if ev_dict:
                    ev_type = ev_dict.get("type")

                    if ev_type in ["tool_call", "ai_message"]:
                        llm_wait_end = now_ts
                        llm_wait_dur = llm_wait_end - llm_wait_start
                        trigger = ev_dict.get("tool") if ev_type == "tool_call" else "final_response"
                        logger.info(
                            f"[LLM_WAIT_END][User:{get_uid()}] reason={llm_wait_reason} trigger={trigger} "
                            f"ts={llm_wait_end:.4f} dur={llm_wait_dur:.4f}s"
                        )
                    current_ts = ev_dict.get("timestamp", now_ts)

                    if ev_type == "tool_call":
                        call_id = ev_dict.get("call_id")
                        tool_name = ev_dict.get("tool", "unknown")
                        if call_id:
                            ai_overhead = current_ts - perf_last_action_ts
                            logger.info(f"[PERF][{time.time():.6f}][User:{get_uid()}] LLM Thinking Time (Decision => {tool_name}): {ai_overhead:.4f}s")
                            perf_tool_starts[call_id] = current_ts

                            if ai_overhead > max_duration:
                                max_duration = ai_overhead
                                max_process_name = "AI Logic (Pre-Tool)"

                    elif ev_type == "tool_response":
                        call_id = ev_dict.get("call_id")
                        if call_id and call_id in perf_tool_starts:
                            start_time = perf_tool_starts.pop(call_id)
                            tool_duration = current_ts - start_time
                            tool_name = ev_dict.get("tool", "unknown")
                            logger.info(f"[PERF][{time.time():.6f}][User:{get_uid()}] Tool Round Trip (Network + Execution) for {tool_name} ({call_id}): {tool_duration:.4f}s")
                            perf_last_action_ts = current_ts

                            if tool_duration > max_duration:
                                max_duration = tool_duration
                                max_process_name = f"Tool: {tool_name}"

                        llm_wait_start = now_ts
                        llm_wait_reason = f"post_tool:{ev_dict.get('tool', 'unknown')}"
                        logger.info(
                            f"[LLM_WAIT_START][User:{get_uid()}] reason={llm_wait_reason} session={session_id} ts={llm_wait_start:.4f}"
                        )

                    elif ev_type == "ai_message" and not ev_dict.get("_partial", False):
                        ai_overhead = current_ts - perf_last_action_ts
                        logger.info(f"[PERF][{time.time():.6f}][User:{get_uid()}] LLM Thinking Time (Response Generation): {ai_overhead:.4f}s")
                        perf_last_action_ts = current_ts

                        if ai_overhead > max_duration:
                            max_duration = ai_overhead
                            max_process_name = "AI Logic (Response)"

                    captured_events.append(ev_dict)

                if event.is_final_response() and event.content and event.content.parts:
                    final_part = event.content.parts[0]
                    if hasattr(final_part, 'text') and final_part.text:
                        final_text = str(final_part.text).strip()
                        logger.debug(f"[_RUNNER][{time.time():.6f}][User:{get_uid()}] Final response captured: {final_text[:500]}...")

                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.debug(f"[_RUNNER][{time.time():.6f}][User:{get_uid()}] Runner iteration cancelled for {session_id}")
            raise
        except Exception as runner_err:
            logger.error(f"[_RUNNER][{time.time():.6f}][User:{get_uid()}] Error during runner execution for {session_id}: {runner_err}", exc_info=True)
            final_text = final_text or f"Partial response: {str(runner_err)[:200]}"

        finally:
            if hasattr(runner, 'aclose') and not cancel_scope.is_cancelled():
                try:
                    await runner.aclose()
                except:
                    pass

        return {
            "final_text": final_text,
            "events": captured_events,
            "max_duration": max_duration,
            "max_process_name": max_process_name
        }

    async def invoke(self, message: str, session_id: str, user_id: str,
                     cancel_event: Optional[asyncio.Event] = None) -> str:
        perf_start_tat = time.time()

        logger.info(f"[INVOKE][{perf_start_tat:.6f}][User:{get_uid()}] user={user_id} session={session_id} msg='{message[:100]}...'")

        if session_id not in self.sessions:
            await self._new_runner(session_id, user_id)

        sess = self.sessions[session_id]
        runner = sess["runner"]
        events = sess["events"]
        runner_id = sess.get("runner_id")

        recovery_needed = has_incomplete_tool_calls(events) or sess.get("dirty", False)
        if recovery_needed:
            old_runner_id = runner_id or id(runner)
            logger.warning(
                f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Recovery triggered for session_id={session_id} | "
                f"closing old runner id={old_runner_id}"
            )
            await self._safe_close_runner(runner, "recovery")

            preserved = copy.deepcopy(events)
            recovery_count = sess.get("recovery_count", 0) + 1

            preserved_context = sess.get("context", {})

            runner = await self._create_clean_session(
                session_id=session_id,
                user_id=user_id,
                preserved_history=preserved
            )
            new_runner_id = id(runner)

            sess["events"] = deduplicate_ai_messages(preserved)
            sess["runner"] = runner
            sess["dirty"] = False
            sess["recovery_count"] = recovery_count
            sess["last_cleanup"] = time.time()
            sess["session_service"] = runner.session_service
            sess["runner_id"] = new_runner_id
            sess["context"] = preserved_context

            logger.info(
                f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Recovery #{recovery_count} completed for {session_id} | "
                f"old_runner_id={old_runner_id} → new_runner_id={new_runner_id} | "
                f"preserved_events={len(preserved)}"
            )

        user_msg = types.Content(role="user", parts=[types.Part(text=message)])
        final_text = ""
        interruption_occurred = False

        cancel_scope = GracefulCancelScope(cancel_event)

        events.append({
            "type": "user_message",
            "message": message,
            "timestamp": time.time()
        })

        sess["events"] = deduplicate_ai_messages(events)

        session_max_stats = (0.0, "None")

        try:
            async with cancel_scope:
                runner_task = asyncio.create_task(
                    self._run_runner_with_cancel(runner, user_id, session_id, user_msg, cancel_scope)
                )

                while not runner_task.done():
                    if cancel_scope.is_cancelled():
                        logger.info(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Cancellation detected during streaming for {session_id}")
                        interruption_occurred = True
                        sess["dirty"] = True
                        runner_task.cancel()
                        try:
                            await asyncio.wait_for(runner_task, timeout=1.0)
                        except asyncio.CancelledError:
                            logger.debug(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Runner task cancelled for {session_id}")
                        except Exception as cancel_err:
                            logger.warning(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Error during cancel: {cancel_err}")
                        break

                    await asyncio.sleep(0.05)

                try:
                    result = await asyncio.wait_for(runner_task, timeout=5.0)
                    final_text = result.get("final_text", "")

                    session_max_stats = (
                        result.get("max_duration", 0.0),
                        result.get("max_process_name", "None")
                    )

                    if "events" in result:
                        events.extend(result["events"])
                except asyncio.TimeoutError:
                    logger.warning(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Runner task timed out during completion for {session_id}")
                    interruption_occurred = True
                    sess["dirty"] = True
                except asyncio.CancelledError:
                    logger.debug(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Runner task was cancelled for {session_id}")
                    interruption_occurred = True
                    sess["dirty"] = True
                except Exception as task_err:
                    logger.error(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Runner task failed for {session_id}: {task_err}", exc_info=True)
                    interruption_occurred = True
                    sess["dirty"] = True

        except Exception as e:
            logger.error(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Outer error for {session_id}: {e}", exc_info=True)

            if "cancellederror" in str(e).lower() or "cancelled" in str(e).lower():
                logger.info(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Expected cancellation for {session_id}: {e}")
                interruption_occurred = True
                sess["dirty"] = True
                final_text = final_text or "Session interrupted. All progress preserved."
            elif "tool_calls" in str(e).lower() or "litellm" in str(e).lower():
                sess["dirty"] = True
                final_text = f"Session recovered from processing issue: {str(e)[:100]}"
            else:
                final_text = f"Unexpected error: {str(e)[:100]}"
                sess["dirty"] = True

            await self._safe_close_runner(runner, "exception_or_cancel")

        events = deduplicate_ai_messages(events)
        sess["events"] = events

        if final_text and not interruption_occurred and len(final_text.strip()) > 10:
            existing_final = None
            for event in reversed(events[-3:]):
                if (event.get("type") == "ai_message" and
                        event.get("message") and
                        final_text in event.get("message") and
                        abs(event.get("timestamp", 0) - time.time()) < 10):
                    existing_final = event
                    break

            if not existing_final:
                events.append({
                    "type": "ai_message",
                    "message": final_text,
                    "timestamp": time.time()
                })
            sess["events"] = deduplicate_ai_messages(events)

        elif interruption_occurred:
            final_text = final_text or "Request stopped. Session state preserved for continuation."
            logger.info(f"[INVOKE][{time.time():.6f}][User:{get_uid()}] Stop completed for {session_id} - preserved {len(events)} events")
            sess["dirty"] = True

        perf_end_tat = time.time()
        total_tat = perf_end_tat - perf_start_tat
        logger.info(f"[PERF][{perf_end_tat:.6f}][User:{get_uid()}] Total Turnaround Time: {total_tat:.4f}s")
        if session_max_stats[0] > 0:
            logger.info(f"[PERF][{time.time():.6f}][User:{get_uid()}] Max Latency Driver: {session_max_stats[1]} ({session_max_stats[0]:.4f}s)")

        return final_text

    async def end_session(self, session_id: str, user_id: str, force: bool = False) -> bool:
        if session_id not in self.sessions:
            logger.warning(f"[END_SESSION][{time.time():.6f}][User:{get_uid()}] Session {session_id} not found")
            return False

        sess = self.sessions.pop(session_id)
        runner = sess.get("runner")
        events = sess.get("events", [])
        recovery_count = sess.get("recovery_count", 0)

        ctx = sess.get("context")

        ctx["user_id"] = user_id

        logger.info(f"[END_SESSION][{time.time():.6f}][User:{get_uid()}] Using stored context for {session_id}: tenant={ctx['tenant_id']}, user={ctx['user_id']}, role={ctx['role_id']}, ws={ctx['workspace_id']}")

        final_events = deduplicate_ai_messages(events)

        if force or not sess.get("dirty", False):
            save_events = final_events
        else:
            save_events = final_events
            logger.info(f"[END_SESSION][{time.time():.6f}][User:{get_uid()}] Dirty session - saving all {len(save_events)} events")

        if runner:
            await self._safe_close_runner(runner, "end_session")

        try:
            summary_events = final_events.copy()
            if len(summary_events) > 1:
                if summary_events:
                    openai_client = get_summary_client()
                    summary_model = get_summary_model()

                    logger.info(f"[SUMMARY][{time.time():.6f}][User:{get_uid()}] Summary Model: {summary_model}")

                    llm_start = time.time()
                    logger.info(f"[LLM_START][{llm_start:.6f}][User:{get_uid()}] end_session_summary model={summary_model}")

                    resp = openai_client.chat.completions.create(
                        model=summary_model,
                        messages=[
                            {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                            {"role": "user", "content": json.dumps(summary_events, default=str)},
                        ],
                    )

                    llm_end = time.time()
                    logger.info(f"[LLM_END][{llm_end:.6f}][User:{get_uid()}] end_session_summary model={summary_model} dur={(llm_end-llm_start):.4f}s")

                    summary = resp.choices[0].message.content.strip()
                else:
                    summary = f"Session with {len(final_events)} events (tools only)"
            else:
                summary = f"Brief session with {len(final_events)} events"
        except Exception as summary_error:
            logger.warning(f"[END_SESSION][{time.time():.6f}][User:{get_uid()}] Summary failed: {summary_error}")
            summary = f"Session with {len(final_events)} events"

        long_term_data = {
            "events": save_events,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": time.time(),
            "recovery_count": recovery_count,
            "final_status": "clean" if not sess.get("dirty", False) else "recovered"
        }

        try:
            append_long_term(
                ctx["tenant_id"],
                ctx["user_id"],
                ctx["role_id"],
                ctx["workspace_id"],
                long_term_data
            )
            logger.info(f"[LONG_TERM][{time.time():.6f}][User:{get_uid()}] Saved {len(save_events)} chronological events for {session_id} to ws={ctx['workspace_id']}")
        except Exception as lt_error:
            logger.error(f"[LONG_TERM][{time.time():.6f}][User:{get_uid()}] Save failed for {session_id}: {lt_error}")

        try:
            append_short_term(
                ctx["tenant_id"],
                ctx["user_id"],
                ctx["role_id"],
                ctx["workspace_id"],
                {
                    "summary": summary,
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "recovery_count": recovery_count,
                    "total_events": len(final_events)
                }
            )
            logger.info(f"[SUMMARY][{time.time():.6f}][User:{get_uid()}] Saved: {summary[:80]}... ({len(final_events)} events) to ws={ctx['workspace_id']}")
        except Exception as st_error:
            logger.warning(f"[SUMMARY][{time.time():.6f}][User:{get_uid()}] Short-term save failed for {session_id}: {st_error}")

        logger.info(f"[END_SESSION][{time.time():.6f}][User:{get_uid()}] Session {session_id} ended - {len(final_events)} events, {recovery_count} recoveries")
        return True

    def mark_dirty(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions[session_id]["dirty"] = True
            logger.info(f"[MARK_DIRTY][{time.time():.6f}][User:{get_uid()}] Session {session_id} marked dirty")
            return True
        return False
