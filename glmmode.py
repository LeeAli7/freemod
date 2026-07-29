#!/usr/bin/env python3
"""GLMMode — OpenAI-compatible API via chat.z.ai (Playwright headless).

Routes requests through chat.z.ai (GLM-4.7) using Playwright.
Features:
  - Pool of browser pages with Semaphore-based concurrency
  - SSE interception via page.on("response")
  - page.goto(URL) reset after each request
  - Tool calls via JSON prompt injection
  - Environment-based configuration
"""

import asyncio, json, sys, os, time, re, hashlib, logging, uuid
from dataclasses import dataclass, field
from typing import Optional
from playwright.async_api import async_playwright, Page, BrowserContext

# ─── Configuration ──────────────────────────────────────────────
URL = os.getenv("GLMMODE_URL", "https://chat.z.ai")
POOL_SIZE = int(os.getenv("GLMMODE_POOL_SIZE", "1"))
SSE_TIMEOUT = int(os.getenv("GLMMODE_SSE_TIMEOUT", "90"))
MAX_ATTEMPTS = int(os.getenv("GLMMODE_MAX_ATTEMPTS", "2"))
CREATE_PAGE_DELAY = float(os.getenv("GLMMODE_CREATE_PAGE_DELAY", "0.2"))
HEADLESS = os.getenv("GLMMODE_HEADLESS", "true").lower() == "true"
API_KEY = os.getenv("GLMMODE_API_KEY", "")
MODEL = os.getenv("GLMMODE_MODEL", "glm-4.7")
PORT = int(os.getenv("GLMMODE_PORT", "5001"))

log = logging.getLogger("glmmode")

_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--disable-web-security",
    "--no-first-run",
    "--no-default-browser-check",
]

# ─── Page State ─────────────────────────────────────────────────
@dataclass
class PageState:
    page: Page
    busy: bool = False
    last_used: float = field(default_factory=time.time)

# ─── Helpers ────────────────────────────────────────────────────

async def _create_page(ctx: BrowserContext, model: str) -> Page:
    page = await ctx.new_page()
    await page.goto(URL, wait_until="domcontentloaded", timeout=20000)
    await asyncio.sleep(1)
    # Select model if needed
    if model:
        try:
            btn = await page.query_selector("button[aria-label='Select a model']")
            if btn:
                txt = await btn.inner_text()
                if txt.strip() != model:
                    await btn.click()
                    await asyncio.sleep(0.2)
                    opt = await page.query_selector(f"[class*='option']:has-text('{model}')")
                    if opt:
                        await opt.click()
                        await asyncio.sleep(0.2)
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.2)
        except:
            pass
    return page

async def _refresh_page(ctx: BrowserContext, page: Page, model: str) -> Page:
    try:
        await page.close()
    except:
        pass
    await asyncio.sleep(1)
    return await _create_page(ctx, model)

def _detect_captcha(body_text: str) -> bool:
    bl = body_text.lower()
    return "security verification" in bl or "drag the slider" in bl

async def _type_and_send(page: Page, text: str) -> None:
    """Type text and click send on chat.z.ai."""
    await page.evaluate("""(t) => {
        const ta = document.getElementById('chat-input');
        if (!ta) return;
        const s = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        s.call(ta, t);
        ta.dispatchEvent(new Event('input', {bubbles: true}));
    }""", text)
    await asyncio.sleep(0.1)
    await page.evaluate("""() => {
        const b = document.getElementById('send-message-button');
        if (b) b.click();
    }""")

def _parse_sse(raw: str) -> dict:
    """Parse SSE from chat.z.ai into {text, reasoning}."""
    if not raw:
        return {"text": "", "reasoning": ""}
    reasoning = []
    answer = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line.startswith('data: '):
            continue
        try:
            ev = json.loads(line[6:])
        except:
            continue
        d = ev.get('data', {})
        phase = d.get('phase', '')
        delta = d.get('delta_content', '') or ''
        if phase == 'thinking':
            reasoning.append(delta)
        elif phase == 'answer':
            answer.append(delta)
        elif phase == 'done':
            break
    return {"text": ''.join(answer).strip(), "reasoning": ''.join(reasoning).strip()}

async def _wait_for_response(ctx: BrowserContext, page: Page, prompt: str, model: str, timeout: int) -> tuple[dict, Page]:
    """Send prompt and wait for response via SSE or DOM fallback."""
    bodies: dict[str, Optional[str]] = {"raw": None}
    response_event = asyncio.Event()

    async def capture(resp) -> None:
        if "/api/v2/chat/completions" in resp.url and bodies["raw"] is None:
            try:
                bodies["raw"] = await asyncio.wait_for(resp.text(), timeout=timeout)
                response_event.set()
            except:
                pass

    page.on("response", capture)

    try:
        await _type_and_send(page, prompt)

        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        if bodies["raw"] is not None:
            parsed = _parse_sse(bodies["raw"])
            return parsed, page

        # Check for captcha
        try:
            bt = await asyncio.wait_for(page.evaluate("() => document.body.innerText"), timeout=5)
        except:
            bt = ""
        if _detect_captcha(bt):
            return {"text": "[GLMMode] Captcha detected", "reasoning": ""}, page

        # DOM fallback
        try:
            text = await asyncio.wait_for(page.evaluate("() => document.body.innerText"), timeout=5)
        except:
            text = ""
        return {"text": text.strip(), "reasoning": ""}, page

    finally:
        try:
            page.remove_listener("response", capture)
        except:
            pass

# ─── Prompt Builder ─────────────────────────────────────────────

def _build_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "user":
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                parts.append("\n".join(texts))
            else:
                parts.append(str(content))
        elif role == "assistant":
            tc = m.get("tool_calls")
            if tc:
                for t in tc:
                    fn = t.get("function", {})
                    parts.append(f'{{"tool": "{fn.get("name", "")}", "arguments": {fn.get("arguments", "{}")}}}')
            elif isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                parts.append("\n".join(texts))
            else:
                parts.append(str(content))
        elif role == "tool":
            parts.append(f"[Tool result]\n{content}")
    return "\n\n".join(parts).strip()

# ─── Tool Call Parser ───────────────────────────────────────────

def _parse_tool_call(text: str) -> tuple[Optional[str], Optional[dict]]:
    idx = 0
    while True:
        tool_pos = text.find('"tool"', idx)
        func_pos = text.find('"function"', idx)
        if tool_pos == -1 and func_pos == -1:
            break
        pos = tool_pos if tool_pos != -1 and (func_pos == -1 or tool_pos < func_pos) else func_pos
        idx = pos + 1
        brace = text.rfind('{', 0, pos)
        if brace == -1:
            continue
        i = brace
        depth = 0
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    chunk = text[brace:i+1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            name = obj.get('tool') or obj.get('function')
                            if name:
                                return name, obj.get('arguments', {})
                    except:
                        pass
                    break
            i += 1
    # Fallback: code block
    m = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                name = obj.get('tool') or obj.get('function')
                if name:
                    return name, obj.get('arguments', {})
        except:
            pass
    return None, None

def _format_chat_result(content_text: str, reasoning: str) -> dict:
    tool_name, tool_args = _parse_tool_call(content_text)
    if tool_name:
        tc_id = f"call_{uuid.uuid4().hex[:8]}"
        resp = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
            }],
        }
    else:
        resp = {"role": "assistant", "content": content_text}
    if reasoning:
        resp["reasoning_content"] = reasoning
    return resp

# ─── Pool ───────────────────────────────────────────────────────

class GLMModePool:
    def __init__(self, size: int = POOL_SIZE, model: str = MODEL, headless: bool = HEADLESS):
        self.size = size
        self.model = model
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.ctx: Optional[BrowserContext] = None
        self._states: list[Optional[PageState]] = [None] * size
        self._lock = asyncio.Lock()
        self._available = asyncio.Semaphore(size)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=_LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
        self.ctx = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="en-US",
            timezone_id="America/New_York",
        )
        await self.ctx.add_init_script("""\
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
""")

        for i in range(self.size):
            for attempt in range(5):
                try:
                    page = await _create_page(self.ctx, self.model)
                    self._states[i] = PageState(page=page)
                    log.info(f"Page {i+1}/{self.size} ready")
                    break
                except Exception as e:
                    log.warning(f"Page {i} create failed (attempt {attempt+1}): {e}")
                    await asyncio.sleep(1)
            else:
                log.error(f"Page {i} UNAVAILABLE after 5 attempts")
                self._states[i] = None
            if i < self.size - 1:
                await asyncio.sleep(CREATE_PAGE_DELAY)

        log.info(f"Pool ready ({sum(1 for s in self._states if s is not None)}/{self.size} pages) | Model: {self.model}")
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _check_health(self, page: Page) -> bool:
        try:
            bt = await page.evaluate("() => document.body.innerText.substring(0, 300)")
            if _detect_captcha(bt) or not bt.strip():
                return False
            return True
        except Exception:
            return False

    async def _recreate_page(self, idx: int) -> None:
        old = self._states[idx]
        if old:
            try:
                await old.page.close()
            except Exception:
                pass
        try:
            page = await _create_page(self.ctx, self.model)
            self._states[idx] = PageState(page=page)
            log.info(f"Page {idx} recreated")
        except Exception as e:
            log.error(f"Page {idx} recreate failed: {e}")
            self._states[idx] = None

    async def _heartbeat(self) -> None:
        while not self._shutdown:
            try:
                await asyncio.wait_for(asyncio.sleep(15), timeout=20)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            for i in range(self.size):
                if self._shutdown:
                    break
                state = self._states[i]
                if state is None:
                    log.warning(f"Heartbeat: page {i} missing, recreating")
                    async with self._lock:
                        await self._recreate_page(i)
                    continue
                if state.busy:
                    continue
                if not await self._check_health(state.page):
                    log.warning(f"Heartbeat: page {i} unhealthy, recreating")
                    async with self._lock:
                        await self._recreate_page(i)

    async def execute(self, prompt: str) -> dict:
        await self._available.acquire()
        try:
            async with self._lock:
                idx = -1
                for i, state in enumerate(self._states):
                    if state is not None and not state.busy:
                        idx = i
                        state.busy = True
                        state.last_used = time.time()
                        break
                if idx == -1:
                    return {"text": "[GLMMode] No available pages", "reasoning": ""}

            state = self._states[idx]

            if not await self._check_health(state.page):
                async with self._lock:
                    await self._recreate_page(idx)
                    state = self._states[idx]
                    if state is None:
                        return {"text": "[GLMMode] Page unavailable", "reasoning": ""}
                    state.busy = True

            try:
                result, new_page = await _wait_for_response(
                    self.ctx, state.page, prompt, self.model, SSE_TIMEOUT
                )
                async with self._lock:
                    if new_page != state.page:
                        try:
                            await state.page.close()
                        except Exception:
                            pass
                        self._states[idx] = PageState(page=new_page, busy=False)
                    else:
                        state.busy = False
                        state.last_used = time.time()
                return result
            except Exception as e:
                log.error(f"Page {idx} error: {e}")
                async with self._lock:
                    state.busy = False
                    await self._recreate_page(idx)
                raise
        finally:
            self._available.release()

    async def chat(self, messages: list[dict], tools: Optional[list] = None) -> dict:
        last_content = _build_prompt(messages)
        if not last_content:
            return {"role": "assistant", "content": "[GLMMode] No message"}

        if tools:
            desc = "\n".join(
                f"- {t['function']['name']}({', '.join(f'{p}: {v}' for p, v in t['function'].get('parameters', {}).get('properties', {}).items())})"
                for t in tools
            )
            prompt = (f"You are an AI agent with tools:\n{desc}\n\n"
                      f"When calling a tool, respond ONLY with:\n{{\"tool\": \"name\", \"arguments\": {{...}}}}\n\n"
                      f"Otherwise respond normally.\n\n{last_content}")
        else:
            prompt = last_content

        result = await self.execute(prompt)
        return _format_chat_result(result.get("text", ""), result.get("reasoning", ""))

    async def close(self) -> None:
        self._shutdown = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        for state in self._states:
            if state:
                try:
                    await state.page.close()
                except Exception:
                    pass
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

# ─── Server ─────────────────────────────────────────────────────

def run_server(port: int = PORT) -> None:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException, Header
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    pool: Optional[GLMModePool] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal pool
        pool = GLMModePool(size=POOL_SIZE, model=MODEL)
        await pool.start()
        logging.basicConfig(level=logging.INFO)
        yield
        await pool.close()

    app = FastAPI(title="GLMMode API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("GLMMODE_CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        model: str = MODEL
        messages: list
        stream: bool = False
        tools: Optional[list] = None

    def _verify_auth(authorization: Optional[str]) -> None:
        if not API_KEY:
            return
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = authorization.replace("Bearer ", "").strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, authorization: Optional[str] = Header(None)):
        from fastapi.responses import StreamingResponse
        _verify_auth(authorization)
        if pool is None:
            raise HTTPException(status_code=503, detail="Pool not ready")

        result = await pool.chat(req.messages, tools=req.tools)

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        content = result.get("content", "")
        reasoning = result.get("reasoning_content", "")
        has_tc = bool(result.get("tool_calls"))
        finish = "tool_calls" if has_tc else "stop"

        if not req.stream:
            msg = {"role": "assistant", "content": content}
            if has_tc:
                msg["tool_calls"] = result["tool_calls"]
            if reasoning:
                msg["reasoning_content"] = reasoning
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": req.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}]
            }

        async def gen():
            if reasoning:
                d = {"role": "assistant", "reasoning_content": reasoning}
                yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":d,"finish_reason":None}]})}\n\n'
            if content:
                words = content.split(" ")
                for i in range(0, len(words), 5):
                    chunk = " ".join(words[i:i+5])
                    d = {"content": chunk + " "}
                    yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":d,"finish_reason":None}]})}\n\n'
                    await asyncio.sleep(0.01)
            if has_tc:
                tc = result["tool_calls"][0]
                d = {"tool_calls": [{"index": 0, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}
                yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":d,"finish_reason":None}]})}\n\n'
            yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{},"finish_reason":finish}]})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        if pool is None:
            return {"status": "not_ready"}
        healthy = sum(1 for s in pool._states if s is not None)
        return {"status": "ok", "pages": {"total": pool.size, "healthy": healthy}}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "glmmode"}]
        }

    print(f"\n[GLMMode] API Server on http://0.0.0.0:{port}", file=sys.stderr)
    print(f"[GLMMode] Pool: {POOL_SIZE} pages | SSE capture", file=sys.stderr)
    print(f"[GLMMode] Model: {MODEL}", file=sys.stderr)
    print(f"[GLMMode] POST /v1/chat/completions\n", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    if "--server" in sys.argv:
        idx = sys.argv.index("--server")
        port = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else PORT
        run_server(port)
    else:
        print("Usage: python glmmode.py --server [port]", file=sys.stderr)
        sys.exit(1)
