#!/usr/bin/env python3
import asyncio, json, sys, os, time, re, hashlib, logging
from playwright.async_api import async_playwright

URL = "https://chat.z.ai"
log = logging.getLogger("glmmode")

SSE_TIMEOUT = 120
MAX_ATTEMPTS = 2
POOL_SIZE = 3
CREATE_PAGE_DELAY = 0.5

_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--disable-web-security",
    "--no-first-run",
    "--no-default-browser-check",
]

async def _create_page(ctx, model):
    page = await ctx.new_page()
    await page.goto(URL, wait_until="domcontentloaded", timeout=20000)
    await asyncio.sleep(4)
    if model:
        try:
            btn = await page.query_selector("button[aria-label='Select a model']")
            if btn:
                txt = await btn.inner_text()
                if txt.strip() != model:
                    await btn.click()
                    await asyncio.sleep(0.5)
                    opt = await page.query_selector(f"[class*='option']:has-text('{model}')")
                    if opt:
                        await opt.click()
                        await asyncio.sleep(0.5)
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.5)
        except:
            pass
    return page

async def _refresh_page(ctx, page, model):
    try:
        await page.close()
    except:
        pass
    await asyncio.sleep(2)
    return await _create_page(ctx, model)

def _detect_captcha(body_text):
    bl = body_text.lower()
    return "security verification" in bl or "drag the slider" in bl

def _type_and_send(page, text):
    return page.evaluate("""(t) => {
        const ta = document.getElementById('chat-input');
        if (!ta) return;
        const s = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
        s.call(ta, t);
        ta.dispatchEvent(new Event('input', {bubbles:true}));
    }""", text)

def _click_send(page):
    return page.evaluate("""() => {
        const b = document.getElementById('send-message-button');
        if (b) b.click();
    }""")

def _parse_sse(raw):
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

async def _send_prompt_once(ctx, page, prompt, model):
    prev_handler = None
    for attempt in range(MAX_ATTEMPTS):
        if prev_handler is not None:
            try:
                page.remove_listener("response", prev_handler)
            except:
                pass

        bodies = {}
        bodies[attempt] = None

        async def capture(resp, _a=attempt):
            if "/api/v2/chat/completions" in resp.url and bodies.get(_a) is None:
                try:
                    bodies[_a] = await asyncio.wait_for(resp.text(), timeout=SSE_TIMEOUT)
                except:
                    pass

        prev_handler = capture
        page.on("response", capture)

        await _type_and_send(page, prompt)
        await asyncio.sleep(0.3)
        await _click_send(page)

        for _ in range(SSE_TIMEOUT // 2):
            await asyncio.sleep(2)
            if bodies.get(attempt) is not None:
                try:
                    page.remove_listener("response", capture)
                except:
                    pass
                return _parse_sse(bodies[attempt]), page

            bt = await page.evaluate("() => document.body.innerText")
            if _detect_captcha(bt):
                print(f"[GLMMode] Captcha, refreshing (attempt {attempt+1})", file=sys.stderr)
                page = await _refresh_page(ctx, page, model)
                await asyncio.sleep(5)
                break
        else:
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(3)
                page = await _refresh_page(ctx, page, model)
                continue

    text = await page.evaluate("() => document.body.innerText")
    return {"text": text.strip(), "reasoning": ""}, page

def _build_prompt(messages):
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
                    parts.append(f'{{"tool": "{t["function"]["name"]}", "arguments": {t["function"]["arguments"]}}}')
            elif isinstance(content, list):
                for b in content:
                    if b.get("type") == "text":
                        parts.append(b.get("text", ""))
            else:
                parts.append(str(content))
        elif role == "tool":
            parts.append(f"[Tool result]\n{content}")
    return "\n\n".join(parts).strip()

def _parse_tool_call(text):
    """Extract tool call JSON from anywhere in model output.
    Searches for "tool" or "function" keys, then does balanced brace matching."""
    idx = 0
    while True:
        # Find "tool" or "function" key anywhere in text
        tool_pos = text.find('"tool"', idx)
        func_pos = text.find('"function"', idx)
        if tool_pos == -1 and func_pos == -1:
            break
        pos = tool_pos if tool_pos != -1 and (func_pos == -1 or tool_pos < func_pos) else func_pos
        idx = pos + 1
        # Find the opening brace before the key
        brace = text.rfind('{', 0, pos)
        if brace == -1:
            continue
        # Balanced brace matching (handles nested {})
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
    # Last resort: markdown code block
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

def _format_chat_result(content_text, reasoning):
    tool_name, tool_args = _parse_tool_call(content_text)
    if tool_name:
        tc_id = f"call_{hashlib.md5(tool_name.encode()).hexdigest()[:8]}"
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

class GLMModePool:
    def __init__(self, size=POOL_SIZE, model="glm-4.7", headless=True):
        self.size = size
        self.model = model
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.ctx = None
        self._pages = [None] * size
        self._queue = None
        self._heartbeat_task = None
        self._replenishing = set()

    async def start(self):
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
        await self.ctx.add_init_script("""
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
""")
        self._queue = asyncio.Queue()
        for i in range(self.size):
            for attempt in range(5):
                try:
                    self._pages[i] = await _create_page(self.ctx, self.model)
                except Exception as e:
                    print(f"[GLMMode] Page {i} create failed: {e}", file=sys.stderr)
                    await asyncio.sleep(3)
                    continue
                if await self._check_health(self._pages[i]):
                    break
                print(f"[GLMMode] Page {i} unhealthy, recreating...", file=sys.stderr)
                await self._recreate_page(i)
                await asyncio.sleep(3)
            else:
                print(f"[GLMMode] Page {i} UNAVAILABLE after 5 attempts", file=sys.stderr)
            self._queue.put_nowait(i)
            print(f"[GLMMode] Page {i+1}/{self.size} ready", file=sys.stderr)
            if i < self.size - 1:
                await asyncio.sleep(CREATE_PAGE_DELAY)

        print(f"[GLMMode] Pool ready ({self.size} pages) | Model: {self.model}", file=sys.stderr)
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _check_health(self, page):
        try:
            bt = await page.evaluate("() => document.body.innerText.substring(0, 300)")
            if _detect_captcha(bt):
                return False
            return True
        except Exception:
            return False

    async def _recreate_page(self, idx):
        old = self._pages[idx]
        if old:
            try:
                await old.close()
            except:
                pass
        await asyncio.sleep(1)
        self._pages[idx] = await _create_page(self.ctx, self.model)

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(30)
            for i in range(self.size):
                if i in self._replenishing:
                    continue
                if not await self._check_health(self._pages[i]):
                    print(f"[GLMMode] Heartbeat: page {i} unhealthy, recreating", file=sys.stderr)
                    await self._recreate_page(i)
                    await asyncio.sleep(5)

    async def _replenish(self, idx):
        """Close old page and create a fresh one in background."""
        self._replenishing.add(idx)
        # Close old page
        old = self._pages[idx]
        if old:
            try:
                await old.close()
            except:
                pass
        self._pages[idx] = None
        # Create new page
        for attempt in range(5):
            try:
                new_page = await _create_page(self.ctx, self.model)
                if await self._check_health(new_page):
                    self._pages[idx] = new_page
                    self._queue.put_nowait(idx)
                    self._replenishing.discard(idx)
                    return
                try:
                    await new_page.close()
                except:
                    pass
            except Exception as e:
                print(f"[GLMMode] _replenish({idx}) attempt {attempt+1} failed: {e}", file=sys.stderr)
            await asyncio.sleep(3)
        print(f"[GLMMode] _replenish({idx}) UNAVAILABLE after 5 attempts", file=sys.stderr)
        self._pages[idx] = None
        self._replenishing.discard(idx)

    async def execute(self, prompt):
        idx = await self._queue.get()
        page = self._pages[idx]

        if page is None or not await self._check_health(page):
            print(f"[GLMMode] Page {idx} unhealthy on acquire, recreating", file=sys.stderr)
            await self._recreate_page(idx)
            page = self._pages[idx]

        try:
            result, _ = await _send_prompt_once(self.ctx, page, prompt, self.model)
            # Close page immediately and start creating a fresh one
            asyncio.create_task(self._replenish(idx))
            return result
        except Exception as e:
            print(f"[GLMMode] Page {idx} error: {e}, recreating", file=sys.stderr)
            await self._replenish(idx)  # await — need slot ready for next request
            raise

    async def chat(self, messages, tools=None):
        last_content = _build_prompt(messages)
        if not last_content:
            return {"role": "assistant", "content": "[GLMMode] No message"}
        if tools:
            desc = "\n".join(
                f"- {t['function']['name']}({', '.join(f'{p}: {v}' for p,v in t['function'].get('parameters',{}).get('properties',{}).items())})"
                for t in tools
            )
            prompt = f"You are an AI agent with tools:\n{desc}\n\nWhen calling a tool, respond ONLY with:\n{{\"tool\": \"name\", \"arguments\": {{...}}}}\n\nOtherwise respond normally.\n\n{last_content}"
            result = await self.execute(prompt)
        else:
            result = await self.execute(last_content)
        return _format_chat_result(result.get("text", ""), result.get("reasoning", ""))

    async def close(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        for p in self._pages:
            if p:
                try:
                    await p.close()
                except:
                    pass
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

def run_server(port=5001):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    pool = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal pool
        pool = GLMModePool(size=POOL_SIZE, model="glm-4.7")
        await pool.start()
        logging.basicConfig(level=logging.INFO)
        yield
        await pool.close()

    app = FastAPI(title="GLMMode API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class ChatRequest(BaseModel):
        model: str = "glm-4.7"
        messages: list
        stream: bool = False
        tools: list | None = None

    async def get_pool():
        nonlocal pool
        return pool

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        from fastapi.responses import StreamingResponse
        p = await get_pool()
        result = await p.chat(req.messages, tools=req.tools)

        cid = f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"
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
                d = {"content": content}
                yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":d,"finish_reason":None}]})}\n\n'
            if has_tc:
                tc = result["tool_calls"][0]
                d = {"tool_calls": [{"index": 0, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}
                yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":d,"finish_reason":None}]})}\n\n'
            yield f'data: {json.dumps({"id":cid,"object":"chat.completion.chunk","created":created,"model":req.model,"choices":[{"index":0,"delta":{},"finish_reason":finish}]})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "glm-4.7",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "glmmode",
                }
            ]
        }

    print(f"\n[GLMMode] API Server on http://0.0.0.0:{port}", file=sys.stderr)
    print(f"[GLMMode] Pool: {POOL_SIZE} pages", file=sys.stderr)
    print(f"[GLMMode] POST /v1/chat/completions\n", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    if "--server" in sys.argv:
        port = int(sys.argv[sys.argv.index("--server") + 1]) if "--server" in sys.argv and len(sys.argv) > sys.argv.index("--server") + 1 else 5001
        run_server(port)
    else:
        print("Usage: python glmmode.py --server [port]", file=sys.stderr)
        sys.exit(1)
