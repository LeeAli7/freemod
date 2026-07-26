# GLMMode

OpenAI-compatible API server that routes requests through **chat.z.ai** (GLM-4.7) using Playwright headless browser automation.

## How It Works

GLMMode launches a headless Chromium browser pointing to chat.z.ai, maintains a pool of pages, intercepts SSE responses from the `/api/v2/chat/completions` endpoint, parses them back into OpenAI-compatible chat completion responses, and supports tool calls via JSON prompt injection.

```
Client (opencode, curl, etc.) -> GLMMode :5001 -> Playwright -> chat.z.ai
```

## Requirements

- Python 3.10+
- Chromium (installed by Playwright)

## Installation

```bash
# 1. Clone
git clone https://github.com/LeeAli7/glmmode.git
cd glmmode

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright Chromium
playwright install chromium
```

## Usage

```bash
# Default port 5001
python glmmode.py --server

# Custom port
python glmmode.py --server 5001
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

### Quick test

```bash
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5001/v1/models

curl http://127.0.0.1:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

### With tool calls

```bash
curl http://127.0.0.1:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "What is 15 * 7?"}],
    "tools": [{
      "function": {
        "name": "calculator",
        "description": "Evaluate math expressions",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string"}
          },
          "required": ["expression"]
        }
      }
    }]
  }'
```

## Configuration

No config file needed. All settings are in `glmmode.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `URL` | `https://chat.z.ai` | Target chat URL |
| `POOL_SIZE` | `3` | Number of concurrent browser pages |
| `SSE_TIMEOUT` | `120` | SSE response wait timeout (seconds) |
| `MAX_ATTEMPTS` | `2` | Retry attempts on failure |

## opencode Integration

Add to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "glmmode": {
      "name": "GLMMode",
      "npm": "@ai-sdk/openai-compatible",
      "env": ["GLMMODE_API_KEY"],
      "options": {
        "apiKey": "public",
        "baseURL": "http://127.0.0.1:5001/v1",
        "timeout": 300000
      },
      "models": {
        "glm-4.7": {
          "id": "glm-4.7",
          "name": "GLM-4.7",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 128000, "output": 4096 },
          "cost": { "input": 0, "output": 0 }
        }
      }
    }
  }
}
```

## Linux Service (systemd)

Create `/etc/systemd/system/glmmode.service`:

```ini
[Unit]
Description=GLMMode API Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/glmmode
ExecStart=/path/to/glmmode/venv/bin/python glmmode.py --server 5001
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now glmmode
sudo systemctl status glmmode
```

## Notes

- The server profile is ephemeral — each restart gets a fresh session
- Rate limits are IP/session-based (typically reset daily ~21h after first request)
- Captcha detection triggers automatic page refresh
