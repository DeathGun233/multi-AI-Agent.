# FlowPilot

FlowPilot is a multi-agent AI workflow lab focused on execution rather than retrieval. It complements a RAG knowledge base project by emphasizing task orchestration, tool calling, human review, prompt experiments, and AI observability.

## What It Does

- Runs four workflow types:
  - `sales_followup`
  - `marketing_campaign`
  - `support_triage`
  - `meeting_minutes`
- Orchestrates `Planner / Operator / Analyst / Content / Reviewer`
- Uses `LangGraph` for state-driven workflow execution
- Supports prompt profiles and model routing
- Tracks prompt, model, tokens, latency, and estimated cost
- Provides a monthly cost board and budget summary
- Runs batch prompt/model A/B experiments
- Turns manual review feedback into reusable evaluation samples
- Supports role-based access for `viewer / operator / reviewer / admin`

## Stack

- Python 3.11
- FastAPI
- LangGraph
- SQLAlchemy
- SQLite or MySQL
- Redis
- OpenAI Python SDK
- DashScope OpenAI-compatible endpoint

## Environment Variables

Key variables:

- `DASHSCOPE_API_KEY`
- `OPENAI_API_KEY`
- `OPEN_AI_KEY`
- `MODEL_NAME`
- `MODEL_BASE_URL`
- `DATABASE_URL`
- `REDIS_URL`
- `FLOWPILOT_SECRET_KEY`
- `FLOWPILOT_SESSION_COOKIE`
- `FLOWPILOT_USERS_JSON`
- `FLOWPILOT_MONTHLY_BUDGET_USD`

Default model endpoint:

- `https://dashscope.aliyuncs.com/compatible-mode/v1`

## Local Run

```bash
python -m uvicorn app.main:app --reload
```

Open:

- [http://127.0.0.1:8000/login](http://127.0.0.1:8000/login)

Demo accounts:

- `admin / admin123`
- `reviewer / reviewer123`
- `operator / operator123`
- `viewer / viewer123`

## Optional MySQL + Redis

```bash
docker compose -f docker-compose.infra.yml up -d
```

Then set:

```env
DATABASE_URL=mysql+pymysql://flowpilot:flowpilot@127.0.0.1:3306/flowpilot
REDIS_URL=redis://127.0.0.1:6379/0
```

## Tests

```bash
python -m pytest -q
```

## Main Pages

- `/dashboard`
- `/runs`
- `/reviews`
- `/compare`
- `/evaluations`
- `/prompts`
- `/costs`
- `/batches`

## Main APIs

- `GET /api/health`
- `GET /api/session`
- `GET /api/experiments/catalog`
- `GET /api/experiments/compare`
- `GET /api/workflows`
- `GET /api/workflows/graph`
- `GET /api/workflows/review-queue`
- `POST /api/workflows/run`
- `POST /api/workflows/{id}/review`
- `GET /api/evaluations`
- `GET /api/feedback-samples`
- `GET /api/costs/summary`
- `GET /api/batches`
- `GET /api/batches/{id}`
- `POST /api/batches`
- `POST /api/prompts`
- `PUT /api/prompts/{id}`

## Recent AI-Focused Additions

- Prompt history management and prompt profile editing
- Model routing policies for different agent roles
- Automated evaluation runs against a baseline
- Cost tracking and monthly budget dashboard
- Batch prompt/model A/B experiment runs
- Human review feedback loop that becomes evaluation data

## Docs

- [docs/多Agent工作-项目设计.md](./docs/多Agent工作-项目设计.md)
- [docs/第1步-MySQL与Redis升级.md](./docs/第1步-MySQL与Redis升级.md)
- [docs/第2步-LangGraph状态流升级.md](./docs/第2步-LangGraph状态流升级.md)
- [docs/第3步-前端工作台升级.md](./docs/第3步-前端工作台升级.md)
- [docs/第4步-多页面与权限升级.md](./docs/第4步-多页面与权限升级.md)
- [docs/第5步-数据库认证升级.md](./docs/第5步-数据库认证升级.md)
- [docs/第6步-AI运行指标升级.md](./docs/第6步-AI运行指标升级.md)
- [docs/第7步-Prompt版本管理与对比页.md](./docs/第7步-Prompt版本管理与对比页.md)
- [docs/第8步-Prompt路由与自动评测升级.md](./docs/第8步-Prompt路由与自动评测升级.md)
- [docs/第9步-成本看板批量实验与反馈闭环.md](./docs/第9步-成本看板批量实验与反馈闭环.md)
