# FlowPilot

`FlowPilot` 是一个偏执行型的企业 AI 工作流自动化项目，用来补足“企业级 RAG 知识库搜索”之外的能力面。它不以检索为中心，而以任务拆解、多 Agent 协作、工具调用、人工接管和流程可观测为核心。

## 当前能力

- `Planner / Operator / Analyst / Content / Reviewer` 五类 Agent 协作
- 销售分析与跟进计划
- 营销内容工厂
- 客服工单智能分流
- 会议纪要转执行系统
- 真实模型接入：默认 `qwen3-max`
- 真实数据库兼容：`SQLite / MySQL`
- 缓存兼容：`Redis`
- 多页面工作台：仪表盘、运行历史、审核中心、详情页
- 图形化执行时间线
- 登录态与角色权限：`viewer / operator / reviewer / admin`

## 技术栈

- `Python 3.11`
- `FastAPI`
- `SQLAlchemy`
- `SQLite / MySQL`
- `Redis`
- `OpenAI Python SDK`
- DashScope OpenAI 兼容接口

## 环境变量

样例见 [.env.example](./.env.example)。

关键变量：

- `DASHSCOPE_API_KEY`
- `OPENAI_API_KEY`
- `MODEL_NAME`
- `MODEL_BASE_URL`
- `DATABASE_URL`
- `REDIS_URL`
- `FLOWPILOT_SECRET_KEY`
- `FLOWPILOT_SESSION_COOKIE`
- `FLOWPILOT_USERS_JSON`

默认模型端点：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`

## 本地启动

```bash
python -m uvicorn app.main:app --reload
```

启动后打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

默认演示账号：

- `admin / admin123`
- `reviewer / reviewer123`
- `operator / operator123`
- `viewer / viewer123`

## 启动 MySQL 和 Redis

如果你想切到更接近正式环境的方式，可以先启动基础设施：

```bash
docker compose -f docker-compose.infra.yml up -d
```

然后把环境变量切到：

```env
DATABASE_URL=mysql+pymysql://flowpilot:flowpilot@127.0.0.1:3306/flowpilot
REDIS_URL=redis://127.0.0.1:6379/0
```

## 测试

```bash
python -m pytest -q
```

## API

- `GET /`
- `GET /login`
- `POST /login`
- `POST /logout`
- `GET /dashboard`
- `GET /runs`
- `GET /reviews`
- `GET /runs/{id}`
- `GET /api/health`
- `GET /api/session`
- `GET /api/workflows/templates`
- `GET /api/workflows`
- `GET /api/workflows/review-queue`
- `GET /api/workflows/graph`
- `GET /api/workflows/{id}`
- `POST /api/workflows/run`
- `POST /api/workflows/{id}/review`

## 项目说明

当前版本重点展示三件事：

- 多工作流、多 Agent 的执行编排
- 真实模型增强业务分析与审核
- MySQL + Redis 兼容的工程化持久化方案
- 工作台从单页升级为多页面后台，补上运行历史、审核中心和详情页
- 基于角色的登录态与审核权限控制
- 任务详情页提供图形化执行时间线，方便演示执行轨迹

## 文档

- 总体设计：[docs/多Agent工作-项目设计.md](./docs/多Agent工作-项目设计.md)
- 第一步升级说明：[docs/第1步-MySQL与Redis升级.md](./docs/第1步-MySQL与Redis升级.md)
- 第二步升级说明：[docs/第2步-LangGraph状态流升级.md](./docs/第2步-LangGraph状态流升级.md)
- 第三步升级说明：[docs/第3步-前端工作台升级.md](./docs/第3步-前端工作台升级.md)
- 第四步升级说明：[docs/第4步-多页面与权限升级.md](./docs/第4步-多页面与权限升级.md)
