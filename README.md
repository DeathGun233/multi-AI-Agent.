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

默认模型端点：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`

## 本地启动

```bash
python -m uvicorn app.main:app --reload
```

启动后打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

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
- `GET /api/health`
- `GET /api/workflows/templates`
- `GET /api/workflows`
- `GET /api/workflows/{id}`
- `POST /api/workflows/run`

## 项目说明

当前版本重点展示三件事：

- 多工作流、多 Agent 的执行编排
- 真实模型增强业务分析与审核
- MySQL + Redis 兼容的工程化持久化方案

## 文档

- 总体设计：[docs/多Agent工作-项目设计.md](./docs/多Agent工作-项目设计.md)
- 第一步升级说明：[docs/第1步-MySQL与Redis升级.md](./docs/第1步-MySQL与Redis升级.md)
