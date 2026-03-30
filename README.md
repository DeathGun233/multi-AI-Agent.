# FlowPilot

`FlowPilot` 是一个偏执行型的企业 AI 工作流自动化项目，用来补足“企业级 RAG 知识库搜索”之外的能力面。它不以检索为中心，而以任务拆解、多 Agent 协作、工具调用、人工接管和流程可观测为核心。

## 覆盖能力

- `Planner / Operator / Analyst / Content / Reviewer` 五类 Agent 协作
- 销售分析与跟进计划
- 营销内容生成与 A/B 版本建议
- 客服工单智能分流与升级建议
- 会议纪要转行动项与会后邮件
- 工作流状态机、执行日志、人工接管判断

## 技术栈

- `Python 3.11`
- `FastAPI`
- `Jinja2`
- `SQLite`
- `OpenAI Python SDK`
- DashScope OpenAI 兼容接口

## 模型与数据库

- 默认模型：`qwen3-max`
- 默认端点：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 默认数据库：本地 `flowpilot.db`
- 环境变量样例见 [.env.example](./.env.example)

模型环境变量优先级：

1. `DASHSCOPE_API_KEY`
2. `OPENAI_API_KEY`
3. `OPEN_AI_KEY`

如果模型调用不可用，系统会自动回退到本地规则逻辑，保证工作流仍能执行。

## 快速启动

```bash
python -m uvicorn app.main:app --reload
```

启动后打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

## 测试

```bash
python -m pytest -q
```

## API

- `GET /api/health`
- `GET /api/workflows/templates`
- `GET /api/workflows`
- `GET /api/workflows/{id}`
- `POST /api/workflows/run`

## 项目说明

这个版本是一个可运行 MVP，重点展示工程思路与产品包装方式：

- 用工作流模板覆盖多个企业场景，避免和现有 RAG 项目重复
- 用状态机和 Agent 日志表现执行链路
- 用 Reviewer Agent 做人工接管判断，模拟企业级可控性
- 用 SQLite 持久化每次运行记录
- 用真实模型增强 Analyst / Content / Reviewer 三类 Agent
- 用结构化输出方便继续接数据库、消息队列或内部系统

## 详细设计文档

完整设计思路见 [docs/多Agent工作-项目设计.md](./docs/多Agent工作-项目设计.md)。
