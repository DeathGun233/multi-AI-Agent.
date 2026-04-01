# FlowPilot

FlowPilot 是一个面向企业场景的多 Agent AI 工作流实验平台，核心目标不是“检索信息”，而是“把任务执行完”。  
它适合和你已有的企业级 RAG 知识库项目形成互补：前者解决“找信息”，这个项目解决“做任务、跑流程、做评测、看成本、做优化”。

## 项目定位

项目聚焦以下 AI 应用开发能力：

- 多 Agent 协作执行
- LangGraph 状态流编排
- 工具调用与工作流路由
- Prompt 版本管理
- 多模型路由与对比
- AI 可观测性与成本统计
- 自动评测与人工反馈回流
- 企业工作台、审核流和执行轨迹展示

## 当前能力

### 1. 多 Agent 工作流执行

系统内置 5 个核心角色：

- `PlannerAgent`：任务规划与步骤拆解
- `OperatorAgent`：工具调用与结构化执行
- `AnalystAgent`：结果分析与行动建议
- `ContentAgent`：业务输出补全与内容生成
- `ReviewerAgent`：质量判断与人工接管建议

当前支持 4 条示例工作流：

- `sales_followup`：销售分析与跟进计划
- `marketing_campaign`：营销内容生成
- `support_triage`：客服工单分流
- `meeting_minutes`：会议纪要转执行项

### 2. AI 应用实验能力

- Prompt 历史版本新增与编辑
- Prompt 方案切换与对比
- 多模型路由策略
- Prompt / 模型批量 A/B 实验
- 自动评测集运行与基线对比
- 人工审核反馈自动回流成评测样本

### 3. AI 可观测性与成本管理

- 记录每次 LLM 调用的 `模型 / Prompt / Tokens / 耗时 / 回退情况`
- 汇总工作流级 AI 调用指标
- 成本统计与预算看板
- 预算阈值提醒
- 成本趋势展示
- 批量实验结果排序与冠军方案标记

### 4. 企业工作台能力

- 运行历史页
- 人工审核页
- 执行轨迹页
- 中文化多页面前端工作台
- 登录态与角色权限
- 工作流删除能力

支持角色：

- `viewer`
- `operator`
- `reviewer`
- `admin`

## 技术栈

- Python 3.11
- FastAPI
- LangGraph
- SQLAlchemy
- SQLite / MySQL
- Redis
- OpenAI Python SDK
- 阿里 DashScope OpenAI 兼容接口
- Jinja2 多页面前端

## 模型接入

默认使用阿里兼容 OpenAI 协议的模型接口，当前可直接接入：

- `qwen3-max`

默认兼容端点：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`

支持通过环境变量切换模型和端点。

## 环境变量

常用环境变量如下：

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

如果你已经配置了 `OPEN_AI_KEY` 或 `DASHSCOPE_API_KEY`，项目可直接调用模型。

## 本地启动

```bash
python -m uvicorn app.main:app --reload
```

打开：

- [http://127.0.0.1:8000/login](http://127.0.0.1:8000/login)

默认演示账号：

- `admin / admin123`
- `reviewer / reviewer123`
- `operator / operator123`
- `viewer / viewer123`

## 可选基础设施

如果你想切到 MySQL + Redis 运行，可先启动基础设施：

```bash
docker compose -f docker-compose.infra.yml up -d
```

然后设置：

```env
DATABASE_URL=mysql+pymysql://flowpilot:flowpilot@127.0.0.1:3306/flowpilot
REDIS_URL=redis://127.0.0.1:6379/0
```

## 测试

```bash
python -m pytest -q
```

## 主要页面

- `/dashboard`：总览面板
- `/runs`：运行历史
- `/reviews`：审核中心
- `/compare`：Prompt / 模型对比页
- `/evaluations`：评测结果页
- `/prompts`：Prompt 管理页
- `/costs`：成本看板
- `/batches`：批量实验页

## 主要接口

- `GET /api/health`
- `GET /api/session`
- `GET /api/workflows`
- `GET /api/workflows/graph`
- `GET /api/workflows/review-queue`
- `POST /api/workflows/run`
- `POST /api/workflows/{id}/review`
- `DELETE /api/workflows/{id}`
- `GET /api/experiments/catalog`
- `GET /api/experiments/compare`
- `GET /api/evaluations`
- `GET /api/feedback-samples`
- `GET /api/costs/summary`
- `GET /api/batches`
- `GET /api/batches/{id}`
- `POST /api/batches`
- `POST /api/prompts`
- `PUT /api/prompts/{id}`

## 使用建议

如果你是第一次体验，建议按下面顺序看：

1. 登录 `/login`
2. 到 `/dashboard` 发起一个销售或客服工作流
3. 到 `/runs` 查看执行记录
4. 进入详情页查看执行轨迹、AI 调用指标和结果 JSON
5. 如果任务进入人工接管，到 `/reviews` 处理审核
6. 到 `/compare`、`/evaluations`、`/costs`、`/batches` 查看实验、评测与成本表现

## 适合简历/面试怎么讲

这个项目最适合强调的方向是：

- 我不只做 RAG，也做 Agent 执行与工作流编排
- 我能把 Prompt、模型、Tokens、耗时、成本做成可观测指标
- 我支持 Prompt 管理、多模型路由、自动评测、人工反馈闭环
- 我能把 AI 应用做成企业工作台，而不是单一聊天 Demo

## 文档索引

- [多Agent工作-项目设计](./docs/多Agent工作-项目设计.md)
- [第1步-MySQL与Redis升级](./docs/第1步-MySQL与Redis升级.md)
- [第2步-LangGraph状态流升级](./docs/第2步-LangGraph状态流升级.md)
- [第3步-前端工作台升级](./docs/第3步-前端工作台升级.md)
- [第4步-多页面与权限升级](./docs/第4步-多页面与权限升级.md)
- [第5步-数据库认证升级](./docs/第5步-数据库认证升级.md)
- [第6步-AI运行指标升级](./docs/第6步-AI运行指标升级.md)
- [第7步-Prompt版本管理与对比页](./docs/第7步-Prompt版本管理与对比页.md)
- [第8步-Prompt路由与自动评测升级](./docs/第8步-Prompt路由与自动评测升级.md)
- [第9步-成本看板批量实验与反馈闭环](./docs/第9步-成本看板批量实验与反馈闭环.md)
