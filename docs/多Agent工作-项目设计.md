# 多Agent工作项目设计

## 1. 项目定位

`多Agent工作` 是一个偏执行型的企业级 AI 工作流平台，用来补足传统 RAG 项目“能找信息但不一定能把事情做完”的短板。项目重点不是知识检索，而是：

- 任务拆解
- 多 Agent 协作
- 工具调用
- 业务流程自动化
- 人工接管
- 结果评测与流程可观测

一句话定位：

**一个面向企业运营、销售、市场和客服场景的多智能体执行平台，重点解决任务自动拆解、工具调用、流程编排、人工审核与执行闭环。**

## 2. 为什么这个项目适合岗位要求

相比重复做一个 RAG 变体，这个项目更能覆盖以下能力关键词：

- `Agent`
- `Multi-Agent`
- `Function Call`
- `Tool Use`
- `Workflow / Agentic Workflow`
- `Context Engineering`
- `Prompt Engineering`
- `Python`
- `FastAPI`
- `MySQL / Redis / SQLite`
- `数据分析`
- `营销内容生成`
- `企业级 AI 应用落地`

它和已有的企业级 RAG 项目可以形成互补：

- RAG 项目证明“会找信息、会检索、会知识问答”
- 多 Agent 工作项目证明“会拆任务、会执行流程、会做自动化协同”

## 3. 核心业务场景

项目围绕 4 类典型企业工作流设计：

### 3.1 销售分析与跟进计划

输入销售周期、区域和销售人员，系统自动：

- 分析销售漏斗
- 找出高风险客户
- 识别转化瓶颈
- 生成下一步跟进建议
- 输出日报摘要给管理者

### 3.2 营销内容工厂

输入产品信息、目标用户、投放渠道和核心卖点，系统自动：

- 生成小红书文案
- 生成抖音脚本
- 生成微信内容摘要
- 生成 A/B 测试版本
- 给出投放建议和人工审核提示

### 3.3 客服工单智能分流

输入一批工单文本，系统自动：

- 判断工单类型
- 识别优先级
- 生成回复草稿
- 对高风险工单触发人工接管
- 输出升级建议

### 3.4 会议纪要转执行系统

输入会议纪要，系统自动：

- 抽取行动项
- 分配负责人
- 提取截止时间
- 生成会后总结邮件
- 对时间不明确的事项打回人工确认

## 4. 多 Agent 角色设计

项目采用 5 个职责清晰的 Agent：

### 4.1 Planner Agent

职责：

- 理解用户目标
- 生成工作流执行计划
- 定义每一步预期输出

### 4.2 Operator Agent

职责：

- 调用工具执行任务
- 聚合工具输出结果
- 记录执行日志

### 4.3 Analyst Agent

职责：

- 解读工具结果
- 提炼洞察
- 生成行动建议

该角色已接入真实模型，可用阿里兼容 OpenAI 接口生成中文业务结论。

### 4.4 Content Agent

职责：

- 把分析结果转成业务可直接使用的输出
- 生成日报、营销备注、交接说明、会后邮件等内容

该角色已接入真实模型。

### 4.5 Reviewer Agent

职责：

- 判断结果是否完整
- 判断是否需要人工接管
- 给出审核分数和原因

该角色已接入真实模型，同时保留稳定的回退规则。

## 5. 系统架构

### 5.1 当前实现架构

- 前端：Jinja2 模板页，用于选择工作流模板并提交 JSON
- API 层：FastAPI
- 编排层：`WorkflowEngine`
- Agent 层：Planner / Operator / Analyst / Content / Reviewer
- 工具层：销售分析、内容工厂、工单分流、会议行动项提取
- 持久化：SQLite
- 模型层：OpenAI Python SDK + DashScope OpenAI 兼容接口

### 5.2 后续可扩展架构

- 前端可替换为 React / Next.js
- 数据库可切换为 MySQL
- 缓存与队列可接 Redis / Celery
- Agent 编排可升级到 LangGraph
- 工具中心可接 CRM、ERP、邮件、飞书、钉钉等真实系统

## 6. 数据模型设计

当前数据库核心表为 `workflow_runs`，已经实现持久化。

字段包括：

- `id`
- `workflow_type`
- `status`
- `current_step`
- `objective`
- `input_payload`
- `plan_json`
- `result_json`
- `review_json`
- `logs_json`
- `created_at`
- `updated_at`

后续建议扩展的表：

- `workflow_templates`
- `agent_runs`
- `tool_calls`
- `human_reviews`
- `metrics_snapshots`

## 7. API 设计

当前已实现：

- `GET /`
- `GET /api/health`
- `GET /api/workflows/templates`
- `GET /api/workflows`
- `GET /api/workflows/{id}`
- `POST /api/workflows/run`

后续建议补充：

- `POST /api/workflows/retry`
- `POST /api/review/submit`
- `GET /api/metrics/summary`
- `GET /api/workflows/{id}/logs`

## 8. 为什么要接真实模型

为了让这个项目不只是“规则引擎 + 模板拼接”，需要在关键角色上接入真实大模型：

- Analyst Agent：生成业务分析结论
- Content Agent：生成业务可用输出
- Reviewer Agent：做审核判断和解释

这样项目更接近真实企业场景，也更匹配岗位中的：

- LLM 应用
- Prompt Engineering
- Agent
- Workflow
- 多智能体协作

## 9. 当前模型接入策略

模型优先级：

1. `DASHSCOPE_API_KEY`
2. `OPENAI_API_KEY`
3. `OPEN_AI_KEY`

默认模型：

- `qwen3-max`

默认兼容端点：

- `https://dashscope.aliyuncs.com/compatible-mode/v1`

同时提供降级策略：

- 如果环境变量缺失，则自动回退到本地规则逻辑
- 如果模型请求失败，则自动回退到本地规则逻辑
- 测试环境下默认禁用真实模型，保证自动化测试稳定

## 10. 当前数据库接入策略

当前默认采用本地 `SQLite` 文件：

- 默认路径：`flowpilot.db`
- 可通过 `DATABASE_URL` 覆盖

这样做的原因：

- 你当前已经配了模型相关环境变量，但没有给数据库连接变量
- SQLite 是真正的持久化数据库，足够支撑 MVP 演示和面试展示
- 未来迁移到 MySQL 成本较低

## 11. 工作流状态机

当前工作流状态包括：

- `created`
- `planning`
- `executing`
- `reviewing`
- `waiting_human`
- `completed`
- `failed`

其中：

- 销售分析通常可直接 `completed`
- 营销内容默认建议人工品牌复核，因此常为 `waiting_human`
- 紧急工单默认进入 `waiting_human`
- 会议纪要若缺截止时间也会进入 `waiting_human`

## 12. 人工接管设计

这是项目区别于普通 AI Demo 的关键点之一。

人工接管触发条件示例：

- 工单涉及生产故障
- 内容需要品牌或合规复核
- 会议纪要关键信息缺失

这样可以体现：

- 企业级落地思维
- 风险控制意识
- 流程可控性

## 13. 面试可讲亮点

### 13.1 项目不重复

它不是再做一个 RAG，而是补齐 Agent 执行和流程自动化能力。

### 13.2 真实模型接入

不是纯本地模板，而是把阿里兼容 OpenAI 接口接到了业务 Agent 上。

### 13.3 持久化和可追踪

每一次工作流运行都会写入数据库，保留输入、计划、日志、审核和最终结果。

### 13.4 结构化输出

所有 Agent 都尽量产出结构化 JSON，便于后续接入前端、数据库、BI 或审批系统。

### 13.5 工程化回退

即使模型不可用，系统仍能靠回退策略完成核心流程，不至于完全瘫痪。

## 14. 下一阶段升级路线

如果继续往下做，建议按这个顺序升级：

### 阶段一：数据库增强

- 拆分 `workflow_runs` 为多张表
- 增加 `tool_calls` 和 `agent_runs`
- 增加分页和筛选查询

### 阶段二：模型与 Prompt 管理

- 抽 Prompt 模板
- 记录 Prompt 版本
- 增加模型切换能力

### 阶段三：前端增强

- 增加运行历史列表
- 增加执行轨迹可视化
- 增加人工审核操作面板

### 阶段四：真实业务集成

- 接 CRM 数据
- 接邮件通知
- 接飞书/钉钉任务
- 接企业内部工单系统

### 阶段五：评测能力

- 引入任务完成率
- 引入人工接管率
- 引入工具成功率
- 引入平均执行时长和模型成本统计

## 15. 结论

`多Agent工作` 适合作为你现有 RAG 项目的补充型主项目。它更强调：

- AI 如何真正参与企业流程执行
- 多智能体如何协同完成业务任务
- 如何在真实工程里兼顾自动化、可控性和可扩展性

从简历和面试视角看，这个项目的价值在于它能展示你不只是会“做问答”，而是会“做可落地的 AI 系统”。
