# 第 1 步：MySQL 与 Redis 升级说明

## 这一步做了什么

这次升级把原本只支持本地 `SQLite` 的版本，改造成了一个更接近正式工程的持久化架构：

- 数据层改为 `SQLAlchemy`
- 数据库连接改为 `DATABASE_URL`
- 原生支持 `MySQL`
- 保留 `SQLite` 作为本地默认运行方式
- 增加 `Redis` 缓存层
- 健康检查接口暴露数据库类型和 Redis 可用性

## 为什么这样改

前一个版本虽然已经能跑通工作流，但持久化能力更偏 MVP：

- 数据库实现写死在 `sqlite3`
- 无法平滑切换到 MySQL
- 没有缓存层
- 无法体现“正式项目的基础设施兼容性”

这一步的目标不是把所有生产能力一次性做满，而是先把底座换掉，让后续第二步 `LangGraph` 和第三步前端增强都能接在更合理的架构上。

## 当前实现结构

### 1. 数据库层

新增：

- [app/db.py](D:/explore_MANG/app/db.py)

职责：

- 定义 `SQLAlchemy` 的 `Base`
- 定义 `workflow_runs` ORM 模型
- 创建数据库引擎与会话
- 支持 `SQLite` 和 `MySQL`

### 2. 缓存层

新增：

- [app/cache.py](D:/explore_MANG/app/cache.py)

职责：

- 对接 `Redis`
- 缓存单条工作流运行结果
- Redis 不可用时自动回退到进程内存

### 3. 仓储层

重写：

- [app/repository.py](D:/explore_MANG/app/repository.py)

变化：

- 由原生 `sqlite3` 改为 `SQLAlchemy Session`
- 保存与查询都走 ORM
- 单条工作流查询优先读缓存，再回落数据库

### 4. 配置层

更新：

- [app/config.py](D:/explore_MANG/app/config.py)

新增配置：

- `DATABASE_URL`
- `REDIS_URL`
- `database_backend`

### 5. 启动层

更新：

- [app/main.py](D:/explore_MANG/app/main.py)

变化：

- 启动时初始化 `Database`
- 启动时初始化 `CacheStore`
- `health` 接口返回数据库类型和 Redis 状态

## 支持的运行模式

### 模式 A：本地快速运行

使用默认配置：

- `DATABASE_URL=sqlite:///flowpilot.db`
- `REDIS_URL=redis://127.0.0.1:6379/0`

即使本地没起 Redis，也能运行，因为会自动回退。

### 模式 B：正式一些的本地环境

用 `docker-compose.infra.yml` 拉起：

- MySQL 8.4
- Redis 7.4

然后设置：

```env
DATABASE_URL=mysql+pymysql://flowpilot:flowpilot@127.0.0.1:3306/flowpilot
REDIS_URL=redis://127.0.0.1:6379/0
```

## 这一步带来的价值

### 1. 和岗位要求更贴近

现在项目已经不只是“会做工作流页面和接口”，而是能明确展示：

- `MySQL`
- `Redis`
- `SQLAlchemy`
- 持久化设计
- 缓存回退
- 基础设施兼容能力

### 2. 给后续升级留好了接口

后面做第二步和第三步时，不需要再返工底层：

- 第二步做 `LangGraph` 可以直接沿用仓储层
- 第三步做历史页、轨迹页、审核页时也有稳定数据源

### 3. 保留了本地易用性

虽然新增了 MySQL 和 Redis 兼容能力，但仍然没有牺牲本地开发体验：

- 不配 MySQL 也能跑
- 不起 Redis 也能跑
- 测试环境也能跑

## 当前边界

这一步还没有做这些内容：

- 多表拆分
- Redis 队列
- 任务异步消费
- 数据库迁移工具
- 更细粒度的 Agent / Tool Call 表

这些留给后续继续升级时处理。

## 验证方式

本地验证项：

- `python -m pytest -q`
- `GET /api/health`
- 提交至少一个工作流并确认数据库有记录

接口期望：

- `database_backend` 正确显示
- `redis_enabled` 能反映当前 Redis 是否可用
- 工作流结果仍然可正常返回
