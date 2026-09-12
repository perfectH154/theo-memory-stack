# Theo Memory Stack

这是 Theo 当前使用的“记忆库 + 语义检索”代码快照，只包含记忆相关部分，不包含 Telegram、PWA、Claude 桥接或其他伴侣业务代码。

## 内容

- `memory/ombre-brain`：Ombre Brain 记忆核心源码。负责记忆桶、情绪坐标、遗忘曲线、MCP 工具和记忆写入接口。
- `retrieval/recall-sidecar`：只读检索侧车。读取已有 Ombre Markdown 桶，生成/使用本地语义索引，并提供检索 API。
- `integration/recall-client.js`：当前桥接层使用的召回客户端，包含模型主导的召回判断、候选检索、重排和上下文注入逻辑。
- `deploy/systemd`：脱敏后的服务单元示例。
- `config/recall.env.example`：召回相关环境变量示例，不含任何密钥。

## 明确不包含

本仓库没有、也不应加入：

- 真实 Ombre 记忆桶 Markdown；
- `recall.sqlite3` 或任何向量数据库；
- Telegram 聊天、附件和会话状态；
- `.env`、API key、Telegram token、Bridge token 或登录态；
- 日志、缓存、虚拟环境、依赖目录、备份和构建产物；
- 指向真实记忆目录的符号链接。

当前 VPS 的记忆库仍然保留在原位置。部署检索侧车时，应把 `--buckets-dir` 指向现有目录，并把索引数据库放在单独的数据目录；不要把数据复制进 Git 仓库。

## 运行关系

```text
Ombre Brain Markdown buckets
          │
          ├── recall-sidecar indexer ──> local recall.sqlite3
          │                              （不进 Git）
          └── recall-sidecar /search
                         ▲
                         │
                 integration/recall-client.js
                         │
              返回少量、经过筛选的 memory_context
```

召回客户端默认要求检索服务返回 `read_only=true`，不会通过检索 API 修改记忆。记忆写入仍由 Ombre Brain 的 `hold`/`grow` 等写入工具负责；本仓库没有带出任何已有记忆内容。

## 上游

Ombre Brain 是引用的上游组件，不是本项目原创。已在 GitHub 核对并记录为 [`P0luz/Ombre-Brain`](https://github.com/P0luz/Ombre-Brain)。详细来源、许可证和本地快照说明见 [`UPSTREAMS.md`](UPSTREAMS.md)。

## 安全默认值

- 服务监听检索 API 时使用 `127.0.0.1`；
- API key 只通过本机环境或密钥管理注入；
- 记忆桶和向量索引与代码仓库分离；
- 任何自动写入策略都应在接入前以 shadow/dry-run 模式验证。
