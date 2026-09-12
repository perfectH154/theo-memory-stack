# Ombre Recall Sidecar

这是一个放在 Ombre Brain 旁边的只读召回索引原型。

设计目标：

- Ombre 的 Markdown bucket 仍然是唯一真实数据源。
- 索引只从 bucket 读取，绝不写回源文件。
- 自动召回不调用现有 `breath`，因此不会修改 `activation_count` 或 `last_active`。
- 使用 BM25/FTS5 与 embedding 结果做混合排序。
- 索引丢失时可以从源 Markdown 完整重建。

目录结构建议：

```text
/var/lib/companion/ombre-brain/buckets/       # 原始记忆，不改动
/var/lib/companion/ombre-brain/recall/        # 可删除、可重建的旁路索引
  recall.sqlite3
```

当前实现分为两部分：

- `indexer.py`：离线扫描 bucket，建立 SQLite FTS5 和可选 embedding 索引。
- `service.py`：只绑定 `127.0.0.1` 的 HTTP 查询服务，供 bridge-v2 调用。

初次部署建议先使用 `--no-embeddings` 验证文件解析和 BM25 结果，再安装本地 embedding 后重建索引。当前 VPS 使用 `BAAI/bge-small-zh-v1.5`，服务返回的分数是当前候选集合内的相对排序分数，不应直接当作概率。

## 本地验证

```powershell
python -m py_compile indexer.py service.py
```

## VPS 运行示例

```bash
python indexer.py \
  --buckets-dir /var/lib/companion/ombre-brain/buckets \
  --index-db /var/lib/companion/ombre-brain/recall/recall.sqlite3 \
  --no-embeddings

python service.py \
  --index-db /var/lib/companion/ombre-brain/recall/recall.sqlite3 \
  --host 127.0.0.1 \
  --port 8788
```

正式部署使用 `systemd/companion-recall-sidecar.service`，索引刷新使用 `systemd/companion-recall-index.timer`。bridge 的 `companion-bridge-v2-recall.conf` 目前只打开 shadow mode；真正注入前应先观察日志中的命中率和延迟。

索引文件应设置为仅运行用户可读。不要把该旁路服务暴露到公网。
