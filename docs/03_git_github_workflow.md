# Git/GitHub 开发流程

## 基本原则

所有开发都走分支和 PR。

不要直接在 `main` 上提交功能代码。`main` 应该始终保持可运行或至少文档一致。

## 第一次同步远端

```bash
git status --short --branch
git fetch origin
git pull --ff-only origin main
```

如果远端是 SSH 地址，需要本机配置 GitHub SSH key。不要把账号密码写进命令、脚本、文档或 remote URL。

推荐使用 SSH：

```bash
git remote -v
ssh -T git@github.com
```

如果使用 HTTPS，GitHub 一般需要 token，而不是账号密码。

## 新建分支

分支命名建议：

```text
docs/learning-roadmap
feat/backend-abstraction
feat/tiny-transformer
feat/contiguous-kv-cache
feat/paged-kv-cache
feat/npu-attention
```

命令：

```bash
git checkout main
git pull --ff-only origin main
git checkout -b docs/learning-roadmap
```

## 提交前检查

```bash
git status --short
git diff
```

如果有测试：

```bash
pytest
```

如果只是文档，至少检查文件是否能被正常打开，链接是否明显写错。

## 提交

```bash
git add README.md docs
git commit -m "docs: add learning roadmap"
```

commit 信息格式建议：

```text
docs: add learning roadmap
feat: add backend abstraction
fix: correct kv cache slot mapping
test: add scheduler tests
refactor: split attention backends
```

## 推送分支

```bash
git push -u origin docs/learning-roadmap
```

## 创建 PR

如果安装了 GitHub CLI 并已登录：

```bash
gh pr create \
  --base main \
  --head docs/learning-roadmap \
  --title "docs: add learning roadmap" \
  --body "Add project learning roadmap, staged development plan, and Git workflow."
```

如果没有 GitHub CLI，就在 GitHub 网页上创建 PR。

## PR 内容模板

```text
## Summary

- Add project goal document.
- Add staged development roadmap.
- Add Git/GitHub workflow.
- Add stage templates.

## Validation

- Documentation-only change.
- Checked file structure and links manually.
```

## 不要做的事情

- 不要把密码、token、cookie、私钥写入仓库。
- 不要把认证信息写入 remote URL。
- 不要提交 `.codex`、本地缓存、模型权重、日志、大文件。
- 不要在一个 PR 里同时做文档、模型、attention、scheduler 多个大改动。

## 推荐 PR 粒度

一个 PR 只做一个阶段。

例子：

- PR 1：只加学习路线文档。
- PR 2：只加 tiny model 最小推理。
- PR 3：只加 backend abstraction。
- PR 4：只加连续 KV cache。
