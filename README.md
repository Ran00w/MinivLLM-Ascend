# MinivLLM-Ascend

这是一个面向学习的 mini vLLM on Ascend/NPU 项目。目标不是一开始追求生产级性能，而是把 LLM 推理引擎拆成可以逐步理解、逐步实现、逐步记录的模块。

推荐从这里开始读：

- [项目总目标](docs/00_project_goal.md)
- [项目结构与模块边界](docs/01_architecture_map.md)
- [迭代开发路线图](docs/02_development_roadmap.md)
- [Git/GitHub 开发流程](docs/03_git_github_workflow.md)
- [阶段文档目录](docs/README.md)

最终学习目标：

- 在 Ascend NPU 上跑通一个单卡 mini-vLLM。
- 支持 Qwen3-0.6B 或同级别小模型。
- 支持 prefill/decode 两阶段推理。
- 支持 KV cache 和 paged KV cache。
- 支持多个 prompt 的简单 batch 调度。
- 优先使用 `torch_npu` 官方算子，不把自写 Ascend C kernel 作为早期目标。

当前推荐开发分支：

```bash
git checkout -b docs/learning-roadmap
```

文档优先级高于代码。每完成一个阶段，都要把“学到了什么、踩了什么坑、下一步为什么这么做”记录下来。
