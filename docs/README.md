# 文档目录

这套文档把项目拆成十个阶段。每个阶段都有固定格式：

- 学习目标
- 本阶段不做什么
- 建议新增或修改的文件
- 任务清单
- 验收标准
- 推荐记录的问题

## 总文档

- [00_project_goal.md](00_project_goal.md)：项目目标、范围、非目标
- [01_architecture_map.md](01_architecture_map.md)：最终项目结构和模块职责
- [02_development_roadmap.md](02_development_roadmap.md)：完整迭代路线
- [03_git_github_workflow.md](03_git_github_workflow.md)：分支、提交、推送、PR 流程
- [04_stage_task_board.md](04_stage_task_board.md)：阶段任务看板

## 阶段文档

- [stage 00：读懂原 MinivLLM](stages/00_read_original_project.md)
- [stage 01：最小 PyTorch 推理闭环](stages/01_minimal_torch_inference.md)
- [stage 02：设备后端抽象与 NPU eager](stages/02_backend_abstraction_npu_eager.md)
- [stage 03：Qwen3 模型结构与权重加载](stages/03_qwen3_model_loading.md)
- [stage 04：连续 KV cache](stages/04_contiguous_kv_cache.md)
- [stage 05：paged KV cache 学习版](stages/05_paged_kv_cache.md)
- [stage 06：Ascend attention 算子](stages/06_ascend_attention_ops.md)
- [stage 07：batch engine 和 scheduler](stages/07_batch_engine_scheduler.md)
- [stage 08：graph 与性能实验](stages/08_graph_performance.md)
- [stage 09：多卡 tensor parallel](stages/09_tensor_parallel.md)

## 模板

- [阶段学习记录模板](templates/stage_note.md)
- [实验记录模板](templates/experiment_log.md)
