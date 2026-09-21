# MiniLOps —— "封装算子"实操 demo 合集

> 定位：领导说的"把模型封装成算子集成到工作流"的**零依赖玩具版**。
> 你在这里体验的每一环，都是那条产线的微观缩影。
>
> ## 环境要求
> - python 3.11 + numpy + flask（你机器上全部已装）
> - **不需要** torch / PIL / onnxruntime（那是 E1~E5 的活，装好后回来衔接）
>
> ## 跑法
> ```bash
> python -X utf8 minilops/run_all.py      # 一键全跑 D1→D4+retrain，最后打印总结
> # D4 服务化单独玩：
> python -X utf8 minilops/server.py       # 起服务（端口 8790）
> curl -X POST http://127.0.0.1:8790/detect -H "Content-Type: application/json" -d "{\"image\": [[0,0,0],[0,255,0],[255,255,0]]}"
> ```
>
> ## 文件地图（按依赖顺序读）
> | 文件 | 对应真实产线的什么 |
> |---|---|
> | `preprocess.py` (D1) | 预处理算子：缩放/归一化——产线算子的输入适配层 |
> | `operator_core.py` (D2) | 算子内核：检测"模型"本体 + **计算图雏形**（列表模拟 ONNX 图） |
> | `workflow.py` (D3) | 工作流引擎：算子注册 + 编排 + 模拟标注产线跑批 |
> | `retrain_feedback.py` | 数据闭环：打回任务→回流→微调→指标上升 |
> | `server.py` (D4) | 算子服务化：HTTP 接口 + schema + 异步队列微观版 |
> | `run_all.py` | 一键串联，打印总结 |
>
> ## 核心设计映射（真实世界 → 玩具版）
> - 模型 checkpoint → `operator_core.DetectorCore` 的 weights（dict）
> - ONNX 计算图 → `graph_forward(ops)` 的 list[dict]（每个 dict 一个节点）
> - 算子注册 → `workflow.REGISTRY`（和 Pointcept registry 同构思想）
> - 产线编排 → `workflow.run_workflow(tasks, ops)`
> - 预标注+人工审查 → D3 的 review 点（置信度不够就打回）
> - 数据回流 → `retrain_feedback.py`（打回的当训练数据）
> - 算子服务 → D4 的 Flask 接口（真实世界是 triton/torchserve/自研网关）
