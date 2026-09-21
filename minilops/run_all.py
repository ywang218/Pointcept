# -*- coding: utf-8 -*-
"""
run_all —— MiniLOps 一键全跑
=========================================
按依赖顺序跑 D1→D3→retrain（自测含在各自文件里），并打印总结。
D4 服务化是常驻进程，不在这里自动跑——README 里有手动玩法。

建议：跑完这个，再按 README 的文件地图顺序逐个精读。
"""

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import preprocess
import operator_core
import workflow
import retrain_feedback


def main() -> None:
    print("═" * 62)
    print(" MiniLOps —— 领导说的『封装算子』全链路玩具")
    print(" 每一步对应真实产线的一环，跑完等于预演一遍")
    print("═" * 62)
    print()

    print("── D1 预处理算子 ──")
    preprocess.self_test()
    print()

    print("── D2 算子内核 + 计算图雏形 ──")
    operator_core.self_test()
    print()

    print("── D3 工作流引擎 + 模拟产线 ──")
    workflow.self_test()
    print()

    print("── 数据闭环 retrain feedback ──")
    t0 = time.time()
    retrain_feedback.self_test()
    print(f"   （闭环总耗时 {time.time()-t0:.1f}s，纯 numpy，无 GPU）")
    print()

    print("═" * 62)
    print(" 总结：你刚才依次经历了")
    print("  D1  预处理写成可测试算子单元（输入契约/归一化/resize）")
    print("  D2  模型内核 + 手写 backward + 计算图执行 + NMS 后处理")
    print("  D3  注册表 + config 驱动编排 + 审查门（预标注的灵魂）")
    print("  RT  打回→回流→微调→采纳率上升（数据飞轮）")
    print("  D4  服务化（下一步：python -X utf8 minilops/server.py）")
    print()
    print(" 真实产线 = 这套骨架 + 真模型(ONNX) + 平台协议 + 规模化运维。")
    print(" 骨架你已经有了——剩下的只是把玩具换大。")
    print("═" * 62)


if __name__ == "__main__":
    main()
