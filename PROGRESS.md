# Pointcept 学习进度指针（每天开工先读这个）

> 最后更新：2026-09-16（🆕 MiniLOps demo 合集上线：`minilops/` 四件套，直接对应"领导说的封装算子"的实操版。恢复方式："读 PROGRESS.md 和 试用期/双线学习总路线.md，我们继续"）
> 用法：开工时让 Claude 读本文件 + 相关补充讲解，即可恢复上下文继续。

---

## 双线学习结构（2026-09-06 起）

- **线A（主线）**：Pointcept 算法侧——领导任务（LiDAR seg 预研/算子封装/参与训练）的主战场
- **线B（并行）**：车端部署侧——《试用期/Code_20260906.txt》方向（非考核但需学）：模型转换/量化/推理引擎/warm-up
- **合流点**："封装成算子"= 训练侧产模型 → 导出 → 推理服务化
- **总路线文档**：`C:\Users\GW00408524\Desktop\试用期\双线学习总路线.md`（含融合后的统一路线、E1~E5 合流实验、三组练习题含答案、环境约束备忘）
- **当前所处位置**：线A Step 4 已收官（2026-09-07）；Step 5 进行中，进入第 2 周后开始 E1~E5 合流实验

---

## 当前状态一览

- **项目**：`C:\Users\GW00408524\Desktop\dev\Pointcept`（3D 点云感知框架，LiDAR 语义分割方向）
- **学习者背景**：大前端图形渲染（WebGL/WebGPU）出身，懂 DL 基本概念、懂 Transformer 原理、无实践；Python/numpy 补课中
- **已完成**：Step 0（环境）、Step 1（open3d 看点云）、Step 2（GridSample + 原理消化）、Step 3（PTv2 attention 三差异落地验证）
- **进行中**：Step 5（翻译 semantic_kitti config 成中文笔记）
- **Step 4 收官记录（2026-09-07）**：
  - ✅ `step4_mini_framework.py` 实机跑通（loss 5.0→2.5 正常下降；Evaluator 假 mIoU 随权重绝对值增大而下降 90→70——彩蛋生效：无梯度方向的下山≠指标变好，呼应"假 backward"设计）
  - ✅ 仓库三处导读完成：`registry.py`（真身多出的部分：build_from_cfg 的 default_args/异常包装、parent/children 作用域链 `type="mmdet.ResNet"`、register_module 的 name/force 参数、infer_scope 自动推断包名）
  - ✅ `engines/train.py`：真主循环 69~90 与迷你版结构一致（before_train→epoch→before_epoch→step 循环→after_epoch→after_train）；train_step 真身比迷你版多：梯度累积（先 zero_grad 只在首步、loss 除以累积步数、凑满才 step）、AMP 混合精度（scaler.scale/unscale_/update）、clip_grad 裁剪——这三样都是"工程加固"，概念骨架没变
  - ✅ `configs/_base_/default_runtime.py`：hooks 列表 7 个钩子（CheckpointLoader→ModelHook→IterationTimer→InformationWriter→SemSegEvaluator→CheckpointSaver→PreciseEvaluator），字符串→HOOKS Registry 查表，与迷你版机制一致
  - 新增认知：`mix_prob=0.8` 不是训练参数而是 **collate 阶段的 batch 混合概率**（`datasets/utils.py:208` point_collate_fn：以 0.8 概率把 batch 内偶数/奇数位置的样本合并拼接、instance id 平移、offset 重算）——点云特色增强：点云靠 offset 分 batch，拼接成本几乎为零
  - 新增认知：hook 挂接用 `weakref.proxy(self)`（train.py:66）防循环引用内存泄漏；钩子必须继承 HookBase 并实现 8 个生命周期方法
  - 新增认知：config 继承用 `_base_ = ["../_base_/default_runtime.py"]`（semantic_kitti config 第 1 行），子 config 只写差异字段——Vue props 默认值覆盖的既视感
  - 新增认知：`DefaultSegmentor.forward`（models/default.py:21）三分支：training→只回 loss；eval+有标签→loss+logits；test→只回 logits。训练循环里取 `output_dict["loss"]`
  - 新增认知：DefaultSegmentor（v1，backbone 直接吐 logits）vs DefaultSegmentorV2（v1.5+，backbone 吐 Point 结构、外面套 seg_head Linear；pooling_parent 链上溯融合多级特征）
- **领导任务对齐**（2026-09-02）：LiDAR seg 预研（完全重合=本路径）、封装算子（高度重合）、参与训练（重合）、红绿灯 2D 预标注（间接重合）。**主线不变，继续 Pointcept。**

## 学习文档全景（按用途分）

| 文档 | 内容 | 性质 |
|------|------|------|
| `ONBOARDING_TODO.md` | 总计划 6 步 + 每晚执行版 | 路线图 |
| `PROGRESS.md` | 本文件,进度指针 | 每日开工入口 |
| `step2_逐行讲解.md` | step2 脚本逐行 + 10 段 numpy 验证命令 | 查阅 |
| `step2_补充讲解_数据链路与Transformer.md` | 补充 A（数据接入两层）+ B（神经网络→反向传播→PTv2） | 原理 |
| `step2_补充讲解C_QKV纠正与预处理动机.md` | QKV 纠正 + 预处理动机 + DFS/memoization 类比延伸 | 原理（核心） |
| `step4_mini_framework.py` | 迷你版 Registry+config+Trainer+钩子（纯 Python 零依赖，已验证可跑） | **Step 4 实验，待跑** |
| `step4_补充讲解_执行机制.md` | forward/backward 的执行机制：Python 指挥/CUDA 演奏、自动微分原理、认知分界线 | 原理 |
| `step4_补充讲解B_forward设计手册.md` | forward 六积木+数学原理、论文=积木组装、三线吃透路径（手搓/数学/逆向） | 原理（进阶地图） |
| `step4_补充讲解C_CUDA与shader对照.md` | CUDA kernel ↔ 自定义 shader 对照表、SIMT 同源、PyTorch 逃逸舱口梯度 | 原理（新） |
| `step5_补充讲解_Python_numpy_API速查.md` | 7 个★心智模型级 numpy API（gather/clip/cumsum/floor/切片/unique/集合运算）+ 随机数/argsort/语言点/torch 六兄弟；按题目撞见处索引，全部带前端对照 | **API 工具书（做题时查）** |

实验脚本另有：`step1_touch_pointcloud.py`、`step2_gridsample.py`（step2 可视化窗口还没正式跑过）。

## 当前断点（下次开工从这继续）

**Step 5 已上线练习场（2026-09-07）：`step5_practice.html`**

- 本地服务器已起（后台）：`http://127.0.0.1:8765/step5_practice.html`；以后手动起服务：在仓库目录跑 `python -m http.server 8765` 然后开浏览器访问该地址（或直接双击 html 也能用，进度存 localStorage）
- 页面内容：热身（misc 4 题）+ A（transform 链 12 题）/ B（model 块 6 题）/ C（criteria 6 题）/ D（test 块 6 题），共 **34 题**
- **题型（v2 改版，2026-09-07）**：单选 15 / 判断 8 / 问答 6 / 代码 5——单选/判断/代码提交后**自动判分**（带解析），问答写完对照要点自评；代码题=预测输出或本地写 GridSample 计数再贴输出比对
- 每题两种模式：✍ 自己作答（客观题自动判，问答勾选要点自评）/ 👀 直接看答案（计入进度不计分）
- 两处勘误已并入题目：RandomRotate 角度是 **±180° 不是 ±57°**；epoch/eval_epoch=50/50 是**每 epoch 评估+存档共 50 次**（loop=1，不是只评估 1 次）
- 进度持久化在浏览器 localStorage（key: step5_quiz_v2）；「导出学习记录」把作答+对错+未命中要点复制成 Markdown，贴回给 Claude 逐题点评
- **🆕 知识定位（v2.1）**：新增「🗺 数据流全景」首页 tab——整份 config 画成训练侧 9 站 + 测试侧 3 站的流水线，每道题是一枚芯片（颜色=战绩：绿对/红错/黄看过/灰未做），点击直达；**每题解析下方新增「📍位置」块**：环节（流程第几站）/ 代码位置（文件:行）/ 作用 / 关联知识点（链到其他题）——答错时顺着它定位概念在整条链路里的坐标
- **🆕🆕 先讲解后习题（v2.2，2026-09-07 晚，回应用户反馈"题和代码对不上、正确率低"）**：每个题组顶部新增**「📖 讲解」折叠卡**——先读章节讲解、再做该组习题。讲解内容全部对着仓库真代码写（config 原文摘录 + transform.py/model/engines 行号），每节末尾有**「🎯 考什么」预告框**（列出该组每题的考点），做题前先扫一眼就知道要抓什么。5 个组全覆盖：warm（misc 块）/ A（transform 10 步管道）/ B（model 块 U-Net 骨架）/ C（criteria 双损失）/ D（test 块推理技巧）。**用法：开一组 → 先展开 📖 讲解读完 → 看 🎯 预告 → 再做题**
- **下次开工**：先问练习场做到哪了/要导出记录，然后逐题点评未命中要点 → Step 6 收尾

**🆕🆕 全程复习站上线（2026-09-08）：`step_all_review.html`**

- 地址：`http://127.0.0.1:8765/step_all_review.html`（同一个 8765 服务器；双击 html 也能用；**打开记得 Ctrl+F5 强刷**）
- 定位：**Step 0→5 全部学过内容的收官复习站**，应"按 讲解→代码→习题 把全部内容过一遍"的请求生成；与 step5_practice.html 相互独立（进度互不影响）
- 结构：**9 章 + 全程地图首页**，每章严格三段式 = 📖 知识点讲解（含仓库代码例子、file:line 引用）→ 🎯 考什么预告 → 习题；9 章对应 S1 点云初见 / S2 GridSample / S3 数据链路 / S4 网络与训练循环 / S5 QKV 与 PTv2 attention / S6 forward 六积木 / S7 执行机制与 CUDA↔shader / S8 工程骨架 / S9 Config 全景
- **48 题**：单选 21 / 判断 6 / 问答 9 / 代码 12——单选/判断/代码自动判分（带解析），问答勾要点自评；每题带 📍 知识定位块（环节/代码位置/作用/关联）；首页地图 9 站芯片直达题目
- 进度 localStorage key：`step_all_review_v1`（独立于 step5 的 `step5_quiz_v2`）；「导出学习记录」生成 Markdown 贴回给 Claude 逐题点评
- 代码题两种：预测输出型（贴运行结果自动比对）+ 手写型（写代码跑通后交输出）；所有代码题均可用 `python -X utf8` 验证
- 验证：node vm 沙箱跑通全部 JS（语法/48 题 kmap 与判分字段/9 站地图/gradeOutput 用例/交互函数全检通过）；S3/S4 曾各有一处 ASCII 引号嵌套导致语法错误，已修

**🆕🆕🆕 动手项目 MiniPT 上线（2026-09-09）：`minipt.py` + `minipt_手册.md`**

- 定位：应"想通过自己写代码感受为什么这么做"的请求搭的**练习场架子**——迷你 Pointcept 全链路，核心函数留 TODO 给用户写，边补边问
- 链路：合成 LiDAR（S1）→ GridSample（S2）→ Collect（S3）→ attention（S5）→ 六积木组装（S6）→ 五步心跳（S4）→ mIoU（S9）；合成场景三类（地面/车/电线杆），CPU 可跑零外部数据
- **6 个 TODO 关卡**（每关带自检）：①make_scene 造标准字典 ②grid_sample 字典版（coord/segment 同步，串线检测）③collect 转 torch ④softmax+attend（v2m2_base.py:122/127 同款）⑤train_step 五步心跳 ⑥miou 混淆矩阵
- 用法：`python -X utf8 minipt.py --check N` 单关自检 / `--check all` 连跑 / `--train` 毕业训练（验证 mIoU≥0.85 出 🎓）/ `--compare` GridSample 对照实验（灵魂：亲手感受不采样有多慢）
- 手册含：每关的为什么/提示/真身对照/常见坑 + bonus 关 B1 offset/B2 inverse 铺回/B3 破坏性实验（注释 zero_grad、去残差等，最快建立直觉）+ 卡住问法模板
- 前置：需装 CPU torch（`pip install torch --index-url https://download.pytorch.org/whl/cpu`，约 200MB）——**装 torch 的命令当时被环境安全检查暂时拦住没跑成，下次开工先确认 torch 已装再让用户开跑第 3 关起**（第 1/2 关纯 numpy 不需要 torch，可立即开始）
- 节奏建议：通关一关 → 回 step_all_review.html 对应章做题，知识与手感闭环

**🆕🆕🆕🆕🆕 MiniLOps demo 合集上线（2026-09-16），学习模式改闯关制（2026-09-17）——`minilops/` + `minilops_手册.md`**

- 应"自己搭 demo 亲身体验封装算子"的请求搭的**零依赖全链路玩具**（numpy + flask 即可跑，不需要 torch/PIL/ORT）——E1~E5 要装 torch 才能跑，这套现在就能跑
- **⚠️ 学习方式（2026-09-17 应用户"要循序渐进"反馈调整）**：`minilops/` 里的五个文件（preprocess / operator_core / workflow / retrain_feedback / server）现在是**参考答案**，学习期间**别通读，过关一关翻一关**；作业写 `my_d1.py` ~ `my_d7.py`，**只准依赖 numpy（关7 另加 flask）和前一关自己写的文件，不许 import 参考答案**
- **闯关结构（详见 `minilops_手册.md`，每关含：为什么做/契约/自检/提示/常见坑/卡住问法模板）**：关1 预处理算子（grayscale/双线性 resize/归一化+契约检查）→ 关2 检测器内核（forward+手写 backward+数值梯度验证，最重一关）→ 关3 后处理（decode/IoU/NMS）→ 关4 工作流引擎（注册表+信封+config 驱动，领导说的"集成到工作流"本体）→ 关5 审查门+模拟产线（预标注的灵魂；⚠️帧尺寸 23 不是 24——预处理 23→卷积 22=标签，尺寸对齐经典坑）→ 关6 数据闭环（打回→回流→微调→采纳率 83%→87%，用关2 的 backward 真训练）→ 关7 服务化（/detect /health /stats，模型只加载一次+契约校验+可观测）
- Bonus 破坏性实验 B1~B3：每请求重建模型看延迟暴涨（铁律验证）/ min_conf 调 0.1 看采纳率虚高 F1 掉（阈值毒化）/ 一个亮斑出 20 框之谜（NMS 阈值与框尺寸耦合）
- 参考答案已全部实测跑通：`run_all.py`（D1→D3+retrain 一键）+ server.py 三接口 curl 验证过（400 契约错误、延迟统计、model_version 可追溯）
- **建议节奏**：一晚 1~2 关共 5 小时；每关流程 = 读手册讲解 → 写 my_N.py → 自检全绿 → 翻参考答案 diff → 喊 Claude 点评 → 下一关；`run_all.py` 只在热身时跑（看输出归位到地图，别读代码）
- **学习价值锚点**：通关后能回答"预标注产线里算子是什么、注册是什么、schema 是什么、工作流是什么、回流是什么"——领导再提算子，脑里有图
- torch 装好后衔接：把 my_d7 里的 kernel 推理换成真 ONNX，全套壳不改（E 系列衔接点）
- **关2 三段式进行中（2026-09-17）**：①high level(why) 已讲（模型=从例子学出来的函数机器；backward=图形学没有的那半边）；②原理层已讲（2×2 卷积 / BCE+sigmoid 手推链式法则 / dz=(p−t)/N 相消 / 数值梯度验证）；③下一站 = 写 `my_d2.py`。**用户点名待讲**（原理吃透后）："5 个参数怎么够检测红绿灯"——答案种子=参数共享，原理层已埋



- 应"讲解和题目下面都提供对话窗口，我提问、程序拿去问 Claude、答完返回页面"的请求上线；新增配套 **`review_server.py`**（仓库根目录）
- **用法变更**：以后起服务不再用 `python -m http.server 8765`，改跑 `python -X utf8 review_server.py`（同端口 8765；既静态伺服又提供问答代理）；打开 `http://127.0.0.1:8765/step_all_review.html` 记得 Ctrl+F5 强刷
- **每章 📖 讲解末尾 + 每道题下面都有 💬 问答窗口**：点开展开对话框，输入问题回车发送 → 服务端调用 claude CLI 无头模式（`claude -p`，允许它 Read/Grep 仓库代码核实后再答）→ 回答渲染回页面；支持多轮追问（带最近对话历史）
- **防剧透机制**：没作答的题，窗口会把"用户还没作答"标进上下文，claude 先给提示和思路引导、不直接说答案；作答后（无论对错）才附上解析原文供深入追问
- 每个窗口自动携带对应讲解全文/题目原文（题干、选项、📍知识定位）作为上下文，不用复制粘贴；回答用中文、走 WebGL/GLSL/前端类比风格（与前几天的讲解一致）
- **优雅降级**：不跑 review_server.py（双击打开页面或用纯静态服务器）时做题功能完全不受影响，💬 窗口显示启动指引；页面启动时会自动探测问答服务是否在线
- 服务端细节：POST /ask 接口；claude 路径自动探测（PATH → npm 全局目录）；单次超时 300 秒；上下文超长自动截断；服务器窗口会打印每次提问日志

## 任务清单状态（Claude 侧 task list）

- ✅ #6 Step 0 环境（numpy/open3d/addict/pyyaml 已装）
- ✅ #7 Step 1 open3d 看点云
- ✅ #8 Step 2 GridSample（实验 + 讲解全部消化完毕）
- ✅ #9 Step 3 PTv2 attention 三差异（落地验证完成）
- ✅ #10 Step 4 工程骨架 ← **2026-09-07 收官：迷你实验跑通 + 仓库三处对照完毕**
- 🔄 #11 Step 5 翻译 semantic_kitti config ← **进行中：练习场 v2 已上线（34 题：单选15/判断8/问答6/代码5，自动判分），做完导出记录给 Claude 点评**
- ⬜ #12 Step 6 收尾（一句话数据流 + 明日待办）
- ⬜ #13 第 2 周 E1~E5 合流实验（手搓 MiniPT→ONNX→warm-up→动态shape→INT8）
- ⬜ #14 MiniLOps 闯关（关1~关7 + B1~B3）← 2026-09-17 改闯关制：作业写 `my_d1.py`~`my_d7.py`（只依赖 numpy+flask 和前一关），参考答案过关一关翻一关；手册在 `minilops_手册.md`；torch 装好后把 my_d7 内核换真 ONNX（E 系列衔接点）

## 挂起事项（别忘）

- **要问同事/领导的四件事**：
  1. GPU 资源在哪？（本机 RTX 500 Ada 4GB 只够推理验证，训练需服务器/集群——train.sh 里有 SLURM 分支）
  2. LiDAR seg 的数据源是 nuScenes / SemanticKITTI / Waymo 还是自有数据？（决定写哪个 dataset 类）
  3. **产线平台的算子接入协议**：HTTP 服务 / Python SDK / docker 镜像？输入输出 schema 长什么样？**有没有现成算子代码可以参考**（起步最快路径——拿到一个现成算子照着写胜过一切文档）
  4. 红绿灯 2D 预标注用的模型是**现成 checkpoint 还是自己训**？（决定要不要补 2D 检测训练链路）
- **领导最新对齐（2026-09-14）**：①近期可能接手**红绿灯产线 2D 预标注**——2D 检测/分割模型**封装成算子集成到工作流**+可参与算子模型训练（等确认再拉人）；②LiDAR seg 回流数据不足以训出提升效果的模型（**估计需几万帧高质量**），先做预研不出活；③概念校准：领导说的"算子"=**产线工作流的算法推理节点**（模型→导出→推理服务→平台注册），**不是** libs/pointops 那种 C++/CUDA 自定义计算算子——两者交集只在"模型带自定义算子导不出时要重写或写 C++ 插件"（2D 模型基本遇不到，LiDAR 模型一定会）；④为此新增 **E6 实验：把 E3 的 ORT 推理脚本重构成 Operator 类**（加载/warm-up/三段式 process/失败处理/版本号），排总路线 E1~E5 之后
- 本机没 conda/WSL/Docker/nvcc，SemanticKITTI（约 80GB）磁盘放不下（剩 71GB）——训练相关的事等 GPU 资源确认
- `libs/pointops` CUDA 编译在 Windows 上坑深，不碰（但读 kernel 源码没问题）

## 学习成果自检（累计）

第 1 天：
1. numpy 心智模型：整批操作、切片、广播、argsort/花式索引
2. GridSample = 体素化 + 空间哈希 + 每格选代表点（图形学 binning/LOD 直觉复用）
3. 数据链路两层：dataset 类管格式翻译、transform 管道管增强与采样
4. QKV 正确图景：三份并列投影、softmax 算权重、权重混合 V
5. 反向传播：loss → 链式法则 → 梯度 → 下山更新
6. 预处理动机：体素=token + 邻域=范围；堆深度换全局（16⁴≈65536）
7. 优化直觉：GridSample/kNN ≈ 剪枝（有损）非 memoization（无损）；类比 SSAO/稀疏矩阵/BVH/LOD

第 2 天（新增）：
8. PTv2 attention 真实代码落地：linear_q/k/v 并列投影、softmax 在 122 行、einsum 权重混合货物、mask 处理 padding 邻居
9. einsum 读法："把指定维度求和消掉，其余保留"
10. 执行机制：Python 指挥/CUDA 演奏（类比 JS draw call vs shader）；自动微分=记账+反向走图+链式连乘；懂"水文学"不懂"水管公式"
11. 领导任务交集：LiDAR seg 预研=完全重合；算子封装/参与训练=高度重合；红绿灯 2D=间接重合
12. forward 设计手册：六大通用积木（残差/MLP/归一化/激活/DropPath/attention动态权重）+各自数学；论文=积木新组装；三线吃透路径（手搓/数学按需/逆向拆解+预测-验证闭环）
13. CUDA kernel ↔ 自定义 shader：动机相同（宿主语言慢千倍）、执行模型相同（SIMT=compute shader）、分层同构；knn_query 就是"每线程一个点+小顶堆"的 shader；PyTorch 逃逸舱口梯度（标准算子→Triton→裸CUDA）
