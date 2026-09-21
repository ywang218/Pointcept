# MiniPT 手册 —— 你边补边问的作战地图

> 项目:`minipt.py`(迷你 Pointcept)。一句话:**亲手搓通一条点云语义分割全链路**,把复习站 9 章的知识全部"过手"一遍。
> 节奏:一关一关来,**先自己写 → 跑自检 → 过**。卡住 15 分钟就来问(问法模板见 §5)。写完一关,顺手回复习站做对应章的题,知识就闭环了。

---

## 一、全景:你要搓的这条链

```
合成 LiDAR(S1) → GridSample(S2) → Collect(S3) → attention(S5)
     → 六积木组装(S6) → 训练五步心跳(S4) → mIoU 评估(S9)
```

毕业标准:`python -X utf8 minipt.py --train` 跑到 **验证 mIoU ≥ 0.85**,会打印 🎓。
全程 CPU 可跑、零外部数据(场景是合成的:地面 + 一辆车 + 一根电线杆,三类)。

**为什么值得做**:复习站解决"懂",MiniPT 解决"会"——`GridSample 串线了会怎样`、`漏了 zero_grad 会怎样`、`不采样有多慢`,这些"为什么这么做"的问题,自己踩一次比看十遍讲解都牢。

## 二、六关攻略(每关:为什么 / 提示 / 对照真身 / 常见坑)

### 第 1 关 `make_scene` —— 合成一帧"LiDAR 扫描"(对应 S1)

- **为什么**:Pointcept 一切的入口是"一帧点云 = 标准字典"(S1/S3)。真身 `semantic_kitti.py` 读 `.bin` 得 [N,4] 翻译成标准字典;我们造一个迷你世界,体会"字典里每个数组必须同长度同顺序"这个贯穿全仓库的约定。
- **提示**:车 = `rng.normal(中心, (1,1,0.4), (n,3))` 后 z 做 `np.clip(car[:,2], 0, None)`;杆 = 角度 `rng.uniform(0, 2π, n)`、高 `rng.uniform(0, 4, n)`,x = 杆位 + 半径×cos(角度),y = 杆位 + 半径×sin(角度);拼接用 `np.concatenate([...], axis=0)`,segment 用 `np.zeros/ones/full` 三段拼。
- **对照真身**:step1_touch_pointcloud.py 的车 blob 和极坐标采样,全是写过的。
- **常见坑**:① segment 忘了转 `np.int64`;② 车的 z 没 clip 穿到地下(自检会抓)。

### 第 2 关 `grid_sample` —— 字典版 GridSample(对应 S2)

- **为什么**:19k 点做 attention 是 N² 灾难;体素化后 ~1000 token,CPU 也能秒训。新难点:**coord 抽走哪些点,segment 必须跟走哪些点**——对不齐 = 训练全毁,真实工程事故第一名。
- **提示**:五步食谱在函数 docstring 里,每步一行共 ~6 行。你写过的两处升级:①`np.random.randint` → `rng.integers`(随机纳入种子管理);②多带一个 segment 数组,同一个 `idx` 花式索引两个数组。
- **对照真身**:transform.py:840;step2_gridsample.py 实验三(核心六行一模一样)。
- **常见坑**:① 忘了 `gc -= gc.min(axis=0)`(负数打包会撞车);② idx 只用在了 coord 上忘了 segment(自检专门抓这个)。

### 第 3 关 `collect` —— 进 torch 世界(对应 S3,最短的一关)

- **为什么**:transform 管道是 numpy 世界,模型是 torch 世界,Collect 是两岸的桥:① feat 按顺序拼起来(coord 本身也进 feat!)② 转 tensor。
- **提示**:`torch.from_numpy(...).float()` / `.long()`。
- **对照真身**:transform.py:54。真身 `feat_keys=["coord","strength"]` 拼 (N,4) → `in_channels=4`;我们 `["coord"]` → (N,3) → `in_channels=3`。**真身的 offset(batch 分段账本)是 batch>1 才需要的,MiniPT 全程 batch=1,挪到 bonus 关 B1。**
- **常见坑**:segment 转 float32 了(交叉熵的 target 必须 int64/long,自检会抓)。

### 第 4 关 `softmax` + `attend` —— attention 的心脏(对应 S5)

- **为什么**:attention = "token 间开会":打分 → softmax 变权重 → 加权混货。score 已替你算好,你完成后两步——正是 GroupedVectorAttention 的 :122 和 :127 两行。
- **提示**:softmax 先减 max 再 exp(数值稳定);attend 两步:`w = softmax(score)` 然后 `(w.unsqueeze(-1) * value).sum(dim=1)` 消掉 S 维。
- **对照真身**:v2m2_base.py:122/127 —— `(attn.unsqueeze(-1) * v).sum(2)`,只是维数从 (B,H,N,K) 降到 (N,S)。
- **常见坑**:① softmax 忘减 max,大数直接 inf(自检喂 1000.0 抓);② sum 的 dim 错了(S 维是 dim=1,因为 shape 是 (N, S, C))。

### 第 5 关 `train_step` —— 五步心跳(对应 S4,全项目画龙点睛)

- **为什么**:torch 里所有训练循环都是同一心跳:`zero_grad → forward → loss → backward → step`。亲手写一次,并亲眼看 backward 之后每个权重都拿到 `.grad`。
- **提示**:每步一行共五行,docstring 里就是答案的形状。注意 forward 写 `model(feat)` 不写 `model.forward(feat)`(走 `__call__` 才带钩子,双线路线 Q1 考过)。
- **对照真身**:engines/train.py train_step。真身多三样工程加固:梯度累积/AMP/clip_grad(s8q4 考的就是这三样)。
- **常见坑**:① 顺序错(比如 backward 在 loss 前);② 忘 `loss.item()` 返回了 tensor(自检抓);③ zero_grad 放最后(等于这步白清)。

### 第 6 关 `miou` —— 分割的真考分(对应 S9)

- **为什么**:ground 类大、pole 类小,准确率会被 ground 淹没;IoU 每类平等,骗不了人——这正是 config 里 class_weight 从 1.3(vegetation)到 10.2(motorcyclist)的动机。
- **提示**:混淆矩阵 `conf = torch.zeros(3, 3); conf[target, pred] += 1`(行=真值 列=预测);每类 `iou = 对角 / (行和 + 列和 - 对角)`;分母为 0 的类跳过。
- **对照真身**:evaluate.py SemSegEvaluator(挂在 after_epoch)。
- **常见坑**:① `conf[target, pred] += 1` 是 scatter 加法,不是赋值(重复下标要累加);② 把"没出现过的类"算成 0 拉低平均(应跳过)。

## 三、通关之后:毕业训练 + 对照实验

```bash
python -X utf8 minipt.py --train              # 200 步 → mIoU ≥ 0.85 毕业
python -X utf8 minipt.py --train --steps 400  # 没到就加步数
python -X utf8 minipt.py --compare            # 不做 GridSample 有多惨(亲手感受"为什么")
```

`--compare` 是这个项目的灵魂实验:同样的 40 步,GridSample 版(≈1000 token)vs 不采样版(≈19k token)的耗时对比——N² 的 cdist/topk 在 CPU 上会教你做人。这就是 config 里 GridSample 排在模型前面的原因:**token 数是 attention 计算量的平方项,采样 = 用 1/20 的点换几百倍速度,语义信息几乎不丢**。

## 四、Bonus 关(通关后可选,做了更稳)

- **B1 offset**:把 Collect 升级成 batch=2:两帧点云拼成一个大数组,`offset = cumsum` 记账本;模型里用 `offset` 切回两帧。真身 datasets/utils.py 的 point_collate_fn(mix_prob=0.8 的拼接魔法就发生在那,s8q5)。
- **B2 inverse 铺回**:GridSample 加 `return_inverse=True`(np.unique 顺手给的),测试时对全量点预测后 `pred[inverse]` 一行铺回——evaluator.py:187 同款(s9q4 的代码题变大活)。
- **B3 破坏性实验**(最快建立直觉的一组,每行一条命令):
  - 把第 5 关的 `zero_grad()` 注释掉 → 看 loss 曲线抽风(累加语义);
  - 把 TransformerBlock 的 `x +` 去掉 → 加深 n_blocks 看训不动(残差,s6q1);
  - 把 `grid_size` 改成 2.0 → 看 mIoU 崩(过粗采样丢语义);
  - 把车的 class 权重调成 0.1 → 看车 IoU 掉但总 loss 还行(类别不均衡)。

## 五、卡住了怎么问(模板)

> "第 N 关,我按 XXX 思路写了 XXX(贴代码),自检报 XXX(贴报错)。我猜是 XXX 但不确定。"

这样问 30 秒内能定位。反过来,通关后想验证理解,可以问:"第 N 关为什么必须 XXX?如果 XXX 会怎样?"——我可以用 B3 的破坏性实验带你亲手看。

## 六、文件清单

| 文件 | 角色 |
|---|---|
| `minipt.py` | 练习场本体:6 个 TODO + 自检系统 + 毕业训练 + 对照实验 |
| `minipt_手册.md` | 本手册:攻略、坑、问法 |
| `step_all_review.html` | 复习站(做完一关回对应章做题) |
| `step5_补充讲解_Python_numpy_API速查.md` | API 工具书(撞见生疏 API 就查) |
