"""
Step 4 迷你实验: 亲手造一个"迷你版 Pointcept 骨架"
=================================================

只用纯 Python(不需要 numpy/torch!),把 Pointcept 的三大工程件各造一个迷你版:

  1. Registry   —— 组件注册表     (对应 pointcept/utils/registry.py)
  2. config     —— 字典声明组件树  (对应 configs/*.py 里的 python 配置)
  3. Trainer    —— 训练主循环+钩子 (对应 pointcept/engines/train.py + hooks/)

跑法:
    cd /c/Users/GW00408524/Desktop/dev/Pointcept
    python -X utf8 step4_mini_framework.py

每个类比都写在注释里,标注了 [类比] 的是你前端世界里的对应物。
"""
import time

# ══════════════════════════════════════════════════════════════
# 第一件: Registry —— 组件注册表
# ══════════════════════════════════════════════════════════════
# [类比] Vue 的 app.component('MyButton', {...}) 全局注册表,
#        模板里 <component :is="'MyButton'"> 按字符串取组件。
# [类比] 你也可以想成 JS 里 const components = { MyButton: {...} },
#        渲染函数从字典里按名字取出来 new。

class Registry:
    """Pointcept 的 Registry 剥掉所有工程细节后的核心,就这么多。"""

    def __init__(self, name):
        self.name = name
        self._module_dict = {}          # 本质: {字符串: 类} 的字典,没了

    def register_module(self, cls):     # 当装饰器用
        # [类比] @app.component 注册:把类塞进字典,再原样还回去
        self._module_dict[cls.__name__] = cls
        return cls

    def build(self, cfg: dict):
        """cfg 形如 {"type": "XXX", 其他字段...} —— 按名字找类,其余字段当构造参数"""
        cfg = dict(cfg)                 # 复制一份,不改坏原config(好习惯)
        type_name = cfg.pop("type")     # 拿走 type 字符串
        cls = self._module_dict[type_name]   # 字典查表 → 类
        return cls(**cfg)               # new 实例,剩余字段全当构造参数
        # [类比] h(componentName, props) —— 组件名+props 造 vnode

MODELS = Registry("models")     # 模型注册表(仓库里还有 DATASETS/TRANSFORMS/HOOKS/TRAINERS...)
HOOKS = Registry("hooks")       # 钩子注册表

# ══════════════════════════════════════════════════════════════
# 第二件: 用装饰器注册几个"模型"和"钩子"
# ══════════════════════════════════════════════════════════════
# 仓库里每个模型文件开头都有 @MODELS.register_module() —— 现在你知道那行了在干嘛了

@MODELS.register_module
class MiniSegmentor:
    """对应仓库的 DefaultSegmentor(models/default.py): 组装 backbone + loss"""
    def __init__(self, backbone, criteria):
        # ★ 关键:嵌套的 backbone 字典,由我自己再调 MODELS.build 递归实例化
        # [类比] 父组件的 setup 里渲染子组件 —— 组件树递归构建
        self.backbone = MODELS.build(backbone)
        self.criteria = criteria
        self.weights = 0.0        # 假装这是几百万个权重(用一个数模拟)

    def forward(self, pointcloud):
        feat = self.backbone(pointcloud)
        return feat               # 假装输出"每点19类分数"

    def compute_loss(self, pred, answer):
        # 假装是 CrossEntropy:预测和答案差多少,返回一个数字
        return abs(pred - answer)

    def backward_and_step(self, loss, lr=0.1):
        # 假装的反向传播:真实版是 loss.backward() + optimizer.step()
        # 这里用"loss 对权重的导数=常数1"模拟下山:权重 -= lr * 梯度
        gradient = 1.0
        self.weights -= lr * gradient      # [类比] 每帧 camera.position -= speed * dt
        return gradient


@MODELS.register_module
class MiniPTv2:
    """对应仓库的 PointTransformerV2:吃点云,吐特征"""
    def __init__(self, num_layers=4, embed_dim=48):
        self.num_layers = num_layers
        self.embed_dim = embed_dim

    def __call__(self, pointcloud):
        # 假装做了 4 级 GridPool+attention,输出一个"预测分数"
        return self.num_layers * 10 + pointcloud * 0.5


@HOOKS.register_module
class LogHook:
    """对应仓库的 InformationWriter:每个 after_step 打日志"""
    def __init__(self, every=10):
        self.every = every
        self.step = 0

    def before_train(self, trainer): print(f"  [LogHook] 训练开始,共 {trainer.max_epoch} 个 epoch")

    def after_step(self, trainer):
        self.step += 1
        if self.step % self.every == 0:
            print(f"  [LogHook] epoch {trainer.epoch} step {self.step}: loss={trainer.last_loss:.3f}")

    def after_epoch(self, trainer): pass


@HOOKS.register_module
class CheckpointSaver:
    """对应仓库的 CheckpointSaver:每个 epoch 存一次档"""
    def __init__(self, save_freq=1): self.save_freq = save_freq

    def after_epoch(self, trainer):
        print(f"  [CheckpointSaver] epoch {trainer.epoch} 结束,保存 model_epoch{trainer.epoch}.pth (weights={trainer.model.weights:.2f})")

    def before_train(self, trainer): pass
    def after_step(self, trainer): pass


@HOOKS.register_module
class Evaluator:
    """对应仓库的 SemSegEvaluator:每个 epoch 评估一次 mIoU(假装)"""
    def before_train(self, trainer): pass
    def after_step(self, trainer): pass

    def after_epoch(self, trainer):
        miou = 100 - abs(trainer.model.weights) * 10      # 假装权重越接近0越好
        print(f"  [Evaluator]   epoch {trainer.epoch} 评估: mIoU={miou:.1f}")
        trainer.best_metric = max(trainer.best_metric, miou)


# ══════════════════════════════════════════════════════════════
# 第三件: config —— 用字典声明整个实验
# ══════════════════════════════════════════════════════════════
# [类比] 这就是一份"组件树声明",和 Vue 模板声明组件嵌套一个意思
# 仓库里 configs/*.py 里的 model = dict(...) 长得一模一样,只是字段更多

config = dict(
    model=dict(
        type="MiniSegmentor",              # ← Registry 按这个名字查类
        backbone=dict(                     # ← 嵌套子组件
            type="MiniPTv2",
            num_layers=4,
            embed_dim=48,
        ),
        criteria="CrossEntropy+Lovasz",    # 普通字符串字段,直接传给构造函数
    ),
    hooks=[
        dict(type="LogHook", every=5),
        dict(type="CheckpointSaver", save_freq=1),
        dict(type="Evaluator"),
    ],
    max_epoch=3,
    steps_per_epoch=10,
    lr=0.1,
)

# ══════════════════════════════════════════════════════════════
# 第四件: Trainer —— 主循环 + 钩子分发(对应 engines/train.py)
# ══════════════════════════════════════════════════════════════
# 主循环自己【不知道】要打日志/存档/评估 —— 全部通过钩子触发点分发出去
# [类比] 生命周期钩子:组件本身只管 render,挂载/卸载时的副作用写在钩子里

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = 0
        self.max_epoch = cfg["max_epoch"]
        self.best_metric = -1
        self.last_loss = 0

        # 按配置实例化模型和钩子 —— 和仓库的 build 流程一致
        self.model = MODELS.build(cfg["model"])
        self.hooks = [HOOKS.build(h) for h in cfg["hooks"]]

    # ---- 钩子分发:对每个已注册钩子调用同名方法(没有就跳过) ----
    def _trigger(self, event):
        for h in self.hooks:
            method = getattr(h, event, None)
            if method: method(self)

    # ---- 主循环:仓库 train.py 69~90 行的骨架 ----
    def train(self):
        self._trigger("before_train")
        for self.epoch in range(1, self.max_epoch + 1):
            self._trigger("before_epoch")
            for step in range(self.cfg["steps_per_epoch"]):
                self.train_step(step)
                self._trigger("after_step")
            self._trigger("after_epoch")
        self._trigger("after_train")

    # ---- 五步心跳:仓库 train.py 195~230 行的骨架 ----
    def train_step(self, step):
        input_data = step * 1.0                    # ① 假装取了一帧点云
        pred = self.model.forward(input_data)      # ① 前向
        answer = 47.0                              #    假装这是 .label 里的标准答案
        loss = self.model.compute_loss(pred, answer)   # ② 算损失
        self.model.backward_and_step(loss, self.cfg["lr"])  # ③④反向 ⑤更新
        self.last_loss = loss


# ══════════════════════════════════════════════════════════════
# 跑起来!
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("迷你版 Pointcept:Registry + config + Trainer + 钩子")
    print("=" * 62)

    print("\n① 注册表里有什么(仓库里 MODELS.module_dict 就是这个):")
    print(f"   MODELS: {list(MODELS._module_dict.keys())}")
    print(f"   HOOKS:  {list(HOOKS._module_dict.keys())}")

    print("\n② 按 config 实例化 Trainer(config 声明了整棵组件树):")
    trainer = Trainer(config)
    print(f"   模型: {type(trainer.model).__name__}")
    print(f"   递归构建的 backbone: {type(trainer.model.backbone).__name__}"
          f"(num_layers={trainer.model.backbone.num_layers})")
    print(f"   钩子: {[type(h).__name__ for h in trainer.hooks]}")

    print("\n③ 开始训练(观察钩子如何被触发):")
    trainer.train()

    print(f"\n④ 训练完成: best_metric={trainer.best_metric:.1f}")
    print("\n" + "=" * 62)
    print("观察要点:")
    print("  1. Trainer 主循环里没有一行日志/存档/评估代码 —— 全在钩子里")
    print("     [类比] 想加'训练完发通知'?写个新 Hook 类注册进 config,主循环零改动")
    print("  2. config 里的 type='MiniPTv2' 字符串 → Registry 查表 → 类 → 实例")
    print("     [类比] <component :is> 按名字渲染组件")
    print("  3. MiniSegmentor.__init__ 里自己 build 了 backbone —— 组件树递归构建")
    print("     [类比] 父组件渲染子组件")
    print("=" * 62)
