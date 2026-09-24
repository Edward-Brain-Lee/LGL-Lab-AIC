# HANDOFF — LGL-Lab AIC 算法大赛（噪声标签细粒度识别）

> 交接文档。最后更新：**2026-09-24**。
> 面向：接手继续推进的工程师。假设你会 PyTorch、用过 Linux 命令行，但不了解这个赛题和之前发生过什么。
>
> **最新状态看 §0 和 §16。** §16 是本轮的全新结果，其中有一个**决定性实验**。
> §15 是上一轮的复赛实测，§6~§13 是初赛阶段的记录 —— 仍然有效，但要结合后面读。

---

## 0. 30 秒现状

| 项目 | 状态 |
| --- | --- |
| 当前阶段 | **复赛（750 类）**，提交截止 **2026-10-07** |
| 提交名额 | **全队 2 次/天**，排行榜取历史最高分 —— **这是最紧的资源**，比 GPU 时间紧得多 |
| **当前最好成绩** | **66.2456**（`sub_tta4.csv`，4 视图 TTA）—— 见 §16.2 |
| 同模型无 TTA 的分数 | 64.16（`sub_ep20.csv`）⇒ **TTA 单项值 +2.09 点** |
| 初赛得分（参考） | 68.02 —— **不同数据集，不可直接比较** |
| 基线自测指标 | `val_acc = 0.7070`（**高估 4.45 点**；TTA 之前是 6.54，TTA 把缺口缩小了） |
| 已完成 | `outputs_r2` / `outputs_old`（各 20 ep）、`outputs_ema`（224 + 教师权重）、死类诊断、8 视图 TTA 代码 |
| **进行中** | `outputs_288`（`--img-size 288`）—— 每轮 362 秒（224 是 235 秒，1.54×），20 轮约 2 小时 |
| **288 的早期信号** | **ep1~ep4 的 `val_acc` 每一轮都比 224 高 1.3~2.1 点**（见 §16.6）⇒ 光训练端就值约 **+1.6** |
| 下一步 | ① 288 训完 → 生成 `288 + TTA` ② 明天的 2 个名额交 `sub_288_tta4` 和 `sub_tta8` |

### 本轮（2026-09-24）的三个结论

**① H1 成立，H3 出局 —— 训练轮次这条路封死了。**（决定性实验，见 §16.1）

| 提交 | val_acc | 平台分 |
| --- | --- | --- |
| `sub_ep4.csv` | 0.6568 | **64.1598** |
| `sub_ep20.csv` | 0.7070 | **64.16** |
| **Δ** | **+5.02** | **+0.0002** |

16 个 epoch 让 `val_acc` 涨了 5 个点，让真实分数涨了 **0.0002**。缺口从 **1.52 点涨到 6.54 点** ——
而 H3（域偏移）要求缺口**恒定**，所以 H3 被排除；缺口是被训练自己撑大的 ⇒ **H1（记忆结构化噪声）成立**。

> ⇒ **不要再纠结「交哪个 epoch」**：ep4 到 ep20 在排行榜上是同一条直线。

**② TTA 是唯一已验证有效的提分手段：+2.09 点。**（§16.2）

**③ 死类那条线**（上一轮 §15.5 称为"最有希望的方向"）**已放弃。**（§16.3）
实测 4257 个错误分散在 **3394 个无序类对**里，最集中的 20 个有序对只占 **2.9%** —— 没有可利用的集中结构。
其中少数 `asym = +1.00` 的类是**训练集里的系统性标注错误**（folder A 里的图真身是 B），训练和推理都救不回来。

**最重要的一句话**：**模型在这套配方下的天花板是 64.16，TTA 把它抬到了 66.2456。**
想再往上，只能换**输入表征**（288，正在跑）或**换机制** —— 靠调轮次、修死类都不行，两条路都已用硬证据封死。

---

## 1. 赛题与硬性约束

**赛题**：面向噪声标签数据的细粒度图像识别鲁棒微调（自然动植物细粒度分类）。
**当前阶段**：初赛。**赛道**：AIC-算法挑战赛（赛马制）。
**参赛编号**：`AIC-2026-88900760`，队伍 `LGL-Lab`。
**排行榜**：`reg.aicomp.cn/special/phb/list`（赛马制，成绩只在排行榜显示；**该域名无法用工具抓取，必须用浏览器打开**）。

### 硬性约束（违反 = 成绩作废）

1. 骨干**必须**是 CLIP **ViT-B/32**，只能用 **OpenAI 官方公开权重**（`openai/clip` 或 HuggingFace Transformers 对应版本）。
2. **不得**使用其他视觉基础模型、商业闭源 API、在线大模型推理接口。
3. **不得**多模型集成 / 模型融合 / 多模型投票。最终结果必须来自**单一模型、单一推理流程**。
   - 注意：EMA 教师（权重指数滑动平均）算**同一个模型**，不算集成。但如果你要引入任何"训练两个独立模型再合并"的做法，那是违规的。
4. 每个阶段**只能用当前阶段官方数据**。复赛不得用初赛数据，半决赛不得用初赛+复赛数据。
5. **测试数据不得以任何方式参与训练**（包括自监督、无监督）。
6. **不得引入任何额外数据集**（公开的、私有的、人工补标注的都不行）。
7. 训练过程必须**可由提交的代码完整复现**。**人工数据清洗不能是必要前置步骤**——噪声筛选必须是自动算法。
8. 指标：**Top-1 Accuracy** = 预测正确数 / 测试总数。

### 1.1 合规性对照（已逐条审查）

| 规则 | 我们的做法 | 判定 |
| --- | --- | --- |
| 五.1 / 十一(一).1 必须用 CLIP ViT-B/32 | `ViT-B-32-quickgelu` + `pretrained='openai'`（同一架构、同一套官方权重） | ✅ **已在代码里硬检查**，见 §8 第 8 项 |
| 十一(二).3 仅限 OpenAI 官方公开权重 | HF 的 `timm/vit_base_patch32_clip_224.openai` 或 Azure CDN 官方 `.pt` | ✅ `hf-mirror.com` 只是下载镜像，不是推理 API |
| 十一(一).2 不得用其他基础模型/闭源 API/在线推理接口 | 只用 `open_clip` 本地前向 | ✅ |
| 五.2 允许 PEFT / 鲁棒损失 / 噪声过滤 / 伪标签 / 表征约束 | LoRA + APL/GCE + `LabelTrustTracker` + 原型对比 + 锚定 | ✅ 全部落在明确列举的范围内 |
| 五.4 / 十一(一).3 不得集成，结果须来自单一模型单推理流程 | `infer.py` 只建一个 `Net`、加载一个 checkpoint、单次前向，**无 TTA** | ✅ |
| 十一(二).1 只用当前阶段数据；测试集不得参与训练 | 训练只读 `--data`；验证集从**训练集**切 10%；测试集只在 `infer.py` 前向（`model.eval()` + `no_grad`） | ✅ |
| 十一(二).2 不得引入额外数据集 | 无。RandAugment 等是增强算子，不是数据集 | ✅ |
| 五.6 噪声筛选必须自动、不能依赖人工清洗 | `noise.py` 是纯自动算法，无人工清洗环节 | ✅ |
| 五.5 可复现 | 种子 3407、`cudnn.deterministic=True`、`generator`+`worker_init_fn`、checkpoint 存 `args` 与 RNG | ✅ |

**EMA 教师不违规**：它是学生权重的在线滑动平均（`teacher = deepcopy(model)` 后每步 `0.995·t + 0.005·s`），**不是独立训练的模型**，而且**推理时完全不用**。赛题文档十(二) 原文写明鼓励"结合**教师模型**、自训练等方法增强鲁棒性"。

**两处灰色地带**（不是违规，但要知道边界在哪）：

1. **按排行榜选 checkpoint**（§9.5 的做法）。不是集成、不是投票，规则未禁止，但严格说测试集信息影响了模型选择。
   站得住的理由：初赛规则九.1 说"初赛主要用于参赛者熟悉任务、验证方案"；复现协议明确（"训练 20 轮取第 8 轮"，确定性种子下可重现）。
   ⚠️ **对外表述时讲成"确定最优训练轮数"这个超参的消融，不要说成"挑分数最高的交"。** 总决赛是线下答辩，会正面问这个。
2. **TTA**。单模型多视图平均算不算"单一推理流程"规则没写死。**我们现在完全不用，就别加。**

### 1.2 ⚠️ 后续阶段的真陷阱：复赛不得热启动初赛权重

规则十一(二).2 原文：

> 不得在训练过程中引入任何形式的额外数据集（包括公开的、私有的、或人工补充标注的等）。这里的限制包括：**复赛期间不得使用初赛的数据**，半决赛期间不得使用初赛和复赛的数据。

**推论：复赛时不能拿初赛训出来的 `outputs2/best.pt` 做初始化或继续微调**——那里面编着初赛数据的信息。复赛必须**从 OpenAI 官方权重重新开始训**。

这一条很容易无意中踩到（"拿初赛权重初始化收敛快"是很自然的想法），但按最保守的读法它是违规的。

**可以带走的**：算法配方、超参、代码。
**不能带走的**：权重、以及任何从上一阶段数据里统计出来的东西（包括原型、噪声跟踪器的状态）。

### 提交格式

- 文件名**必须**是 `pred_results.csv`，压缩成**一个 zip**提交。
- 每行两个字段：`图片文件名,类别编号`，**4 位数字、不足前补 0**，**无表头**。
- 图片文件名必须与测试集**完全一致**（含大小写和扩展名）。
- 赛题文档的示例里逗号后有个空格（`xxx.jpg, 0001`），但**实测不带空格可以通过**（我们的 68.02 用的就是无空格格式，平台状态 `DONE`）。继续用无空格。

### 数据规模（来自赛题文档）

| 阶段 | 类别数 | 训练样本 | 测试样本 | 特点 |
| --- | --- | --- | --- | --- |
| **初赛（当前）** | 500 | 103218 | 24967 | 标签含噪 |
| 复赛 | 1500 | 297282 | 74896 | 含噪 + 长尾 |
| 半决赛 | 1000 | 180274 | 49857 | 含噪 + 长尾 |

初赛不计入最终线上综合成绩（复赛 40% + 半决赛 60%），**但初赛是用来试方案、验证算法的**，所以现在的每一轮实验都是在为后两个阶段铺路。**跨阶段泛化能力是评分重点**。

数据特点（原文）：标签含噪（"错误标注、**弱相关标注**及难样本干扰"）、细粒度性强、长尾分布、规模大。另有提示：部分图片文件被截断、文件系统里显示不正常，但**都能被 Pillow 正常读取**。

### 训练数据的目录结构

```
<data_root>/
├── 0000/   *.jpg
├── 0001/   *.jpg
├── ...
└── 0499/   *.jpg
```

类文件夹名就是 0~499 的编号，**没有语义类名**。这一点很关键：**无法做 zero-shot 文本分类**（不知道每个类是什么动植物），所以只能用有监督微调。`infer.py` 直接把文件夹名转成 4 位 label 输出。

代码用 `sorted()` 遍历文件夹得到 `class_to_idx`，**顺序即编号**。只要目录名是 `0000`~`0499`，`class_to_idx` 就是恒等映射。

---

## 2. 代码结构

9 个 Python 文件，加 3 个文档。全部在仓库根目录，**没有子模块、没有包结构**。

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `train.py` | 752 | 训练主程序：模型定义（LoRA/余弦头/原型头）、数据、训练循环、评估、存档；**`resize_positional_embedding()` / `val_resize()` 也在这里**（`infer.py` / `analyze.py` / `valmetrics.py` 都从这里 import） |
| `infer.py` | 189 | 用 checkpoint 跑测试集，生成官方格式 `pred_results.csv`；**`--tta` 多视图平均**在这里 |
| `analyze.py` | 208 | **错误分析**：混淆结构（结构化噪声 vs 真·细粒度困难）、逐类准确率、疑似错标清单 |
| `datastats.py` | 216 | **数据集画像**：类别分布 / 长尾诊断 / 采样权重 / 是否有类名。只用标准库，可本地跑 |
| `valmetrics.py` | 120 | 对**已有 checkpoint** 算 micro / macro 准确率并排对比，零训练成本。内置自检：`val_acc` 必须等于训练日志里的值 |
| `losses.py` | 110 | 鲁棒损失：`ce` / `gce` / `nce` / `rce` / `apl` |
| `noise.py` | 150 | `LabelTrustTracker`：逐样本标签可信度跟踪、噪声过滤、伪标签纠正 |
| `selftest.py` | 552 | **改代码后必须先跑这个**：约 15 秒，不需要数据/GPU/CLIP 权重。本轮新增 `check_pos_embed_resize()` 和 `check_img_size_transforms()` |
| `probe_resolution.py` | 188 | **本轮新增**。分辨率探针：问 open_clip「换尺寸会不会自动插值」「插值后特征还对得上吗」。**不需要 GPU 也能跑**（但加载模型要 ~1.3GB 内存，无卡模式 2GB 跑不动） |
| `README.md` | — | 赛题原文（不要改） |
| `README_AUTODL.md` | — | AutoDL 平台操作手册 + 方法说明 + 参数表（中文，给操作者看的） |
| `HANDOFF.md` | — | 本文件 |

> ⚠️ **`train.py` 里那两个分辨率辅助函数是共享的**：改它们会同时影响 `infer.py` / `analyze.py` /
> `valmetrics.py`。改完必须跑 `selftest.py` —— `check_pos_embed_resize()` 会验 token 数、
> CLS 令牌不被改、斜坡方向不反转（抓转置/翻转，纯形状检查抓不到）、非法尺寸被拒。

> `datastats.py` 额外的好处：**它不需要 torch**，所以在本地 Windows 上直接
> `py -3 datastats.py --zip D:\BaiduDisk\train.zip` 就能分析，不用等 28 GB 传完。
> 每次换数据集（复赛、半决赛）**先跑它**，5 分钟，决定要不要上长尾方法。

### `infer.py` 的依赖方向

`infer.py` 从 `train.py` 导入 `Net` / `CLIP_MEAN` / `CLIP_STD` / `IMG_EXTS`，并重新构造一遍模型再加载 checkpoint 里的可训练权重。**改了 `Net` 的结构就要确认 `infer.py` 还能加载**（`selftest.py` 的端到端测试覆盖了这条）。

### checkpoint 的内容

```python
{
  'model': <只含可训练参数的 state_dict>,   # 几 MB，不含 350MB 的冻结 CLIP
  'classes': {'0000': 0, ..., '0499': 499},
  'optim': ..., 'epoch': ..., 'args': ...,
  'model_name': 'ViT-B-32-quickgelu',
  'val_acc': ..., 'val_acc_hi': ..., 'rng': ...
}
```

**「只存可训练参数」是刻意设计**：冻结的 CLIP 权重从官方权重重建，所以 checkpoint 从 ~350MB 降到几 MB。`train.py` 的 `Net.trainable_state_dict()` 负责过滤（`train.py:254-259`）。

---

## 3. 方法：训练配方详解

核心思路：**冻结 CLIP，只训很少的参数，让预训练先验不被噪声带偏**（赛题注意事项第 1 条明确说"不宜直接采用无约束的全参数微调"）。

训练参数总量：**约 1.141M**（LoRA 36 层 + 余弦头 + 原型）。日志里会打印 `trainable params=1.141M`。

### 3.1 模型（`train.py`）

```
CLIP ViT-B/32 视觉塔（冻结）
   └── 每层 nn.Linear 外面包一层 LoRALinear（可训练 B@A，rank=8, alpha=16）
         ↓ encode_image → F.normalize → 512-dim 特征 z
   ├── CosineClassifier：归一化权重 + 可学习温度，输出 500 类 logits
   └── EMA 类原型（ProtoHead）：对比学习用
```

四个要点：

1. **LoRALinear 必须是 `nn.Linear` 的完全替身**（`train.py:82-130`）。`B` 零初始化，所以训练开始时 LoRA 是恒等变换。`enabled=False` 时退回原始 CLIP 层——这是 `anchor_feat()` 拿到"冻结 CLIP 特征"而不必在显存里再放一份视觉塔的办法。

   > ⚠️ 这里踩过坑，见 §6.2。`nn.MultiheadAttention` **不调用** `out_proj(x)`，而是直接读 `out_proj.weight` 交给底层函数。所以 `weight` 必须返回**合并后**的矩阵。

2. **`anchor_feat(x)`**（`train.py:241-252`）：把 LoRA 关掉、模型切 eval，跑一遍视觉塔拿到"冻结 CLIP 该有的特征"。用来做锚定正则，抑制表征漂移/灾难性遗忘。

3. **`CosineClassifier`**（`train.py:161-170`）：`logit_scale.exp().clamp(1,100) * cos(z, W)`。注意 `logit_scale` 是可学习的、初始 `log(20)`，会慢慢长大——**这意味着"绝对置信度阈值"在训练早期不成立**（§6.3 的核心原因）。

4. **`ProtoHead`**（`train.py:173-220`）：每类一个 512 维 EMA 原型，用**冻结 CLIP 特征**的类均值初始化（warm-up 期间累积，`train.py:532-535`），之后**只用被信任样本**的教师特征做 EMA 更新。

### 3.2 数据（`train.py:265-354`）

- **`ImageFolderNoisy`**：按类文件夹读图，**分层**切出验证集。切分在**每类内部**做 `n = max(1, int(len * val_ratio))`，所以每类都按比例留出。
- 关键细节：**验证集的切分大小不能依赖"正在构建哪个 split"**，否则训练集会静默吞掉整个验证集。这个 bug 修过（见 §6.5）。
- 返回 `(image, target, index)`，`index` 是在**训练项列表**里的下标，噪声跟踪器和原型 bootstrap 都按这个 index 索引。
- **`TwoView`**：同一张图做**两次独立增强**，返回双视图。用于一致性正则。
- 增强：`RandomResizedCrop(224, scale=(0.55, 1.0))` + 水平翻转 + `RandAugment(2, 9)`。验证集：`Resize(256) + CenterCrop(224)`。
- 归一化用 **CLIP 自己的** mean/std（`train.py:49-50`），不是 ImageNet 的。
- 采样：`WeightedRandomSampler`，权重 `1/sqrt(类频次)`。**这是为复赛/半决赛的长尾阶段准备的**，`--sampler balanced` 是默认值。
- 读不出来的图 → 灰色图兜底，不跳过（否则 CSV 会缺行 → 提交直接无效）。

### 3.3 训练循环（`train.py:451-563`）

**阶段一：纯 CE warm-up（默认 3 epoch）**
只做监督学习，用原始标签，不做任何噪声处理。目的是让模型先"热"起来，教师才有意义。

在 warm-up 的最后一个 epoch 结束时，用累积的冻结 CLIP 特征类均值 seed 原型：

```python
means, present = prototype_bootstrap(boot_sum, boot_cnt)
model.proto.init_from_means(means, present)
```

**阶段二：EMA 教师驱动（第 4 epoch 起）**

每个 batch：

```
1. anchor = model.anchor_feat(x1)              # 冻结 CLIP 特征（LoRA off）
2. out, z   = model(x1)   ;  out2, z2 = model(x2)      # 两个视图
3. tout, tz = teacher(x1)                      # EMA 教师，no_grad
4. tracker.update(idx, softmax(tout))          # 免费：教师已经跑过了
5. hard  = tracker.label[idx]                  # 可能被纠正过的标签（只给原型用）
   mix_t = tracker.target(idx, y)              # 混合伪标签 (B,C)，给鲁棒项
   soft  = smooth_target(mix_t, 0.05, C)       # 再加标签平滑，给 CE 项
   w     = tracker.weight[idx] * max_p^2       # 逐样本权重
   w     = w / w.mean()                        # 归一化，稳定有效步长
6. loss = 0.5*(w*CE(out,soft) + w*CE(out2,soft))              # 标注项
       + robust_weight * w * APL(out/out2, mix_t)             # 鲁棒项 0.5
       + consistency_weight * MSE(z, z2.detach())             # 一致性 0.05
       + anchor_weight * (1 - cos(z, anchor))                 # 锚定 0.1
       + proto_weight * w * CE(proto.logits(z), mix_t)        # 原型对比 0.5
7. 反向 + 梯度裁剪(1.0) + AdamW
8. EMA 更新教师权重: tp = 0.995*tp + 0.005*sp   （只更新 requires_grad 的参数）
9. model.proto.update(tz, hard, trusted)       # 只用被信任样本；原型 EMA 仍要硬标签
```

> **`soft` 和 `mix_t` 为什么是两个不同的目标**：标签平滑只作用于 CE 项。`losses.nce`
> 的 `-log(p)` 分支是无界的，750 类下每类 `0.05/750` 的平滑质量加起来会产生一个很大
> 的、与鲁棒性无关的"推向均匀"梯度。两者在 one-hot 时都精确退化为原来的公式
> （`selftest.check_losses` 有断言）。

每个 epoch 开始时（非 warm-up）调用一次 `tracker.refresh()` 重算标签和权重。

优化器 `AdamW(lr=2e-4, weight_decay=0.05)` + `CosineAnnealingLR`（周期 = `--epochs`）。
AMP 默认 **bf16**（4090 是 Ada 架构，比 fp16 + GradScaler 更稳更快，也省掉梯度缩放的随机性）。

### 3.4 噪声处理（`noise.py`）——本项目最核心的算法

`LabelTrustTracker` 维护**每个训练样本**的教师后验 EMA（`momentum=0.9`）。规则：

| 条件 | 目标 | 权重 |
| --- | --- | --- |
| `argmax p == 给定标签`（教师**同意**） | 硬 one-hot `y` | `1.0` |
| 不同意 且 `max(p) >= tau_conf`(0.8) | `(1-mix)·y + mix·argmax p`，`mix=0.5` | `w_relabel` = 0.5 |
| 不同意 且 `max(p) < 0.8` | 硬 one-hot `y` | `w_noise` = 0.1 |
| 采样器还没抽到过 | 硬 one-hot `y` | `1.0` |

> **第二行是混合，不是覆盖**（`--relabel-mix`，默认 0.5）。赛题说噪声含"弱相关标注"
> ——给定标签是错的但**相关**。这种噪声下教师"自信地不同意"经常不是标签错了，而是
> 两个类本来就难分。硬覆盖会把这种混淆提升为 ground truth，正好教会模型我们要避免
> 的错误。保留一部分给定标签的证据、让教师只拉一部分，才是对的。

**判据是教师的"首选类别"，不是它的绝对置信度**——这一点极重要，见 §6.3。

**不做删除**，只降权：这样采样计划（`WeightedRandomSampler` 的 `num_samples`）在全程都有效。

`max_noise_frac=0.4` 是保险丝：教师没收敛时不许把超过 40% 的样本判为不可信。超出的部分按 trust 升序保留最低的那批为"不可信"，其余救回。被救回的数量记在 `stats['capped']`。

**统计口径必须自洽**：`clean + relabel + noisy + unseen == N`。日志里出现 `noise stats: {...}`，四个数加起来必须等于训练集大小（93102）。**这是判断噪声模块是否正常工作的第一指标。**

`clean` 的定义是"按全权重训练的样本"= 上面两个分支的剩余，所以被判为 noisy 又被上限救回的样本会算进 `clean`。

### 3.5 鲁棒损失（`losses.py`）

都返回**逐样本**的 `(B,)` 向量，方便外面乘权重。

- **`gce`**：`(1 - p_y^q)/q`，上界 `1/q`，`p_y→1` 时梯度消失，所以"已经背下来的错样本"不再主导更新。
- **`nce`**（APL 的主动项）：`p <= k` 时是 `-log(p)`；`p > k` 时替换成 `-log(p)` 在 `p=k` 处的**切线** `A·p^B + C`（`A=-1/(B·k^B)`，`C=1/B-log k`）。梯度永不消失，所以叫"主动"损失。
  - ⚠️ **`k=0.2` 时损失下界约 −2.39，所以 warm-up 之后训练 loss 打印成负数是正常的，不是发散。**
- **`rce`**（APL 的被动项）：`-log(eps)·(1-p_y)`，`eps=1e-4` → `≈ 9.21·(1-p_y)`，范围 `[0, 9.21]`，类似 MAE。
  - ⚠️ **这个函数原来写错了**，见 §6.4。
- **`apl`** = `nce + rce_scale·rce`，默认 `rce_scale=1.0`。

### 3.6 评估（`train.py:360-384`）

```python
evaluate() -> (loss, acc, acc_hi)
```

- `val_acc`：整个验证集上的准确率，**与"给定标签"比对**。
- `val_acc_hi`：只统计 `max(softmax) >= 0.8` 的高置信样本。

⚠️ **两个都不是真实测试准确率的可靠代理**，见 §7。

---

## 4. 运行环境（AutoDL）

### 4.1 平台与实例

- **AutoDL**（`autodl.com`）租用 **RTX 4090 24GB**，20 vCPU。
- 数据盘（持久、便宜）：`/root/autodl-tmp/`——**所有代码、数据、权重、输出都放这里**。
- 系统盘 `/root/` 容量小，放大数据会被清。

**省钱要点**：AutoDL 支持**无卡模式**开机（很便宜）。传数据、装依赖、下权重全部在无卡模式下做完，再关机切到 GPU 模式训练。

### 4.2 已装好的环境

```
torch 2.3.0+cu121
torchvision
open_clip_torch 3.3.0
Pillow
```

重装命令（一般不需要，已就位）：

```bash
cd /root/autodl-tmp
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.3 文件上传（这是最常卡住的一步）

**用 FileZilla（SFTP）**：

| 字段 | 填什么 |
| --- | --- |
| 主机 | 实例的 `region.autodl.com`（在控制台实例卡片上） |
| 端口 | **SSH 登录指令里 `-p` 后面的那个数字**（不是 22！） |
| 协议 | SFTP |
| 用户 | `root` |
| 密码 | 实例密码 |

**上万个小文件用 FileZilla 传极慢**（每个文件一次往返）。做法：**本地先打成 zip，上传后解压**。

```powershell
# Windows 本地打包（PowerShell）
tar -a -c -f train.zip train
tar -a -c -f test.zip test
```

```bash
# 云端解压
cd /root/autodl-tmp
apt install unzip -y        # 若缺
unzip train.zip -d /root/autodl-tmp/
unzip test.zip -d /root/autodl-tmp/
```

小文件（5 个 .py + README）直接用 FileZilla 拖过去即可，很快。

**下载结果文件**（CSV/zip）同样用 FileZilla，从云端拖回本地。

### 4.4 CLIP 权重（国内最容易卡的一步）

`open_clip` 3.3.0 改成从 **HuggingFace** 拉 OpenAI 权重（不再是 Azure CDN），国内直连会 `Network is unreachable`。

```bash
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc    # 一劳永逸
```

设好后第一次运行会自动下载 `timm/vit_base_patch32_clip_224.openai`（605 MB），落在 `~/.cache/huggingface/`，之后走缓存不再联网。

**备用方案**（走 Azure CDN 手动下官方 `.pt`）：

```bash
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18e219bf2a342f68e27b3960b7620019/ViT-B-32.pt \
     -O /root/autodl-tmp/ViT-B-32.pt
# 然后训练时加 --pretrained /root/autodl-tmp/ViT-B-32.pt
```

> ⚠️ **必须用 `ViT-B-32-quickgelu`**（已是代码默认值）。OpenAI 官方 ViT-B/32 用的是 **QuickGELU** 激活；如果建成普通的 `ViT-B-32`（GELU），open_clip 会打印 `QuickGELU mismatch ...` —— **这不是无害警告，精度会掉**。详见 §6.1。

### 4.5 当前实测性能

- **约 144 秒 / epoch**（93102 训练样本，batch 128，12 workers）。
- 20 epoch ≈ **48 分钟**。
- 显存占用约 **8.9 GB / 24 GB**，GPU 利用率 **95%**。
- 如果 GPU 利用率长期低于 60%，是数据加载瓶颈 → 把 `--workers` 加到 12~16。

---

## 5. 完整流程（从零到提交）

### Step 0 — 确认 GPU 没有被占

```bash
pgrep -af train.py
```

空输出 = 没有训练在跑。

### Step 1 — 上传代码

把 `train.py` / `infer.py` / `losses.py` / `noise.py` / `selftest.py` / `requirements.txt` 传到 `/root/autodl-tmp/`。

### Step 2 — 环境自检

```bash
cd /root/autodl-tmp
nvidia-smi                                          # 应看到 RTX 4090 24GB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 3 — 设 HF 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Step 4 — **跑自检（一定要先跑）**

```bash
python selftest.py
```

必须看到 **`ALL CHECKS PASSED`** 才继续。约 10 秒，不需要数据/GPU/权重（它用桩模型替换了 `open_clip`，但桩模型里有真的 `nn.MultiheadAttention`，所以能守住 LoRA 那条回归）。它会跑完一次完整的 4 epoch 训练 + 推理 + CSV 格式校验。

### Step 5 — 冒烟测试（1 epoch 的前 20 步）

```bash
python train.py --data /root/autodl-tmp/train --out ./smoke \
  --epochs 1 --warmup-epochs 1 --limit-batches 20 --batch-size 64 --workers 8
```

应该打印 `first batch ok: ...` 然后正常结束。另开终端 `watch -n 1 nvidia-smi` 看利用率。

### Step 6 — 正式训练

```bash
cd /root/autodl-tmp
export HF_ENDPOINT=https://hf-mirror.com

nohup python -u train.py --data /root/autodl-tmp/train --out ./outputs2 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  > train2.log 2>&1 &

tail -f train2.log
```

> 云端**没有 `tmux`**（试过，`command not found`），所以用 `nohup`。
> AutoDL 的**网页终端会吞掉多行粘贴**——一行一行粘，或先写好脚本再执行。

监控：

```bash
tail -f train2.log
grep -c "^epoch" train2.log     # 已完成多少轮
nvidia-smi
```

中断了就地续训（记得带上其余参数）：

```bash
python train.py --data /root/autodl-tmp/train --out ./outputs2 --resume ./outputs2/last.pt ...
```

### Step 7 — 推理

```bash
unzip test.zip -d /root/autodl-tmp/          # → /root/autodl-tmp/test/*.jpg

python infer.py --test /root/autodl-tmp/test \
                --checkpoint outputs2/best.pt \
                --output pred_results.csv

wc -l pred_results.csv        # 初赛 24967 / 复赛 37444（见 §14）
head -3 pred_results.csv      # 形如 00012f3f....jpg,0083
```

### Step 8 — 打包提交

```bash
zip submission.zip pred_results.csv
```

**提交前清单**：

- [ ] 行数 = 测试集图片数（初赛 24967 / 复赛 37444）
- [ ] 无表头
- [ ] 每行恰好 2 个字段，文件名与测试集完全一致
- [ ] 标签是 4 位数字、共 500 个不同值
- [ ] 文件名是 `pred_results.csv`
- [ ] zip 里只有这一个文件

用 FileZilla 把 zip 拖回本地，去平台提交。

### Step 9 — 在排行榜看分

浏览器打开 `reg.aicomp.cn/special/phb/list`（**代码工具抓不到这个域名，必须用浏览器**）。提交记录页有【详情】按钮。

---

## 6. 踩过的坑（按踩到的顺序）

### 6.1 QuickGELU 失配（静默掉精度）

**现象**：`open_clip.create_model('ViT-B-32', pretrained='openai')` 打印

```
QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)
```

**原因**：OpenAI 官方 ViT-B/32 是用 **QuickGELU** 训出来的。`ViT-B-32` 这个名字对应的配置是普通 GELU。于是模型拿 GELU 去跑按 QuickGELU 训练的权重。

**修法**：模型名改成 **`ViT-B-32-quickgelu`**。已加 `--model` 参数（默认值就是这个），checkpoint 里存了 `model_name`，`infer.py` 自动读取。

**教训**：open_clip 的 warning 不都是无害的。

### 6.2 `AttributeError: 'LoRALinear' object has no attribute 'weight'`

**现象**：训练跑到 `z = visual(x)` 崩掉，栈顶在 `nn.MultiheadAttention.forward` 里的 `self.out_proj.weight`。

**原因**：`nn.MultiheadAttention` **不调用** `out_proj(x)`，而是**直接读** `out_proj.weight` / `.bias` 传给底层函数。原来的 `LoRALinear` 只实现了 `forward`，所以属性访问失败。

**修法**：给 `LoRALinear` 加 `weight` / `bias` / `in_features` / `out_features` 属性（`train.py:109-125`）。

> ⚠️ **关键细节**：`weight` 必须返回**合并后**的矩阵 `base.weight + (B@A)*scale`。如果只转发 `base.weight`，不会报错，但**注意力路径上的 LoRA 会静默失效**。

**连带教训**：原来的 `selftest.py` 桩模型是个普通 MLP，**根本不经过 MHA 路径**，所以没抓到这个 bug。已把桩模型改成结构上模仿 open_clip（含真的 `nn.MultiheadAttention`），作为回归守卫。

### 6.3 噪声筛选每轮都撞上限（**影响最大的 bug**）

**现象**：训练日志里 `noisy` **每个 epoch 都恰好等于 37240**。而 37240 = `int(0.4 × 93102)` = `max_noise_frac × 训练集大小`。连续 20 轮一模一样，不可能是巧合。

**原因**：旧判据是

```python
clean = trust >= tau_clean          # tau_clean = 0.8，要求 p[y] >= 0.8
noisy = ~clean & ~relabel
```

它要求教师"**既同意又自信到 0.8**"才肯相信一个样本。但 500 类的余弦头 `logit_scale` 是**慢慢长起来**的，训练早期绝对概率普遍偏低。实测**只有 16.6% 的样本 `p[y] >= 0.8`**，其余 83%——**包括教师明确同意、只是 `p[y] ≈ 0.5` 的样本**——全被扔进"不可信"，压到 0.1 权重。然后 40% 上限每轮都触发。

**另一个佐证**：日志里 `clean 15449 + relabel 430 + noisy 37240 + unseen 6 = 53125`，而训练集是 **93102**。四个数加起来对不上——**统计口径本身就不自洽**，被上限救回的那批样本哪个类别都没算。

**修法**（`noise.py:66-111`）：判据改成看教师的**首选类别**：

```python
agree   = pred == self.y
relabel = judged & ~agree & (conf >= self.tau_conf)
noisy   = judged & ~agree & (conf <  self.tau_conf)
clean   = judged & ~relabel & ~noisy       # 剩余，保证严格划分
```

**同意就是信任的证据，只有反对才是反对的证据。** 同时加了 `capped` 计数器，四个数现在严格划分整个训练集。

`--tau-clean` 参数已删除；`--tau-conf`（默认 0.8）现在只用于"教师**否定**给定标签时，要多自信才敢改标注"。

**怎么早点发现**：`noisy` 连续多轮**等于某个整数**就是撞上限的信号。README_AUTODL 里已写明这条。

### 6.4 `rce` 实现成了熵（和自己的文档相反）

**现象**：训练 loss 一路降到 **−0.53**，`val_loss` 从 1.38 **涨**到 1.77，`val_acc_hi` 从 0.975 **掉**到 0.873。

**原因**：`rce` 的实现是 `-(p·log p)`，即预测的**熵**，而它的 docstring 写着"damps over-confidence"。**最小化熵是在鼓励过度自信**，和文档说的正好相反。

**修法**（`losses.py:43-62`）：改回 APL 论文（Ma et al., ICML 2020）的定义

```
RCE = -Σ_k p_k log q_k = -log(eps)·(1 - p_y)      # eps=1e-4 → ≈ 9.21(1-p_y)
```

即 MAE 型的项，范围 `[0, 9.21]`。

**这个 bug 和 §7 的诊断直接相关**：熵项在主动把模型推向过度自信，正好是在喂噪声记忆。

### 6.5 验证集切分依赖"正在构建哪个 split"

**原因**：旧代码里 `n` 的算法在构建训练集和验证集时算出了不同的值，导致训练集静默吞掉整个验证集（或反之）。

**修法**：切分大小只由 `val_ratio` 和类内样本数决定，与 `val=True/False` 无关（`train.py:288-291`）。`selftest.py` 里断言了 `tr ∩ va == ∅` 且 `len(tr) + len(va) == 全集`。

### 6.6 推理丢图 → 提交缺行 → 直接判无效

**原因**：原代码 `except: pass`，读不出的图直接跳过，CSV 行数就少于测试集图片数。

**修法**：读不出来用**灰色图兜底**，保证每张测试图都有且仅有一行（`infer.py:72-77`）。

赛题文档也提到"部分数据可能由于图片文件部分截断问题导致…无法正常显示"，所以这条必须防。

### 6.7 其他环境坑

| 坑 | 解法 |
| --- | --- |
| `tmux: command not found` | 用 `nohup python -u ... > log 2>&1 &` |
| 多行粘贴被 AutoDL 网页终端吞掉（`tail: cannot open 'train.log'`） | 一行一行粘，或先写成 `.sh` |
| CLIP 权重下载 `Network is unreachable` | `export HF_ENDPOINT=https://hf-mirror.com` |
| 赛题文档 CSV 示例有空格 `xxx.jpg, 0001` | **实测不用加空格**，无空格可通过 |
| 提交后平台回传的 `pred_results.csv` | 那只是**你自己文件的回显，不含分数**，别当评分找 |
| 成绩不在提交记录页 | 赛马制，成绩在**排行榜**：`reg.aicomp.cn/special/phb/list`，必须浏览器打开 |

---

## 7. ⚠️ 核心发现：`val_acc` 会**高估**真实水平

这是目前最重要的认知，直接决定后续策略。

### 事实

| 指标 | 值 |
| --- | --- |
| 验证集准确率 `val_acc`（与**含噪**验证标签比对） | **0.7305** |
| 平台真实得分（**人工精确标注**的测试集） | **0.6802** |

**差了 5.03 个点，而且方向是"验证集虚高"。**

### 为什么这个方向很反直觉

验证集是从训练集里切出来的，**带着同一套噪声**。设真实准确率 `a`、噪声比例 `η`：

- 干净样本（`1-η`）：模型预测真实标签 = 给定标签 → 一致 ✓
- 错标样本（`η`）：模型预测真实标签 ≠ 给定标签 → 不一致 ✗

**如果噪声是随机的、且模型完全抵抗住了它**，则

```
val_acc ≈ (1-η)·a   （必然小于 a）
```

但实测 `0.7305 = (1-η) × 0.6802` 解出 **`1-η = 1.074 > 1`**，**不可能**。

### 结论

模型在验证集上"答对"的一部分，**其实是它把给定标签里的错误也学了过去**。而且这种学习**泛化到了没见过的验证集图片**——随机噪声做不到这一点（验证图片没参与训练）。唯一解释是：**噪声是结构化的、类间一致的**。

这正好对上赛题文档 §四(三) 的描述：**"错误标注、弱相关标注及难样本干扰"**。所谓"弱相关标注"就是"某一类图片被系统性地标成了另一类"——模型学会的是这个**错误映射**，它在验证集上同样成立，在干净的测试集上就失效。

### 日志里的佐证（基线运行）

| 指标 | 第 4 轮 | 第 20 轮 | 含义 |
| --- | --- | --- | --- |
| `clean`（`p[y]≥0.8`） | 8797 | **43859** | 模型"认同"的给定标签越来越多 |
| `mean_trust` | 0.297 | **0.628** | 一路涨，**没有饱和** |
| 训练 loss | 0.97 | **−0.53** | 极度自信 |
| `val_loss` | 1.62 | 1.70（第 9 轮峰值 1.77） | 不降反升 |

**如果模型只学真实信号，认同率应该在某个点饱和。** 它一路涨到训练结束，说明它一直在吃噪声。

### 推论

1. **`val_acc` 不能当优化目标**——它奖励的正是我们要避免的行为。
2. **优化转向**：需要多个候选模型，**用排行榜选**，不能靠 `val_acc` 选。已加 `--save-every 4`（默认值）每 4 轮存一个 `epN.pt`。
3. 一个**可验证的实验**：如果较早轮次（比如 `ep4`）的分数**更高**，就证明"吃噪声"这个判断成立；如果总是 `ep20` 最好，那说明测试集本来就比验证集难，方向要换。
4. 注意 `--select` 默认是 `val_acc`，所以 **`best.pt` 反而是最可疑的那个**（它是按虚高指标挑的）。

### 基线完整曲线（`./outputs`，有 bug 的版本）

| epoch | val_acc | val_acc_hi | val_loss | train loss |
| --- | --- | --- | --- | --- |
| 1 | 0.6211 | 0.9748 | 1.7746 | 2.8873 |
| 3 | 0.6924 | 0.9698 | 1.3807 | 1.4196 |
| 4 | 0.6866 | 0.9092 | 1.6170 | 0.9686 |
| 6 | 0.6977 | 0.8754 | 1.7601 | −0.0063 |
| 8 | 0.7054 | 0.8702 | 1.7635 | −0.1850 |
| 10 | 0.7114 | 0.8693 | 1.7605 | −0.2933 |
| 20 | **0.7305** | 0.8728 | 1.6962 | −0.5257 |

另一个可疑信号：**噪声模块生效（第 4 轮）之前**，`val_acc` 每轮涨 **+0.026**；**之后**只有 **+0.003 ~ +0.005**。也就是说那套"噪声处理"上线后，进展反而慢了近一个数量级。

---

## 8. 已经做完的改动（**尚未在真实数据上验证**）

改动集中在 3 个 bug 上，加一个存档功能。全部只过了 `selftest.py`，**没有跑过真实训练**。

| # | 文件 | 改动 | 状态 |
| --- | --- | --- | --- |
| 1 | `train.py:109-125` | `LoRALinear` 暴露合并后的 `weight`/`bias`/`in_features`/`out_features` | 已验证（selftest 含 MHA 回归） |
| 2 | `train.py:576` | 模型名 `ViT-B-32` → `ViT-B-32-quickgelu`，新增 `--model` | 已验证（跑通训练） |
| 3 | `noise.py:66-111` | 噪声判据改为基于**教师首选类别**；`clean` 改为剩余项；新增 `capped`；删掉 `tau_clean` | **只过了 selftest** |
| 4 | `losses.py:43-62` | `rce` 改为 `-log(eps)·(1-p_y)` | **只过了 selftest** |
| 5 | `train.py:592-596` + `train.py:552-559` | 新增 `--save-every`（默认 4），每 4 轮存 `epN.pt` | **只过了 selftest** |
| 6 | `selftest.py` | 桩模型改用真 MHA；新增回归测试 3b；测试 4 断言划分自洽；端到端断言 `ep4.pt` | 已跑通 |
| 7 | `README_AUTODL.md` | 补 HF 镜像、quickgelu、参数表、三个 bug 的说明、`val_acc` 虚高的警告 | — |
| 8 | `train.py:52-67`, `train.py:404`, `infer.py:41` | **合规护栏**：`check_backbone()` 白名单，`--model ViT-L-14` 之类直接 `SystemExit`；`--pretrained` 非 `openai` 时告警 | 已加 selftest 断言 |

**改动还没有上传到云端。** 云端 `/root/autodl-tmp/` 里现在是**旧的、有 bug 的版本**。

---

## 9. 现在要做什么（立即下一步）

### 9.1 上传修复后的代码

用 FileZilla 把本地这 6 个文件覆盖到 `/root/autodl-tmp/`：

```
train.py    infer.py    noise.py    losses.py    selftest.py    README_AUTODL.md
```

> `infer.py` 也改了（合规护栏），**别漏传**，否则训练出来的 checkpoint 在推理端不做骨干检查。

### 9.2 确认 GPU 空闲 → 跑自检

```bash
cd /root/autodl-tmp
pgrep -af train.py                     # 应为空
export HF_ENDPOINT=https://hf-mirror.com
python selftest.py                     # 必须 ALL CHECKS PASSED
```

### 9.3 启动修复后的训练

**注意输出目录用 `./outputs2`**，`./outputs` 是 68.02 的基线，留着对比，**不要覆盖**。

```bash
cd /root/autodl-tmp
export HF_ENDPOINT=https://hf-mirror.com

nohup python -u train.py --data /root/autodl-tmp/train --out ./outputs2 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  > train2.log 2>&1 &

tail -f train2.log
```

约 50 分钟。

### 9.4 训练过程中要盯的两个指标

1. **`noise stats` 的四个数必须加起来等于 93102**（`clean+relabel+noisy+unseen`）。对不上 = 统计口径又坏了。
2. **`noisy` 不应该等于 37240**（= `0.4 × 93102`，撞上限）。如果又撞上了，说明新判据在这个模型上也不成立，需要调 `--tau-conf` 或 `--max-noise-frac`。

另外对比基线的第 4/8/20 轮：`0.6866 / 0.7054 / 0.7305`。修复后应该明显更好，否则说明我们对 bug 的判断有误。

### 9.5 用排行榜选轮次（关键步骤）

```bash
cd /root/autodl-tmp
for ep in 4 8 12 16 20; do
  mkdir -p sub_ep$ep
  python infer.py --test /root/autodl-tmp/test \
                  --checkpoint outputs2/ep$ep.pt \
                  --output sub_ep$ep/pred_results.csv
  (cd sub_ep$ep && zip ../submission_ep$ep.zip pred_results.csv)
done
ls -la submission_ep*.zip
```

每个 zip 依次去平台提交，记录分数。

- **如果靠后的轮次分数更低** → 证实"模型在吃噪声"，那么最优轮次在早期，后续应该减少 epoch 或加强正则。
- **如果一直 `ep20` 最好** → "测试集比特难"这个解释成立，方向要换（可以参考 §10）。

> ⚠️ **提交次数配额未知**（平台页面没写明）。如果配额紧张，先提交 `ep8` 和 `ep20` 探两端，再按结果补中间。**接手时请先确认配额。**

### 9.6 预期

**不要相信任何乐观预测。** 上一轮我们预测测试分应该**高于** `val_acc`（估 0.75~0.84），实际是 0.6802 —— 方向就错了。

基于目前唯一的实测数据点：`val_acc` 比真实分**高约 5 个点**。如果修复后 `val_acc` 落在 0.78~0.80，真实分大约在 0.73~0.75。**这只是推算，不是承诺。**真正的判据是 §9.5 那 5 个快照在排行榜上的分布。

---

## 10. 后续可选优化方向（按预期收益排序）

1. **保存并评估 EMA 教师权重。**
   `teacher` 是模型权重的完整 EMA 副本（`--ema 0.995`），但它**从来没有被评估或保存过**（`train.py:516-521` 只做 EMA 更新）。
   EMA 权重通常比原始学生权重泛化更好——**正好对症我们诊断出的"过拟合噪声"**。
   代价：每 epoch 多一次验证前向。EMA 是同一模型的权重滑动平均，**不算集成，不违规**。
   落地方式：`evaluate(teacher, ...)`，如果比学生好就存 `teacher.pt`。

2. **减少 epoch / 更强正则。**
   如果 §9.5 证实早期轮次更好，就直接调整 `--epochs`，或加大 `--weight-decay`、调小 `--lr`。

3. **`--select` 换个选择标准。**
   现在按 `val_acc` 选 `best.pt`，而 `val_acc` 是虚高的。可以考虑按 `val_loss` 选（基线里 `val_loss` 第 9 轮就见底了，比 `val_acc` 的峰早得多）。

4. **消融实验**（README_AUTODL §4 有清单，每项单独跑，别一次全改）：
   - `--robust-loss gce` vs `apl`
   - `--w-noise 1 --w-relabel 1`（等于关掉噪声筛选，只保留置信度重加权）
   - `--proto-weight 0`（关原型对比）
   - `--anchor-weight 0`（关锚定）
   - `--lora-target mlp`（LoRA 只加 MLP，更保守）

5. **为复赛/半决赛做准备。**
   初赛不计入最终成绩。复赛 1500 类、长尾，半决赛 1000 类、长尾，两者合计决定线上综合分（40% + 60%）。
   `--sampler balanced`（`1/sqrt(freq)` 类均衡采样）**已经默认开着**，就是为长尾准备的。**应当尽早验证它在长尾数据上是否真的有效**，因为跨阶段泛化是评分重点。

6. **提交前的复现性检查。**
   赛题明确写了"若赛事方无法基于提供的代码、环境和赛事数据复现主要实验结果，将取消相应成绩"。目前的可复现性措施：固定种子 `3407`、`cudnn.benchmark=False`、`cudnn.deterministic=True`、DataLoader 传 `generator` + `worker_init_fn`、bf16、checkpoint 存 `args` 和 RNG 状态。**提交前建议在干净环境跑一遍 `selftest.py` + 一次短训练验证。**

---

## 11. 参数速查表

### 命令行

```bash
python train.py \
  --data /root/autodl-tmp/train \      # 必需。类文件夹根目录
  --out ./outputs2 \                   # 输出目录（checkpoint + 日志）
  --epochs 20 \                        # 默认 20
  --warmup-epochs 3 \                  # 纯 CE 预热轮数，之后启用噪声处理
  --batch-size 128 \
  --workers 12 \                       # 数据加载进程。GPU 利用率低于 60% 就加大
  --model ViT-B-32-quickgelu \         # 别改，见 §6.1
  --pretrained openai \                # 或本地 .pt 路径
  --lr 2e-4 --weight-decay 0.05 \
  --seed 3407 \
  --amp bf16 \                         # bf16 / fp16 / none
  --val-ratio 0.1 \                    # 定稿冲分时可降到 0.05（更多训练数据）
  --sampler balanced \                 # 1/sqrt(freq) 类均衡，长尾用
  --select val_acc \                   # 选 best.pt 的依据（⚠️ 虚高，见 §7）
  --save-every 4 \                     # 每 4 轮存 epN.pt（0 = 关闭）
  --resume ./outputs2/last.pt          # 断点续训（记得带上其余参数）
```

### 噪声处理

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--tau-conf` | 0.8 | 教师**否定**给定标签时，要多自信才敢改标注 |
| `--w-noise` | 0.1 | 判为不可信样本的损失权重 |
| `--w-relabel` | 0.5 | 被改标注样本的损失权重 |
| `--max-noise-frac` | 0.4 | 最多判多少比例不可信（保险丝） |
| `--noise-momentum` | 0.9 | 教师后验的逐样本 EMA 动量 |
| `--conf-gamma` | 2.0 | 置信度重加权的指数 |
| `--relabel-mix` | 0.5 | 改标注时有多少目标质量移到教师的选择上（1.0 = 硬覆盖，旧行为） |
| `--label-smooth` | 0.05 | CE 项的标签平滑（只作用于 CE 项，见 §3.3 的说明） |

### 鲁棒损失

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--robust-loss` | apl | `apl` / `gce` / `nce` / `ce` |
| `--robust-weight` | 0.5 | 鲁棒项权重 |
| `--apl-k` | 0.2 | NCE 拐点（`p>k` 后走切线） |
| `--apl-b` | 1.0 | NCE 切线幂次 |
| `--apl-rce` | 1.0 | RCE 缩放（调小更温和） |
| `--gce-q` | 0.7 | GCE 的 q（上界 `1/q`） |

### 其他损失项权重

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--proto-weight` | 0.5 | 原型对比损失 |
| `--proto-temp` | 0.1 | 原型 softmax 温度 |
| `--proto-momentum` | 0.99 | 原型 EMA 动量 |
| `--proto-min-weight` | 0.5 | 更新原型所需的最低样本权重 |
| `--anchor-weight` | 0.1 | 冻结 CLIP 锚定 |
| `--consistency-weight` | 0.05 | 两视图一致性 |
| `--lora-rank` | 8 | LoRA 秩（alpha 固定为 2×rank） |
| `--lora-target` | all | `all` / `mlp` |

### `infer.py`

```bash
python infer.py \
  --test /root/autodl-tmp/test \       # 必需，递归搜索
  --checkpoint outputs2/best.pt \      # 必需
  --output pred_results.csv \
  --batch-size 256 \
  --pretrained '' \                    # 空 = 用 checkpoint 里记录的
  --lora-rank 0 \                      # 0 = 用 checkpoint 里记录的
  --tta plain flip wide tight \        # 多视图 logits 平均；不写则与旧行为逐字节相同
  --img-size 0                         # 0 = 用 checkpoint 里记录的（一般别动）
```

**注意**：`infer.py` 会自动从 checkpoint 读取 `model_name` / `lora_rank` / `lora_target` /
`pretrained` / **`img_size`**，一般**不需要手动指定**。

**`--tta` 视图**（详见 §16.2 和 `README_AUTODL.md` Step 7b）：
`plain`(87.5% 裁剪，= 训练取景框) / `flip` / `wide`(整幅画面，= open_clip 官方预处理) /
`tight`(70% 裁剪)。视图是**比例**，自动跟着 `--img-size` 缩放。

> 实测 **4 视图 = 66.2456 vs 无 TTA = 64.16（+2.09）**。
> ⚠️ 但**单个 `wide` 视图单独用是 −2.16（62 分）** —— 收益来自多视图平均，不是"修正取景框"。

---

## 12. 排错手册

| 症状 | 原因 / 解法 |
| --- | --- |
| `Network is unreachable` 下载权重失败 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `QuickGELU mismatch` 警告 | 模型名被改成了 `ViT-B-32`，改回 `ViT-B-32-quickgelu` |
| `'LoRALinear' object has no attribute 'weight'` | `LoRALinear` 的 `weight` 属性丢了，见 §6.2 |
| `noise stats` 四个数加起来 ≠ 训练集大小 | 噪声统计口径坏了，见 §6.3 |
| `noisy` 连续多轮等于同一个整数 | 撞上 `max_noise_frac` 上限，判据对该模型不成立，调 `--tau-conf` |
| warm-up 之后训练 loss 是负数 | **正常**，NCE 的下界约 −2.39，见 §3.5 |
| `val_loss` 不降反升、`val_acc_hi` 一直掉 | 过度自信 / 在吃噪声，检查鲁棒损失和噪声权重 |
| GPU 利用率长期 < 60% | 数据加载瓶颈，`--workers` 加到 12~16 |
| `tmux: command not found` | 用 `nohup` |
| 多行命令粘贴后没反应 | AutoDL 网页终端吞多行，一行一行粘 |
| 提交后 CSV 行数 ≠ 24967 | 有图读不出来被跳过（老 bug）或测试集解压不完整 |
| 平台回传的 CSV 里没有分数 | 那只是文件回显，分数在排行榜 |
| 排行榜打不开 | 必须用**浏览器**，`reg.aicomp.cn` 工具抓不到 |

---

## 13. 交接清单

**接手时请确认：**

- [ ] AutoDL 实例是否还在（欠费会释放，数据和权重都会没）——如果没了，按 §4 重建
- [ ] `/root/autodl-tmp/` 下是否还有 `train/`（93102+10116 张图）和 `test/`（24967 张图）
- [ ] `~/.cache/huggingface/` 里 CLIP 权重是否还在（约 605 MB）
- [ ] `./outputs/best.pt` 是否还在（68.02 的基线，**别删**）
- [ ] 提交次数配额是多少（平台页面没写，**目前未知**）
- [ ] 本地 `HANDOFF.md` / 代码是否已同步到云端（**云端现在是旧版本**，待传 6 个文件：`train.py` `infer.py` `noise.py` `losses.py` `selftest.py` `README_AUTODL.md`）

**第一件事：跑 §9 的流程。**

---

## 14. 复赛阶段（750 类）—— 2026-09-21 更新

### 14.1 实际数据集：**和赛题文档写的完全不同**

赛题文档 `README.md` 写的是"复赛 1500 类 / 297282 训练样本 / 74896 测试样本 / 长尾分布"。
**实际拿到的复赛数据不是这样**，以实际数据为准。`datastats.py` 的输出：

| 指标 | 初赛 | **复赛（实际）** |
| --- | --- | --- |
| 类别数 | 500 | **750**（`0000`~`0749`） |
| 训练图片 | 103218 | **148695** |
| 每类均值 / 中位数 | ~206 | **198.3 / 196** |
| 最少 / 最多 | — | **5 / 248**（49.6×） |
| p10 / p90 | — | **159 / 239** |
| 头部 10% 类占比 | — | 12.3%（均衡值 10%） |
| 尾部 50% 类占比 | — | 42.9%（均衡值 50%） |
| 少于 50 张的类 | — | 仅 **7** 个 |
| **测试图片** | 24967 | **37444**（`test/*.jpg`，扁平结构，无子目录） |
| 测试集每类 | — | **约 50 张**（37444/750 = 49.9，确认测试集是均衡的） |
| **非图片文件** | — | **0 个** |

**结论一：这批数据基本均衡，不是长尾。** 均值 198.3 ≈ 中位数 196，80% 的类落在
159~239 的窄区间。文档里"复赛存在长尾分布"在这版数据上不成立。

> ⇒ **不要投入长尾方法**（逆频采样、logit adjustment、类平衡损失）。
> 现有的 `--sampler balanced`（`1/sqrt(freq)`）是温和修正，保留即可，别加码。

**结论二：没有类名映射文件（非图片文件 0 个）。**

> ⇒ **所有依赖文本提示的方法全部不可用**——而这恰好涵盖了 2025 年这个方向上的主流工作：
> TrustCLIP 的 Semantic Label Verification、NLPrompt 的 PromptOT、DEFT 的正负文本提示、
> Robust-CLIP 的 prompt 预筛选。**它们都需要用真实类名构造文本提示，我们构造不出来。**
> 只能走纯视觉路线。

**结论三：224 分辨率不浪费，提到 288 有真实收益空间。**

`datastats.py --sample` 解析 JPEG 头得到的尺寸分布（抽 300 张）：

| | 训练集 | 测试集 |
| --- | --- | --- |
| 宽 × 高中位 | 620 × 500 | 480 × 500 |
| 短边中位 | 480 | 375 |
| **短边 < 224 的比例** | 11.1% | **1.0%** |

推论：

- 预处理是 `Resize(256) + CenterCrop(224)`，**先把短边缩到 256**，所以两边的绝对尺度差
  在管线里就被抹平了，**不构成 domain gap**。
- 测试集只有 1% 的图短边不足 224 → **224 分辨率完全没有浪费源图细节**，不必担心"分辨率过高"。
- 训练 480 / 测试 375 的短边都远超 288 → **提到 288 微调 + 288 推理是有细节可挖的**，
  §14.5 的 P3 项由此从"猜"变成"有数据支持"。走这条路**必须训练和推理同分辨率**，
  且需要处理 pos-embed 插值（先写冒烟验证 `open_clip` 3.3.0 的行为再改主流程）。

### 14.2 必须在复赛代码里守住的规则

**⚠️ 复赛不能复用初赛的任何权重。** 规则十一(二).2："复赛期间不得使用初赛的数据"。
一个在初赛数据上训出来的 checkpoint 里编着初赛数据的信息，拿它热启动即违规。

- ❌ 不能用 `outputs2/best.pt` 做初始化或 `--resume`
- ❌ 不能用初赛的原型、噪声跟踪器状态
- ✅ 可以复用**算法配方和超参**（那是对方法的验证，不是数据）

**必须从 OpenAI 官方权重重新开始训。**

### 14.3 复赛训练配置

数据集变了，**噪声阈值从来没在新数据上校准过**，所以这些参数要盯：

```bash
cd /root/autodl-tmp
export HF_ENDPOINT=https://hf-mirror.com

nohup python -u train.py --data /root/autodl-tmp/train \
  --out ./outputs_r2 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  > train_r2.log 2>&1 &
```

**注意**：

| 项 | 初赛 | 复赛 | 说明 |
| --- | --- | --- | --- |
| 输出目录 | `./outputs2` | **`./outputs_r2`** | 别混 |
| 每轮步数 | 727 | **1161** | 148695 / 128 |
| 预计每轮耗时 | ~144 s | **~230 s** | ⇒ 20 轮约 **77 分钟** |
| 显存 | ~8.9 GB | 略高 | `tracker.prob` 是 (148695, 750) fp32 = **446 MB**，24 GB 卡无压力 |
| 测试集行数 | 24967 | **37444** | `wc -l pred_results.csv` 必须等于它 |
| 推理耗时 | — | 37444 张，batch 256 | 比初赛多 50% |

> 提交前校验：`wc -l pred_results.csv` → **37444**；`awk -F, '{print $2}' | sort -u | wc -l` → **750**；
> 无表头；每行 2 字段；标签 4 位数字（`0000`~`0749`）。

**代码本身不需要为 750 类改任何东西**：`class_to_idx` 从排序后的文件夹名生成，
`infer.py` 输出 `f'{int(name):04d}'`，`0749` 仍是 4 位 ✓。

### 14.4 ⚠️ 新数据集上要盯的三个信号（train.py 现在会自己报警）

`train.py` 在每轮打印 `noise stats` 之后会**自动检查三种退化情况并打印 `!! 警告`**
（这是上一轮"40% 上限连撞 20 轮、日志一句话不说、白烧一次完整 GPU"的教训）：

1. **四个统计量加起来 != 训练集大小** → 划分逻辑坏了，本次结果不可信。
2. **`capped > 0`** → 有样本撞到 `--max-noise-frac` 上限，判据对这个数据集不成立
   → 调 `--tau-conf` 或 `--max-noise-frac`。
3. **`relabel == 0` 且已过 warm-up 两轮** → **这条在 750 类下最可能触发**：
   类数越多，argmax 置信度越难达到 `--tau-conf=0.8`，伪标签纠正可能完全不发生。
   → **考虑调低 `--tau-conf`（试 0.6 / 0.5）**。

第 3 条是这个数据集带来的新风险：初赛 500 类时 `relabel` 还有 430 个，
750 类下可能直接归零，而**伪标签纠正是整套噪声处理里最有价值的一环**。
如果报警了，用 `--tau-conf 0.6` 重跑对比。

### 14.5 复赛的推进顺序

| 优先级 | 做什么 | 命令 / 成本 |
| --- | --- | --- |
| **P0** | 数据集画像（本地就能跑） | `py -3 datastats.py --zip D:\BaiduDisk\train.zip` |
| **P0** | 上传 `train.py` `infer.py` `noise.py` `losses.py` `selftest.py` `analyze.py` `datastats.py`，跑 `python selftest.py` | 10 s |
| **P0** | 从官方权重起训复赛（`./outputs_r2`） | 77 min |
| **P1** | 训练中盯 §14.4 的三条警告 | 看日志 |
| **P1** | 训完跑 `analyze.py` → 看混淆结构是否单向、有多少疑似错标 | 5 min |
| **P1** | `--save-every` 快照逐个提交，排行榜选轮次 | 每次提交 |
| **P2** | 若 §14.4 第 3 条触发 → `--tau-conf 0.6` 重跑 | 77 min |
| **P2** | `--robust-loss rce` 直臂（CVPR 2025 的 MAE 结论） | 77 min |
| **❌** | 长尾方法 | 数据是均衡的，没收益 |
| **❌** | 文本提示 / zero-shot 相关的一切 | 没有类名，做不了 |
| **❌** | MixUp / CutMix | 文献：细粒度上收益有限甚至为负 |
| **❌** | 生成模型造数据 / 换骨干 | 违规 |

### 14.6 本轮算法改动（2026-09-22，针对复赛）

#### 先纠正一个上一轮的推理错误

上一轮我写过一个判断："初赛有效数据率只有 52%（`43859×1.0 + 2000×0.5 + 37240×0.1 = 48583`），
所以**把被过滤掉的 40% 捞回来是最大的杠杆**"。**这个推论和证据是矛盾的**：

> `val_acc 0.7305` **高于** `test 0.6802` ⇒ 模型在**过拟合** ⇒ 容量有富余。
> 过拟合说明模型不缺数据；把被压到 `w=0.1` 的 40% 捞回来，只会让它记忆得更多。

所以真正的问题是**记忆结构化噪声**（赛题说的"弱相关标注"），优化方向是**抗记忆**，
不是"用更多数据"。据此本轮做了下面这些改动，**全部围绕抗记忆**。

#### 改动清单

| # | 改动 | 文件 | 为什么 |
| --- | --- | --- | --- |
| 1 | **标签平滑** `--label-smooth 0.05` | `train.py` `smooth_target()` | 直接压住"单个（可能是错的）标签能把模型推多自信"。过拟合噪声的前提就是模型对错标签变得自信，平滑把这个上限砍掉。最便宜、风险最低的一条 |
| 2 | **混合伪标签** `--relabel-mix 0.5` | `noise.py` `refresh()` / `target()` | 弱相关标注下，教师"自信地不同意"经常是**两个类难分**而不是标签错。硬覆盖 = 把这个混淆提升为 ground truth，正好教模型犯我们想避免的错。改成 `(1-mix)·y + mix·argmax`，保留一半给定标签的证据 |
| 3 | **损失接受软目标** | `losses.py` `as_dist()` | 上面两条的前提。五个损失全部改成对 `(B,C)` 分布计算，one-hot 时**精确退化**为原公式（`selftest.check_losses` 有断言）。CE 项吃平滑后的目标，鲁棒项吃未平滑的混合目标——理由见 §3.3 的引用块 |
| 4 | **后处理 logit 调整** | `infer.py --logit-adjust` | 赛题明确写"测试集类别分布均衡"而训练集不是。一次前向算出 logits，之后只是重新 argmax，所以 **一次推理能出多个提交文件**，每个 tau 都能上排行榜试。注意：本轮训练集实测近乎均衡（§14.1），**这条预期收益很小**，是廉价的 tie-breaker，不是主力 |
| 5 | **快照只存推理需要的东西** | `train.py` `thin()` | `tracker.prob` 是 `148695×750` fp32 = **446 MB**，加上优化器状态，7 个 `ep*.pt`/`best.pt` 要吃掉 **~3 GB 数据盘**——而推理一个字节都不读它。`best.pt`/`ep*.pt` 现在只存权重（约 5 MB），`last.pt` 保持完整以支持 `--resume` |

#### 怎么跑

```bash
cd /root/autodl-tmp
export HF_ENDPOINT=https://hf-mirror.com

nohup python -u train.py --data /root/autodl-tmp/train \
  --out ./outputs_r2 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  > train_r2.log 2>&1 &
```

新参数**都用默认值**（`--label-smooth 0.05 --relabel-mix 0.5`），不用显式写。
想回到旧行为的对照实验：

```bash
# 旧行为（硬覆盖 + 无平滑），用来 A/B
python -u train.py ... --out ./outputs_r2_old --label-smooth 0 --relabel-mix 1.0
```

推理时把多个 tau 一次跑出来：

```bash
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_r2/best.pt \
  --output pred_results.csv --logit-adjust 0 0.25 0.5
# → pred_results.csv / pred_results_tau025.csv / pred_results_tau050.csv
```

#### 风险与诚实的预期

这三条改动（1/2/3）**从没在真实数据上验证过**。它们的方向有文献和本项目自己的测量支撑
（`val_acc > test_acc` 证明存在记忆），但**幅度不可预测**：

- 悲观：数据本身的噪声率就是很高，模型已经在容量上限附近，平滑只是让它变保守 → **掉 1~2 分**
- 中性：小幅缓解记忆 → **+0.5~1.5 分**
- 乐观：记忆是主要瓶颈 → **+2~3 分**

所以**务必保留 `--save-every` 的轮次快照，并且把 `outputs_r2` 和 `outputs_r2_old` 的
同一轮次都提交一次做对照**，而不是只看 `val_acc` ——§7 已经证明 `val_acc` 在这个数据集上
是会骗人的。

---

## 15. 复赛实测结果 —— 2026-09-23 更新

> **一句话**：两条训练线都跑完了，第一次提交得 **64.16**。本会话最重要的产出**不是分数**，
> 而是**证伪了我自己提出的一个假设**（§15.4）——它本来会让接下来两周的方法论走错方向。

### 15.1 两条训练线，都训完了

| | **分支1** `outputs_r2`（现默认） | **分支2** `outputs_old`（旧行为） |
| --- | --- | --- |
| 参数 | `--label-smooth 0.05 --relabel-mix 0.5` | `--label-smooth 0 --relabel-mix 1.0` |
| 日志 | `train_r2.log` | `train_old.log` |
| 每轮耗时 | 210 ~ 220 s | 210 ~ 216 s |
| 20 轮总时长 | ~73 min | ~73 min |
| best `val_acc` | **0.7070** | **0.7128** |

其余参数两边完全一致：`--epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12
--val-ratio 0.1 --save-every 4`。（`--data` 一条写绝对路径一条写相对路径，无影响。
注意 `--workers` 曾经一条 12 一条 8，那会污染对照——见 §15.8 第 6.11 条。）

**完整 `val_acc` 曲线**

| epoch | 分支1 | 分支2 | Δ |
| --- | --- | --- | --- |
| 1 | 0.5847 | 0.5849 | −0.0002 |
| 2 | 0.6387 | 0.6387 | 0 |
| 3 | 0.6542 | 0.6547 | −0.0005 |
| 4 | 0.6568 | 0.6571 | −0.0003 |
| 5 | 0.6626 | 0.6637 | −0.0011 |
| 6 | 0.6694 | 0.6721 | −0.0027 |
| 7 | 0.6734 | 0.6771 | −0.0037 |
| 8 | 0.6784 | 0.6815 | −0.0031 |
| 9 | 0.6860 | 0.6871 | −0.0011 |
| 10 | 0.6891 | 0.6924 | −0.0033 |
| 11 | 0.6934 | 0.6962 | −0.0028 |
| 12 | 0.6970 | 0.6997 | −0.0027 |
| 13 | 0.6968 | 0.7028 | −0.0060 |
| 14 | 0.6998 | 0.7019 | −0.0021 |
| 15 | 0.7047 | 0.7081 | −0.0034 |
| 16 | 0.7043 | 0.7102 | −0.0059 |
| 17 | 0.7048 | 0.7095 | −0.0047 |
| 18 | 0.7067 | 0.7120 | −0.0053 |
| 19 | 0.7069 | 0.7127 | −0.0058 |
| 20 | **0.7070** | **0.7128** | −0.0058 |

**结论**：分支2 从第 5 轮起**每一轮都更高**，结尾 +0.58 点。

> ⚠️ 但**先别读成"旧行为更好"**：§15.4 已经证明 `val_acc` 高估 6.54 点，
> 而 `--label-smooth 0` 恰恰是**更容易硬记标签**的那一套。分支2 的 `val_acc` 更高，
> **完全可能是它记噪声记得更多**。这一条只能靠排行榜裁决。

### 15.2 ⚠️ 这个对照实验其实只测了一件事

`noise stats`：

```
分支2 ep20: clean 103078 (76.8%)  relabel 2275 (1.70%)  noisy 28812 (21.5%)  unseen 0
            mean_trust 0.6312  mean_weight 0.7982
分支1 ep4 : clean  79269 (59.1%)  relabel  172 (0.13%)  noisy 47905 (35.7%)  unseen 6819
            mean_trust 0.2533  mean_weight 0.6780      ← 注意是 ep4，不是同期，只能粗看
```

（两行四个数之和都等于 **134165** = 训练集 × 0.9，口径自洽。分支1 的 ep20 那行没取到，
要补 `grep "noise stats" train_r2.log | tail -3`。）

**关键点**：`relabel` 只占 **1.70%**（2275 / 134165）。

> 也就是说——**即使把 `--relabel-mix` 从 0.5 提到 1.0（硬覆盖），也只有 1.7% 的样本
> 真的被改标注。两个分支实际上主要在比 `--label-smooth` 0.05 vs 0。**

推论：

1. 伪标签纠正在两边都**近乎空转**，它现在不是变量。
2. §14.4 第 3 条警告（`relabel == 0`）**没有触发，但离触发很近**。`--tau-conf 0.8`
   在 750 类下只放行了 1.7% —— **调低 `--tau-conf` 的优先级应该从 P2 提到 P1**，
   因为这套机制现在等于没开。
3. 想测"抗记忆三件套"的整体效果，这个 A/B **并不干净**——它只测了其中一件。

### 15.3 分支2 已经收敛 ⇒ 20 轮是够的

分支2 的训练末尾：

```
epoch 16/20 loss=0.1922 val_loss=2.2697 val_acc=0.7102 val_acc_hi=0.8379 lr=1.91e-05 time=211.7s
epoch 17/20 loss=0.1747 val_loss=2.2683 val_acc=0.7095 val_acc_hi=0.8399 lr=1.09e-05 time=210.9s
epoch 18/20 loss=0.1587 val_loss=2.2633 val_acc=0.7120 val_acc_hi=0.8375 lr=4.89e-06 time=209.9s
epoch 19/20 loss=0.1483 val_loss=2.2637 val_acc=0.7127 val_acc_hi=0.8394 lr=1.23e-06 time=211.5s
epoch 20/20 loss=0.1398 val_loss=2.2649 val_acc=0.7128 val_acc_hi=0.8389 lr=0.00e+00 time=210.6s
```

`val_loss` 在第 18 轮触底后**微升**，`lr` 同时归零 ⇒ **已收敛**。

> ⇒ **§14.5 里"跑 35 轮"的计划作废**（至少对这组配置）。
> **20 轮 ≈ 73 分钟 = 一次完整实验的单位成本**，按这个做预算。

（分支1 的末尾在更早的会话里看到 `lr=0.00e+00` 但 `val_loss` 仍在降，说明它**没有**
完全收敛。两分支 `val_loss` 不可比（见 §15.7 第 2 条），但**分支内部**可比。
建议补跑 `grep "^epoch" train_r2.log | tail -5` 确认。）

### 15.4 ⭐ 本会话最重要的产出：证伪 H2

#### 起因

第一次提交 **64.16**，而那一轮的 `val_acc` 是 **0.7070** ⇒ **缺口 6.54 点**。
初赛那次是 `val_acc 0.7305` vs 排行榜 `0.6802` ⇒ **缺口 5.03 点**。
§7 把这种缺口解释成"记忆结构化噪声"。

我提出了一个**竞争假设 H2**：`val_acc` 可能只是**度量口径不对**。依据是切分代码
（`train.py:322-324`）：

```python
for i in range(len(classes)):
    n = max(1, int(len(by_cls[i]) * val_ratio))
    self.items.extend(by_cls[i][:n] if val else by_cls[i][n:])
```

**验证集是按类分层抽的**，原样保留训练集的类分布；而赛题明确写测试集**类别均衡**。所以：

- `val_acc` = 按**出现频率**加权的准确率（**micro**）
- 排行榜 = 按**类别等权**的准确率（**macro**）

如果 H2 成立，那就是**模型没问题、我们一直在看错的数**——而且 `macro` 会立刻成为
**免费的本地代理**，后面两周所有消融实验都不必再烧提交名额。

#### 做法

新增 `valmetrics.py`（本会话新文件，已在本地仓库）。它**只读已有的 checkpoint**，
零训练成本，把 micro 和 macro 并排打出来。**脚本内置自检**：`val_acc` 必须等于
`train.py` 日志里那个值，否则一切免谈。

```bash
python valmetrics.py --checkpoint outputs_r2/ep4.pt outputs_r2/ep8.pt \
    outputs_r2/ep12.pt outputs_r2/ep16.pt outputs_r2/ep20.pt
python valmetrics.py --checkpoint outputs_old/ep20.pt
```

#### 结果（val 集 14530 张）

| checkpoint | checkpoint 内部 `epoch` | `val_acc`（micro） | **`macro`** |
| --- | --- | --- | --- |
| `outputs_r2/ep4.pt` | 3 | 0.6568 | 0.6495 |
| `outputs_r2/ep8.pt` | 7 | 0.6784 | 0.6712 |
| `outputs_r2/ep12.pt` | 11 | 0.6970 | 0.6898 |
| `outputs_r2/ep16.pt` | 15 | 0.7043 | 0.6973 |
| `outputs_r2/ep20.pt` | 19 | **0.7070** | **0.7000** |
| `outputs_old/ep20.pt` | 19 | 0.7128 | 0.7052 |

**自检六个全部精确复现**（0.6568 / 0.6784 / 0.6970 / 0.7043 / 0.7070 / 0.7128）——
说明切分和前向都复现了，`macro` 列可信。

（注意 `epN.pt` 的内部 `epoch` 是 `N-1`，因为保存时 `ep` 从 0 开始。`ep20.pt` ↔ 日志里
`epoch 20/20`，是同一个模型。）

#### 结论：**H2 死了**

**micro 与 macro 只差 0.7 点（0.7070 vs 0.7000），而缺口是 6.54 点。**

> **H2 只解释了 6.54 点里的 0.7 点（约 11%）。它不能作为主要解释。**

连带作废的是那个"大奖"：**`macro` 不能当本地代理**。它照样是在**带噪、同分布**的
验证集上算的，只是函数形式对了一点。

> ⇒ **从今以后，任何和轮次 / 配置有关的问题，都只能用排行榜回答。**
> 这一条直接决定了 §15.6 的名额经济学是当前的**首要约束**。

#### 剩下的 5.84 点只能来自

- **H1 记忆**：验证集与训练集共享噪声，学好噪声在 val 上得分、在测试集上丢分。
- **H3 域偏移**：验证集抽自训练分布，测试集是另一批采集。

**H1 和 H3 指向相反的动作，且只能用排行榜分开。**

### 15.5 ⭐ 第二条线索：有若干"卡死"的类

`valmetrics.py` 顺带打出每轮最差的 5 个类。`n` = 该类在 val 里的张数；val 抽 10%，
所以 **`n=24` 意味着原始类有约 248 张 —— 是训练集里最大的那一档**。所有条目的召回率**都是 0.00**：

| checkpoint | 最差的 5 个类（类号 / val 张数） |
| --- | --- |
| `ep4` | 0413(2) 0550(16) **0730(23)** **0728(21)** 0535(15) |
| `ep8` | 0413(2) 0039(18) **0730(23)** 0535(15) 0550(16) |
| `ep12` | 0039(18) 0671(4) **0730(23)** **0728(21)** 0589(4) |
| `ep16` | 0179(22) 0039(18) 0270(16) 0589(4) 0046(1) |
| `ep20` | 0270(16) **0364(24)** **0728(21)** 0723(10) 0046(1) |
| `outputs_old/ep20` | **0364(24)** 0179(22) 0305(3) 0324(12) 0270(16) |

**三个特征，每一个都反常**：

1. **召回率恒为 0.00** —— 这些类**从来没有被预测对过一次**，连偶然命中都没有。
2. **不是尾部类** —— `n` 从 1 到 24，`0364` 的 `n=24` 是**最大的一档**。
   不是"训练样本太少"能解释的。
3. **跨轮次持续** —— `0730` 在 ep4/ep8/ep12 连续三档垫底；`0728` 在 ep4/ep12/ep20；
   `0039` 在 ep8/ep12/ep16；`0270` 在 ep16/ep20（以及分支2 ep20）。
   **从第 4 轮到第 20 轮，同一批类一直坏。**

如果这样的类有十几二十个，它们在 macro 里就是 **2~3 个百分点**——和整个缺口同一个量级。

**待诊断**：新增 `perclass.py`（本会话新文件，已在本地仓库）会打出召回率分布直方图，
以及每个坏类**被误判成了什么**。

```bash
python perclass.py --checkpoint outputs_r2/ep20.pt --top 20
```

判读：

| 观察 | 结论 | 对策 |
| --- | --- | --- |
| 坏类**总是被预测成同一个类** | 两类图混在一起（合并），不是标签噪声 | **训练救不回来**，得从数据/推理层面想 |
| 坏类的预测**分散** | 标签噪声集中在这些类 | `noise.py` 那套有办法，调 `--tau-conf` |

> 注意：**"卡死"和"记忆"是两个独立的假设**，不要混。H1 解释的是"整体高估 5.84 点"，
> 这一条解释的是"某些类永远为 0"。它们可以同时成立。

### 15.6 名额经济学（本轮新增的硬约束）

**平台每天只能提交 2 次；排行榜分数取历史最高分，不是每次分数。**
**提交截止约在 2026-10-07（两周后）。**

推论：

- 一次"差"的提交只损失**一个名额**，**不损失分数** ⇒ 可以放心做探测性提交。
- 但 14 天 × 2 = **28 次**要覆盖所有消融 ⇒ **必须挑信息量最大的用**。
- 结合 §15.4（`macro` 不能当本地代理），**这是目前最紧的约束，比 GPU 时间紧得多**。
  GPU 一次实验 73 分钟，随便跑；名额一次就没了。

#### 第一次提交的定案

第一次提交后**始终无法确定交的是哪一份**，直到补跑 `md5sum`：

```
7678f2e4492fd29d682391bf1f54e0f8  sub_ep20.csv
7678f2e4492fd29d682391bf1f54e0f8  pred_results.csv     ← 提交的就是这一份
```

⇒ **64.16 对应 `outputs_r2/ep20.pt`，`val_acc 0.7070`。**

全部快照的 md5（后续对比用）：

| 文件 | md5 |
| --- | --- |
| `sub_ep4.csv` | `14d73462833637e087effe10f49cc274` |
| `sub_ep8.csv` | `78fbe9bf63c22841fe0286c75b3263cb` |
| `sub_ep12.csv` | `041a3b8beb0dfef26f37ed1a91667949` |
| `sub_ep16.csv` | `1a1c4a058a8a021fa08c5b93511ab6e1` |
| `sub_ep20.csv` | `7678f2e4492fd29d682391bf1f54e0f8` |
| `sub_ep20_tau025.csv` | `eb41a1362d51c5363876e2b380d6c8c4` |
| `sub_ep20_tau050.csv` | `61a1a32e26565ce2ebb5d77a5d87b090` |

> **教训**：**每次打包前都跑 `md5sum sub_epXX.csv pred_results.csv`，两个值必须相等。**

#### 下一次提交该交什么（决策 + 理由）

**推荐交 `sub_ep4.csv`，而不是 `sub_old_ep20.csv`**（后者本来是想先交的，改了主意）：

| 候选 | `val_acc` | 若 H1（记忆）成立 | 若 H3（域偏移）主导 |
| --- | --- | --- | --- |
| `sub_ep4.csv` | 0.6568 | **高于 64.16** | ≈ 59.6（缺口恒定） |
| `sub_old_ep20.csv` | 0.7128 | 略低于 64.16 | ≈ 64.7 |

`ep4` 的摆动幅度约 **5 点**，`old_ep20` 只有约 **0.6 点**（接近排行榜噪声）。
**先打信号强的那个**——它同时回答"记忆是否随轮次增长"和"该交哪个轮次"。

**结果的分叉**：

- `sub_ep4` **高于** 64.16 ⇒ **H1 成立**，后面 16 轮主要在记噪声 ⇒ 转向早轮次 + 更强正则
- `sub_ep4` **低于** 64.16 ⇒ 训练是真的在涨 ⇒ 继续训练路线，H3 权重上升

### 15.7 修正我自己之前说错的两处

#### (1) `log range 4.03` ≠ "56 倍类失衡"

我在复盘时说过："`infer.py` 打出 `log range 4.03` → 最大类 / 最小类 ≈ 56 倍，
所以 §14.1 的『近乎均衡』是错的"。**这个判断是错的。**

`log range 4.03` 完全可以算出来：训练 split 每类的张数是全量 × 0.9（val 按 `int(n*0.1)` 抽），
所以最大 `248 × 0.9 ≈ 224`、最小 `5 × 0.9 ≈ 4`（被 `max(1, ...)` 保底），
`ln(224 / 4) = 4.02` ✓。

**它完全由那 7 个少于 50 张的极端尾部类造成，不代表分布长尾。**
§14.1 的 `p10/p90 = 159/239`（80% 的类挤在窄区间）才是主体事实。

> ⇒ **§14.1 的判断和 `README_AUTODL.md` Step 7「预期收益很小」是对的，不要改。**
> `--logit-adjust` 仍然是廉价的 tie-breaker，不是主力。

（另一处交叉验证：val 每类张数 = `int(n × 0.1)`，范围 1 ~ 24、ratio 24× ——
`valmetrics.py` 实测打出的正是 `min 1 / max 24 (ratio 24x)` ✓）

#### (2) `val_loss` 不能跨分支比

两个分支 `--label-smooth` 不同 ⇒ **损失函数定义不同** ⇒ `val_loss` 是**两把不同的尺子**。

> **跨分支唯一可比的是 `val_acc`。** 分支**内部**跨轮次比 `val_loss` 仍然有效
> （§15.3 就是这么用的）。

### 15.8 本会话踩的坑（接 §6）

| # | 坑 | 表现 | 教训 |
| --- | --- | --- | --- |
| 6.8 | **`valmetrics.py` 把 3 元组写成 4** | `ValueError: not enough values to unpack (expected 4, got 3)` | `ImageFolderNoisy.__getitem__` 返回 **3** 元组（`train.py:336-338`）；返回 **4** 元组的是两视图包装类 `TwoView`（`train.py:350-353`）。`evaluate()` 用的是 `for x, y, _ in loader:`（`train.py:405`）。**读代码时先确认是哪个类** |
| 6.9 | **`adjusted_path()` 返回类型随参数变** | 自检断言失败 | `tau==0` 返回 `base`（`str`），否则返回 `Path`；而 `Path('x.csv') == 'x.csv'` 是 **False**。生产代码从没出错（`open()` 两者都收），**只有自检抓到了** |
| 6.10 | **`!!` 在双引号里被 history expansion 展开** | `grep "!!" train.log` 实际执行成了上一条命令 | 用**单引号**：`grep '!!' train.log` |
| 6.11 | **`--workers` 变量污染** | 两条对照实验一条 `--workers 12` 一条默认 8 | `seed_worker` 用**每个 worker 自己的 `torch.initial_seed()`** 播种，worker 数不同 ⇒ **增强随机流不同**，消融实验混入变量。**对照实验必须逐项核 config，别用"省参数的短命令"** |
| 6.12 | **heredoc 粘贴会丢空行** | 本地 136 换行 / 服务器 135，md5 对不上 | `syntax OK` 已排除代码行损坏（两行合并必然是语法错误），**丢失的只能是空行**，而空行在 Python 里是惰性的。**别靠 md5 判断对错，靠脚本自带的自检**（`val_acc` 必须等于日志值） |
| 6.13 | **从错误目录启动训练** | 在 `~` 下 `nohup python -u train.py ...`，而 `train.py` 在 `/root/autodl-tmp/` | 秒死。**启动前先 `cd /root/autodl-tmp; pwd`**，然后 `sleep 20; tail -5 <log>` 确认看到 `first batch ok` 和 `classes=750` |
| 6.14 | **记错输出目录名** | 让人跑 `outputs/ep4.pt`，实际是 **`outputs_r2`** | 复赛目录是 `outputs_r2`（日志 `train_r2.log`），旧行为对照是 `outputs_old`。**先 `ls -d outputs*` 确认** |
| 6.15 | **提交文件名** | `sub_ep20.zip` 不符合要求 | 规则十三：**必须命名为 `pred_results.csv`**，再压成一个 zip。流程见 §15.6 |
| 6.16 | **长命令更容易被粘贴撕开** | `mkdir -p ... && screen -S move` 变成 `mkdir: invalid option -- 'S'`；`cd/root/autodl-temp` 少空格 | **一律用短命令**，充分利用默认值；每条命令前先 `cd /root/autodl-tmp` |

### 15.9 复赛推进顺序（**修正版，替换 §14.5**）

| 优先级 | 做什么 | 成本 |
| --- | --- | --- |
| **P0** | 提交 `sub_ep4.csv` → 分开 H1 / H3（§15.6） | 1 次名额 |
| **P0** | `python perclass.py --checkpoint outputs_r2/ep20.pt --top 20` → 查"卡死的类"是合并还是噪声 | 2 min |
| **P0** | 后台跑 `--save-every 1` 的复现训练（`outputs_dense`）→ 拿到 ep1~ep20 **完整阶梯**，好在知道拐点后立刻跟进 | 73 min |
| **P1** | 提交 `sub_old_ep20.csv` → 若 H1 成立应**低于** 64.16 | 1 次名额 |
| **P1** | **调低 `--tau-conf`**（0.8 → 0.6 / 0.5）——目前 `relabel` 只占 1.7%，这套机制等于没开（§15.2） | 73 min |
| **P2** | 提交 `sub_ep20_tau050.csv` | 1 次名额 |
| **P2** | 消融阶梯：`--robust-loss gce` / `--proto-weight 0` / `--anchor-weight 0` / `--lora-target mlp` | 各 73 min |
| **P2** | 补 `grep "noise stats" train_r2.log \| tail -3`（同步分支1 的噪声统计） | 秒 |
| **❌** | 长尾方法（逆频采样、logit adjustment 当主力） | §14.1 + §15.7：主体是均衡的 |
| **❌** | 35 轮长跑 | §15.3：已收敛 |
| **❌** | 靠 `macro` 在本地排序配置 | §15.4：macro 也在带噪 val 上算，不是代理 |
| **❌** | 调 `--relabel-mix` | §15.2：它只影响 1.7% 的样本 |

### 15.10 诚实的现状

- **分数**：**64.16**（复赛第一次提交）。
  **但不知道复赛的公开基线是多少**——规则九.4 说低于基线判无效，
  **需要去平台确认 64.16 不是无效分**。
- **归因**：6.54 点的缺口里，**度量口径只值 0.7 点**；剩下的 **5.84 点尚未归因**。
- **主线假设**（记忆结构化噪声）**既没被证实也没被证伪**——`macro` 做不到这件事，
  只能靠排行榜。
- **最值得跟的新方向**是 §15.5 那批"卡死的类"：**跨 16 个轮次、恒零召回、还是大类**，
  三个特征都不像训练不足。
- **下一个决定性实验是 `sub_ep4.csv`**：它同时回答"记忆是否随轮次增长"和"该交哪个轮次"。
- **不要浪费名额**。§15.4 已经把"用本地指标代替排行榜"这条路堵死了，28 次名额
  是接下来两周最稀缺的资源。

---

## 16. 本轮结果 —— 2026-09-24

> **一句话**：这一轮做了上一轮计划的那个决定性实验，**答案出来了** —— H1 成立、H3 出局，
> **训练轮次这条路正式封死**；同时找到并验证了**第一个真正有效的提分手段（TTA，+2.09 点）**，
> 并把上一轮"最有希望"的死类方向**用数据否定掉**。
>
> 当前最好成绩从 **64.16 → 66.2456**。

### 16.1 ⭐ 决定性实验：`sub_ep4` vs `sub_ep20`

上一轮 §15.6 设计的实验，用来分开 H1（记忆）和 H3（域偏移）。**结果是干净的、单侧的。**

| 提交 | `val_acc` | 平台分 |
| --- | --- | --- |
| `sub_ep4.csv` | 0.6568 | **64.1598** |
| `sub_ep20.csv` | 0.7070 | **64.16** |
| **Δ** | **+5.02** | **+0.0002** |

**16 个 epoch 让 `val_acc` 涨了 5 个点，让真实分数涨了 0.0002（≈ 0.07 张图）。**

#### 为什么这同时判了 H3 死刑

看**缺口**（`val_acc` − 平台分）随轮次的变化：

| 轮次 | `val_acc` | 平台分 | **缺口** |
| --- | --- | --- | --- |
| ep4 | 0.6568 | 64.1598 | **1.52 点** |
| ep20 | 0.7070 | 64.16 | **6.54 点** |

**H3（域偏移）要求缺口恒定。缺口从 1.52 涨到 6.54，是被训练自己撑大的** ⇒ 不是域偏移。

⇒ **H1（记忆结构化噪声）成立。** 那 5 个点的 `val_acc` 全部来自记住训练集里的结构化噪声
（同分布切分的验证集也带这套噪声，所以在 val 上"答对"），而**干净测试集上的真实能力，
从第 4 轮起就没有再涨过**。

#### 这条实验关闭了什么

| 关闭 | 原因 |
| --- | --- |
| ❌ 纠结交哪个 epoch | **ep4 == ep20**，从第 4 轮起是一条直线 |
| ❌ 训练更久 | 后 16 轮 = 5 点虚高 + 0 点真收益 |
| ❌ 靠 `val_acc` 选模型 | 硬证据：它涨 5 点，真分不动 |
| ❌ 调低 `--tau-conf`（上一轮 §15.9 的 P1） | 训练轮次整体封死，改噪声处理也只是换一种记法 |

> **结论**：**64.16 不是"还没训够"，而是这套配方在当前设置下的天花板。**

### 16.2 ⭐ TTA：第一个真正有效的提分手段（+2.09）

`infer.py --tta plain flip wide tight`（4 视图 logits 平均，单模型单流程）：

| 提交 | 分数 | Δ |
| --- | --- | --- |
| `sub_ep20.csv`（无 TTA） | 64.16 | — |
| **`sub_tta4.csv`（4 视图）** | **66.2456** | **+2.09** |

缺口同时从 **6.54 → 4.45 点**。

#### 一个待验证的假设：可能是「取景框」在起作用

4 个视图是 `plain flip wide tight`，其中 **`wide` = `Resize(224) + CenterCrop(224)`**。

⚠️ **这是本轮的一个意外发现**：`open_clip` 官方给出的预处理就是 `Resize(224) + CenterCrop(224)`
（短边缩到 224，裁剪覆盖**整幅画面**），而**本项目一直用的是 `Resize(256) + CenterCrop(224)`**
（只裁 **87.5%**，而且被放大）。也就是说，**模型一直看的是一个比 CLIP 预训练时更"放大、更裁边"的视角** ——
而赛题恰恰强调"最大程度保留和利用 CLIP 预训练先验"。

**`sub_wide.csv`（单视图、纯 `wide`）就是这个假设的解耦实验。**

#### ✅ 已定案：取景框假设**被推翻**，收益来自多视图平均

两个 CSV 都提交了，经比对确认：**`sub_wide.csv` = 62**，**`sub_tta4.csv` = 66.2456**。三份数据摆在一起：

| 提交 | 视图 | 分数 | Δ vs 无 TTA |
| --- | --- | --- | --- |
| `sub_ep20.csv` | `plain` 单视图 | 64.16 | — |
| **`sub_wide.csv`** | `wide` 单视图 | **62** | **−2.16** |
| **`sub_tta4.csv`** | `plain flip wide tight` 平均 | **66.2456** | **+2.09** |

**结论一：`wide` 单独用是负收益（−2.16）。** 这符合预期 —— 模型是在 `plain` 取景框上训练的，
单独把它换成 CLIP 原生取景框就是**训练/测试失配**。

> ❌ **"用 `Resize(224)+CenterCrop(224)` 重训"这条路不要做** —— 会掉分。

**结论二（更重要）：平均后的分数比它任何一个组成部分都高。**

```
最好的单个视图（plain） 64.16
最差的单个视图（wide）  62
四个视图的平均          66.2456   ← 比最好的单视图还高 2.09
```

**四个视图的错误高度不相关，平均起来把错误抵消掉了** —— 这是个很强的集成效应，
说明**收益来自多视图平均本身，不是来自"修正取景框"**。

> ⇒ **可操作推论：加更多样的视图，可能继续涨。** 纯推理改动，不用训练。

#### 复现性检查的附带收获

在 `outputs_ema/ep20.pt` 上不写 `--tta` 重跑 `infer.py`，md5 得到 `a91f6c6467f4ec18a9afe6f6cdaee7`，
**与上一轮 §15.6 记录的 `7678f2e4492fd29d682391bf1f54e0f8` 不同** —— 这是**预期的**（本轮用的是
4090 **D**，上一轮是 4090，硬件不同）。**md5 不是判据**，真正的判据是 `val_acc` 曲线：

| | 本轮（4090 D） | 上一轮 §15.1（4090） |
| --- | --- | --- |
| epoch 1 | **0.5849** | 0.5849 |
| epoch 20 | **0.7076** | 0.7070 |

⇒ **复现质量：epoch 1 完全相同，epoch 20 差 9 张图 / 14530。** 训练管线是可信的。

### 16.3 死类方向：**放弃**（上一轮 §15.5 的估计作废）

上一轮把 §15.5 那批"恒零召回的类"列为最有希望的方向，估计值 **~2.7 点**。本轮的 `analyze.py` 输出否定了它：

```
=== 混淆结构 ===
总共 4257 个错误，分布在 3598 个有序类对 / 3394 个无序类对
最集中的 20 个有序对贡献了 123 个错误 = 2.9%
```

**4257 个错误撒在 3394 个无序类对里，最集中的 20 对只占 2.9%。** 如果有"少数几个类被系统性搞混"
这种问题，top-20 会占 30%+。**2.9% 说明错误是普遍的细粒度困难，没有可利用的集中结构。**

死类本身确实存在（`n≥10` 且准确率 ≈ 0 的类约 **14 个**，都是大类），但它们的成因是：

```
0179 -> 0247   A->B=13  B->A=0   asym=+1.00   <== 单向
0039 -> 0446   A->B=10  B->A=0   asym=+1.00   <== 单向
0640 -> 0502   A->B=8   B->A=0   asym=+1.00   <== 单向
```

**`asym = +1.00` 且反向为 0**，这正是 `analyze.py` 自己文档里写的判据：

> *"A labelled B happening far more often than B labelled A cannot come from visual similarity
> (that is symmetric); it is the fingerprint of a **systematic annotation error**."*

⇒ **文件夹 `0179` 里的图，真身是 `0247`。这是训练数据里的标注错误，不是模型能力问题。**

**训练救不回来**（没有正确标签做信号），**推理端也救不回来**（没有任何信息能把它们分开）。
**别把资源投进这条线。**

> 另外注意：`p(y_given) < 0.1` 的样本占 **24.25%**，与噪声跟踪器独立统计的 `noisy 23.2%` 吻合 ——
> **模型自己知道那 24% 的标签是错的**，它和验证集标签的分歧里有一部分是**正确的分歧**。

### 16.4 EMA 教师：ep20 上已收敛，无收益

本轮给 `train.py` 加了 `--save-teacher`，第一次拿到教师权重的完整轨迹：

| 轮次 | 教师 | 学生 | 差距 |
| --- | --- | --- | --- |
| ~ep4 | 0.6658 | 0.6544 | **+1.14** |
| ~ep5 | 0.6764 | 0.6614 | **+1.50** |
| ~ep8 | 0.6922 | 0.6869 | +0.53 |
| ~ep12 | 0.7005 | 0.6986 | +0.19 |
| ~ep16 | 0.7067 | 0.7056 | +0.11 |
| **ep20** | **0.7072** | **0.7076** | **−0.04** |

**优势从 +1.5 点衰减到 0。** 这是 EMA 的典型行为：早期学生抖动大，平滑收益明显；后期学生几乎不动，
教师就等于学生（`--ema 0.995` 的记忆只有约 200 步）。

⇒ **`teacher_last.pt` / `teacher_ep20.pt` 基本没有额外价值。**
⇒ 但 **`teacher_ep4~8.pt`** 值得一试 —— 那是「教师优势最大」**且**「还没开始记噪声」的交叉点，
可以配 `+TTA` 花 1 个名额。

### 16.5 本轮代码改动

| # | 文件 | 改动 |
| --- | --- | --- |
| 1 | `train.py` | 新增 `--save-teacher`（默认关）：每轮评估教师 + 存 `teacher_epN.pt` / `teacher_last.pt` |
| 2 | `train.py` | 新增 `--img-size` + `resize_positional_embedding()` + `val_resize()`，变换与灰图兜底参数化 |
| 3 | `infer.py` | 新增 `--tta`（多视图 logits 平均）+ `--img-size`，从 checkpoint 读分辨率 |
| 4 | `infer.py` | `TTA_VIEWS` 存**比例**而非绝对像素，跟着 `img_size` 缩放 |
| 5 | `analyze.py` / `valmetrics.py` | 从 checkpoint 读分辨率 —— **否则 288 训练后分析会用 224 的变换，静默出错** |
| 6 | `selftest.py` | 新增 `check_pos_embed_resize()`、`check_img_size_transforms()`、TTA 的 4 条断言 |
| 7 | `probe_resolution.py` | **新文件**：分辨率探针 |

**所有改动默认关闭** ⇒ 不加新开关时行为与上一轮**逐字节相同**（`selftest` 里的
`legacy_tf` 断言守着这条：`--tta plain` 必须等于改动前的变换）。

> ⚠️ **`--save-teacher` 不改变训练数学**，只多一次验证前向 + 一次保存。所以带不带它，
> 学生权重应当一致 —— 本轮用它做复现验证，同时白拿教师权重。

### 16.6 分辨率 288：可行性已验证，代码就绪

#### 探针结论（`probe_resolution.py`）

```
=== forward at each candidate size ===
  224x224 (grid 7x7): OK  -> output (1, 512)
  256x256 (grid 8x8): FAILED  RuntimeError: a (65) vs b (50)
  288x288 (grid 9x9): FAILED  RuntimeError: a (82) vs b (50)
  320x320 (grid 10x10): FAILED  RuntimeError: a (101) vs b (50)
  352x352 (grid 11x11): FAILED  RuntimeError: a (122) vs b (50)

=== retrying [256, 288, 320, 352] after manual positional interpolation ===
  256: OK  -> cosine(feat@224, feat@256) = +0.9917
  288: OK  -> cosine(feat@224, feat@288) = +0.9820
  320: OK  -> cosine(feat@224, feat@320) = +0.9683
  352: OK  -> cosine(feat@224, feat@352) = +0.9581
```

**三个事实，都是实测的，不要再猜：**

1. **`open_clip` 的 `VisionTransformer` 不做任何自动插值** —— 非 224 尺寸直接抛形状错误
   （`b = 50 = 7×7 + 1`）。好消息是**它报错而不是静默出错**，比"悄悄掉分"安全。
2. **位置编码的属性名是 `positional_embedding`，形状 `(50, 768)`** —— 这是 **open_clip 自己的命名**
   （`open_clip.transformer.VisionTransformer`），**不是** timm 的 `pos_embed`。形状没有 batch 维，CLS 在第 0 位。
3. **`336` 对 ViT-B/32 不合法**（`336 / 32 = 10.5`）。那个"336"来自 **CLIP ViT-L/14@336px**（patch 14）。
   合法值只有 32 的倍数：**224 / 256 / 288 / 320 / 352**。

**手动双三次插值后全部可用，且余弦相似度单调下降**（0.9917 → 0.9820 → 0.9683 → 0.9581）——
尺寸跳得越远误差越大，**这正是健康信号**（写错的话这几个数会随机跳）。**288 的 0.9820 说明位置完全对得上。**

#### 代码状态

`resize_positional_embedding(visual, img_size)` 已实现并自检通过：双三次插值 patch 网格、
**CLS 令牌单独保留**、同步更新 `image_size` / `grid_size` 记账。

> ⚠️ **必须用 `setattr` 替换属性，不能用 `pe.copy_()`** —— `copy_` 是原地拷贝并要求形状相同，
> 它会拒绝这个函数存在的唯一理由（把 50 变成 82）。这个 bug 是 `selftest` 抓出来的，见 §16.7。

#### 正在跑

```bash
nohup python -u train.py --data /root/autodl-tmp/train --out ./outputs_288 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  --img-size 288 --save-teacher > train_288.log 2>&1 &
```

**启动成功的判据（两条，缺一不可）**：

```
img_size=288: positional grid resampled to 9x9        ← 网格插值成功
first batch ok: x1=(128, 3, 288, 288)                 ← 数据真的是 288（这条才是真证明）
```

**实测**：每轮 **362 秒**（224 是 235 秒，**1.54×**）⇒ 20 轮约 **2 小时**。

#### ⭐ 早期信号：每一轮都比 224 高

同一台机器、同一份代码、同一批数据，只有 `--img-size` 不同：

| epoch | 224 `val_acc` | **288 `val_acc`** | Δ |
| --- | --- | --- | --- |
| 1 | 0.5849 | **0.6061** | **+2.12** |
| 2 | 0.6398 | **0.6531** | **+1.33** |
| 3 | 0.6552 | **0.6734** | **+1.82** |
| 4 | 0.6544 | **0.6731** | **+1.87** |

**四轮全部领先，幅度 1.3~2.1 点。** 而且 `val_acc_hi` 也同步更高（ep3：0.9725 vs 0.9696；
ep4：0.9021 vs 0.8966），`loss` 也更低 —— **三个指标同时更好，不是孤立数字。**

#### 为什么这次 `val_acc` 可以信

`val_acc` 会骗人 —— 但那是在 **ep20**，那时缺口被记忆撑到 6.54 点。
**在 ep1~ep4 缺口只有约 1.5 点**（§16.1 实测），记忆还没开始污染指标。

#### 投影（是投影，不是测量）

```
224 @ ep4:  val_acc 0.6568  →  实测测试分 64.1598   ⇒  缺口 1.52 点
288 @ ep4:  val_acc 0.6731  →  按同样缺口推算 ≈ 65.8
```

⇒ **288 光训练端就值约 +1.6 点**；再叠上已验证的 TTA（+2.09），如果叠加不打折就是奔 **~68** 去的。

> ⚠️ **两个假设都还没验证**：① 288 的缺口是否也是 1.5 点；② TTA 在 288 上能否等量叠加。
> **只有提交能回答。**

#### 推理命令

```bash
# 4 视图（裸 --tta 现在 = 已验证的那 4 个视图）
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_288/ep20.pt \
  --output sub_288_tta4.csv --tta

# ep4 版本是便宜的备选 —— ep4 与 ep20 在排行榜上已实测是同一条直线（§16.1）
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_288/ep4.pt \
  --output sub_288_ep4_tta4.csv --tta
```

日志里应打印 **`at 288px`** —— 分辨率是从 checkpoint 的 `args` 自动读回来的，
所以**训练和推理不可能失配**。

### 16.7 本轮踩的坑（接 §15.8）

| # | 坑 | 表现 | 教训 |
| --- | --- | --- | --- |
| 7.1 | **`Tensor.copy_` 不能改形状** | `RuntimeError: a (50) must match b (82)` | 要增长张量只能**替换属性**（`setattr`），不能拷进去。open_clip 每次前向都读 `self.positional_embedding`，替换即生效 |
| 7.2 | **selftest 的桩模型每次都是随机权重** | 两次 `infer.main` 预测不同，断言失败 | `selftest.py:89` 的 `_stub.create_model = lambda ...: _StubCLIP()` 每次新建随机骨干；而**冻结权重的设计就是不进 checkpoint**。⇒ **不能在 selftest 里跨 `infer.main` 调用比 CSV**，要断言就断言**变换**（确定性） |
| 7.3 | **无卡模式 2GB 跑不动探针** | `Killed`（连 traceback 都没有） | 光加载 CLIP 模型就 **1249 MB**。探针需要 GPU 模式 |
| 7.4 | **数据盘 50G 装不下 zip + 解压结果** | 峰值 61G > 50G | **借系统盘（30G 全空）分摊**：`test.zip` 挪过去、train 分两半（`00*~03*` 进数据盘、`04*~07*` 进系统盘），删 zip 后再合回来。见 §16.8 |
| 7.5 | **该主机不支持扩容数据盘** | 弹窗显示「当前主机可扩容容量为0」 | 这是**租用时选的主机**决定的，事后无解。**下次租机器时就把数据盘选够** |
| 7.6 | **按量计费关机后 GPU 被释放** | 开机弹「该主机空闲GPU不足」 | 关机才释放卡、也才能抢卡；但**连续关机 15 天会释放实例**。节奏：保持关机等卡 + 每隔几小时「无卡模式开机」续命 |
| 7.7 | **定时关机差点杀掉训练** | 外出设了定时关机 | 长训练要么别设，要么设到预计时间的 1.5 倍之后 |

### 16.8 环境搭建（无卡模式 + 50G 数据盘的实际做法）

本轮在一台**数据盘 50G 且主机不支持扩容**的实例上从零建起了完整环境。记录过程，下次复用：

```
1. 上传 9 个 .py + requirements.txt + train.zip + test.zip 到 /root/autodl-tmp/
2. pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
3. echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
4. python selftest.py            # 必须 ALL CHECKS PASSED，纯 CPU 十几秒
5. 分半解压（因为 50G 装不下 61G 峰值）：
     mv test.zip /root/                                  # 挪走小的，腾出数据盘
     unzip -q train.zip "train/00*" ... "train/03*" -d /root/autodl-tmp/   # 前半 → 数据盘
     mkdir -p /root/rest && unzip -q train.zip "train/04*" ... "train/07*" -d /root/rest  # 后半 → 系统盘
     rm /root/autodl-tmp/train.zip                       # 腾出 29G
     mv /root/rest/train/* /root/autodl-tmp/train/       # 合回来
     unzip -q /root/test.zip -d /root/autodl-tmp/ && rm /root/test.zip
6. 核对：ls train | wc -l → 750；find train -type f | wc -l → 148695；ls test | wc -l → 37444
```

> `unzip` 是**单线程**的，无卡模式 0.5 核解压 29G 约 30~60 分钟 —— 慢但能忍。
> **`train.zip` 到最后一步才删**，前面任何一步失败都能重来。

### 16.9 推进顺序（替换 §15.9）

| 优先级 | 做什么 | 成本 |
| --- | --- | --- |
| **P0** | **288 训完 → 生成 `288 + TTA` 并提交** | 1 名额 |
| **P0** | **提交 `sub_tta8.csv`**（8 视图）—— 测更多视图能否突破 66.2456 | 1 名额 |
| **P1** | 若 288 有收益 → **`288 + 8 视图`**（两条已验证的线叠加） | 1 名额 |
| **P1** | `288_ep4 + TTA`（零成本备选；ep4 与 ep20 已实测同分） | 1 名额 |
| **P2** | `teacher_ep8.pt + TTA`（教师优势最大 × 还没开始记噪声的交点） | 1 名额 |
| **❌** | 交任何**无 TTA** 的 ep4~ep20 快照 | ep4 == ep20，同一条直线（§16.1） |
| **❌** | **用 `Resize(224)+CenterCrop(224)` 重训** | `wide` 单视图实测 **−2.16（62 分）**，取景框假设已推翻（§16.2） |
| **❌** | 修死类 | §16.3：系统性标注错误，错误又极度分散，不可修 |
| **❌** | 调低 `--tau-conf` | 训练轮次整体封死（§16.1） |
| **❌** | 训练更久 / 更多 epoch | §16.1 |
| **❌** | 用 **ep20 的** `val_acc` 选模型 | §16.1，硬证据（**注意：ep1~ep4 的 `val_acc` 仍可用**，那时缺口只有 1.5 点 —— §16.6 就是靠这个判断 288 的） |

### 16.10 诚实的现状

- **分数**：**66.2456**（`sub_tta4.csv`，4 视图 TTA）。**已确认不是无效分**（远高于 64.16 的历史分）。
- **归因**：`val_acc` 与平台分的缺口 **6.54 → 4.45 点**，TTA 吃掉 2.09。**H1 已证实，H3 已排除。**
- **已封死的路**：训练轮次、死类修复、`val_acc` 选型。
- **唯一活着的方向**：**换输入表征**（288 正在跑；取景框修正待 `sub_wide` 判定）。
- **最紧的约束仍然是提交名额**（全队 2 次/天），不是 GPU。
- **天花板认知**：**64.16 是这套配方的天花板，TTA 把它抬到 66.2456。** 再往上必须换结构性变量。
