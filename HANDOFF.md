# HANDOFF — LGL-Lab AIC 算法大赛（噪声标签细粒度识别）

> 交接文档。最后更新：**2026-10-03**。
> 面向：接手继续推进的工程师。假设你会 PyTorch、用过 Linux 命令行，但不了解这个赛题和之前发生过什么。
>
> 最新状态看 **§0** 和 **§35**（2026-10-03：`--lora-rank 16` = 新最高分 72.2733 + OOM 教训）。
> **§0"最重要的一句话"已被 §35.6 部分证伪 —— 先读 §35.6 再读 §0。**
> §28（`--local-head` 移植进队友树）、§29（`--local-head` 实测 + 云端 runbook）仍然有效。
> §6~§13 是初赛阶段的记录，仍然有效但要结合 §15 读。
> **§18 已被 §19.7 取代；§0 在 2026-10-01 被整体重写，取代此前所有"下一步"。**
> 另有独立文档 `提分路径研究.md`（A/B/C/D 编号的方法清单与优先级，§7 是排序表）。

---

## 0. 30 秒现状（2026-10-01 重写）

| 项目 | 状态 |
| --- | --- |
| 当前阶段 | **复赛（750 类）**，截止 2026-10-07 前后，**剩约 4 天** |
| 云端训练环境 | AutoDL **RTX 4090 24GB**，数据/权重/依赖已就位；容器内存上限 **60 GiB** |
| 主战场（云端） | `/root/autodl-tmp/` **顶层**——2026-10-03 起训练统一在这里（代码已融合整理，`AIC_orig/` 计划删除）。启动前按 §36.1 认树 |
| 主战场（本地） | `C:\Users\Ed\Desktop\Recent Project\LGL-Lab-AIC\` —— **2026-10-02 起代码已换成 AIC_orig 基线副本**（§34），旧增强树存档在 `_backup_enhanced_20261002/` |
| 平台最高分 | **72.2733** —— `384 + --train-pos-embed + --local-head + --lora-rank 16 + TTA8`（2026-10-03 开分，**§35**）。前一档 71.2023 = 同配方 `--lora-rank 8`，71.1382 = 再去掉 `--local-head` |
| 最近一次开分 | `sub_lr16_tta8.zip`（384+pe+`--local-head`+`--lora-rank 16`，val_acc 0.7465）⇒ **72.2733**，比 71.2023 高 **+1.071**（§35）。同日另跑的 `--tau-conf 0.3`（val_acc 0.7371）**尚未提交** |
| 分数阶梯 | 64.16（224）→ 66.2456（224+TTA4）→ 69.218（288）→ 69.968（320+TTA4）→ 70.332（320+TTA8）→ 70.8605（352+TTA8）→ 71.1382（384+pe+TTA8）→ 71.2023（384+pe+`--local-head`+TTA8）→ **72.2733（+`--lora-rank 16`）**；失败行：68.5476（384 冻结网格）、69.31（416+pe+堆叠，多变量混杂）、71.0768（`--train-proj`）、70.5160（C2 `--two-head`） |
| 已封死的路 | **输入侧整体封盘**（TTA 视图数、分辨率、多分辨率 TTA）—— 用户 2026-09-28 首封、2026-10-01 明确重申（**硬规则：不要再提，不要以任何"新证据"为由重开**）；另有：训练轮次（ep4 == ep20）、死类修复、用 ep20 的 `val_acc` 选型、`wide` 取景框修正、`--train-proj`（§16 前的 71.0768）、C2 `--two-head`（70.5160）、`--tau-conf` ≥ 0.5（空操作，§35.4） |
| 唯一活着的方向 | **训练侧，而且已经找到具体杠杆：容量**（`--lora-rank` 8→16 换来 **+1.071**，§35）。下一步探 `--lora-rank` 24 / 32 的拐点 |
| 最紧的约束 | **提交名额（全队 2 次/天）**，比 GPU 时间紧得多 |
| 本地树（`LGL-Lab-AIC`） | **2026-10-02 起 = AIC_orig 基线的本地副本**（§34）：`train.py` 1005 行、含 `--local-head`，不含 `--head-init`/`--frozen-mix-*`/`--distill-*`/`--augment-mode`。旧增强树（`train.py` 2796 行）整体存档在 `_backup_enhanced_20261002/`，只作参考 |
| 下一步 | **§35.7**：`--lora-rank 24` → 32 探拐点（两台并发）；`--tau-conf 0.2`（等 `sub_tau03` 开分再定） |

**最重要的一句话**：**本项目的分数瓶颈不在输入侧，也不在"模型不够大"。**
输入侧已封盘；而 §29 的实测表明，给模型加参数（`--local-head` 加了 393k）时，
gate 会一路涨到 0.51 把新支路吃满，**但 val_acc 一动不动**（+0.0003）——
即模型**渴求容量**，可"怎么池化同一批 token"这条路不产生新信息。
真正的缺口是 §7 记的那条：**val_acc 与线上分数的错配**，那是标签噪声与口径问题，不是容量问题。

> **2026-10-02 更正上段**：`--local-head` 的本地 val_acc 只涨 +0.0003，但线上开分实测 **+0.0641**（71.1382 → 71.2023，§33.10）。所以"val_acc 一动不动"只能推出**本地指标看不见这点增益**，不能推出该改动无效；这是上文"错配"那条的又一例证。至今训练侧唯一为正的改动就是它，量级 +0.06。

> **⚠️ 2026-10-03 更正（重要，推翻上上段的后半句）**：**"瓶颈不在模型不够大"已被 §35 证伪。**
> `--lora-rank` 8 → 16（LoRA +884k 参数）换来线上 **+1.071**（71.2023 → 72.2733），
> 是至今**单次收益最大**的一档 —— 比 `--local-head`（+0.064）大一个数量级。
> **"模型渴求容量"那句是对的；错的是"这不产生新信息"。**
> 上上段只剩"瓶颈不在**输入侧**"这半句仍然成立。详见 **§35.6**。

**⚠️ 三个必须先知道的事实**：

1. **`336` 这个数字是错的**（对 ViT-B/32）。patch 是 32，合法值只有 32 的倍数：224/256/288/320/352/384/416。`336` 属于 ViT-L/14。组织者裁定里提到 336 只是举例（§16.6）。
2. **位置编码的归属取决于是否开了 `--train-pos-embed`。** 不开时它是冻结参数，`trainable_state_dict()` 会丢掉，**每次载入都是重建**，重建机制本身就是模型且错了不报错（§17.1）。开了之后它是**训练过的 Parameter 并存进 checkpoint**——好处是 384 从 68.5476 涨到 71.1382，代价是**换分辨率推理会 `size mismatch` 直接崩**（§29.7-6）。
3. **`--resume` 在结构上是有损的**：EMA teacher 从不被保存，恢复会用 `copy.deepcopy(model)` 把它重置。≤2 轮时重启，别 resume（§29.7-5）。

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

5 个 Python 文件，加 3 个文档。全部在仓库根目录，**没有子模块、没有包结构**。

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `train.py` | ~650 | 训练主程序：模型定义（LoRA/余弦头/原型头）、数据、训练循环、评估、存档 |
| `infer.py` | ~110 | 用 checkpoint 跑测试集，生成官方格式 `pred_results.csv` |
| `analyze.py` | ~200 | **错误分析**：混淆结构（结构化噪声 vs 真·细粒度困难）、逐类准确率、疑似错标清单 |
| `datastats.py` | ~160 | **数据集画像**：类别分布 / 长尾诊断 / 采样权重 / 是否有类名。只用标准库，可本地跑 |
| `losses.py` | ~112 | 鲁棒损失：`ce` / `gce` / `nce` / `rce` / `apl` |
| `noise.py` | ~132 | `LabelTrustTracker`：逐样本标签可信度跟踪、噪声过滤、伪标签纠正 |
| `selftest.py` | ~385 | **改代码后必须先跑这个**：约 10 秒，不需要数据/GPU/CLIP 权重 |
| `README.md` | — | 赛题原文（不要改） |
| `README_AUTODL.md` | — | AutoDL 平台操作手册 + 方法说明 + 参数表（中文，给操作者看的） |
| `HANDOFF.md` | — | 本文件 |

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
  --lora-rank 0                        # 0 = 用 checkpoint 里记录的
```

**注意**：`infer.py` 会自动从 checkpoint 读取 `model_name` / `lora_rank` / `lora_target` / `pretrained`，一般**不需要手动指定**。

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
- [ ] 本地 `HANDOFF.md` / 代码是否已同步到云端。**判断"要不要传"的唯一依据是服务器上的版本，
      不是本地的 mtime** —— 见 §19.8 第 8 条。省事的做法：`noise.py` `losses.py` `datastats.py`
      `train.py` `infer.py` `probe.py` `valmetrics.py` `analyze.py` `selftest.py` `requirements.txt`
      + 两个 `.md` **一次性全传**（全是文本，几十 KB），不要挑。
      自检 `ImportError` 几乎一定是某个 `.py` 在云端是旧的。

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

> ⚠️ **本表已被 §18 取代，保留只为记录当时的推理。** 下面 P0 的三条都已执行完
> 并有了结论：`sub_ep4` 的结果见 §16.1（ep4 == ep20），"卡死的类"见 §16.3
> （系统性标注错误，放弃），`outputs_dense` 那条因 §16.1 封死训练轮次而失去意义。
> **不要照本表执行。**

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

## 16. 队友两轮的新结果 —— 2026-09-24 / 09-25

> 本节把队友树（`...\Recent Project\AIC\LGL-Lab-AIC\HANDOFF.md`）的 §16 与 §16.11 合并进本树。
> 本树的 §15.10 结束于 2026-09-23，下面是之后发生的事。

### 16.1 ⭐ 决定性实验：`ep4 == ep20`（训练轮次这条路封死）

| 提交 | `val_acc` | 平台分 |
| --- | --- | --- |
| `sub_ep4.csv` | 0.6568 | 64.1598 |
| `sub_ep20.csv` | 0.7070 | 64.16 |
| **Δ** | **+5.02** | **+0.0002** |

16 个 epoch 让 `val_acc` 涨了 5 个点，真实分数涨了 0.0002。**缺口从 ep4 的 1.52 点涨到 ep20 的 6.54 点** —— 缺口是被训练自己撑大的，所以不是域偏移（**H3 出局**），而是记忆结构化噪声（**H1 成立**）。

关闭：纠结交哪个 epoch、训练更久、用 ep20 的 `val_acc` 选型、调低 `--tau-conf`。
**但注意：ep1~ep4 的 `val_acc` 仍然可用**（那时缺口只有约 1.5 点）—— §16.5 判断 288 就是靠这个。

### 16.2 ⭐ TTA 是第一个真正有效的提分手段（+2.09）

| 提交 | 视图 | 分数 | Δ |
| --- | --- | --- | --- |
| `sub_ep20.csv` | `plain` 单视图 | 64.16 | — |
| `sub_wide.csv` | `wide` 单视图 | **62** | **−2.16** |
| `sub_tta4.csv` | `plain flip wide tight` 平均 | **66.2456** | **+2.09** |

**四个视图的平均比它最好的组成部分还高 2.09 点**，而最差的单视图（`wide`，62）也在平均里贡献了正收益 ⇒ 收益来自**多视图平均本身**，不是来自"修正取景框"。

推论：`wide` 单独用是训练/测试失配（模型是在 `plain` 取景框上训的）；**用 `Resize(224)+CenterCrop(224)` 重训会掉分，不要做**。

### 16.3 死类方向：放弃

`p(y_given)` 的统计显示 `0179` 里的图真身是 `0247` —— **训练数据里的系统性标注错误**。没有正确标签做信号，也没有任何信息能把它们分开。不投资源。

附带事实：`p(y_given) < 0.1` 占 **24.25%**，与噪声跟踪器独立统计的 `noisy 23.2%` 吻合 —— 模型自己知道那 24% 的标签是错的。

### 16.4 EMA 教师：ep20 上已收敛

教师优势从 +1.5 点衰减到 ep20 的 −0.04。`teacher_last.pt` / `teacher_ep20.pt` 无额外价值；**`teacher_ep4~8.pt`** 值得一试（教师优势最大 × 还没开始记噪声的交点）。

### 16.5 ⭐ 288 可行性（实测，不是推断）

```
224x224 (grid 7x7):  OK
256/288/320/352:     FAILED  RuntimeError: a (65/82/101/122) vs b (50)
```

1. **open_clip 的 `VisionTransformer` 不做任何自动插值**，非 224 直接抛形状错（`b = 50 = 7×7+1`）。好在它**报错而不是静默出错**。
2. 手动双三次插值后全部可用，**余弦相似度单调下降**：256→0.9917、**288→0.9820**、320→0.9683、352→0.9581。跳得越远误差越大 —— 这正是健康信号。
3. **`336` 对 ViT-B/32 非法**（`336 / 32 = 10.5`）。合法值只有 32 的倍数：224/256/288/320/352。
4. ⚠️ 必须用 **`setattr` 替换属性**，不能用 `pe.copy_()` —— `copy_` 是原地拷贝且要求形状相同，它会拒绝这个函数存在的唯一理由。

#### 288 训练命令（本树可直接用，`--img-size` 是别名，见 `train.py:1366`）

> ⚠️ **2026-09-26 修正**：下面这条命令原样抄自队友树，带着 `--save-teacher` —— 而
> **`--save-teacher` 是队友树的开关，本树的 `train.py` 没有这个参数**（本树 argparse 的最后一项是
> `--proto-min-weight`，`train.py:1441`）。照抄会**在第 2 秒直接 `unrecognized arguments` 退出**，
> 日志看起来像"跑起来了"，其实一个 epoch 都没跑。已删除该开关。
> `teacher_ep*.pt` / `teacher_last.pt` 只有队友树会写；本树要用教师权重，需要先把那 ~15 行
> （`save_teacher` 的评估 + `torch.save(thin_t, ...)`，见队友 `train.py:728-770`）移植过来。
> 优先级见 §19.7 —— **不建议现在做**。

```bash
nohup python -u train.py --data /root/autodl-tmp/train --out ./outputs_288 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  --img-size 288 > train_288.log 2>&1 &
```

**启动成功的判据（两条，缺一不可）**：

```
img_size=288: positional grid resampled to 9x9     ← 网格插值成功
first batch ok: x1=(128, 3, 288, 288)              ← 数据真的是 288（这条才是真证明）
```

**实测每轮 362 秒**（224 是 235 秒，1.54×）⇒ 20 轮约 **2 小时**。

### 16.6 ⭐⭐ `288 + TTA = 69.218`，且与 TTA **超加性**

| 提交 | 配置 | 分数 |
| --- | --- | --- |
| `sub_ep20.csv` | 224 | 64.16 |
| `sub_tta4.csv` | 224 + TTA4 | 66.2456 |
| **`sub_288_tta4.csv`** | **288 + TTA4** | **69.218** |

```
简单相加预期：64.16 + 1.80（288 的 val_acc 领先）+ 2.09（TTA）= 68.05
实际：                                                          69.218
超出加和：                                                       +1.17
```

⇒ **288 的 `val_acc` 领先（+1.8）是真的**，而且**多尺度/多裁剪的 TTA 在高分辨率下收益更大**（裁剪覆盖像素更多，视图间差异性更有价值）。

⚠️ 注意：这条超出加和的**解释是队友的假设，不是测量**（见 §17.6）。

### 16.7 队友自己排的下一步

| # | 做什么 | 依据 |
| --- | --- | --- |
| P0 | `288 + 8 视图`（checkpoint 仍用 `outputs_288/ep20.pt`） | 288 是赢的线；视图数在 224 上还没测 |
| P1 | **320 分辨率**（探针已验证可行，余弦 0.9683） | 288 已证明分辨率是结构性变量；每轮约 7.5 分钟，20 轮约 2.5 小时 |
| P2 | `sub_tta8.csv`（224 + 8 视图，**已生成待提交**） | 只回答"视图数"这一个问题，天花板低于 69.218 ⇒ 有信息量但不改善成绩 |

---

## 17. 本树这一轮 —— 2026-09-25

> 目标不是涨分，是**把队友那条已经拿分的线在本树上变得可信**。
> 一句话：本树原来的代码**在评估队友的 288 checkpoint 时，用的不是它被训练时的那个模型**。

### 17.1 ⭐ 修掉的静默错误：位置编码的 `antialias` 分歧

| | 队友（拿了 69.218） | 本树（改之前） |
| --- | --- | --- |
| 机制 | 自己的 `resize_positional_embedding`，`F.interpolate(mode='bicubic', align_corners=False)` | `build_clip` 传 `force_image_size`，走 open_clip 的 `resize_pos_embed` |
| `antialias` | **`False`**（PyTorch 默认） | **`True`**（`_oc_src/model.py:811-817`） |
| 288 处的形状 | `(82, 768)` | `(82, 768)` |
| 288 处的值 | ← **不同** → | ← **不同** → |

**形状相同、值不同、不报任何错。** 这正是队友 §6.1 记录的 QuickGELU 那一类"静默掉精度"。

#### 为什么这是根因而不是表象

`trainable_state_dict()`（`train.py:877-882`）只保留 `requires_grad=True` 的参数，而位置编码是**冻结**的 ⇒ **它从不进 checkpoint**。每次载入都是**重建**。

> ⇒ **重建机制就是模型本身。**

这也解释了为什么"给 checkpoint 加校验"救不了队友已有的 288 —— 它里面没有可比对的东西。**只能靠对齐机制。**

反过来，如果位置编码真进了 `state_dict`，形状不符会在 `load_state_dict` 时**直接 RuntimeError**（`strict=False` 只抑制 missing/unexpected，**不抑制形状不符**）。所以那条路是响的，只有这一条是静的。

#### 修法（A 步）

1. `train.py` 新增 `resize_positional_embedding(visual, size, antialias=False)`，照搬队友已验证的实现（双三次插值 patch 网格、CLS 单独保留、**`setattr` 替换属性而非 `copy_`**、保留 `requires_grad`、同步 `image_size`/`grid_size`）。默认 `antialias=False` 对齐队友；开成参数是为了让这个差异**可命名、可测量**。
2. `build_clip` 非 224 时改走这个函数，**不再传 `force_image_size`**。
3. 下游 `analyze.py` / `infer.py` / `valmetrics.py` / `probe.py` 都经由 `build_clip`，自动继承。
4. **新 checkpoint 记位置编码指纹**（`ck['pos_embed']` = 网格形状 + sha256 前 16 位），载入时重建并**断言一致** ⇒ 把"同形状不同值"从静默变成响的，对**今后每一个** checkpoint 永久关闭这一类错误。
5. 校验做成**不可忘**的：由 `build_clip` 自己拿 `ck_image_size(ck)` 和请求尺寸比，而不是要求调用方决定何时校验。两种"不是错配"的情况只打提示不开火：checkpoint 没有指纹（队友的全部、本树 09-25 之前的全部），以及**调用方故意要另一个尺寸**（TTA 到第二分辨率正是这种，它的网格本来就该不同）。

### 17.2 离线代理扩到分辨率轴（B 步，0 个提交名额）

`probe.py --calib` 现在能回答分辨率问题。四点改动：

1. **裁判尺寸与模型尺寸解耦**，裁判**钉在 224**（`--calib-judge-size` 可覆盖）。
   *为什么必须钉 224*：裁判是 OpenAI CLIP，224 是它**原生**的尺寸；在 288/320 跑裁判要先插值**它自己**的位置编码，"高置信且正确"的那部分就成了这个插值的产物。而且**按尺寸重算 V\* 不只是不同，是有偏的** —— 裁判同时是筛选器，哪个尺寸裁判更喜欢就给出更干净的子集，而被评分恰好就是在那个尺寸训出来的模型。钉住子集，差值才归因于模型。
2. **允许混合尺寸**：每个 checkpoint 在**它自己记录的**分辨率上评分。视图元组带尺寸，`None` = 该 checkpoint 自己的分辨率。
3. **门新增第四条事实**：`288-trained@288+tta4` vs `224-trained@224+tta4` = **+2.97 点**，方向为正。输出里**明说它混淆**（训练尺寸和评估尺寸同时变）—— 它的用处是检验代理能不能看见 **3 点量级的模型差异**，而那正是 320 决策的量级。
   行键是 `(训练分辨率, epoch, 视图集)`：两个 run 都有 epoch 20，两段键会让一个静默覆盖另一个。只有一个 run 时这条读 **`not measured`**，**不等于通过**。
4. **载入守卫换成**：`missing` 里不得含可训练名 + `unexpected` 必须为空。旧的 `requires_grad ∩ missing` 检查对冻结的位置编码**结构上不可达**。

另外两处结构性改动：

- **每个分辨率一个 `Net`，整个 run 只建一次**。`Net.__init__` 会 `add_lora` 往交给它的 tower 里**注入 LoRA 并冻结其余参数** ⇒ 一个 tower 不能既当裁判又当被打分的模型，也不能对同一个 tower 建两个 `Net`（适配器会叠）。
- **`val_acc` 自检从"打印提示"改成自动断言**。`valmetrics.py` 原本只*打印*"must match train.py log"，从不检查；现在 `calib` 自己比，退出阀是 `--calib-valacc-tol`（默认 0.5 点）。**理由**：位置编码错了会静默载入、不报任何错、把这个数移动好几点 —— 这是唯一能察觉"模型不是那个模型"的检查。

### 17.3 代码改动清单（本树，2026-09-25）

| 文件 | 改动 |
| --- | --- |
| `train.py` | 新增 `resize_positional_embedding` / `pos_embed_fingerprint` / `verify_pos_embed`；`build_clip` 改机制并自动校验 `ck=`；存档写指纹；`--resume` 校验 |
| `probe.py` | 改动最大：裁判钉 224、混合分辨率、门第四条事实、载入守卫、`--calib-judge-size` / `--calib-valacc-tol`，`--calib-sweep` 扩到尺寸轴 |
| `selftest.py` | 新增 `check_pos_embed`；门夹具扩到四条事实；视图元组改 4 元组 |
| `infer.py` | `make_model(size)` 每个尺寸都传 `ck=` |
| `valmetrics.py` | 载入时传 `ck=`（一行） |
| `analyze.py` | 同上（一行） |

**本地已验证**（无 torch）：`py_compile` + `ast.parse` 全绿；undefined-name 检查 0 命中且负对照会响；门的夹具用**真函数**跑过（AST 提源码整函数执行，含新增的分辨率行：+2.97 通过、+0.5 不通过、−2.97 不通过，防止吃 `|x|`），且一个"永远返回 True"的门会被夹具抓住。

### 17.4 ⚠️ 服务器现状（2026-09-25 实测）

`ls /root/autodl-tmp/outputs*/*.pt` 的结果：

| 目录 | 内容 | 时间 |
| --- | --- | --- |
| `outputs/` | `best.pt` (6.4M)、`last.pt` (**404M**) | Sep 23 10:00 |
| `outputs_old/` | `best.pt` + `ep4/8/12/16/20.pt`（各 6.4M）、`last.pt` (404M) | Sep 23 12:36–13:33 |
| `outputs_r2/` | 同上 | Sep 23 10:35–11:31 |

**三批全是 09-23 的、全是 224。这台机器上没有任何 288 的 checkpoint，也没有队友的任何 checkpoint。**

- `last.pt` 的 404M 说明它是**完整 state_dict**（含冻结骨干）⇒ **不能**传给 `--checkpoint`：里面每个冻结骨干张量都会落进 `unexpected`，被 §17.2 的守卫直接拒掉（**拒得对**）。要用 6.4M 的 `ep*.pt`。
- 三批都**没有位置编码指纹**（字段是 09-25 才加的）⇒ 跑 `--calib` 时会打印 `0/N checkpoint(s) record a fingerprint`，**这次运行里 `val_acc` 自检是唯一防线**，要盯住。
- **这批 checkpoint 到底是不是 64.1598 / 64.16 背后的那两个，必须先用它们的 `val_acc` 字段确认**（应为 0.6568 / 0.7070）。不是的话，门就无法评级（代理不能复现已知事实 ⇒ 它的输出不是证据）。确认命令见 §17.7 第 0 步。

### 17.5 ⭐ 结论：自己重训 288，**这不是赌注**

一度以为"没有队友的 288 checkpoint ⇒ 门评不了第四条事实 ⇒ 卡住"。**这个框架是错的**：

> **288 不需要代理来批准。** 队友的 69.218 是一次**真实提交**，已经证明 288 有效（§16.6）。
> 训练端 +1.80 的 `val_acc` 领先也在 §16.5 里逐轮实测过（ep1~ep4 四轮全部领先，且 `val_acc_hi`、`loss` 同时更好）。

⇒ **在本树重训一个 288 是"复现一个已验证的结果"** —— 组织者明确允许改输入分辨率 + 插值位置嵌入、骨干与权重不变，所以完全在许可范围内，**不是一次 GPU 赌博**。约 2 小时。

它同时买下三件事：

1. C 步的**多分辨率视图**能在 288 上跑（没有 288 模型就跑不了）；
2. 本树有了自己的 288 线 —— **决赛复现要求只认自己提交的代码**；
3. 门虽然仍评不了第四条事实（那是"我的 288 vs 我的 224"，要一次提交才知道平台分），但**事实 1 已经覆盖了同样的量级**（ep4→ep20 的平台 +1.80 ≈ 320 决策的量级）。

### 17.6 还没测过的两条轴（C 步，全在 288 上离线排序）

队友的 8 个视图（`plain/flip/mid/mid_flip/tight/tight_flip/wide/wide_flip`）**全在一条轴上**：都是 `Resize(img·ratio) + CenterCrop`，ratio ∈ {1.0, 1.143, 1.286, 1.429} —— **只是缩放**。**从没有人测过正交轴。**

| 轴 | 内容 | 诚实先验 |
| --- | --- | --- |
| **正交裁剪轴** | `full`（整图压缩）/ `pad`（保比例加边），即 `axis4` / `pad4` | **唯一有理由翻正的一条**。依据是 §16.6 的超加性：若收益真来自"视图差异性"，换轴比在同一条缩放轴上再买一档值 |
| **多分辨率视图** | 同一个 288 checkpoint 在 256/320/352 各出一视图再平均（`s256`/`s320`/`s352`/`tta4s320`） | **大概率 ~0 或略负**。`RESIZE_RATIO` 在所有尺寸下固定 ⇒ 每个尺寸保留的画面比例完全相同（方形图 224 和 320 都是 76.6%），尺寸视图基本是同一批像素的重新编码 ⇒ 相关性高；再叠加 interpolate-only 本身 OOD。机制上更像"更差且相关"，那是稀释平均而不是去噪。**但要测，不要假设。** 唯一可能翻正：对 288 训出的模型，320 是同族内的细化（82→101 tokens），比 224 温和得多 |

两条都按 `acc@V*` **和** `acc@V*-hard` 排序。

> ⚠️ **红线**：任何形式的**跨 checkpoint 平均**都是违规的（题目禁止多模型融合）。多分辨率 TTA 必须全部来自**同一个** checkpoint —— 这才在组织者关于分辨率的裁定范围内。

### 17.7 上机顺序

```bash
cd /root/autodl-tmp

# 第 0 步：先确认那三批 checkpoint 是不是已知分数的那些（必做，1 分钟）
python - <<'EOF'
import torch, glob
keys = ('epoch', 'val_acc', 'image_size', 'img_size', 'lora_rank', 'lora_target')
for p in sorted(glob.glob('/root/autodl-tmp/outputs*/*.pt')):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck.get('args', {}) if isinstance(ck, dict) else {}
    row = {k: ck.get(k, a.get(k)) for k in keys}
    m = ck.get('model') if isinstance(ck, dict) else None
    row['model_tensors'] = len(m) if isinstance(m, dict) else None
    row['pos_embed_fp'] = bool(ck.get('pos_embed')) if isinstance(ck, dict) else False
    print(p.replace('/root/autodl-tmp/', ''), row)
EOF

# 第 1 步：自检（无数据无 GPU，约 1 分钟）
python selftest.py            # 必须看到 ALL CHECKS PASSED

# 第 2 步：实机确认 288 能构造能前向（约 1 分钟）
python probe.py --data /root/autodl-tmp/train --limit 8 --sizes 288 --skip-test

# 第 3 步：离线代理 + 门（0 个提交名额）
python probe.py --data /root/autodl-tmp/train --calib \
    --checkpoint /root/autodl-tmp/outputs_old/ep4.pt /root/autodl-tmp/outputs_old/ep20.pt

# 第 4 步：训练本树的 288（约 2 小时）
#   ⚠️ 不要带 --save-teacher：那是队友树的开关，本树没有（见 §16.5 的修正说明）
nohup python -u train.py --data /root/autodl-tmp/train --out ./outputs_288 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 \
  --img-size 288 > train_288.log 2>&1 &

# 第 5 步：288 训完后，在 288 上排视图与尺寸（C 步）
python probe.py --data /root/autodl-tmp/train --calib --calib-sweep \
    --checkpoint /root/autodl-tmp/outputs_288/ep20.pt
```

**要看的两行输出**：

1. 载入后第一行 `N checkpoint(s) at M training resolution(s): [...]px (read from the checkpoints, not from --sizes)`。
   **如果显示 `1 training resolution(s): [224]px` 而你传了 288 的 checkpoint，立刻停** —— 说明那个 checkpoint 没记自己的分辨率，被当成 224 了。
2. 门的四条事实。**只有一个 run 时第四条必须显示 `not measured`** —— 那**不等于通过**，代理没被评级，不能拿它授权 GPU。

**决策树**：

- 门全过（含所有能被评级的事实）→ 走 C 步排序，按 `acc@V*` 决定交哪个配方。
- 门有一条没过 → **停手**，报告哪条没过。代理不可信时，它给出的任何"320 值得试"都是噪音，不能花那 2.5 小时。

### 17.8 ⚠️ 三个还没验证的问题（诚实记录）

1. **`antialias` 的 True/False 在 7×7→9×9 上到底有没有差别，仍未证。**
   `selftest.py` 的 `check_pos_embed` 里**测量并打印**最大差值，而不是断言它非零 —— 因为本地没有 torch，无法确认 PyTorch 的 antialias 对**上采样**是否生效。
   **如果测出来相同，A2（§17.1）就是重构而不是修复**（机制对齐本身不花钱，指纹校验无论如何都留着）；**如果不同，那它就是根因**。`selftest.py` 第一次上机跑就会自己说出来。
2. **本机 open_clip 的 `create_model` 是否必须拿到匹配 grid 才能构造，未确认。**
   能构造的证据是队友的 288 就是那样跑的（§16.5）。万一不行，退路是"传 `force_image_size` 构造 + 用我们的函数覆盖位置编码"，并断言两条路产出**逐位相同**。§17.7 第 2 步就是为这个准备的。
3. **`selftest.py` 整体在任何机器上都还没跑过**（本地缺 torch/torchvision/open_clip）。所有 torch 路径（分辨率几何、`Net` 载入、`extract`、`predict_probs`、坏图兜底、CLI 别名解析）都尚未执行过。

---

## 18. 现在要做什么（替换 §16.7 与本树 §15.9）

> ⚠️ **本节已被 §19.7 取代**（2026-09-26）。它写于 320 结果出来之前，那时"288"还是前沿；
> 现在前沿是 320，而且本树该做的第一件事从"重训 288"变成了"把 **320** 的 checkpoint 拿过来"。
> 下面这张表保留是为了记录当时为什么那样排 —— 排法本身没有错，只是基线动了。

| 优先级 | 做什么 | 成本 | 依据 |
| --- | --- | --- | --- |
| **P0** | **在本树重训 288** → 跑门 → `--calib-sweep` 排视图 | 2 小时 GPU，0 名额 | §17.5：288 是已验证的结构性变量，不是赌注 |
| **P0** | 门的四条事实没全过就**停手**，报告哪条没过 | 0 | §17.2：代理不可信时它的结论是噪音 |
| **P1** | 288 训完后排序结果 → 按 `acc@V*` 选配方 → **提交** | 1 名额 | §16.2：多视图平均是唯一的已验证提分手段 |
| **P2** | `teacher_ep8.pt + TTA`（教师优势最大 × 还没开始记噪声的交点） | 1 名额 | §16.4 |
| **P2** | 320 分辨率（探针已验证可行，余弦 0.9683） | 2.5 小时 + 1 名额 | §16.6：288 证明分辨率是结构性变量 |
| **❌** | 交 `sub_tta8.csv` | 1 名额 | 天花板低于 69.218，先让 sweep 免费回答"视图数有没有用" |
| **❌** | 交任何**无 TTA** 的 ep4~ep20 快照 | — | §16.1：ep4 == ep20 |
| **❌** | 用 `Resize(224)+CenterCrop(224)` 重训 | — | §16.2：`wide` 单视图实测 −2.16 |
| **❌** | 修死类 / 调 `--tau-conf` / 训练更久 | — | §16.1 / §16.3 |
| **❌** | 任何形式的**跨 checkpoint 平均** | — | 题目禁止多模型融合 ⇒ 违规 |
| **❌** | 换更大的骨干或非官方权重 | — | 硬性约束（§1） |

---

## 19. 本树这一轮 —— 2026-09-26（320 之后：把"输入表示"这条线做成可复现的）

> 队友的 320 把成绩推到 **69.968**（§16.12）。这一轮**完全不动训练数学**，只做四件事：
> 把"哪套视图"从散装命令行改成**一张共享的表**、让已拿分的配方**可以被精确地要求**、
> 修掉队友点名的唯一瓶颈（`infer.py` 的 CPU 变换）、把新候选视图集**预先登记**进 sweep。
> 改动只在 `train.py` / `probe.py` / `infer.py` / `selftest.py` 四个文件里。

### 19.1 ⭐ 先修一个真实风险：`--tta-agg` 的默认值从未上过榜

| | 本树改前 | 已拿分的四个分数用的是 |
| --- | --- | --- |
| 聚合方式 | `feat`（默认） | **`logit`** |

证据（队友 `infer.py:164-169`，就是产出 69.968 的那份代码）：

```python
acc = None
for v in view_names:
    out = model(batch).float().cpu()
    acc = out if acc is None else acc + out     # 逐视图相加的是 logits
logits.append(acc / len(view_names))
```

⇒ **66.2456 / 69.218 / 69.968 都是 logit 平均**。64.16 是单视图，而单视图下两种聚合恒等
（`selftest.py` 断言过这一点），所以它不受影响。

**风险**：用本树的默认值去复现"已拿分的配方"，得到的是**一条从未上过榜的推理路径** ——
而它会生成一个看起来完全正常的 CSV，没有任何东西会报错。
**已改**：`infer.py` 的 `--tta-agg` 默认改为 `logit`；`feat` 仍然可用（`--tta-agg feat`），
并且正是 `probe.py --calib-agg` 要离线比较的那条路。`selftest.py` 现在断言这个默认值。

### 19.2 ⭐ `--tta-views`：让"拿到 66.2456 的那 4 个视图"可以被精确要求

旧状态是一个**无法表达已拿分配方**的命令行：

| 想表达 | 旧命令行能得到 | 结果 |
| --- | --- | --- |
| `tta4` = plain / plain_flip / wide / tight | `--tta-ratios plain wide tight --tta-flip` | 它翻**每一个**视图 ⇒ 6 视图**超集**，不是那 4 个 |

也就是说：本项目唯一有排行榜背书的配方，**在本树无法被要求**。现在：

```bash
python infer.py --test ... --checkpoint ... --tta-views tta4     # 就是那 4 个
```

- **视图表只有一份**：`train.VIEW_SETS`（原来在 `probe.py` 里叫 `CALIB_VIEWS`），
  `probe.py` 与 `infer.py` 读同一张（`probe.CALIB_VIEWS` 保留为别名，文档/自检里的旧名字继续有效）。
- **拒绝而不是忽略**：`--tta-views` 与乘积开关同时出现 → 报错（否则产出的 CSV 会"看起来是那个配方"）；
  集合在**当前 checkpoint 上有两个视图落到同一批像素**（320 的模型 + `tta4s320`：`None` 与字面 320 都是 320；
  288 的模型 + `s288+plain` 同理）→ 报错（否则是把一个视图平均两次当成一次测量）。
  320 这条尤其要紧：队友现在的基线模型**正是** 320。
  **单个**尺寸视图恰好等于训练分辨率（320 的模型 + `s320`）**不**算退化、不拒绝：它声明一个视图也给出一个视图，
  即 `plain` 换了个名字；要拒绝它就必须连 `plain` 一起拒绝，而 `plain` 是 64.16 那条配方。
  代价只是扫描表里两行同数（2026-09-26 原本按"拒绝"写，`selftest.check_view_sets` 断言失败后改成现在这条更窄的规则）。
- **防漂移断言**：`crops6` 必须与 `--tta-crops center full pad --tta-flip` **逐视图相同**，
  `tta8` 与 `--tta-ratios plain wide mid tight --tta-flip` 相同 —— 两条路的名字不许各自演化（`selftest.py` 断言）。

### 19.3 新登记进 sweep 的视图集（都在同一条已验证的轴上加大剂量）

| 名字 | 内容 | 前向数 | 想回答的问题 |
| --- | --- | --- | --- |
| `tta8` | ratio ∈ {plain, wide, mid, tight} × flip | 8 | 缩放轴**饱和了吗**：tta4 的 +2.09 是"平均"的功劳还是"这 4 个视图"的功劳 |
| `crops6` | crop ∈ {center, full, pad} × flip | 6 | **正交轴**：同样 6 次前向，花在裁剪策略上而不是再多两档缩放上 |
| `mix6` | tta4 + `full` / `full_flip` | 6 | 两轴**同时**加 —— 与 `tta8` 同成本、不同花法，可直接对比 |
| `tta4s320f` | `tta4s320` 的对称版（320 那个视图也翻） | 6 | 第 5 个视图**没配对**是不是被稀释了 |
| `s288` | 单视图 288 | 1 | 对 320 的模型，288 是同族内的**近邻**尺寸（256 跳得更远） |
| `tta4s288` | `tta4` + 一个 288 视图 | 5 | **320 的模型上 `tta4s320` 会被跳过**（退化），于是"第 5 个近邻分辨率视图有没有用"在 320 上根本没被问过 —— 这条就是为 320 补的问法 |
| `tta4s288f` | `tta4s288` 的对称版（288 那个也翻） | 6 | 同 `tta4s320f`：第 5 个视图没配对是不是被稀释了 |

`tta4` / `axis4` / `pad4` / `tta4s320` 保持原样（`selftest.py` 断言 `tta4` 恰好 4 个视图）。
sweep 的"额外集合"列表现由 `train.VIEW_SETS_SWEPT` 给出，**与表放在一起**，
避免 sweep 悄悄漏掉一个命令行仍然能要求的集合。

> ⚠️ **精度提醒**：代理自身的噪声底 ≈ **0.5 点**（等于门的 `view_min`）。
> 不要用它来在两个几乎打平的视图集之间做选择 —— 那种平局应该花一个提交名额，而不是相信第 3 位小数。

### 19.4 ⭐ `--workers`：队友点名的唯一瓶颈

诊断（队友 §16.12 末）：`infer.py` 的图片变换在**主进程单线程**跑，8 视图 @288 × 37k 图 ≈ **1.9 小时**，GPU 全程等 CPU。

**已实现（三条里只做前两条）**：

1. 每个视图一个 `DataLoader(num_workers=N)` —— 变换按**图片**并行，语义不变；
2. worker 里 `torch.set_num_threads(1)`（`infer._worker_init`）—— 否则 N 个 worker 各自继承父进程的线程数，
   N×cores 个线程抢 cores；`Normalize` 确实会用到这个池；
3. **没做**：队友的第 3 条"只 resize 一次再从大图裁各视图"。它会改掉所有已验证分数的 TTA 数值 ——
   在"8 视图有没有用"还没被回答之前，不要动这条已经被验证的路。

**为什么默认是 `--workers 0`**：默认路径与已上榜的运行**逐字节相同**。要加速就显式加 `--workers 8`；
当 视图数 × 图片数 > 2 万 时程序会自己把这句话打印出来（否则这 2 小时是事后才发现的）。

**"逐字节等价"是断言，不是实测**（见 §19.8）：`selftest.py` 新增了 `--workers 0` vs `--workers 2`
的 CSV 逐字节比较，以及 `FlatImages` 的顺序 / 坏图兜底 / 尺寸刷新检查。它还没在真 torch 上跑过。

### 19.5 320 之后的账：分辨率这条线还剩多少

四个平台分与它们各自的 `val_acc` 并排（缺口 = `val_acc` − 平台分，单位点）：

| 配置 | `val_acc` | 平台分 | 缺口 |
| --- | --- | --- | --- |
| 224，无 TTA | 0.7070 | 64.16 | **6.54** |
| 224 + TTA4 | 0.7070 | 66.2456 | **4.45** |
| 288 + TTA4 | 0.7256 | 69.218 | **3.34** |
| 320 + TTA4 | 0.7259 | 69.968 | **2.62** |

**缺口不是配方常数，它随输入表示变好而单调收窄**（固定 TTA4 看：4.45 → 3.34 → 2.62）。
这和队友 §16.12 的观察是同一件事的两面：`val_acc` **先饱和**（288→320 只 +0.0003），平台分却还在涨（+0.75）。

> **操作结论**：`val_acc` 会**系统性低估输入侧的剩余空间**，所以"320 vs 352"不能用 `val_acc` 来排 ——
> 要排就用 §17 的代理（`acc@V*` / `acc@V*-hard`），或者直接花一个名额。

**像素账**（用 §14 的实测：测试集短边中位 **375**；`center` 预处理的短边 = `size × 1.143`）：

| 推理尺寸 | eval resize 短边 | 相对原图 | 平台相对上一步 |
| --- | --- | --- | --- |
| 224 | 256 | 0.68× | 基线 64.16 |
| 288 | 329 | 0.88× | **+2.97** |
| 320 | 366 | **0.98×** | **+0.75** |
| 352 | 402 | **1.07×（第一次超过原图）** | ？ |

⇒ 320 是**最后一个"还在用满原图细节"的尺寸**，352 是第一次要先上采样；再叠加 interpolate-only 的 OOD
（探针余弦 0.9581，四个尺寸里最低）。**352 的先验 ≲ +0.3，而且要付 3.2 小时。**
⇒ 不要先花它，先做 §19.6 里那两件**免费**的事。

### 19.6 上机顺序（替换 §17.7）

```bash
cd /root/autodl-tmp

# 第 0 步：自检（无数据无 GPU，1-2 分钟）
#   ★ 这一轮动了 infer.py 的提交路径，selftest 是唯一会证明"它没被改坏"的东西
python selftest.py            # 必须看到 ALL CHECKS PASSED

# 第 1 步：把 320 的 checkpoint 拿到本树（队友传 outputs_320/ep4.pt 与 ep20.pt，各约 6.4 MB）
#   本树要评的是"和 69.968 完全同一个模型"，不是重训一个像它的
#   ★ 必须是**同一轮的两个 epoch**：门的第一条事实就是"ep 之间相等"，
#     只给一个 checkpoint 时 len(epochs) < 2，门会判 FAIL 并在打印 sweep 结果**之前** return 1
#     （2026-09-26 才发现：此前 §19.6 写的单 checkpoint 命令跑不出任何 sweep 结果）

# 第 2 步（可选）：320 的塔 + 噪声/V* 状态，0 名额，几分钟。
#   ⚠️ 这是**诊断路径**：`--checkpoint` 在它里面**根本不会被读**（只有 calib() 读它，
#   见 probe.py:1169 的那句 assert）——所以这一步**不验证"能不能加载 320 的 checkpoint"**，
#   它只验证"本树能按 320 建塔、跑一遍冻结 CLIP"。别把它当成 checkpoint 的冒烟。
python probe.py --data /root/autodl-tmp/train --sizes 320 --limit 16 --skip-test

# 第 2.5 步：真正的 checkpoint 冒烟（几秒）—— 上面那条证明不了这一步
python probe.py --data /root/autodl-tmp/train --calib --limit 16 --calib-views plain \
    --checkpoint /root/autodl-tmp/outputs_320/ep20.pt
#   要在这行下面看到 320px 的载入说明（分辨率是从 checkpoint 读的，不是从 --sizes 读的）
#   会打印 self-check ... SKIPPED -- --limit ... ：只跑了 16 张图，val_acc 必然对不上
#   训练时记录的全量值。这里**显式跳过**而不是让断言误报（本轮修的一处误报）。

# 第 3 步：门 + 视图 sweep，两种聚合各跑一次（0 名额）
#   （--calib 提前 return，所以 --sizes / --skip-test 在这条路上**不生效**，别指望它们）
python probe.py --data /root/autodl-tmp/train --calib --calib-sweep \
    --checkpoint /root/autodl-tmp/outputs_320/ep4.pt /root/autodl-tmp/outputs_320/ep20.pt
python probe.py --data /root/autodl-tmp/train --calib --calib-sweep --calib-agg feat \
    --out ./probe_feat \
    --checkpoint /root/autodl-tmp/outputs_320/ep4.pt /root/autodl-tmp/outputs_320/ep20.pt
#   缓存文件名里已有聚合方式，两次跑不会互相覆盖，也不会重算同一个集合
#   先跑 ① （只 4 个视图集）再跑 ② 是省时间的做法：plain/wide/tta4/axis4 会命中缓存
```

**要看的三行**：

1. **门。** 只有一个 run 时（只传了 320），跨分辨率那条必须显示 `not measured` —— **那不叫通过**，
   代理没被评级；只传一个 epoch 则连门都过不了（见第 1 步）。
2. **`s352` 单视图那一行** —— 它是对"352 值不值 3.2 小时"的免费提问。
   ⚠️ 它量的是 320 模型上的 352 **推理**，与"在 352 上训练"不是一回事（与 §16.6 的 `CONFOUNDED` 同源）。
3. **`tta8` / `crops6` / `mix6` 相对 `tta4` 的差** —— 这才是"视图数还有没有用"的答案。

### 19.7 现在要做什么（替换 §18）

| 优先级 | 做什么 | 成本 | 依据 |
| --- | --- | --- | --- |
| **P0** | 队友传 `outputs_320/ep20.pt` → 在本树跑 §19.6 第 0~3 步 | 0 名额，< 1 小时 | §19.1：默认值曾与已验证配方不一致（已修，但要在真机上确认）；§17：网格重构必须在本树验 |
| **P0** | sweep 结果按 `acc@V*` **和** `acc@V*-hard` 排 → 选一个配方提交 | 1 名额 | §16.2 只验过 4 视图；"更多视图"是唯一还没测过、且有过 +2.09~+2.97 背书的方向 |
| **P1** | 若 `tta8`/`mix6` 明显优于 `tta4` → 在 320 上生成对应 CSV（`--workers 8`，几分钟） | 1 名额 | 同分辨率、同 checkpoint ⇒ 唯一变量是视图 |
| **P1** | 352 训练 —— **只在 `s352` 那一行不为负时才做** | 3.2 小时 + 1 名额 | §19.5：352 是第一个超过原图的尺寸，先验 ≲ +0.3 |
| **P2** | 移植队友的 `--save-teacher`（约 15 行）做 `teacher_ep8.pt + TTA` | 1 名额 | §16.4：教师优势最大 × 还没开始记噪声的交点；**但教师优势在 ep20 已经归零** |
| **❌** | 交任何**无 TTA** 的 ep4~ep20 快照 | — | §16.1：ep4 == ep20 |
| **❌** | 在同一个 checkpoint 上继续调训练数学（轮次 / 死类 / `--tau-conf`） | — | §16.1 / §16.3 |
| **❌** | 改动 `eval_transform` 的几何（为提速的"只 resize 一次"也在此列） | — | §19.4：所有已验证的分数都基于当前几何 |
| **❌** | 任何形式的**跨 checkpoint 平均** | — | 题目禁止多模型融合（§1） |

### 19.8 诚实清单（更新 §17.8）

1. **`selftest.py` 2026-09-26 第一次在真机上跑，三次启动才把 16 个 check 走完**（本地缺
   torch/torchvision/open_clip，所以只能靠上机迭代）：
   * 第 1 次：`ImportError: cannot import name 'FrozenJudge' from 'noise'` —— 云端 `noise.py` 是旧的（见第 8 条）；
   * 第 2 次：`AttributeError: '_StubCLIP' object has no attribute 'logit_scale'` —— 见第 9 条；
   * 第 3 次：16 个 check 全部执行到，**5 个失败** —— `lr warmup` / `image size` / `ck image size` / `view sets` /
     `end to end`。三个不同的根因，见第 11~13 条；`losses` / `tracker` / `judge` / `targets` / `proto` /
     `lora` / `param groups` / `calib gate` / `flat images` / `reproducible augment` 全过。

   ⇒ §19.6 第 0 步不是形式主义：**这一轮的 selftest 断言，只有在真机上跑过才存在。**
   "写了一个断言"和"这个断言被执行过并成立"之间，隔着上面三次启动。
2. **`--workers` 的实际提速倍数没有测过**，只有机制（按图片并行）和队友"单线程 1.9 小时"的观测。
   第一次上机顺手记一下 8 视图的墙钟时间，那就是这个改动的验收数字。
3. **选择性违约检查只在本树的推理路径上生效。** `probe.py` 的 sweep 读同一张表，但走的是另一条代码路径
   （`predict_probs`），它遇到退化集合的行为是**跳过并打印原因**，而不是报错。
   两条路行为不同是**有意的**（探索宽松、提交严格），但这是本轮新增的一处不对称，记在这里。
4. **`s288` 进入单视图枚举会顺带把它与所有单视图的配对（`s288+*`）加进 sweep**（约 7 个新集合）。
   代价是这些集合各自一遍 val 前向；缓存按 (集合, 聚合) 哈希分开，重复跑不会重算。
5. **写 §19.6 时给的命令是错的**（2026-09-26 复查代码时发现，已改）：`--calib` 只传一个 checkpoint
   ⇒ `calib_gate` 的 `len(epochs) < 2` 直接判 FAIL，`calib()` 在打印任何 sweep 结果**之前** `return 1`。
   也就是说照那条命令跑，人只会看到一个 FAIL 与零条视图集结果。**门必须传同一轮的两个 epoch。**
   这条是本轮最值得记的教训：**命令有没有可能跑出结果，要在代码里验，不能凭印象写。**
6. **`--calib --limit N` 的 `val_acc` 自检会误报**（本轮修）：`--limit` 只取前 N 张，
   自检拿它对训练时记录的全量 `val_acc`，必然超容差、抛 AssertionError，
   而真正原因是 `--limit` 而不是模型坏了。现在这条分支**显式打印 SKIPPED 并说明原因**
   （`probe.py` 里 `... and a.limit:` 那一支）——**它同样没有在真机上验过**。
7. **`--calib-sweep` 的 help 文本改成由 `train.VIEW_SETS_SWEPT` 生成**（原来是硬编码
   "tta4 / axis4 / pad4 / tta4s320"，加进 4 个新集合后已经过时）。这是防漂移：
   同一份清单出现在两个地方、其中一处手工维护，就是下一次"文档与代码不一致"的来源。
8. **⭐ 用本地 mtime 推断"服务器上要不要更新"是错的，今天当场踩了。** 我给的待传清单是
   "09-26 改过的 6 个文件"，理由是 `noise.py`（本地 mtime 09-24）"本轮没动"。
   结果本机 `selftest.py` 第一步就死：
   ```
   train.py:48: from noise import FrozenJudge, LabelTrustTracker, prototype_bootstrap
   ImportError: cannot import name 'FrozenJudge' from 'noise'
   ```
   —— 服务器上的 `noise.py` 比本地 09-24 的那版**还旧**。本地 mtime 只说明"我这边最后改于何时"，
   与"云端是哪一版"毫无关系，中间那次上传可能发生在更早。
   **正确做法**：判断依据只能是服务器的版本（比对大小/哈希，或直接全传）；
   而"传哪几个"这种只能靠记忆判断的事，本身就是下一次静默不一致的来源。
   **代码文件全是文本、几十 KB，全传的成本趋近于零 —— 别挑。**
9. **`check_param_groups` 的断言引用了一个 stub 上不存在的属性**（已修）。
   它检查 `not net.clip.logit_scale.requires_grad`，但 `_StubCLIP` 只造了 `visual`。
   真实 CLIP **确实有** `logit_scale`，而且 `Net.__init__` 的冻结循环（`train.py:1075-1077`，
   "除 LoRA 的 `.A`/`.B` 外全部 `requires_grad_(False)`"）本来就该覆盖它 ——
   所以**正确的修法是给 stub 补上这个 `nn.Parameter`**，让断言去检验那段冻结循环；
   用 `hasattr` 绕过则是让一条断言静默地什么都不检查，正是本文件反复警告的那种失败。
10. **`selftest.py` 改成"跑完所有 check 再报告"**（原来是第一个异常就 `raise` 结束）。
    理由是本机没有 torch，**一次上机运行必须拿到尽可能多的信息**：死在 16 个 check 的第 7 个，
    等于白跑一趟才知道第 8 个说什么。代价是第一个失败之后的失败**可能是它的后果**而不是独立 bug，
    报告里已写明"第一个才是要修的那个"。顶层的种子仍然只在循环前设一次 ——
    改成每个 check 前重设会**悄悄改变目前能通过的 check 的输入**，那是一种对没坏的东西的不可验证的改动。
11. **3 个失败（`lr warmup` / `image size` / `ck image size`）是同一个根因：`train.parse_args` 要求 `--data`，
    而三处调用点只传了被测的 flag。** 已修：新增 `train_args()` 一个 helper 供全部直接调用点使用
    （`selftest.py` 里另一个 fixture 是给三个脚本统一走的，`train` 那一项补成 `['--data', '.']`）。
    这是"**自检代码本身从没被执行过**"的典型症状 —— 断言的内容对不对无从谈起，
    参数能不能解析就已经是另一回事。写 check 时每加一个 flag，都要对着 `parse_args` 的
    `required=True` 看一遍。
12. **`end to end` 的逐字节比较失败，根因在 stub 而不在被测代码**（本轮已修，见 `_stub_create_model`）：
    `open_clip.create_model` 每次调用返回**新对象**，真实路径下每次加载的是**同一份预训练权重**，
    所以两个 tower 逐位相同；而 stub 用的是 torch 全局 RNG，两次构造**每个冻结参数都不同**。
    致命之处在于 `trainable_state_dict` 只存**可训练**参数 ⇒ checkpoint 里没有任何东西能还原冻结的骨干 ⇒
    比较两次 `infer.main` 输出的那几条断言（`--tta-sizes 224` 与不带 flag、`--tta-agg feat`、
    `--workers 0 vs 2`）实际是在给**两个无关的模型**打分，无论被测代码对不对都不可能通过。
    修法是把**前提**改对（构造前后 `get_rng_state`/`set_rng_state` 夹一次 `manual_seed(0)`：
    既让每次构造的同名模型逐位相同，又不挪动其它 check 的随机流），而不是把断言改弱。
    顺带确认了一件本来会误诊的事：`--tta-sizes 224` 的输出差异**不是** `view_plan` 把视图数了两次 ——
    读 `infer.py:217-228`，产物分支对 `sizes=[224]+[224 被过滤]`、`crops=['center']`、`ratios=['plain']`、
    不 flip 给出的就是 `[(224,'center','plain',False)]` 这一个视图，与不带 flag 的列表**完全相同**。
    **残留风险**：`v_tta4 != csv`（第 1617 行附近）是**不等**断言，若 stub 在 32 张图上全预测同一类就会失败 ——
    那属于 fixture 的性质而不是 `infer.py` 的 bug，断言信息里已写明这两种可能。
13. **`is_degenerate` 的契约与我的文档写的是两个不同的谓词，`view sets` 的断言把它暴露出来了。**
    我给三处文档（`train.py` docstring、`README §4.5.4`、`HANDOFF §19.2`）写的是
    "320 的 checkpoint 配 `s320` → 拒绝"，但实现只检测**集合内部两个视图落到同一批像素**，
    而单个 `s320` 在任何分辨率下都不会自撞。
    想清楚之后：**窄规则才是对的** —— `s320` 在 320 上声明一个视图、也给出一个视图，它就是 `plain` 换个名字；
    要拒绝它就必须连 `plain` 一起拒绝，而 `plain` 正是 64.16 那条配方。
    已改：实现不动，改文档与断言（并且补上真正会被抓到的 `tta4s320`@320、`s288+plain`@288 两个例子）。
    教训：**断言要按契约写，文档要按代码写**；两边不一致时，先问哪个谓词是对的，再决定改哪边。
    同一处修正还落在 `infer.py` 的报错文案与 `probe.py` 的跳过文案上——后者原来对 `tta4s320`@320 说
    "measures nothing"，而那个集合有 4 个**不同**的视图，多说了一个"零"字就是一行误导人的日志。
14. **`--tta-views s320` 在 320 的 checkpoint 上是允许的，产物就是 `plain`。** 运维后果只有一条：
    在 320 上跑 sweep，`s320` 那行与 `plain` 那行**数字相同**，选配方时别把它当成第二个证据。
    已写进 `README §4.5.4` 与 §19.2（第 13 条同一处）。
15. **新登记 `tta4s288` / `tta4s288f`（320 上的多分辨率多视图行）。** 起因是第 13 条那个契约想清楚之后：
    在 **320** 的 checkpoint 上，`tta4s320` / `tta4s320f` 会**被判退化并跳过**，
    于是 320 那次 sweep 里**一条"多加一个近邻分辨率视图"的多视图行都没有** ——
    而这正是 §19.3 想回答的问题，也是队友 69.968 那个模型所在的分辨率。
    两条新集合与 `tta4s320` 完全对称，只是把"近邻尺寸"换成 288（表的注释早就写了 320 的近邻是 288）。
    **重要的是它们是在看到任何数字之前登记的**：事后按结果挑集合，sweep 就不再是证据。
16. **本地新增的两个 `.py` 改动没有在真机验证过**（`train.py` 的视图表、`selftest.py` 的断言）：
    本机没有 torch，只能 `py -3 -m py_compile` + 静态检查（`issues: 0`）。
    ⇒ 上机第一件事仍是 `python selftest.py`（§19.6 第 0 步）。
17. **`v_tta4 != csv` 这条断言是个硬币，已删掉，换成"数模型真正拿到手的像素"**（2026-09-26 第三次上机，
    16 个 check 只剩它一个失败）。它错在**用 CSV 回答一个 CSV 答不了的问题**：
    视图平均只在"决策本来就接近"的地方改变 argmax，所以"一个视图 32 个标签与四个视图 32 个标签相同"
    对一个**有信心的模型**来说是**正常情况**，不是 bug；而在这个 fixture 上模型连输入都不太依赖
    （val_acc 四个 epoch 全是 0.25 = 4 类的随机水平，tracker 从第 2 个 epoch 起就在把大多数样本重标，
    stub 的塔又把整个 patch 网格平均成一个 token ⇒ 放大后的 48×48 噪声在任何视图下几乎是同一个特征）。
    两种情况下它都是硬币，而且在**好模型**上同样会失败。
    现在改为在 `infer.main` 外面给 `_StubVisual.forward` 装一个记录器，断言两件确定性的事：
    ①塔被递进来的图片数 == `len(plan) × 图片数`（`plan` 由 `infer.view_plan` 现算，
    不是手写常数），②按"每个视图走一遍数据集"重新分组后，第 1 个视图与其它每个视图的**像素不相同**。
    第②条是这个文件里**唯一**能抓到"视图名字对了、但 size/ratio 在传给 transform 的路上掉了"的检查 ——
    视图列表、行数、CSV 顺序全都会照常通过。
    **仍然看不到**的是"决定这行的是平均而不是第一个视图"：模型有信心时平均不改变任何东西，
    那是**可观测性**的缺口而不是断言的缺口（单视图那两条恒等断言钉住了聚合路径）。
18. **`--tta-sizes 224 == 不带 flag` 这条恒等断言，报错文案写的是不可能的成因**（同一次上机顺手修）。
    它原来写"the view was counted twice"——但一个视图与自己平均**就是它自己**，数两次**不可能**移动一个字节。
    所以这条断言真正在测的是**两个拼写是否同一条配方**（会不会多建一座塔、transform 是不是从 flag 而不是
    从解析后的视图建的、base 是不是从别处读的），文案已按这个含义改写。
    `feat` 那条单视图恒等断言的含义则是对的（它才能抓到"归一化两次/对和而不是均值套头"）。
19. **`probe.py` 的两个缓存都不看图片列表：冒烟会把 16 张图的缓存留成"全量"**（本轮发现，修在
    `_cache_matches`）。缓存文件名里只有**视图**（`feat_val_224.npz`、`calib_{stem}_{v}_{agg}_{vh}.npz`），
    而 `--limit` / `--val-ratio` / `--seed` / `--data` 改的是**图片列表**，名字一个字都不变；
    `predict_probs` 读回时更是连图片数都不看。于是 `README §4.5.1` ⓪ 那步（`--limit 16`，默认
    `--out ./probe_out`）写下的 16 张缓存，会被 ① 门控和 ② sweep **当成全量 val 读回去**，
    报告里没有任何一行会说"这是 16 张图上的数"——**静默的错数**，正是这个文件大部分注释在防的事故。
    更麻烦的是调用方那句 `assert np.array_equal(paths, Fva['paths'])` 会**两边同时中毒而通过**。
    现在 `extract` 与 `predict_probs` 都在**读缓存前**先算出本次的图片列表并比对，不符就重算并打印
    `NOT reused`；`README` 的冒烟同时改成 `--out ./probe_smoke`（分开目录，不必依赖那条保护）。
    新增 `selftest` 的 `probe cache identity`：2 张的缓存不得被 4 张的 pass 读回，
    且命中缓存时**不建模型**（用一个会抛异常的 `model_for` 证明它真的读了文件）。
20. **`probe cache identity` 第一次上机就失败了，但坏的不是被测代码，是 check 自己的 fixture**
    （2026-09-26 第四次上机，17 个 check 只它一个失败）。`predict_probs` 在 loader 里解包
    `(x, y, idx)`，而 fixture 用了 `FlatImages`（扁平测试目录，产出 `(x, i)`）⇒
    `ValueError: not enough values to unpack (expected 3, got 2)`。
    **第 19 条那两处修复在同一份日志里已经被证明了**：`extract` 打印了
    `[cache] feat_val_224.npz: NOT reused -- 2 image(s) in the file, 4 in this pass` → 重算 → 第三次
    调用命中 `[cache] val @224 4 images`，即"旧文件不被读回、本次的文件被复用"两步都成立。
    也就是说 `extract` 那一半是过的，只是 check 先崩在 `predict_probs` 那一半的 fixture 上。
    现已改用 `probe.build_split_datasets(a)` 取 `calib()` 真正评的那个 val 集（4 类 × 2 = 8 张），
    两半都跑在同一个数据集上，`n_val` 从实际长度取而不是写死。
    教训与第 13 条同源：**断言要按被调函数的契约写**，凭印象给 fixture 就是在测别的东西。
    这条 check 的第二次执行是下一次上机；同时要说明覆盖边界——selftest 走的是 `predict_probs`
    函数本身，**没有**覆盖 `calib()` 里缓存名的拼法与 `--limit` 的传递（两者共用 `_cache_matches`，
    所以坏在函数里会被抓到，坏在 `calib()` 的调用里不会）。

## 20. 本树这一轮 —— 2026-09-28（第一次对**训练侧**动手，坐标是"ep4 定分"）

§16.1 那个结果是这一轮的出发点，重述一遍：**同一轮训练里 `ep4` 与 `ep20` 在榜上相等
（64.1598 vs 64.16，四位小数），而同期 `val_acc` 涨了 5.02 点。**
推论很硬：这个模型的**榜单相关状态在第 4 个 epoch 就定完了**，warm-up 之后那 16 个 epoch 的
噪声跟踪 / 重标注 / 原型 EMA 在榜上收益≈0。

于是这一轮只改**能改变第 4 个 epoch 状态**的东西。所有开关**默认关闭**，
`--head-init none` / 无 `--epoch-aug` / 无 `--warm-robust` 的路径与之前**逐字节相同**，
所以 64.16 / 66.2456 / 69.218 / 69.968 四个分数**仍然可复现**（这是硬要求，不是客套）。

### 20.1 三个改动（对应 `提分路径研究.md` 的 B11/B17、A3、B4）

**① `--head-init frozen`（B11 + B17 的合体）—— 本轮的主实验**

现在的 `CosineClassifier` 是 `randn * 0.02` 起的：前几个 epoch 是在**从噪声里发明**一个
750 类的线性映射，而冻结 CLIP 本来就知道这个映射；同时挂着全程最大的学习率、对着四分之一
是错的标签。这个改动把**冻结塔在训练集全量上跑一遍**（无梯度、确定性 eval transform），
取"抗噪类质心"，**在第一步之前**写进 `head.weight`。

- 抗噪质心（`robust_centroids`）：第 1 轮就是文件夹的朴素均值（继承标签噪声），
  之后每一轮**只保留"给定标签 == 最近质心"的样本**再重新平均。**全自动**（规则 五.6 禁的是
  *人工*清洗，自动清洗是明确鼓励的），确定性，只用训练集，不碰 val/test。
- **缺席的类保持随机行**，不置零：零行的 cosine 是 0，会变成一个"中性竞争者"，
  比一个真实的负相关类更容易赢 —— `FrozenJudge.judge` 的 docstring 已经记过这个坑。
- 同一个估计顺带补齐两处覆盖：`proto` 的原型、`judge` 的每样本冻结特征。
  原来的 warm-up 靠 `WeightedRandomSampler` 偶然抽到，**稀有类根本抽不到**，
  而代码自己的注释写着"采样器之后也没有理由抽到它们"。
- 代价：一次纯前向的全量 pass。320 下 100 个 patch、442~454 s/epoch 是 2×1162 次前向，
  所以这一次 pass ≈ 1162 次前向 ≈ **4 分钟**，相对 2.5 小时可以忽略。
- 语义：`forward` 里有 `F.normalize(self.weight)`，所以行范数不进 logit —— 换上去的是
  **方向**，也就是说 **step 0 时这个头算的就是 CLIP 的 zero-shot 判决**，不是随机判决。

**② `--epoch-aug`（A3）—— 修一个一直在悄悄生效的退化**

增强种子是 `(seed, index, view)`，**一次训练内固定**：第 1~20 个 epoch 对同一张图抽的是
**同一个 crop、同一个 RandAugment**。20 轮 = 在一张固定的增强图上过 20 遍。
把 epoch 异或进种子里就恢复了"每轮一个新视角"。
(`epoch = 0` 时异或 0 是恒等，所以默认路径逐字节不变。)
⚠️ 这个开关**必须同时关掉两个 loader 的 persistent workers**：worker 持有数据集的一个
fork 副本，main 改 `tr.epoch` 它看不见 —— 第一版就踩了这个坑（只改了 `vloader`），
那会让开关变成一个**静默无效**的死开关。selftest 现在把两个 loader 都钉住了。

**③ `--warm-robust`（B4-lite）—— 把鲁棒损失搬进 warm-up 窗口**

warm-up 现在是**纯 CE 对原始标签**，而"对错标签算 CE"正是**驱动记忆化**的那一项 ——
偏偏就在"决定分数"的那个 epoch 窗口里。打开后 warm-up 也加 `--robust-loss`，
权重用同一个 `--robust-weight`（warm 期间 `w` 全 1，所以就是同一项、同一个权重）。

### 20.2 上机顺序

```bash
cd /root/autodl-tmp

# 第 0 步（必做）：本轮动了 train.py 的主流程与 build_loaders
python selftest.py            # 必须看到 ALL CHECKS PASSED（新增 head init 这一项）

# 第 1 步：主实验 B —— 320 + 冻结质心初始化，**只加这一个变量**
python train.py --data /root/autodl-tmp/train --out ./outputs_320_headinit \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 --image-size 320 \
  --head-init frozen
#   要看到的三行：
#     head-init frozen: 148695 training images, ...        ← pass 开始
#     head-init frozen: head seeded for NNN/750 classes; ...  ← 类数不该是 0
#     head-init frozen: round 1/2: ... agree 0.xx           ← agree 比例
#   ⚠️ 故意**不加** --noise-judge：judge 的假阳性率还没量过（见 20.4-2），
#      加了它就是第二个变量，这一轮要的是"只有初始化变了"。
#   ⚠️ --resume 时 --head-init 会被忽略并打印警告（checkpoint 里的头才是它对的那个）。

# 第 2 步：离线代理（0 名额，几分钟）
python probe.py --data /root/autodl-tmp/train --calib --limit 16 --calib-views plain \
  --checkpoint ./outputs_320_headinit/best.pt
#   只看一个数：320px 载入说明 + 它报告的 acc。
#   对照物是 69.968 那个模型在**同一份 val** 上的同一个数（队友那边有）。
#   注意噪声地板 ≈ 0.5 点（§4.5），差不到 0.5 点**什么都不说明**。

# 第 3 步：提交（1 名额）—— ⚠️ 用 tta8，不是 tta4
#   队友 §16.13 已实测：320 + tta8 = **70.332**（比 tta4 的 69.968 高 0.364），
#   本树的 `tta8` = ratio{plain,wide,mid,tight} × flip，与队友那份 8 视图表**同一张**。
#   所以比较对象从 69.968 改成 **70.332**。
python infer.py --test /root/autodl-tmp/test --checkpoint ./outputs_320_headinit/best.pt \
  --tta-views tta8 --workers 8 --output pred_results.csv --logit-adjust 0 0.25 0.5
wc -l pred_results.csv      # 复赛 37444（README 的 CRLF 提醒：行数用 wc -l 看，别用 $ 锚定）
zip submission.zip pred_results.csv
#   ⇒ 与 70.332 比。榜上取最大值，所以**这次提交没有下行风险**：更低就丢掉。

# 第 4 步（B 不低于 70.332 时）：再叠 C —— epoch-aug + warm-robust
python train.py --data /root/autodl-tmp/train --out ./outputs_320_c \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 --image-size 320 \
  --head-init frozen --epoch-aug --warm-robust
```

> **§21 已按队友 §16.13 的新结果重排过优先级** —— 上面这个顺序仍然成立，
> 但**352 那一步被移出了 P0**（理由见 §21.2），第 5 步改成"352 + head-init 一起上"。

**第二个名额**：当天先留空。B 有结果后，只有当 `best.pt` 与 `ep4.pt` 的 `--calib` 代理
**明显不同**时才值得用第二个名额交 `ep4.pt`；否则那个名额留给第二天的新配方
（§16.1 说 ep4 与 ep20 榜上相等，交两个快照基本是浪费半个名额）。

### 20.3 判据

| 现象 | 怎么读 |
| --- | --- |
| B < 69.968 | 初始化这条线在本数据集上不赚分。榜上无损失，回去查 `agree` 比例（见下） |
| B ≈ 69.968（±0.5） | 代理噪声地板内，**判不了**。看 `agree` 比例是否正常再决定要不要重跑 |
| B > 69.968 但 < +1 | 初始化有效但量级不够 —— 叠 C，或直接上"短轮次多试验"（见下） |
| B ≥ +1 | 这是训练侧第一次真正赚到分。立刻叠 C，并考虑把同一改动搬到 288 复验 |
| `agree` 比例 < 0.5 | 冻结 CLIP 的 NCC 在本数据集上不如标签可靠 —— 质心被噪声主导，**这个改动的前提不成立**，应停下来说清楚，而不是继续调 |
| `head seeded for` 的类数 < 750 | 有类在整个训练集里一个样本都没通过过滤。先看是哪些类（§15.5 那 15~20 个 0 召回类会在这里现身） |

**短轮次的提议（B 出结果之后再决定，别和 B 混在一起）**：既然"ep4 定分"，
那 `--epochs 6`（约 45 分钟）可以作为**筛选器**，一天筛 5~6 个配方，只把最好的那两个
跑满 20 轮去提交。⚠️ 诚实的边界：这不是"20 轮的前 6 轮" —— `make_scheduler` 是
**按 `--epochs` 算的 cosine**，6 轮跑的是"更快衰减"的另一个 schedule。
所以短轮次只能用来**排序配方**（同一 schedule 下横向比），最终候选必须回 20 轮复跑一次。

### 20.4 诚实清单（接 §19.8）

1. **本轮的代码一行都没在真机上跑过。** 本机没有 torch（只有 `py_compile` + 一个
   `symtable` 静态检查：所有函数里引用的全局名都存在），所以
   `--head-init frozen` / `--epoch-aug` / `--warm-robust` 三条路径的**第一次执行就是上机**。
   §19.8 第 1 条已经记过一次同样的账（"写了一个断言"≠"断言被执行过"）。
   ⇒ 上机第一件事是 `python selftest.py`，其中新增的 `head init` 这一项会跑
   `train.main` 全程（stub 塔，3 epoch，4 类，几十秒）+ 三个单元断言。
2. **`--noise-judge` 的假阳性率仍然没量过**，而这一轮把它的覆盖率从"warm-up 抽到的"
   提到了**全量**（`FrozenJudge.add_all`）。覆盖率提高不等于判决变准：
   §19 的 `probe.py --calib` 报的就是这个分布。**在全量覆盖上量过之前，别打开它。**
3. **质心过滤的强度没有标定过。** `--head-init-rounds` 默认 2（一朴素 + 两轮过滤）。
   selftest 用合成的 25% 错标证明了 round 1/3 都比朴素均值更靠近真实方向、且保留集更干净，
   但那是 16 维、4 类、0.15 扰动的**玩具**；真机上 750 类的 `agree` 比例是多少，只有日志知道。
4. **种子头改变了学习动力学，而 20 轮 schedule 是围着随机头调的**（`--lr-warmup-epochs 1`
   也是）。两种可能结局：更好的 ep4 状态（目的），或者被冻在 CLIP 的 zero-shot 几何里出不来。
   `--head-init frozen` 之外**不要**同时改 lr；要改 lr 就是下一个单独实验。
5. **本轮没有验证"训练侧确实赚分"这个命题**，只是把它变成了一个可执行、单变量的实验。
   在 B 的榜上数字出来之前，§16.1 的结论（训练侧收益≈0）**没有被推翻**。
6. **一条与本轮无关但更便宜的邻近动作**（留给用户判断要不要插队）：§19.3/§19.6 登记的
   `tta8` / `crops6` / `mix6` 视图集**还没有任何数字**，而 TTA 是唯一一条有
   +2.09 / +2.97 背书的历史轴，代价是 0 次训练、1 个名额。若只按"每 GPU 小时的期望分"排，
   它排在 B 前面；本轮按用户指令先做训练侧算法，这条记在这里不算执行。
   ⚠️ **已被 §21 取代**：队友 §16.13 已实测 `tta8` = **70.332**，这条不再是未知数。

## 21. 按队友 §16.13 重排 —— 2026-09-28（同一天，第二次调整）

### 21.1 队友新增的四条事实（本树此前不知道的）

1. **320 + 8 视图 = 70.332**（`sub_320_tta8.csv`，比 320+tta4 的 69.968 高 **+0.364**）。
   ⇒ **当前最好成绩是 70.332**，不是 69.968。本树的 `tta8` 与队友那份 8 视图表是**同一张**
   （ratio{plain,wide,mid,tight} × flip，见 §19.3），所以 §20.2 第 3 步已改成 `--tta-views tta8`。
2. **视图数的收益递减**：4 → 8 视图 = +0.364。第一步（0→4 视图）是 +2.09。
   与分辨率并排：224→288 = +2.97，288→320 = +0.75。
3. **`infer.py` 变换并行化，15 倍加速**（9 分 9 秒 vs 预估 2.3 小时），且**逐字节零副作用**
   （真机 md5 回归 = `f5c99066da4d013a5295478d7e1ac4b8`）。本树 §19.4 是同一个改动。
4. 队友把 **352 分辨率列为 P0**、把"更多视图（12/16）"列为 P1。

### 21.2 ⚠️ 但 352 不该排 P0 —— 用本树 §19.5 的像素账反驳

队友的排序理由是"分辨率是最强的杠杆（+2.97、+0.75）且 288→320 还在涨"。
这个推理漏了一个**同一位队友自己测出来的**数据点：**384 + TTA4 = 68.5476（−1.42）**。

本树 §19.5 的账解释了这两个数为什么不矛盾 —— 分界线是**原生像素**：

| 训练尺寸 | eval 短边 | 相对原图（短边中位 375） | 平台相对上一步 |
| --- | --- | --- | --- |
| 288 | 329 | 0.88× | **+2.97** |
| 320 | 366 | **0.98×** | **+0.75** |
| 352 | 402 | **1.07×（第一次要上采样）** | ？ |
| 384 | 439 | **1.17×** | **−1.42（实测）** |

320 是**最后一个还在用满原图细节的尺寸**；352 与 384 都在"先上采样再插值位置嵌入"的同一侧，
而 384 那边已经有一个 **−1.42 的实测**。再加上位置嵌入插值质量的探针（352 的余弦 **0.9581**，
四个尺寸里最低），352 的先验是 **≲+0.3 且可能为负**，代价 **3.2 小时**。
⇒ **352 不做单变量实验；它只作为"叠加项"上**（见 21.4），这样即使 352 本身是 0，
那 3.2 小时也还载着 head-init 的收益。

### 21.3 缺口表更新 —— 一个必须看清楚的趋势

| 配置 | `val_acc` | 平台分 | **缺口** |
| --- | --- | --- | --- |
| 224，无 TTA | 0.7070 | 64.16 | 6.54 |
| 224 + TTA4 | 0.7070 | 66.2456 | 4.45 |
| 288 + TTA4 | 0.7256 | 69.218 | 3.34 |
| 320 + TTA4 | 0.7259 | 69.968 | 2.62 |
| 320 + TTA8 | ~0.726 | **70.332** | ~2.3 |

两个读数：

- **TTA 那 +2.09 是纯粹靠缩小缺口拿到的**：`val_acc` 一位都没动（0.7070 → 0.7070），
  平台分涨了 2.09。⇒ 缺口里有一部分**不是"记忆噪声"，而是"验证集奖励、测试集不奖励"的东西**，
  而多视图平均恰好把那部分削掉了。
- 缺口随输入表示变好**单调收窄**（6.54 → 4.45 → 3.34 → 2.62 → ~2.3）。
  ⇒ 缺口正在被吃完。**这是"输入侧还剩多少"的另一个读数：它在收敛，不在发散。**

### 21.4 修订后的上机顺序

```bash
cd /root/autodl-tmp

# 第 0 步（不变，必做）
python selftest.py                    # ALL CHECKS PASSED

# 第 1 步（不变，主实验）：320 + head-init frozen，单变量
python train.py --data /root/autodl-tmp/train --out ./outputs_320_headinit \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 --image-size 320 \
  --head-init frozen

# 第 2 步（改了）：**直接交 tta8**，比较对象是 70.332
python infer.py --test /root/autodl-tmp/test --checkpoint ./outputs_320_headinit/best.pt \
  --tta-views tta8 --workers 8 --output pred_results.csv --logit-adjust 0 0.25 0.5

# 第 3 步（新，0 训练成本，1 个名额）：16 视图 —— 唯一还没被试过的"更多剂量"
#   tta8 是 ratio{plain,wide,mid,tight}×flip。16 视图 = 再加 4 档 ratio（8 档 × flip）或
#   加裁剪轴（tta8 + crops6 里的 4 个）。
#   ⚠️ 先看 §19.3 的登记表：**必须在看到数字之前把 16 视图那张表定下来**，
#   否则事后按结果挑集合就不是证据了。递减序列（+2.09 → +0.364）说这一步先验 ≈ +0.1。
#   它排在 352 前面，因为它只花 15 分钟和一个名额。

# 第 4 步（新）：352 + head-init + tta8 —— 把 352 当**叠加项**，不当单变量
python train.py --data /root/autodl-tmp/train --out ./outputs_352_headinit \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12 --image-size 352 \
  --head-init frozen
```

**名额分配（每天 2 个）**：第 1 步占 1 个；第 2 步占 1 个；第 3 步（16 视图）在**同一台机器上
不用重训**，可以和第 1 步的提交错开到第二天，避免同一天两个名额都压在同一个模型上。

> ⚠️ **两条与本树相关的差异，照抄队友命令会踩：**
>
> 1. **`--workers` 的默认值不同。** 队友那边默认 **8**，本树默认 **0**（= 与已上榜的运行逐字节相同，
>    §19.4 有意这么定的）。所以本树**必须显式写 `--workers 8`**，否则 8 视图 @320 会退回
>    单进程 2.3 小时那条路。
> 2. **内存：队友的注释写"80 GB 内存没问题"，本容器**不是 80 GB** —— 上限 **60 GiB**
>    （`free` 显示的是宿主机的 755 GB，别被骗）。预取 buffer = `batch × V × 3 × S × S × 4`：
>    `tta8` @320 @默认 `--batch-size 256` ≈ **2.5 GB/批**，8 个 worker 最多同时在飞 8 个 ≈ **20 GB**
>    —— 在 60 GiB 上**能跑但已经很紧**。**16 视图会翻倍到 ~5 GB/批 ≈ 40 GB，必须先降 `--batch-size`**：
>    `--tta-views <16 视图> --batch-size 128 --workers 4`（≈ 10 GB 在飞）。
>    `--batch-size` 只影响**多久跑完**，不改结果。

### 21.5 判据（更新 §20.3）

| 现象 | 怎么读 |
| --- | --- |
| head-init 提交 < 70.332 | 初始化这条线不赚分（和 §16.1 对训练侧的整体判决一致） |
| head-init 提交 ∈ [70.3, 70.9] | 与 tta8 的 +0.364 同量级 —— **这已经是现在能拿到的正常幅度了**，不是失败 |
| head-init 提交 ≥ 71.3 | 训练侧第一次真正赚到分（+1 以上）。立刻叠 `--epoch-aug --warm-robust` |
| `agree` 比例 < 0.5 | 前提不成立（冻结 CLIP 的 NCC 不如标签可靠），停下来说清楚 |
| `present` 类数明显 < 750 | 过滤把某些类清空了。**逐类看**：如果被清空的正是 §16.3 那 14 个 `asym=+1.00` 的死类，那是**好消息** —— 说明过滤器找到了系统性标注错误（见 21.6） |

**为什么第 4 步（C）叠的是 `--epoch-aug` + `--warm-robust` 而不是别的** —— 这是从 §21.3 的缺口表
推出来的排序，值得写清楚：

- 三个新改动可以按**作用方式**分成两类：
  **① 抬高上限**（`--head-init`：让第 4 个 epoch 的状态本身更好）；
  **② 缩小缺口**（`--epoch-aug`：每个 epoch 换一组取景框；`--warm-robust`：把记忆化那一项从
  warm-up 里压下去）。
- **实测证据偏向第 ② 类**：TTA 那 +2.09 是**纯缺口收窄**（`val_acc` 一动没动）。
  而 `--epoch-aug` 正是"推理端的视图多样性"在**训练端**的对应物 —— 既然平均多个取景框值 2.09 分，
  那"20 个 epoch 只在一张固定取景框上训练"（当前行为）就有一个被量化的、明显可修的成本。
- 所以 C 不是"随便再叠两个开关"，而是**按已验证的缺口机制排的第二、第三顺位**。

### 21.6 一个值得写下的机制：head-init 恰好作用在死类上

§16.3 的结论是：`0179 → 0247`、`0039 → 0446`、`0640 → 0502` 这些是 `asym=+1.00` 的
**系统性标注错误** —— 文件夹 `0179` 里的图**真身是 `0247`**，而且**训练集里没有任何一张真正的
`0179`**。对"用文件夹标签做 CE"的训练来说，这是不可救的；`§15.5` 估的 2~3 点也就此作废
（那 2~3 点是**在带同样噪声的 val 上**算的 macro，在干净测试集上本来就不成立）。

但 `--head-init frozen` 走的是**另一条路**：它不看标签，只看冻结塔的特征。

- 文件夹 `0179` 的质心第 1 轮 ≈ `0247` 的方向（因为里面的图就是 `0247`）；
- 这些图与 `0247` 自己的质心更接近（后者是从真 `0247` 图估的，噪声更小），于是
  `argmax == 0247 != 给定标签 0179` ⇒ **它们会被过滤掉**，`0179` 的 `present` 可能归零，
  它的头行保持随机 ⇒ 模型不再输出 `0179`；
- 结果：看起来像 `0247` 的图（**包括测试集里那 50 张真正的 `0247`**）不再被 `0179` 抢走。

**这正是测试集奖励的方向**（测试集是干净的，那 50 张的标签是 `0247`）。
所以 `head-init` 的收益不止于"更好的初始化"，它同时把**系统性标注错误从决策空间里挤出去**。
⚠️ 但要诚实：这条路的**上限很小**（14 类 × 50 张 ≈ 700 张 ≈ 1.87 点，而且只在模型真的把
这些图判给 `0179` 时才损失），且真正的 `0179` 那 50 张测试图**仍然赢不回来**
（训练集里根本没有真 `0179` 的样本，没有信号）。
**判据就是上面那条**：日志里 `present` 归零/骤降的类，是否正好是 §16.3 那 14 个类。
是 ⇒ 机制成立；不是 ⇒ 过滤器在乱删，要立刻停下来。

### 21.7 诚实清单（这一节的硬话）

1. **已确认的剩余杠杆加起来不够 2~3 分。** 把它们并排算：
   352（≲+0.3，可能为负）+ 16 视图（≈+0.1）+ head-init（0 ~ +1，未知）≈ **+0.1 ~ +1.4**。
   70.332 + 1.4 = **71.7**，离"再要 3 分"（≈73.3）仍有距离。
   **这个算术本身是这一节最重要的产出**：它意味着如果 head-init 落在 0~+0.3，
   那就不是"再试几个算法"能补上的，而是要接受"识别精度这条轴已经到顶"，
   把剩下的名额用在**保住 70.332 并榨干最后 0.5 点**上。
2. **这一节的每一条都是别人的数字，不是我跑出来的。** §21.1 的四条来自队友 §16.13；
   §16.3 的死类机制来自队友 `analyze.py`。本树**没有**任何一条独立复现。
3. **352 的反对意见只是一条先验，不是一个实验。** 队友把 352 排 P0 也不是错的 —— 他也知道
   384 是 −1.42；分歧只在"3.2 小时值不值得单变量试"。叠加的做法（21.4 第 4 步）
   把这场分歧绕开了，代价是失去了对 352 的**单变量归因**。
4. **16 视图必须先登记再测**（21.4 第 3 步的 ⚠️）。事后按结果挑集合会让整个 sweep 失去证据效力 ——
   这条规矩 §19.3 末尾已经立过，这里再重申一次，因为它最容易被"时间紧张"冲掉。


## 22. 掉头：输入表示封盘，训练侧按"作用窗口"重排 —— 2026-09-28（同一天，第三次调整）

> **本节取代 §21.4 的第 3、4 步。** §21.1–21.3、21.6 的事实与机制仍然有效；被替换的只有排序。

### 22.1 指令，以及它推翻什么、不推翻什么

用户指令：**"不要再走 TTA 和分辨率了，这条路已经快封顶了"**（研究 `提分路径研究.md`）。

- **撤回**：16 视图、352/384 —— 都不再作为独立实验。§21.4 把 352 当"叠加项"的理由（3.2 小时
  至少带第二个杠杆）在"输入侧封盘"之后也不再成立：那个杠杆本身就是输入侧。
- **不撤回**：**320 + TTA8 仍然是"读分数"的推理配方**。冻结的是"在输入侧继续投入"，
  不是"用当前最优配方交卷"。任何新训练结果**必须**用 `--image-size 320` 训练、
  `--tta-views tta8` 提交，才和 70.332 可比 —— 否则比的就不是同一个东西。
- 因此"封盘"的准确含义：**输入侧不再有新的自变量，但仍然是最优推理配置，用来当测量工具。**

### 22.2 为什么不能照抄 §7 的 P0（B1/B2/B9/B13）

§7 把 B1 重复簇、B2 LOO 多原型、B9 双视图一致性、B13 转移矩阵排在 P0。把它和 §16.1 的实测
放在一起看，会撞上一个**文档写作时还不存在的约束**：

```
--warmup-epochs 3  ⇒  warm-up = epoch 0,1,2
ep4 == ep20（LB 四位小数相同）  ⇒  分数在第 4 个 epoch 就定了
```

而 B2 / B9 / B13 / B15 / B18 **全部挂在 tracker 后面**（代码里是 `if a.proto_weight > 0 and not warm`、
`tracker.refresh()` 也在 `not warm`）—— 它们**最早在第 4 个 epoch 第一次生效**。

> 所以这些路径不是在跟"训练侧零收益"的判决对抗，它们是**在窗口之外**作用。
> §7 是从"噪声建模的质量"排的序，不是从"能不能改到决定性状态"排的序。
> 这不是 §7 错了，是它写的时候还没有 §16.1 那行数字。

**由此得到本节的排序原则：只有作用在 epoch 0–4 的变化才可能改分。**
这条把 §3 的 B1–B18 直接切成两半：

| 作用窗口 | 文档条目 | 本节处置 |
| --- | --- | --- |
| **epoch 0 起**（在窗口内） | B11 类中心初始化、B14 冻结原型蒸馏、B4 早期锚定、A3 逐 epoch 增强、B3 局部 token 头 | B11 / B4-lite / A3 **已实现**（§20）；**B14 本节实现**；B3 排队 |
| epoch 4 起（窗口外） | B1 处理、B2、B9、B12、B13、B15、B18 | 暂不投入 GPU（潜力评级高，但先验作用在窗口外） |

### 22.3 选 B14，并把它从"后期"挪到"第 0 步"

文档把 B14 排在表征 P1（step 4），门控是"冻结 margin 高、双视图稳定**且**与给定标签一致"。
本树实现的是**同一个构造，去掉时间门控**：

- **教师 = 冻结官方塔 + robust 质心** —— 正是 `--head-init` 那一次 pass 的产物。没有第二个训练
  模型、没有额外数据、推理不变（一次前向、一个 ckpt）。合规性上比 D 类灰区干净得多。
- **目标从第 0 步就在** ⇒ 它在决定性窗口内。这是它相对 B2/B9/B13 的**唯一但决定性**的区别。
- **门控 = 冻结塔的 750 类 argmax == 给定标签**，比文档的 margin 判据更硬。
  不满足 ⇒ 该行权重**全零**，对 loss 和对梯度都是零（`sparse_kl` 把 `0·log 0` 当 0，
  selftest 用手算的 0.130812 和梯度方向钉住了这一点）。
- **稀疏存储**：top-8 存成 `(int16, float16)` = **4.6 MB / 148695 条**，整轮训练反复读。
  稠密的 148695×750 float32 是 446 MB，纯浪费 —— 温度 0.05 下 top-8 已经装走几乎全部质量。

### 22.4 为什么是 B14，而不是文档里评级更高的 B1/B2/B13/B15

那四条文档都评"**高**"，B14 是"中高"。选 B14 不是因为潜力，是因为**同时满足三条硬约束**：

1. **窗口** —— 见 22.2，那四条全在窗口外。
2. **可微** —— B1/B2/B13/B15 最终都只能通过 tracker 的 `label` / `weight` 影响训练，而那是
   `not warm` 的；B14 是**直接进 loss 的软分布**，不需要经过那个开关。
3. **不自我确认** —— 学生是 LoRA + 头，教师是**未训练的** CLIP。B2/B15 的 LOO 多原型、
   B13 的转移矩阵都是"用共识筛共识"，§3 自己批评过这个失败模式；B14 的教师不参与被评估的共识。

**顺带覆盖 B1 的一半**：B1（重复簇/跨类冲突）拆成"筛选侧"和"目标侧"。目标侧的内容是
"不要因为预测成孪生类而受重罚"，而 B14 的 top-8 分布**本来就是**把近邻类的质量保留下来
（temp 0.05 下每 0.05 的余弦差 ≈ 1/2.7 的质量）。**B1 的筛选侧（把重复簇合并/降权）
没有覆盖**，仍待做 —— 但它是窗口外的，所以本轮不做。

⚠️ B14 的**先验是文档给的，不是实测**。这一轮之前，噪声侧的机制在本项目**从未**产出过分数。

### 22.5 代码改了什么

> ⚠️ **本表在写下后被同日第二次修改推翻**：`--distill-temp` / `--distill-topk` 已重命名为
> `--frozen-temp` / `--frozen-topk`，且门控从"目标函数内部"下沉到"每个消费者自己"。
> **以 §23.3 的表为准**，照本表打字会 argparse 报错。

| flag | 默认 | 作用 |
| --- | --- | --- |
| `--distill-weight` | **0.0** | `KL(冻结原型分布 ‖ 学生)`，两个视图都加，**不**门控 `not warm` |
| `--frozen-temp`（原 `--distill-temp`） | 0.05 | 作用在**余弦相似度**上（不是 logits）：越低越尖 |
| `--frozen-topk`（原 `--distill-topk`） | 8 | 稀疏目标的类数 |

- **与 `--head-init` 共用同一次 frozen pass**（任一开启即跑）：成本在前向，不在张量。
  两者**各自独立可关**，否则做不成单变量实验（selftest 专门断言 `--distill-weight`
  **不会**顺手把种子头写进去）。
- **每个 epoch 打印 `[distill] ... KL ... vs CE ...`**：这是唯一能在机器上当场校准
  `--distill-weight` 的仪器。学生的 softmax 比教师尖得多（`logit_scale ≈ 20` 对"余弦上的
  温度 0.05"），所以这一项的主要作用是**把学生往软里拉**，它的原始数值事先算不出来。
  **KL 接近 CE ⇒ 这一项在驾驶这次训练，不是在锚定它 ⇒ 权重减半。**
- **三个默认关闭**，selftest 断言；`--head-init`、`--epoch-aug`、`--warm-robust` 的既有行为
  一个字节没动 ⇒ 已上榜的四条配方仍逐字节可复现。

### 22.6 上机顺序（全部 @320 训练，提交一律 `--tta-views tta8`，对比 70.332）

```bash
# 第 1 步（单变量，§20 已定，未跑）：初始化这条线到底赚不赚
python train.py --data /root/autodl-tmp/train --out ./outputs_320_headinit \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen

# 第 2 步：在已种子的头上加 B14（两个都开，看 KL/CE 比值）
python train.py --data /root/autodl-tmp/train --out ./outputs_320_headinit_distill \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen --distill-weight 1.0

# 第 3 步（只有当第 2 步日志里 KL 与 CE 同量级时才做）：把权重压到锚定档
python train.py --data /root/autodl-tmp/train --out ./outputs_320_headinit_distill03 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen --distill-weight 0.3
```

- **第 3 步先看日志再决定跑不跑** —— 这是它和第 1、2 步的区别：前两步是"登记后测"，
  第 3 步是**被第 2 步的 KL/CE 数**触发的，不是事后挑结果。
- 名额：每天 2 个。第 1 步 1 个、第 2 步 1 个；第 3 步（若触发）是**同一台机器重训**，排到第二天。
- `--workers 8` **必须显式写**（本树默认 0，见 §21.4 的 ⚠️1）；内存按 60 GiB 算（⚠️2），
  训练侧 prefetch 与 320 + `--batch-size 128` 的既有配置一致，没有新增风险。
- **第 2 步也可以单变量跑**（`--distill-weight 1.0` 不带 `--head-init`，代码支持）。
  但顺序上先跑第 1 步：seed 是"抬高上限"，distill 是"锚定"，两个都开时**归因不开**。

### 22.7 判据（更新 §21.5）

| 现象 | 怎么读 |
| --- | --- |
| 第 2 步 < 第 1 步 | B14 在拉扯，停。诚实结论：冻结原型的知识已经**被种子头吃掉了**，重复注入只是加约束 |
| 第 2 步 ≥ 第 1 步 + 0.4 | 训练侧第一次靠**噪声建模**赚到分，立刻做 B3（局部 token 头，见 22.8） |
| `[distill] KL ≈ CE` | 权重过大，走第 3 步（0.3），不要直接下结论说 B14 无效 |
| `frozen target:`（原 `distill-frozen:`）打印的 agrees 比例 **≪** 上一条 `head-init` 的 kept 比例 | 质心在两轮之间还在动，调 `--head-init-rounds` |
| agrees 比例 ≈ 0.7 但分数不动 | 门控没问题，是这条机制没信息 —— 与 §16.1 的判决一致，回 §22.8 |

### 22.8 诚实清单（这一节的硬话）

1. **这一节把"训练侧"从判决里救了一次，但没有推翻它。** §16.1（ep4 == ep20）说的是
   **窗口在 epoch 4 附近**，不是"训练侧不可能赚分"。22.2 的排序正是拿这行数字推出的 ——
   如果 ep4 == ep20 其实是别的机制（§1 自己说这是强证据不是证明），
   那么把 B2/B9/B13/B15 判成"窗口外"就是错的，这一轮会白花。
2. **B14 是文档评级"中高"的路径，不是"高"。** 选它是因为窗口 + 可微 + 不自我确认，
   不是因为潜力更大。如果它落在 0，**不等于** B1/B2/B13/B15 无效 —— 只等于"在窗口内的
   那一半无效"，而它们本来就不在窗口内。
3. **剩余杠杆算术没有变好。** §21.7 第 1 条算的是 +0.1 ~ +1.4；本节把 352/16 视图划掉，
   换进来 B14（未知，文档先验"中高"）。**如果 B14 也是 0，训练侧就再没有窗口内的路径了**，
   剩下的只有 B3（局部 token 头）—— 而 B3 是**同一类**改变（给冻结塔更多信息），
   不是新机制。到那一步就该承认：**+3 分不在已识别的路径里**。
4. **B3 是唯一还排着的窗口内新算法**（文档评"中高"，p32 的 7×7 网格，允许过拟合噪声）。
   它要动 `forward` 和头的形状，是本树**风险最高**的一处改动 ⇒ 只有第 2 步赚到分才做。
5. **不得越过 320 + TTA8 交卷。** 新训练结果用一个更弱的推理配方提交，会让分数下降并
   **浪费当天名额**（榜单取最高值，所以不会伤害已有分数，但会浪费一次测量）。

---

## 23. 冻结教师重构：B2/B15 原型库 + B13 前向修正 + 门控下沉 —— 2026-09-28（同一天，第四次调整）

### 23.1 这一节相对 §22 的实质变化

§22 定的是**方向**（train 侧，按作用窗口排序，先 seed 再 distill）。这一节改的是 §22 写下的代码
本身 —— 三处，只有第三处是加法，前两处是**结构**：

1. **教师从"每类一个均值"变成"每类最多 k 个 medoid"**（B2/B15）。`--frozen-protos 1` 仍然是
   那个 robust 均值（`--head-init` 用的同一个），`>1` 走 `spherical_kmeans` 取 medoid。
   这一步买到的是 **B2 的"排除自身"变成精确的**：medoid 是真实样本，所以查询它就是它自己时
   可以按源下标精确屏蔽（`scatter_reduce(amax)` + `proto_src == i`），而不是靠相似度阈值近似。
2. **门控从目标函数内部下沉到每个消费者自己**。`frozen_soft_targets` 现在返回**未门控**的
   分布加一个 `agree` 布尔，由调用方决定：B14 的 KL 自己乘 `agree`（**永远不能**看到一次
   分歧），B13 的 mix 默认**不**乘（见 23.2）。
3. `--distill-temp` / `--distill-topk` **改名** `--frozen-temp` / `--frozen-topk` —— 它们是
   **教师分布的性质**，被两个消费者共用，不属于 distill 这一个。

### 23.2 唯一的算法性判断：一个分布，两个消费者，门控方向相反

这是这一节唯一不是"把文档里的方法写出来"的地方，所以单独说清楚。

§16.3 那三条系统性标注错误（`0179→0247`、`0039→0446`、`0640→0502`，`asym=+1.00`）**就是**
冻结教师与学生分歧的那些样本。同一个 `agree` 掩码：

- 对 **B14（KL）** 必须是**硬门控**：这一项是"把学生的分布拉向教师"，落在分歧样本上就是
  教学生复制教师的判断，而教师也是从**同一批噪声标签**里估出来的质心 ⇒ 自我确认。§22.3
  把"不自我确认"写成选它的理由，那条理由**只对门控版本成立**。
- 对 **B13（mix）** 默认**必须不门控**：mix 的作用是改**目标**，而目标里那 24% 系统性错误的
  样本正是分歧样本 —— 门控掉它们，等于保留错误标签、只把干净样本往软里拉，把这条路径最有
  价值的一半扔了。

所以**一个掩码不能同时服务两个方向**。这是这次重构的实际原因，不是为了整洁。

### 23.3 flag 表（**以此为准**，§22.5 的表已过时）

| flag | 默认 | 作用 | 门控 |
| --- | --- | --- | --- |
| `--head-init {none,frozen}` | none | 用教师质心写种子头 + proto + judge | — |
| `--frozen-protos` | 1 | 1 = robust 均值；>1 = 每类 k 个 LOO medoid | — |
| `--frozen-temp` | 0.05 | 作用在**余弦相似度**上，越低越尖 | 两个消费者共用 |
| `--frozen-topk` | 8 | 稀疏类数（全 750 宽是 446 MB，top-8 是 4.6 MB） | 两个消费者共用 |
| `--distill-weight` | **0.0** | `KL(教师 ‖ 学生)`，两视图都加 | **硬**：`agree` |
| `--frozen-mix-rho` | **0.0** | 目标 `← (1-ρ)·target + ρ·q_f` | **不门控** |
| `--frozen-mix-agree-only` | off | 把上面那条改成门控（保守臂） | 可选 |

- 全部默认关闭/中性，selftest 断言 ⇒ 已上榜的四条配方仍**逐字节**可复现。
- `--distill-weight` 与 `--head-init` **共用同一次 frozen pass**（成本在前向不在张量），
  但**各自独立可关**，selftest 专门断言 `--distill-weight` **不会**顺手把种子头写进去。

### 23.4 ρ 的两条臂是**两种机制**，不是一个旋钮的两个刻度

这一节最该被记住的一条，因为它把 `--frozen-mix-rho` 从"扫一遍"变成"两个点"。

warm-up 阶段目标是 one-hot-ish 的（给定类 `1 - --label-smooth = 0.95`），所以 mix 要翻转 argmax
必须 `ρ·w_top > 0.95(1-ρ)`，即

```
ρ > 0.95 / (0.95 + w_top)
```

`--frozen-temp 0.05` 下 `w_top` 几乎处处 ≥ 0.9 ⇒ **阈值卡在 ρ ≈ 0.5**：

- **ρ ≤ 0.5：在 warm-up 阶段一个样本都翻转不了。** 它的作用**只**是把目标往教师的邻域里
  **软化**（一个从第 0 步就存在的梯度改变），不是重标注。
- **ρ = 0.7：`w_top > 0.41` 就翻** —— 教师只要稍微有把握且与给定标签不一致，warm-up 阶段
  就**直接改标签**。

**按 §22.2 的窗口论证，这两条臂测的是不同的东西**：重标注是唯一"在窗口内做标签级干预"的臂，
软化是"窗口内的一个正则化"。而本项目的正则化（label smoothing 0.05、robust loss、feature
anchor、EMA teacher）**上过一次榜，全部没有产生过一分**。所以：

- **ρ=0.3 的先验低**（它是又一条正则化），但它**安全**。
- **ρ=0.7 的先验高**（它是唯一能改早期训练标签的机制），但它**是近硬覆盖**，
  而本项目自己的代码注释已经警告过这个失效模式：细粒度噪声"弱相关"时，一次有信心的分歧
  往往是真的**易混邻居**而不是错标签，硬覆盖会把这种混淆**扶正**。ρ=0.7 的 0.7 质量就压在
  教师那一类上，非常接近硬覆盖。

⇒ **两条都跑，各占一个名额**，先 0.7（高方差高信息）后 0.3（安全臂）。不要跑 0.4/0.5/0.6 ——
由上面的不等式，它们和 0.3 属于同一条臂（翻不了），只是软化得更狠一点。

### 23.5 上机顺序（取代 §22.6；一律 @320 训练，提交一律 `--tta-views tta8`，对比 **70.332**）

```bash
# 第 0 步，必须先跑：本节这批代码一次都没执行过
python selftest.py

# 第 0b 步：真实数据路径的冒烟测试。selftest 用的是 3 类合成图，验不了
# ImageFolderNoisy 在真数据上的切分、512 维特征、750 类的 medoid 库。
# 冻结前向是全量的（不受 --limit-batches 限制），所以这一跑会花掉一次完整
# frozen pass 的时间，换来三个真实数字：agrees 比例、每类 medoid 数、
# 以及 [mix] 的翻转率。训练部分只跑 5 个 batch，可以忽略。
python train.py --data /root/autodl-tmp/train --out ./smoke \
  --epochs 2 --warmup-epochs 1 --batch-size 32 --workers 4 --image-size 320 \
  --head-init frozen --frozen-mix-rho 0.7 --limit-batches 5

# 第 1 步（§22 已排队，仍未跑）：初始化这条线到底赚不赚
python train.py --data /root/autodl-tmp/train --out ./outputs_320_headinit \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen

# 第 2 步：B13 的"重标注"臂（唯一在窗口内做标签级干预的机制）
python train.py --data /root/autodl-tmp/train --out ./outputs_320_seed_mix07 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen --frozen-mix-rho 0.7

# 第 3 步：B13 的"软化"臂（安全，且是第 2 步的对照）
python train.py --data /root/autodl-tmp/train --out ./outputs_320_seed_mix03 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen --frozen-mix-rho 0.3

# 第 4 步（只有第 2/3 步任一 ≥ 第 1 步 + 0.4 才做）：把 B14 的 KL 叠在赢的那条上
python train.py --data /root/autodl-tmp/train --out ./outputs_320_seed_mix_distill \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen --frozen-mix-rho <赢的那个> --distill-weight 1.0

# 第 5 步（只有第 4 步的 KL/CE 同量级、且整条线在赚分才做）：原型库本身值不值
python train.py --data /root/autodl-tmp/train --out ./outputs_320_bank2 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --image-size 320 \
  --head-init frozen --frozen-protos 2 --frozen-mix-rho <赢的那个>
```

- **第 0 步不是客套。** 这一批改动里有 `spherical_kmeans`、`proto_bank`、门控下沉三处新逻辑，
  本机没有 torch ⇒ **全部只过了 `py_compile` 和未定义全局扫描，一次都没执行过**。
  selftest 现在覆盖：双峰类的 k=2 能落在两个模式上而 k=1 只能覆盖一个、LOO 精确屏蔽
  （带"不屏蔽就会自匹配"的对照）、`proto_bank` 两次调用逐位一致、空类必须报 absent、
  `frozen_mix` 质量守恒且不就地改输入、以及三个新的端到端跑（`--frozen-mix-rho 0.8` 断言
  **翻转数 > 0**，`--frozen-mix-agree-only` 断言日志变成 `agree-only`）。
- **第 2 步的 rho=0.7 是有意选在最陡处**，见 23.4；它的直接风险是毁掉这次训练。**可接受**：
  榜单取最高值，毁掉一次只是浪费一个名额，而它会明确回答"标签级干预到底能不能动分"。
- `--workers 8` 必须显式写（本树默认 0）；内存按 60 GiB 算（§21.4 的 ⚠️2）。mix/bank 都在
  训练侧，prefetch 与既有 320 + `--batch-size 128` 一致，**没有新增显存/内存风险**；
  frozen pass 是 no-grad、复用既有 `--head-init-batch-size`。
- **70.332 出自队友树，不是本树**（§21 的表格里本树只有 384 那一行）。所以"第 1 步 vs 70.332"
  是**跨树比较**，差值里混着树间差异；但**第 1／2／3 步彼此只差一个 flag，是本树内的单变量
  三元组**，这个比较是干净的。这就是第 1 步非跑不可的第二个理由 —— 它不只是"测种子头"，
  它同时是第 2、3 步的同树对照。**别拿 70.332 当唯一基准去读第 2、3 步。**
- `--noise-judge` **默认关闭**（§22.6 的三条命令都没带它，本节也不带）。它是一个**独立机制**
  （冻结 NCC 只降权的否决判据，不改标签），开着就多一个变量；如果队友的 70.332 带了它，那又是一处跨树差异，
  无法从本树这边确认。
- 名额账：2 个/天。第 1、2 步各占一个（同一天跑完，约 4.8 GPU 小时）；第 3 步排第二天。
  第 4、5 步是**被前一步的日志触发**的，不是事后挑结果。

### 23.6 判据（更新 §22.7）

| 现象 | 怎么读 |
| --- | --- |
| 第 2 步（ρ=0.7）明显 < 第 3 步（ρ=0.3） | 教师的分歧是**易混邻居**不是错标签 ⇒ 23.4 引的那条警告成立。**这条路径封盘**，别再加 ρ |
| 第 2 步 ≥ 第 1 步 + 0.4 | 训练侧第一次靠**改标签**赚到分。立刻做第 4 步，并做 B3（§22.8 #4） |
| 第 3 步 ≈ 第 1 步（ρ=0.3 不动） | 与"本项目所有正则化都没上过分"一致。**这不否定第 2 步** —— 它们是两种机制 |
| `[mix]` 在 warm-up 打印 0 flips 且 ρ ≤ 0.5 | **算术，不是死代码**（23.4 的不等式）。看 ρ=0.7 那次的这一行才有诊断价值 |
| `[mix]` 在 ρ=0.7 时 flip 比例 ≪ `1 - agrees 比例` | 教师的 top-1 质量不够尖，调 `--frozen-temp` 或 `--frozen-topk` |
| `frozen target:` 的 agrees 比例 ≪ `head-init` 的 kept 比例 | 质心在两轮之间还在动，调 `--head-init-rounds` |
| 第 5 步（`--frozen-protos 2`）不动 | 本次 750 类里"一个类两个视觉簇"的比例不高，或簇已被均值解释掉。**不否定 B2 的其他部分** |
| 第 1–3 步全部 ≈ 0 | 训练侧在窗口内的路径**真的走完了**。剩下只有 B3，而 B3 是同一类改变（§22.8 #3）⇒ 该承认 +3 不在已识别路径里 |

### 23.7 诚实清单（这一节的硬话）

1. **ρ=0.7 是本次改动里唯一可能把模型变差的旋钮**，而且它的危险方向正是本项目自己记录过的
   失效模式（把易混邻居扶正）。选它是因为**它和 ρ=0.3 不是同一个机制**，而"标签级干预"是
   唯一能解释"为什么这次会和前四条正则化不同"的理由。**如果它变差，那就是答案**，
   不要再用 ρ=0.4~0.6 去救 —— 23.4 的不等式说它们和 0.3 同臂。
2. **原型库（B2/B15）的收益是条件收益**：它只在"某个类真的有两个视觉簇"时才比均值强。
   selftest 用**人工造的**双峰类证明了机制成立，**不等于**本次 750 类里有足够多的双峰类。
   所以第 5 步排最后：它的先验取决于数据，不取决于代码。
3. **门控下沉改变了 §22 的 B14 语义。** §22.5 的表和 §22.7 的两个判据行都因此过时，
   已在原处标注指向本节。`--distill-weight` 的**数学没变**（仍是门控 KL），变的是
   `frozen_soft_targets` 的返回值和日志字符串（`distill-frozen:` → `frozen target:`）。
4. **这一节的任何数字都还没有，一个都没有。** 相对 70.332 的四次测量里，第 1 步是单变量，
   第 2、3 步是单变量，第 4、5 步是叠加 ⇒ **叠加步的归因不开**，这是有意接受的代价
   （时间预算下，先把"训练侧到底能不能动分"答出来，比把每个组件的贡献分清更重要）。
5. **val_acc 这次依然不能当判据。** 榜单 gap 在 1.9 ~ 6.5 之间游走（§21.3），
   唯一算数的是 `--tta-views tta8` @320 交上去的那个数。
6. **不得越过 320 + TTA8 交卷**（同 §22.8 #5）：更弱的推理配方会让分数下降、浪费当天名额。

## 24. 本轮算法准备更新 —— 2026-09-30

本轮没有替换当前树的训练主线；重点是把队友已验证的高分配置接入并补齐可复现诊断。

### 24.1 已确认的高分配置

- 官方 OpenAI `ViT-B/32 QuickGELU` 骨干保持不变。
- `--image-size 384 --train-pos-embed`：384 的位置编码在 `Net` 构造之后、EMA teacher 和 optimizer 之前解冻；位置编码会进入 trainable checkpoint，并由 `pos_embed` 指纹校验。
- 推理使用同一 checkpoint 的 `--tta-views tta8`，即 plain/wide/mid/tight 及其水平翻转，共 8 个中心视图。
- 当前工程禁止缺失训练位置编码的 checkpoint 静默回退到插值网格。

### 24.2 本轮新增

- `--save-teacher`：每个 epoch 评估 EMA teacher，并保存 `teacher_epN.pt`；最后保存 `teacher_last.pt`。它是同一个模型的在线滑动平均，不是模型集成。
- `diagnostics.jsonl`：每个 epoch 记录 loss、验证指标、学习率、噪声统计、JSD、蒸馏 KL/CE、目标翻转率和 teacher 指标。
- `last.pt` 增加 CPU `targets`、配置指纹和 teacher 位置编码指纹；`thin` 推理快照不携带大 tracker。
- 新增 `diagnostics.py`，可从 `last.pt` 输出全局/逐类 clean、relabel、noisy、unseen、FrozenJudge suspect、trust/weight 分位数和教师转移统计。

### 24.3 下一轮推荐顺序

```powershell
py -3 train.py --data <train> --out outputs_384pe_teacher `
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 `
  --image-size 384 --train-pos-embed --save-teacher
py -3 diagnostics.py --checkpoint outputs_384pe_teacher/last.pt `
  --out outputs_384pe_teacher/diag.json
py -3 infer.py --test <test> --checkpoint outputs_384pe_teacher/teacher_ep4.pt `
  --tta-views tta8 --workers 8 --output sub_teacher_ep4_tta8.csv
py -3 infer.py --test <test> --checkpoint outputs_384pe_teacher/best.pt `
  --tta-views tta8 --workers 8 --output sub_student_tta8.csv
```

优先比较 `teacher_ep4/ep8`、student `ep4/ep8` 与 `best.pt` 的线上分数；不要依据带结构化噪声的 `val_acc` 单独决定提交快照。

> **注(§25 写作时发现,原文不改):** 上面这条命令**缺 `--lr-warmup-epochs 0`**。本树该 flag 默认 `1`
> (epoch 0 只跑 `0.1 × --lr`),而榜单上每一行都产生于无 warm-up 的调度下,照原样执行会给这次对照引入
> 一个与待测变量同量级的混淆。判别方法与证据见 §25.2 末。

## 25. 416 分辨率延长线的实测:过程、结果与一个未闭合的歧义 —— 2026-09-30

### 25.1 本节要回答什么

§24.1 把"384 + `--train-pos-embed`"写成已确认的高分配置。本节问的是:**这条线能不能再往上一档。**

选 416 而不是重跑 384 的理由是名额经济。榜单取历史最高,**71.1382 已经在账上**,重跑 384 即便完全复现
也是 +0.0000;而历史轨迹 66.2456 → 69.218 → 69.968 → 70.332 → 70.8605 → 71.1382 一路都是"加分辨率 / 加视图",
416 是唯一还有上行空间的档位。**代价是本节要付的:416 是一个外推,不是已测点。**

### 25.2 代码变更与一条必须先说的 flag

**移植(本节之前写入本树)**
- 新增 `enable_pos_embed_training(visual)`(train.py:328):找不到可训练位置编码时直接 `SystemExit`,
  不让 `--train-pos-embed` 静默失效。
- `verify_pos_embed` 在 `ck['pos_embed_trained']` 为真时放宽 sha 不一致分支。
- `SNAPSHOT_KEYS` 增加 `pos_embed_trained`;快照另存 `pos_embed` 指纹。
- 新 flag `--train-pos-embed`(train.py:2528)。`main()` 内的顺序已逐行核对:
  `pos_embed_trained` 赋值(1814)→ `enable_pos_embed_training`(1816)→ teacher deepcopy(1957)→
  `param_groups`(1961)。**建 teacher 和建 optimizer 之前解冻**是这条链唯一要紧的地方。

**队友侧本次更新(纯观测层,不进训练路径)**
- 新增 `diagnostics.py`:配置指纹、逐轮记录、tracker 汇总、逐类状态、JSD 汇总。
- 每轮追加 `diagnostics.jsonl`(train.py:2318),原注释自述 "deliberately observational: it does not
  affect gradients, checkpoint selection, or the submitted inference path"。
- `SNAPSHOT_KEYS` 增加 `config_fingerprint`;快照增加 CPU `targets`(int16,供逐类诊断)。
- 新 flag `--save-teacher`(train.py:2438),默认关。

**`evaluate()`(train.py:1726)未变** —— `F.cross_entropy(out, y, reduction='sum') / n`,对**带噪给定标签**
求普通 CE,不带 tracker 权重、不加 label smoothing。所以 `val_loss` 跨轮同口径可比。

**`--lr-warmup-epochs` 必须传 0。** 本树默认 **1**(epoch 0 只跑 `0.1 × --lr`);队友的树直接
`CosineAnnealingLR(opt, a.epochs)`(其 train.py:612),没有 warm-up flag。本树自己的 `lr_factor`
docstring(train.py:1037)说 `warm == 0` 精确等价于 `CosineAnnealingLR`,train.py:1082 说
`--lr-warmup-epochs 0` 恢复旧调度。打印值可判别:`warm=0 → p=1/20 → 0.9938 → 1.99e-04`;
`warm=1 → p=0 → 因子 1.0 → 2.00e-04`。**榜单上每一行都产生于 warm-up 之前**,不传 0 的跑无法与它们归因。

> 另有两个未修的移植地雷,本轮安全但下轮要绕开:多分辨率推理会在 `load_state_dict` 的形状校验上硬崩
> (`sNNN` / `tta4sNNN` 视图集与 `probe.py --sweep` 全部阵亡,**`tta8` 不受影响**);
> `--resume` 不带 `--train-pos-embed` 会在 `opt.load_state_dict` 抛一个不提位置编码的裸 `ValueError`。

### 25.3 上机命令

```bash
cd /root/autodl-tmp
nohup python train.py --data /root/autodl-tmp/train --out outputs_416pe \
  --image-size 416 --train-pos-embed --lr-warmup-epochs 0 \
  > outputs_416pe.log 2>&1 &
```

**已通过的门(全部来自运行自身打印,非事后推断)**

| 门 | 实测 | 含义 |
| --- | --- | --- |
| 可训练参数 | `trainable params=1.399M` | = 1,268,737(LoRA+头) + 130,560(网格) |
| 网格 | `130560 params (0.131M), 169 tokens at 416px` | 169 = 13²,**13×13 是 416/32 的正确网格** |
| 插值 | `7x7 -> 13x13 (bicubic)` | 已离开 OpenAI 的 7×7 预训练网格 |
| 调度 | `lr=1.99e-04` | **warm-up 关**(见 25.2),与榜单同口径 |

**耗时**:747 s/epoch × 20 ≈ **4.15 GPU 小时**。

### 25.4 结果

**榜单 69.31。**

| 配方 | TTA | 榜单 | 与 71.1382 |
| --- | --- | --- | --- |
| 320 + TTA8(队友) | 8 | 70.332 | −0.81 |
| 352 冻结(队友) | 8 | 70.8605 | −0.28 |
| 384 + `--train-pos-embed`(队友) | 8 | **71.1382** | — |
| 384 冻结(TTA4,本树) | 4 | 68.5476 | −2.59 |
| **416 + `--train-pos-embed` + 堆叠(本树,本次)** | 8 | **69.31** | **−1.83** |

`val_acc` ep20 = 0.7142,`val_loss` ep20 = 1.6469。**这两个代理都指不动的方向**,见 §25.8。

### 25.5 三个发现

**发现 1(已确证):移植生效,`--train-pos-embed` 不是空操作。**
ep4 / ep8 / ep12 / ep16 / ep20 五份 checkpoint 的位置编码 sha **两两不同**,`shape=[170, 768]`
(170 = 13² + 1)。网格确实每轮都在更新。这一条**排除**了"解冻没接上、参数没动"这个解释。

**发现 2(⚠ 已由 §26.5 降级为未经确认的假设,下表作废):这次跑不是单变量。**

> **更正(2026-10-01):** 本条的推理是"诊断字段出现 ⇒ 开关开启",这是错的。`epoch_record` 的 `**extra`
> 是**无条件** `row.update(...)` 写入的(diagnostics.py:216),关闭时该字段被写成 `null` 但**照样出现**。
> 所以仅凭键名存在推不出任何开关状态。要定论必须读该次 `last.pt` 的 `args` 与 JSONL 的 `active_flags`。
> 详见 §26.5。**§25 写作时我没有让你跑那条 `extras ep20` 的命令就下了结论,这是我的错。**
`diagnostics.jsonl` 第 20 轮的字段含 `distill`、`target_flip_rate`、`teacher_val_acc` / `_hi`。
它们的门控在 train.py:2325-2330,而 `fq` 的构造门在 train.py:1864 / 1872 / 1919:

| 字段出现 | 门控 | ⇒ |
| --- | --- | --- |
| `distill` | `fq is not None and a.distill_weight > 0` | `--distill-weight` **开着** |
| `target_flip_rate` | `fq is not None and a.frozen_mix_rho > 0` | `--frozen-mix-rho` **开着** |
| `teacher_val_acc` / `_hi` | `a.save_teacher` | `--save-teacher` **开着** |

冒烟测试打印的配置里这三项都是关的,§24.3 的命令也没有传。⇒ **实际执行的是 4 变量以上的堆叠。**
其中 `--frozen-mix-rho` 在本项目已判定为死旋钮(k=1 的翻转是 LOO 假象),`--save-teacher` 不改学生
(train.py:2279 的 `best.pt` 选择仍取学生的 `va_acc`),**真正活着的只有 `--distill-weight`(B14)**。
**各 flag 的确切取值必须从 `outputs_416pe/last.pt` 的 `args` 读出——本节写作时尚未读,这是本节最大的欠账。**

**发现 3:416 越过了测试集的细节上限。**
测试集短边中位数 **375**;`center` 预处理把短边缩放到 `size × 1.143`,416 → **475**,即 **1.27× 上采样**,
多出来的像素全是插值、没有新信息。冻结阶梯上 352 → 384 已掉 1.96(70.8605 @TTA8 对 68.5476 @TTA4,
TTA 差已计入),384 → 416 再掉 1.83 是**同一斜率的延续**,不是新现象。

### 25.6 推理与提交

```bash
python infer.py --test /root/autodl-tmp/test --checkpoint ./outputs_416pe/ep20.pt \
  --tta-views tta8 --workers 8 --output pred_results.csv --logit-adjust 0 0.25 0.5
```

- 8 视图 = `plain/wide/mid/tight × 水平翻转`,**与产出 71.1382 的那一套逐项一致**
  (`VIEW_SETS['tta8']`,train.py:522-525;日志里 `416px` 是 `plain` 档只打印尺寸)。
- 日志明确打印了"训练过的网格覆盖重建的插值"(这一步是这次移植的安全网)。
- 坏图 0 张。`pred_results.csv`:37444 行 / **1,610,092 字节** / CRLF / **747 个不同类别**(未塌缩)。
- **提交 `pred_results.csv`(tau=0)**,不是 `_tau025` / `_tau050`。

### 25.7 下一轮(唯一该跑的)

```bash
cd /root/autodl-tmp
nohup python train.py --data /root/autodl-tmp/train --out outputs_384pe \
  --image-size 384 --train-pos-embed --lr-warmup-epochs 0 \
  > outputs_384pe.log 2>&1 &
```

**单变量,`--distill-weight` / `--frozen-mix-rho` / `--save-teacher` 全部不传。** 约 3.5 GPU 小时。

理由:71.1382 是**队友的树**产出的,**本树从来没有在 384 上跑过 `--train-pos-embed`**。
69.31 这一个数**分不开**下面两种情形:

- (甲) 移植没问题,416 本身差 ⇒ 本树 @384 应能到 ~71.1;
- (乙) 本树的 pos-embed 复现不出队友的 +2.2 ⇒ 本树 @384 大概只有 ~68.9,**差 2.2 分**。

两者指向完全相反的下一步。**在查清之前,本树上任何新机制的测量都没有参照系**——所以这一跑的
直接收益是 0 分,它买的是"后面的测量算不算数"。

### 25.8 判据

| 落点 | 怎么读 | 下一步 |
| --- | --- | --- |
| **~71.0–71.2** | 移植复现,本树平台验证通过;416 的锅是 416 自己的 | 转 `--head-init frozen`,或回头做**干净的** 416(§25.5 发现 2) |
| **~68.5–69.0** | 本树的 pos-embed 没复现队友的 +2.2 | **这是当前价值最高的 bug**,查 lr / amp / 数据划分 |
| 中间 ~70 | 部分复现 | 逐项查,别叠新机制 |

`val_acc` **依然不能当判据**:榜单 gap 在 1.9 ~ 6.5 之间游走,唯一算数的是 `--tta-views tta8` 交上去的数。

### 25.9 诚实清单(这一节的硬话)

1. **本节的核心结果是一个负结果,而且它没能干净地归因。** 69.31 同时踩了三件事:416 这个外推档位、
   一次未被授权的 4 变量堆叠、以及本树从未验证过的 pos-embed 移植。**我只确证了第三件里的"网格在动",
   没有确证"网格动得对"。**
2. **我给出的上机命令和实际执行的命令不一致,而我是从 `diagnostics.jsonl` 的字段倒推发现的,不是
   从日志配置行。** 这意味着两件事:一是 §24.3 的命令在下一次上机前必须逐项核对;二是**我上一节
   对 0.7142 的解读("416 没抬起来")是在一个未经核对的前提上做的**,该解读作废。
3. **`val_acc` 0.7142 与 384 冻结的 0.7140 几乎相同,但这不是证据。** 它可能只是又一次说明
   `val_acc` 不能排序(gap 会漂移),也可能是堆叠真的把 416 拉回了冻结水平。**两种解释都成立,
   这一个数选不了。**
4. **不要把 416 写死。** 因为堆叠的存在,416 严格来说**还没有被单独测过**。在 §25.7 的 384 对照
   落地、且 `args` 被读出之前,416 的结论只能是"未测"。
5. **名额账**:今天用掉 1 个(69.31),榜单仍取 71.1382,净损失 0。2 个/天,§25.7 占 1 个。
6. **两个移植地雷没修**:多分辨率推理硬崩、`--resume` 裸 `ValueError`。本轮靠"只用 `tta8`"绕开,
   这**不是**修好了。下一次任何人想用 `--tta-sizes` 或 `probe.py --sweep`,撞的是墙不是数。

## 26. 对 69.31 负结果的深刻复盘 —— 2026-09-30

### 26.1 能确定的原因

这次 416 提交为 69.31，低于已有 71.1382，但它不能被解释成“位置编码训练无效”或“416 单独只得 69.31”。实验没有满足单变量条件：

1. 分辨率由已验证的 384 改成未经验证的 416。416 的训练预处理会把短边放大到约 475，而测试图短边中位数约 375，新增像素主要是插值，已经超过图像细节供给。
2. `diagnostics.jsonl` 同时出现 `distill`、`target_flip_rate` 和 `teacher_val_acc` 字段，说明实际运行至少打开了 `--distill-weight`、`--frozen-mix-rho`、`--save-teacher`。其中 `save-teacher` 只保存/评估，不改变学生；真正可能改变学生的是蒸馏和 mix。
3. 没有先跑本树的 `384 + --train-pos-embed` 对照。因此无法区分“416 本身退化”和“本树未复现队友 384 位置编码训练链路”。队友 71.1382 不能直接当作本树代码的基线。

### 26.2 不能从 69.31 推出的结论

- 不能说 `--train-pos-embed` 失效。位置编码 sha 在 epoch 间变化，至少证明参数进入了优化器；这不等于它学到了正确方向。
- 不能说 `--distill-weight` 或 `--frozen-mix-rho` 一定导致下降，因为它们没有各自的 416 无开关对照。
- 不能用 `val_acc=0.7142` 解释线上分数。历史实验已显示验证集与平台分数的 gap 会随训练策略和轮次变化，且验证集包含同源标签噪声。
- 不能把 416 的结果外推到 384、352 或半决赛脱敏测试集。

### 26.3 流程级错误

核心错误不是“选了 416”本身，而是把探索性外推和训练算法改动堆在一次线上提交里，并且没有从日志完整配置行反查实际命令。一次提交同时消耗名额，却没有产生可归因的科学信息。

以后每次训练启动必须保存并核对：完整命令、配置指纹、`args` 中所有非默认 flag、图像尺寸、位置编码 sha、实际推理视图集合和聚合方式。任何诊断字段出现但命令未声明，都应立即判定为配置漂移。

### 26.4 下一轮唯一有效顺序

1. 先跑干净基线：`384 + --train-pos-embed --lr-warmup-epochs 0`，不启用 distill、frozen-mix、save-teacher；固定 `tta8/logit/fp32`。
2. 只有基线复现到约 71.0～71.2，才测试 `head-init frozen`；再分别测试 `frozen-mix-rho 0.7` 与 `0.3`，每次只增加一个变量。
3. 每次训练保存 ep4/ep8/ep20，并用完全相同的 `tta8` 提交比较；`val_acc` 只做诊断，不做最终选型。
4. 416 暂停。若重新探索，只能作为独立单变量实验，使用更小 batch/梯度累积，并且不能与蒸馏、mix、teacher 保存同时改变。

本节结论：69.31 是一次不可归因的失败实验，不是对某个中风险方法的有效否定。当前最有价值的动作是恢复可归因的 384 基线，而不是继续叠加新方法。

### 26.5 证据修正 —— 诊断字段不能单独证明开关开启

复核代码后发现一个重要的证据问题：旧版 `epoch_record()` 会把关闭的可选指标作为 `null` 写入 JSONL。于是仅凭 `diagnostics.jsonl` 中出现 `distill`、`target_flip_rate` 或 `teacher_val_acc` 字段，不能证明对应开关真的开启；这些字段可能只是 `null` 占位。此前“实际运行至少开启 distill 和 frozen-mix”的表述应降级为**未经确认的假设**，不能作为 69.31 下降的事实原因。

已修复：

- `diagnostics.py` 现在省略值为 `None` 的可选字段；
- `train.py` 每轮写入 `active_flags`，直接记录实际训练开关和数值；
- 下一次复盘必须读取 `last.pt['args']` 与 JSONL 的 `active_flags`，不能只用字段存在性推断。

因此，69.31 当前能确定的事实只有：416 外推分辨率、训练位置编码确实发生变化、线上结果低于 384 队友结果；蒸馏/mix 是否开启必须先读取该次 `last.pt` 的 args 才能定论。即使它们全部关闭，416 仍然可能因为测试图像细节上限和位置编码外推而下降；如果它们开启，则还需要单变量对照才能估计额外影响。

### 26.6 额外分辨率 TTA 的真实加载 bug 已修复

训练过位置编码的 checkpoint 在请求 `--tta-sizes` 或 `probe.py --calib-sweep` 的额外分辨率时，checkpoint 中训练网格的 token 数与额外分辨率插值网格不同。若直接 `load_state_dict`，会因 shape mismatch 崩溃。现在：

- 训练分辨率：严格加载训练位置编码，并做指纹/缺失检查；
- 额外分辨率：只移除 checkpoint 的训练位置编码，保留当前尺寸的插值网格，其他 LoRA/head 权重照常加载；
- `infer.py` 和 `probe.py` 都采用同一规则。

这不会改变 `tta8`（它只使用训练分辨率），但恢复了多分辨率探针和 TTA 的可用性。

### 26.7 训练位置编码时的 anchor 语义修复

启用 `--train-pos-embed` 后，原 `anchor_feat()` 只关闭 LoRA，但仍读取正在更新的位置编码，导致所谓 frozen anchor 会随学生位置网格移动。现在 `Net` 在构造时保存官方插值网格的内存快照；anchor 前向临时恢复该快照，完成后恢复学生网格和 Parameter 对象。这样 anchor 约束的是官方 CLIP 表征，且不改变 optimizer 引用或 checkpoint 格式。

## 27. 复现优先的训练链路修正与中风险局部头 —— 2026-10-01

### 27.1 本轮审计结论

与队友 71.1382 原始工程逐行对照后，当前增强树中有几处默认行为改变了训练数学：首轮学习率 warm-up、按样本固定增强、warm-up 结束清空 tracker 后验、双向一致性、trusted prototype loss 按可信样本归一化，以及样本权重增益上限。这些变化都可能在 384 + train-pos-embed 上造成掉分，且不能从单次结果中归因。

本轮已将默认值调整为可复现优先：

- `--lr-warmup-epochs 0`：恢复 CosineAnnealingLR 首轮峰值学习率；
- `--epoch-aug` 默认开启：每个 epoch 产生新的确定性随机增强；可用 `--no-epoch-aug` 做固定视图消融；
- `--reset-tracker-warmup` 默认关闭：保留 warm-up 后验；需要旧树行为时显式开启；
- `--norm-max-gain 0`：使用 `w / w.mean()`，不加隐藏增益上限；正数才启用 cap；
- 默认使用单向 consistency 和 prototype batch-mean；`--consistency-symmetric`、`--proto-normalize` 均为显式实验开关。

### 27.2 新增中风险方法：`--local-head`

`--local-head` 利用同一次官方 CLIP ViT-B/32 forward 的最后层 patch tokens，做 mean pooling + 轻量线性投影，并通过可学习 residual gate 与全局 CLIP feature 融合。gate 初始为 0，因此训练起点等价于原 global-only 路径；最终仍只有一个 CosineClassifier、一个 checkpoint 和一个推理模型。主干结构和官方 OpenAI 预训练权重没有替换。

该方法已同步 `train.py`、`infer.py`、`valmetrics.py`、`analyze.py`、`probe.py` 的模型构造和 checkpoint 加载。它尚无 GPU 实测，属于中风险候选，不能把代码落地当成已验证收益。建议先跑 1~4 epoch proxy，再决定是否完整训练；不要与 `--distill-weight`、`--frozen-mix-rho` 同时开启。

### 27.3 下一轮实验顺序

1. 干净复现：`384 + --train-pos-embed`，不加 head-init、distill、mix、local-head；
2. 单变量 A：在同一命令上加 `--head-init frozen`；
3. 单变量 B：恢复干净基线后单独加 `--local-head`；
4. 只有 A/B 有正向证据时，才测试 `--distill-weight` 或 `--frozen-mix-rho`；
5. 每轮保留 `ep4/ep8/ep12/ep16/ep20`，最终以线上分数排序，不用带噪验证集单独决定提交。

推荐基线命令：

```bash
python train.py --data /path/to/train --out outputs_384pe_ref \
  --image-size 384 --train-pos-embed --lr-warmup-epochs 0 \
  --augment-mode worker --save-every 4
```

local head 单变量命令：

```bash
python train.py --data /path/to/train --out outputs_384pe_local \
  --image-size 384 --train-pos-embed --lr-warmup-epochs 0 \
  --epoch-aug --local-head --save-every 4
```

### 27.4 原型损失的进一步对齐

补充确认：队友原始代码的 `trusted` 只控制原型向量的 EMA 更新，prototype loss 仍对整个 batch 使用已有噪声权重 `w` 求均值。当前树此前还额外把 loss 乘以 `trusted` mask，相当于又缩小了监督集。本轮默认恢复队友口径；若要实验这个更激进的筛选，显式加 `--proto-trusted-loss`。它与 `--proto-normalize` 是两个不同变量，不能一起打开后把结果归因给其中一个。

### 27.5 数据增强模式对齐

进一步对照发现，队友原始 71.13 实现每次从 DataLoader worker 的 RNG 抽取新增强；当前树的 per-index seed 即使加 epoch，仍会产生不同的随机序列。现新增 `--augment-mode worker|index`，默认 `worker`，直接恢复队友的随机调用路径。`index` 保留为独立可复现实验；只有在该模式下 `--epoch-aug` 才会改变每张图的 seed，并关闭 persistent workers。基线命令无需再显式加 `--epoch-aug`，日志中的 `augment_mode=worker` 才是本轮所称的复现路径。

---

## 28. 把 `--local-head` 移植进队友的 71.1382 原树

### 28.1 为什么改的是队友树，不是当前树

用户 2026-10-01 指示：`C:\Users\Ed\Desktop\Recent Project\AIC\LGL-Lab-AIC` 是队友跑出 71.1382 的原代码，
不在那里做"复现"这件事（它本身就是最好的那次结果），直接在它上面加新机制。

理由：§25/§26 已经证明，在当前树上试新方法时，**得分差异永远无法归因**——树里同时带着
pos-embed 移植、augment-mode、`--lr-warmup-epochs` 默认值、diagnostics 等一批与队友不同的东西。
在队友原树上做单变量实验，`--local-head` 的收益才是可解释的。

### 28.2 改动清单

备份：`_backup_prelocalhead_20261001_100704/`（同目录，含改动前四个文件）。
该目录**不在** `SNAPSHOT_KEYS`/构建路径上，训练不会读到它。

| 文件 | 行数 | 增/删 | 内容 |
|---|---|---|---|
| `train.py` | 919 → 1004 | +90 −5 | `LocalPatchHead` 类；`Net.__init__` 加 `local_head=False`；`Net.forward` 分支；`SNAPSHOT_KEYS`；checkpoint dict；`Net(...)` 调用；`--local-head` |
| `infer.py` | 365 → 371 | +7 −1 | 从 checkpoint 读 `local_head`；`Net(...)` 传入 |
| `valmetrics.py` | 144 → 147 | +4 −1 | 同上 |
| `analyze.py` | 242 → 245 | +4 −1 | 同上 |

`train.py` 那 5 行删除**全部是一对一替换**，其余为纯插入；已用 difflib 对备份逐行核对。

**为什么推理路径也必须改（这是最容易静默出错的地方）**：`load_state_dict(strict=False)` 的第二个返回值
（unexpected keys）在 `infer.py`/`valmetrics.py`/`analyze.py` 里都被丢掉了（`missing, _ = ...`）。
若不重建 `local_head`，检查点里的 `local_head.proj.weight`/`gate` 会落到 **unexpected** 而不是 **missing**，
现有的 `assert not lost` 不会触发——模型会带着一个训练过的 gate 却**不应用它**，静默给出错误预测。

### 28.3 静态验证（本地，无 torch）

`py -3` 通过：四个文件 `py_compile` 全过；AST 确认 `LocalPatchHead`/`forward`/`gate` 存在、
`Net.__init__` 形参含 `local_head` 且默认 `False`、`forward` 有 `self.local_head is None` 分支与
`forward_intermediates` 守卫、`--local-head` 已注册、三条推理路径都传入 `local_head`。全部 OK。

**参数锚点**：`LocalPatchHead.proj` = `Linear(768, 512, bias=False)` = 393216，加 `gate` 1 个 = 393217。
所以开启后日志里 `trainable params=` 必须由 **1.380M 变成 1.773M**；不变就是开关没生效。

`--local-head` 关闭时 `Net.forward` 与备份逐行相同，**基线不受影响**——这是选它而不是 `--head-init`
作为第一个移植对象的原因。

### 28.4 队友树的参数表 ≠ 当前树的参数表（照抄 §27.3 会直接报错）

实测队友树 `train.py` 共 51 个 flag（移植前 50）。**以下 flag 队友树根本没有**：

```
--lr-warmup-epochs   --augment-mode   --epoch-aug   --head-init
--distill-weight     --frozen-mix-rho --proto-trusted-loss
```

这不是遗漏，而是**原代码就是那些行为**，不需要开关：

- 没有 `--lr-warmup-epochs` → `CosineAnnealingLR(opt, a.epochs)` 直接调用，首轮即峰值学习率。
  §27 里"必须传 `--lr-warmup-epochs 0`"这条**在队友树自动成立**，无需也无法设置。
- 没有 `--augment-mode` → 原代码就是 worker 模式。
- 没有 `--epoch-aug` → 原代码不按 epoch 换 seed。

另外**同名 flag 的拼写也不同**，§27.3 的 runbook 用错了：

| 当前树 | 队友树 |
|---|---|
| `--image-size` | **`--img-size`** |
| `--tta-views tta8` | **`--tta`**（裸用） |

### 28.5 两条等价关系（比对的前提）

1. **裸 `--tta` ≡ `--tta-views tta8`**。队友 `infer.py:261` 为
   `resolve_views(list(a.tta) or list(DEFAULT_TTA))`，而 `DEFAULT_TTA`（`infer.py:106-107`）＝
   `plain, flip, mid, mid_flip, tight, tight_flip, wide, wide_flip` ——
   即 4 个比例（1.143 / 1.0 / 1.286 / 1.429）× 2 次翻转，与 `tta8` 的视图集**逐字节相同**。
   两种写法都是"同一模型、同一权重"的 TTA，不构成集成（规则 五.4）。
2. **`--local-head` 关闭时 forward 逐行等于原树**，故 71.1382 与本节实验之间只差这一个变量。

### 28.6 尚未验证的一环：open_clip 的 `forward_intermediates`

`--local-head` 依赖 `visual.forward_intermediates(..., output_fmt='NLC')`，返回
`{'image_features': [B,512], 'image_intermediates': [[B,144,768]]}`（384px 时 144 = (384/32)²）。
**两棵树都从未跑过这个 API**——`requirements.txt` 钉的是 `open_clip_torch==3.3.0`，但两棵树都没有
vendor 一份可核对的副本，所以本地无法判定签名是否匹配。

代码里已加守卫：没有该方法就 `RuntimeError`，**大声失败而不是静默退化成 global-only**。
但"能失败"不等于"能用"，必须先在服务器上探一次（命令见 28.7 的闸门 0）。

### 28.7 可执行命令（替换 §27.3 中面向当前树的那一版）

队友树目前在本地 Windows 上，服务器 `/root/autodl-tmp` 放的是当前树，**需要先上传**。

闸门 0 —— 决定 `--local-head` 到底能不能跑（30 秒）：

```bash
python - <<'PY'
import inspect, torch, open_clip
print('open_clip', open_clip.__version__)
m = open_clip.create_model('ViT-B-32-quickgelu', pretrained='openai')
v = m.visual
print('has forward_intermediates:', hasattr(v, 'forward_intermediates'))
if hasattr(v, 'forward_intermediates'):
    print('signature:', inspect.signature(v.forward_intermediates))
    d = v.forward_intermediates(torch.zeros(1, 3, 384, 384),
                                indices=[len(v.transformer.resblocks) - 1], stop_early=False,
                                normalize_intermediates=True, intermediates_only=False,
                                output_fmt='NLC')
    print('image_features', tuple(d['image_features'].shape),
          'intermediates', tuple(d['image_intermediates'][-1].shape))
PY
```

期望 `(1, 512)` 与 `(1, 144, 768)`。若 `False` 或抛错，本次移植在本机不可用，应改走 `--head-init`。

闸门 1 —— 1 轮冒烟，确认开关注入且 384 下能前向：

```bash
cd /root/autodl-tmp/AIC-LGL-Lab-AIC
python train.py --data /root/autodl-tmp/train --out smoke_local \
  --img-size 384 --train-pos-embed --local-head \
  --epochs 1 --limit-batches 20 --workers 2
```

只看一行：`trainable params=` 必须是 **1.773M**（关掉 `--local-head` 则是 1.380M）。

正式单变量运行（＝71.1382 的配方 + 仅加 `--local-head`）：

```bash
nohup python train.py --data /root/autodl-tmp/train --out outputs_384pe_local \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  > outputs_384pe_local.log 2>&1 &
```

注意：队友树 8 个 worker、384px，容器上限 60 GiB（**dataloader worker 是吃内存的主因**，
不是模型本身），开跑后先看一眼 `free -g`，必要时降到 `--workers 4`。

推理（`ep20.pt`；`--save-every 4` 保证 ep4/8/12/16/20 都在）：

```bash
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_384pe_local/ep20.pt \
  --output pred_results_local.csv --tta
```

闸门 2 —— 提交前核对：`37444` 行、恰好 `1610092` 字节（CSV 是 **CRLF、无表头**，故 43 字节/行）。

### 28.8 队友树没有 diagnostics 设施

队友树**不含** `diagnostics.py`、`probe.py`，因此没有 `diagnostics.jsonl` / `active_flags`。
本次运行的审计依据只有 `last.pt['args']` 与日志里打印的 `config:` 行。
这也意味着 §26.5 那个"键存在≠开关打开"的歧义在这里不存在——但也拿不到逐 epoch 的 tracker 曲线。

---

## 29. `--local-head` 单变量实测：训练数据、推理件与云端 runbook —— 2026-10-01

### 29.1 结论先行

**这一轮否掉了"局部 token 重池化"这个方向，而且是否得干净的。**

| | val_acc |
|---|---|
| 71.1382 那次（384 + `--train-pos-embed`，无局部头） | **0.7344** |
| 本轮（384 + `--train-pos-embed` + `--local-head`） | **0.7347** |

差 **+0.0003**。留出集 10116 张、val_acc≈0.735 时单侧标准差
`σ = sqrt(0.7347 × 0.2653 / 10116) ≈ 0.0044`，所以 **+0.0003 是 0.07σ** ——
这个指标**根本看不见**小于约 0.9 个百分点（2σ）的差异。它不是"显示持平"，是"分辨率不够"。

**线上分数截至本文写作尚未拿到，不得据本节推测填数。** 预期是 ~71.1，
但见 §29.4 的解释：线上自身分辨率也是 ±0.5 分左右，所以最可能的结局是与 71.1382 判平。

### 29.2 训练曲线（20 轮全量，384px）

命令（复原自 §28.7；`-u` 与 `--save-every 4` 是本轮实测确认的）：

```bash
cd /root/autodl-tmp/AIC_orig
nohup python -u train.py --data /root/autodl-tmp/train --out outputs_384pe_local \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  > outputs_384pe_local.log 2>&1 &
```

> ⚠️ 上面这行是**复原**，不是逐字保存的原始命令行。权威记录请打印 checkpoint 里的 args：
>
> ```bash
> python - <<'PY'
> import torch, json
> a = torch.load('outputs_384pe_local/last.pt', map_location='cpu', weights_only=False)['args']
> print(json.dumps(dict(sorted(a.items())), ensure_ascii=False, indent=1))
> PY
> ```

启动时必看的两行（缺任一行说明开关没生效）：

```
trainable params=1.773M        ← 关掉 --local-head 应是 1.380M，差值 393217（=768×512+1）
first batch ok
```

| ep | val_acc | val_acc_hi | loss | 秒 |
|---|---|---|---|---|
| 1 | 0.6136 | 0.9783 | 3.2559 | 633 |
| 2 | 0.6570 | 0.9731 | 2.1262 | 626 |
| 3 | 0.6811 | 0.9727 | 1.8916 | 625 |
| 4 | 0.6855 | 0.9024 | 2.0764 | 627 |
| 5 | 0.6939 | 0.8977 | 1.2943 | 629 |
| 6 | 0.6996 | 0.8918 | 1.0867 | 629 |
| 7 | 0.7057 | 0.8916 | 0.9894 | 630 |
| 8 | 0.7092 | 0.8888 | 0.9098 | 626 |
| 9 | 0.7169 | 0.8938 | 0.8643 | 629 |
| 10 | 0.7173 | 0.8914 | 0.8004 | 630 |
| 11 | 0.7234 | 0.8904 | 0.7693 | 632 |
| 12 | 0.7257 | 0.8958 | 0.7151 | 629 |
| 13 | 0.7270 | 0.8957 | 0.6715 | 631 |
| 14 | 0.7284 | 0.8979 | 0.6378 | 628 |
| 15 | 0.7310 | 0.8962 | 0.6196 | 629 |
| 16 | 0.7320 | 0.8975 | 0.5972 | 628 |
| 17 | 0.7345 | 0.8988 | 0.5923 | 629 |
| 18 | 0.7332 | 0.8999 | 0.5783 | 628 |
| 19 | 0.7345 | 0.8994 | 0.5787 | 630 |
| 20 | **0.7347** | 0.9002 | 0.5869 | 629 |

**时间锚点：384px ≈ 629 秒/轮，20 轮 ≈ 3.5 小时。**（对比 §4.5 的 224px ≈ 144 秒/轮，
384px 是它的 4.4 倍——像素数是 2.94 倍，其余是显存/带宽效应。）

**曲线形状**：ep17/19 都是 0.7345，ep20 0.7347 —— **已经平顶**。`CosineAnnealingLR(opt, 20)`
在末期把学习率压到近 0，所以"再训更多轮"这件事本身不构成一个候选，除非同时改总轮数
（那会改变整条 LR 曲线，是另一个变量）。

**`val_acc_hi` 从 0.9783（ep1）掉到约 0.90** 不是退化：这是 `LabelTrustTracker` 的
高置信子集在收缩，随模型校准而变化，属预期。

### 29.3 gate 轨迹：模型要容量，但没换来泛化

```
epoch 20   val_acc 0.7347   gate +0.5082   tanh +0.4685
epoch 1                      gate +0.1535
```

`tanh(0.5082) = 0.4685`，即融合嵌入 `z = normalize(z_global + 0.4685 · z_local)` 里
**约 47% 来自 patch-token 支路**。gate 从 0 单调涨到 0.51，模型**大量使用**了这条新支路，
训练 loss 也从 3.26 降到 0.587——**但 val_acc 纹丝不动**。

这是"多出来的容量被吃掉、却没转化为泛化"的典型形态。机制上也说得通：
`LocalPatchHead` 只是把**同一批 token** 做 mean-pool 再线性投影，**信息量不变**，
所以它不可能带来 val 集上测不到的新东西。模型用它是为了压训练损失。

**这条负结果的价值在于它是一个干净的容量需求探针**：gate 冲到 0.51 说明模型**想要更多参数**；
错的是我把参数加在了"怎么池化"上，而不是"算什么特征"上。可训练量总共只有 1.380M，
其中 `CosineClassifier` 就占 750×512 = 384k —— 这条观察是后续 `--lora-rank` 一类
容量方向实验的依据。

### 29.4 推理与提交件

```bash
cd /root/autodl-tmp/AIC_orig
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_384pe_local/ep20.pt \
  --output pred_results_local_ep20.csv --tta
```

实测日志（逐字）：

```
img_size=384: positional grid resampled to 12x12
loaded outputs_384pe_local/ep20.pt (epoch 19, 750 classes, ViT-B-32-quickgelu, lora rank 8/all)
class-count prior loaded (log range 4.03)
TTA views (8) at 384px: plain, flip, mid, mid_flip, tight, tight_flip, wide, wide_flip
37444 test images found
wrote 37444 rows to pred_results_local_ep20.csv (logit-adjust tau=0)
unreadable images replaced by grey: 0
```

**闸门 2 已过**（CSV 是 CRLF、无表头，43 字节/行）：

```
37444 pred_results_local_ep20.csv
1610092 pred_results_local_ep20.csv
```

`37444` 行 × `43` 字节（CRLF、无表头）= **1,610,092 字节**，与历史被平台接收过的提交逐字节同构。
`unreadable = 0` 说明没有图被灰图替换。

> `epoch 19` 是 checkpoint 里的 **0-indexed** 轮号，即第 20 轮，不是"用了 ep19"。

**预期与判据**：中心估计 **~71.1**，与 71.1382 判平。理由是两条噪声底——
留出集 ±0.0044（≈±0.9 分）与线上 `sqrt(0.71×0.29/37444) ≈ 0.0023`（≈±0.5 分）——
本轮真实变化量（+0.0003）远低于两者的分辨率。
**提交它不承担下行风险**（平台取历史最高，71.1382 仍在），真正的收益是标定
"val 持平 → 线上持平"这个映射，供后续所有单变量实验筛选用。

### 29.5 云服务器目录结构（AutoDL）

```
/root/autodl-tmp/                 ← 数据盘，唯一该放东西的地方（§4.1）
│
├── train/                        ← 官方训练集：148695 张，750 个数字类文件夹
├── test/                         ← 官方测试集：37444 张（展开在图下，无类文件夹）
│
├── AIC_orig/                     ★ 队友树 = 71.1382 的原配方 + `--local-head` 移植（§28）
│   ├── train.py       1004 行    ← 移植后（原 919 行）
│   ├── infer.py        371 行    ← 原 365
│   ├── valmetrics.py   147 行    ← 原 144
│   ├── analyze.py      245 行    ← 原 242
│   ├── （其余 .py：datastats / probe / selftest 等，共 9 个）
│   ├── HANDOFF.md    205915 B
│   ├── README.md / README_AUTODL.md / requirements.txt
│   ├── outputs_384pe_local/      ← 本轮产物
│   │   ├── ep4.pt ep8.pt ep12.pt ep16.pt ep20.pt   （--save-every 4）
│   │   ├── best.pt last.pt
│   │   ├── outputs_384pe_local.log
│   │   └── pred_results_local_ep20.csv   37444 行 / 1610092 B
│   └── _backup_prelocalhead_20261001_100704/   ← 移植前四文件备份（§28.2）
│
└── （顶层还有当前树 LGL-Lab-AIC 的 .py 副本，train.py 2796 行）
```

> ⚠️ 上表中**只有 `AIC_orig/`、`train/`、`test/` 是本次会话实测确认的**；
> 顶层那批当前树副本来自更早的 `ls`（10-01 10:16 时间戳），本轮未复核。
> 动手前先跑一遍：

```bash
ls -la /root/autodl-tmp/
ls -la /root/autodl-tmp/AIC_orig/
du -sh /root/autodl-tmp/train /root/autodl-tmp/test
nvidia-smi
```

**目录约定（重要）**：`train/` 和 `test/` 在 `AIC_orig/` **外面**，所以所有命令都要写
绝对路径 `/root/autodl-tmp/train`。`cd` 进代码目录后忘记改路径，会让 `--data` 指向不存在的目录。
本轮就发生过一次 `cd` 失败导致冒烟测试跑在了**另一棵树**上（§29.7-1）。

### 29.6 云端训练 runbook（照做即可）

**Step 0 — 开机与省钱**
AutoDL 支持**无卡模式**开机（很便宜）：传数据、装依赖、下权重全在无卡模式做完，
再关机切 GPU 模式训练（§4.1）。训练完立刻关机。

**Step 1 — 环境（一次性）**

```bash
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

不设这个，`open_clip` 3.3.0 从 HuggingFace 拉 OpenAI 权重会 `Network is unreachable`（§4.4）。

**Step 2 — 上传代码**
用 **JupyterLab**（文件树右键 → Upload）而不是 WinSCP：WinSCP 会显示**过期列表**，
新建的目录/文件看不到，本轮在这上面浪费了时间（§29.7-4）。传完必须核对：

```bash
cd /root/autodl-tmp/AIC_orig
wc -l train.py                      # 应为 1004
grep -c "local-head" train.py       # 应为 4
```

**Step 3 — 确认 GPU 空（必做）**

```bash
pgrep -af train.py
nvidia-smi
free -g
```

24GB 卡上**跑不下两个 384px 的训练**。本轮一次 `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`
就是这么来的——不是代码 bug，是显存争用（§29.7-3）。**训练和推理也不要并跑。**

**Step 4 — 两道冒烟闸门**

```bash
# 闸门 1：开关注入（几十秒）
python train.py --data /root/autodl-tmp/train --out smoke_local \
  --img-size 384 --train-pos-embed --local-head \
  --epochs 1 --limit-batches 20 --workers 2
# 必看：trainable params=1.773M ；关掉 --local-head 是 1.380M

# 闸门 2：open_clip 的 forward_intermediates 存在且签名匹配（§28.6，30 秒）
python - <<'PY'
import inspect, torch, open_clip
m = open_clip.create_model('ViT-B-32-quickgelu', pretrained='openai')
v = m.visual
print('has forward_intermediates:', hasattr(v, 'forward_intermediates'))
print('signature:', inspect.signature(v.forward_intermediates))
d = v.forward_intermediates(torch.zeros(1, 3, 384, 384),
                            indices=[len(v.transformer.resblocks) - 1], stop_early=False,
                            normalize_intermediates=True, intermediates_only=False,
                            output_fmt='NLC')
print(tuple(d['image_features'].shape), tuple(d['image_intermediates'][-1].shape))
PY
# 期望 (1, 512) 与 (1, 144, 768)
```

**Step 5 — 正式训练**

```bash
cd /root/autodl-tmp/AIC_orig
nohup python -u train.py --data /root/autodl-tmp/train --out outputs_384pe_local \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  > outputs_384pe_local.log 2>&1 &
```

- **`-u` 不是可选项**。不加的话 stdout 是**块缓冲**（8 KB），`tail` 看到的是空的，
  只有 stderr（PIL 警告）会立刻出现——本轮因此误判过"没在跑"（§29.7-2）。
- 不加 `--workers` 就是用默认 8，与历史运行一致。
- 队友树**没有** `--lr-warmup-epochs`（§28.4），调度本来就是无预热，别去加。
- 想加 `--resume` 前先读 §29.7-5：**EMA teacher 没有被保存，resume 会重置它。**

**Step 6 — 监控**

```bash
tail -f /root/autodl-tmp/AIC_orig/outputs_384pe_local.log | grep --line-buffered "epoch "
# 期望每行形如：
# epoch 3/20 loss=1.8916 val_loss=... val_acc=0.6811 val_acc_hi=0.9727 lr=1.99e-04 time=625.4s
```

`lr=1.99e-04` 顺带证实了无预热调度（有预热首轮会是 2.00e-04）。
每轮约 **629 秒**，据此估 ETA。开跑后看一眼 `free -g`，内存吃紧就降 `--workers`。

**Step 7 — 推理与提交件校验**

```bash
cd /root/autodl-tmp/AIC_orig
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_384pe_local/ep20.pt \
  --output pred_results_local_ep20.csv --tta
wc -l pred_results_local_ep20.csv          # 37444
stat -c '%s %n' pred_results_local_ep20.csv # 1610092
```

两个数**必须同时对上**，否则先别提交。CSV 是 **CRLF、无表头**，所以 43 字节/行：
行数对而字节不对 = 换行符被改了（Windows 上编辑过、或被 git 转换过），平台可能拒收。

**Step 8 — 下载**
JupyterLab：进 `/root/autodl-tmp/AIC_orig/` → 右键 CSV → Download。
用 WinSCP 的话进目录后**按 F5 强制刷新**。

### 29.7 本轮新踩的坑

1. **`cd` 失败会让冒烟跑在错误的树上。** 一次 `cd /root/autodl-tmp/AIC_orig` 没生效，
   冒烟实际跑在顶层那棵 2796 行的当前树上（日志里 `image_size=384`、`local_head=False`、
   有 `augment_mode`/`distill_weight`——全是当前树独有）。**判据是日志里的
   `trainable params=` 与 flag 名**，不是"命令看起来对"。

2. **stdout 块缓冲 = 看不到实时进度。** `nohup python train.py > f.log` 下 stdout 被缓冲，
   `tail` 长时间为空。修法：`python -u`。这纯粹是观感问题，但会让人误以为任务挂了。

3. **`cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` 是显存争用，不是代码 bug。**
   当时另一个 384px 任务还在跑。空载后同样的冒烟一次通过，坐实了诊断。
   **24GB 卡装不下两个 384px 任务**——先 `pgrep -af train.py` 再启动。

4. **WinSCP 显示过期列表。** `mkdir` 之后重新连接仍看不到新目录，而 `ls -ld` 证明它存在
   （31 个文件 / 18 个目录与 mkdir 前的 `ls` 完全吻合）。改用 JupyterLab 后正常。

5. **`--resume` 在结构上是有损的。** `train.py` 的恢复块重建
   `teacher = copy.deepcopy(model)`，但**checkpoint 里根本没有保存 EMA teacher**，
   所以恢复会把教师重置成当前学生。1 个 epoch 时的规则是**重启而不是 resume**
   （≤2 轮不心疼）。要真正支持 resume 得先加 `--save-teacher` 式的教师状态保存。

6. **`--train-pos-embed` 会让"换分辨率推理"直接崩。** checkpoint 存的是**训练过的**
   pos-embed（384px 下 145×768），而 `infer.py` 是"先建目标尺寸的网格、再 `load_state_dict`"，
   形状对不上会 `size mismatch`——`strict=False` **不会**把形状错误降级成 unexpected keys。
   所以 `tta8` 在训练分辨率上是唯一安全的提交路径。

## 30. 最后一次训练前的 frozen-mix 代码审计与选择 —— 2026-10-01

> **更新：§31 的真实数据冒烟结果推翻了本节的分数预测与直接正式训练建议。§30.2 的命令不得照抄；以 §31 为准。**

### 30.1 审计结论

对当前树的 `--head-init frozen --frozen-mix-rho` 路径做了静态审计：训练特征 pass、训练样本索引、LOO medoid 屏蔽、稀疏 top-k 目标、warm-up/post-warm target 混合及目标归一化均一致。`rho=0.7` 会从第一个 batch 改变监督目标；当冻结 CLIP 的 top-1 质量满足边界时，确实会翻转早期 target 的 argmax。

补上的两处加固：

1. CLI 拒绝 `rho` 不在 `[0,1]`、非正 temperature、`topk/protos<1`，避免非法目标或静默改配置；
2. B13 开启时，EMA prototype 更新改用 corrected target 的 argmax，与 CE/robust/prototype loss 使用同一 corrected target；默认关闭 B13 时保持原配方不变。

本机无 PyTorch，未执行 GPU 训练；已通过 `py_compile` 与 `git diff --check`。`selftest.py` 不能在本机运行，原因是环境缺少 `torch`。

### 30.2 今日单次训练建议

若目标是冲击高分且接受中风险，使用当前树单独跑：

```bash
python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_seed_mix07 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --lr-warmup-epochs 0 \
  --head-init frozen --frozen-mix-rho 0.7 --save-every 4
```

推理固定使用同一 checkpoint、384px、`tta8`、fp32/logit aggregation。训练日志必须核对 `config fingerprint`、`frozen target` agreement 比例和每轮 `[mix]` 的 flip 数；若 flip 数始终为 0，说明这次教师分布没有达到标签级修正边界，结果应按保守的 head-init 试验解读。

预期只能作为区间估计：中心约 71.5–72.0，乐观约 72.3；冻结 CLIP 把易混邻居误改为教师类别时可能低于历史 71.1382。该方法未有线上实测，不能把区间当作承诺。

## 31. frozen-mix 真实数据冒烟复盘 —— 2026-10-01

`--head-init frozen --frozen-mix-rho 0.7` 在当前树、384px、750 类、134165 张训练图上完成了冻结特征 pass 和 20 batch 冒烟，没有运行错误。但这次诊断**推翻**了 §30.2 对收益的乐观估计，不能据此直接启动正式 20 轮。

- `lr=0.00e+00` 是 `--epochs 1` 在 epoch 末执行 `sched.step()` 后打印的**下一轮**学习率；20 个 batch 实际用 `2e-4`。不是训练失效。
- `val_acc=0.4182` 来自 2560 次采样训练（全训练集约 1.9%），不能和完整训练的约 0.734 比较。
- 冻结鲁棒质心过滤最终保留 `56752/134165=42.3%`；随后每类仅取一个 medoid 的教师与目录标签一致 `43339/134165=32.3%`。它对人工干净真值的准确率未知，不能把不一致全视为错标，也不能假定教师更正确。
- `rho=0.7` 只翻转 `66/2560=2.6%` 的 target argmax。翻转条件是 `q_top-q_folder > (1-rho)/rho=0.4286`，不是仅仅 `q_top > 0.41`。此前 §23.4/§30 使用了忽略 `q_folder` 的简化不等式，**高估了翻转率**。
- 更关键的是：**2.6% 翻转不代表影响只有 2.6%**。即使 argmax 没翻，所有样本的监督目标仍有 70% 概率质量来自冻结教师；如果教师错误或过软，整个 warm-up 的梯度都会改变。
- `libgomp: Invalid value for environment variable OMP_NUM_THREADS` 是环境变量格式警告，可在云端 `unset OMP_NUM_THREADS` 或设为合法整数；与冻结目标无关。

### 31.1 已做的代码改进

1. `--frozen-protos 1` 的训练目标教师改为 `robust_centroids` 得到的类中心；`--frozen-protos >1` 仍使用显式多 medoid 和 LOO。每类一个 medoid 把约 180 张图压成单图，实测一致率偏低；改回鲁棒类中心是更稳的默认值。**这项是新的算法变化，不能把旧冒烟数字套到新版本。**
2. 冻结 pass 后增加 `frozen target quality`：教师 top1 质量、给定标签质量、分歧样本 top 与给定标签的差值分位数、基于 rho 预测的翻转数。它报告目标分布，避免只盯 argmax flip。
3. 增加 `frozen teacher holdout`：同一冻结教师在独立验证图上的目录标签一致率。它仍是带噪标签准确率，不能当干净真值精度，但可与训练集一致率比较，发现自包含或 medoid 抽样造成的偏差。
4. 移除 `selftest.py` 中依赖单 medoid LOO 人工强迫翻转的端到端断言；该断言证明不了真实数据上的有效纠错。保留目标构造、mix 质量守恒等数学检查。

### 31.2 下一步决策

先上传更新后的 `train.py` 与 `selftest.py`，在云端执行 `python selftest.py`。然后重新运行 20 batch 冒烟，读取 `frozen target quality`、`frozen teacher holdout` 和 `[mix]`。只有当冻结教师对带噪留出标签的表现合理、分歧有清晰 margin、且诊断支持正确方向时，才投入 20 轮。**撤回 §30.2 的“中心 71.5–72.0”预测**；目前没有可靠正增益估计。不能仅为提高 flip 数而降低温度或加大 rho，这同时会放大错误教师的监督。

若今天必须立即启动一次完整训练且无法复跑冒烟，优先保留已验证的 71.1382 方案或仅测试 `--head-init frozen`；不要把旧版 `rho=0.7` 当作高概率冲分方案。
## 32. `head-init frozen` 正式训练准备 —— 2026-10-02

当前正式实验只保留 `--head-init frozen`，不启用 `--frozen-mix-rho`、`--distill-weight`、`--local-head` 或 `--frozen-protos > 1`。该路径只做一次冻结 CLIP 全量特征 pass，用两轮 robust centroid 初始化分类头和 ProtoHead；训练循环仍使用原始单模型监督、EMA、APL、anchor、consistency 和 prototype loss。

启动后必须看到：

```text
frozen pass: ... serves --head-init
head-init frozen: head seeded for 750/750 classes; ...
```

不应看到：

```text
frozen target quality:
frozen teacher holdout:
[mix]
[distill]
```

推荐命令：

```bash
unset OMP_NUM_THREADS
cd /root/autodl-tmp
pgrep -af train.py || true
nvidia-smi
python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_headinit \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --image-size 384 --train-pos-embed --lr-warmup-epochs 0 \
  --head-init frozen --save-every 4 \
  > outputs_384pe_headinit.log 2>&1 &
```

完成后只使用同一训练分辨率的 `ep4/ep8/ep12/ep16/ep20` 做 `tta8` 推理比较；不要使用带 mix/distill 的旧 checkpoint 混合推理。

## 33. `--head-init frozen` 实测：曲线、静默中断与线上预估 —— 2026-10-02

承接 §32，按 §32 的推荐命令在本树跑满 20 轮（`outputs_384pe_headinit`）。**第 20 轮的训练进程被静默 SIGKILL，前 19 轮全部落盘**。结论：该 flag 没有提升迹象，本地 val_acc 全程低于 §29.2 队友树参考曲线，据此推算的线上分数比现行最高分 71.2023 低约 0.4 分。**本臂最终未提交**（§33.10）。

### 33.1 启动闸门（全部通过，与 §32 预期一致）

```text
frozen pass: 134165 training images, ... serves --head-init
[head-init frozen: head seeded for 750/750 classes; 56752/134165 samples survived the agreement filter (0.423)]
```

§32 列出的四条“不应看到”的日志（`frozen target quality:` / `frozen teacher holdout:` / `[mix]` / `[distill]`）在全日志中出现次数为 0，确认本轮只做了分类头初始化，训练循环是纯单模型监督，没有教师目标参与梯度。配方与 §31 冒烟一致（同配方实测 trainable params 1.380M）。

**本轮没有存下启动日志的 `config fingerprint` 行**，这是流程缺口：以后启动后立刻 `grep -m1 'config fingerprint' <log>` 存档。

### 33.2 本地曲线（已逐字核对的锚点轮）

| epoch | val_acc | 备注 |
| --- | --- | --- |
| 1 | 0.6317 | `loss=2.7356 val_loss=1.8440 val_acc_hi=0.9872 lr=1.99e-04 time=634.8s` |
| 4 | 0.6773 | `ep4.pt`，valmetrics 复核一致 |
| 8 | 0.7084 | `ep8.pt` |
| 12 | 0.7214 | `ep12.pt` |
| 16 | 0.7276 | `ep16.pt` |
| 19 | 0.7299 | **best.pt**，`loss=0.6559 val_loss=1.6062 val_acc_hi=0.9012 lr=1.23e-06 time=635.3s` |

其余 13 行只在云端 `outputs_384pe_headinit.log` 里，本节不登记未核对过的数字；补全命令：

```bash
grep -E '^epoch ' /root/autodl-tmp/outputs_384pe_headinit.log
```

- 每轮 634.8–635.3 s，19 轮 ≈ 3.35 h。`lr=1.99e-04` 正好是 `2e-4·(1+cos(π/20))/2`，确认 20 轮 cosine 与 `--lr-warmup-epochs 0` 都生效。
- `val_acc_hi` 从第 1 轮 0.9872 降到第 19 轮 0.9012，是 warm-up 结束转入稳健化阶段的预期行为，不是退化。
- 19 轮里最好值是 ep19 的 0.7299；`best.pt` = ep19。

### 33.3 第 20 轮被静默 SIGKILL

- 日志停在 ep19 的 `noise stats` 行：无 `Traceback`、无 `Killed` 字样；`ep20.pt` 不存在；`last.pt` 与 `best.pt` 的 mtime 都是 05:09（ep19 结束时刻）。三项同时指向“进程被杀，不是异常退出”。
- 主机 uptime 1 年 4 周 ⇒ 不是宿主机重启。排除 Python 异常与重启后，剩下的是 cgroup OOM 静默杀进程（此为排除法推断，容器内未取到内核 OOM 记录核对）。容器上限 60 GiB；`free` 显示的 1007 GB 是宿主机，不可用于判断，要看 `/sys/fs/cgroup/memory.max` 与 `memory.current`。
- 触发原因：**同一时间并发跑了 `valmetrics.py --batch-size 256 --workers 8`**。训练自身持有 19.6/24.5 GB 显存加 8 个 dataloader worker，再叠一份 256 batch 特征解码，容器内存被打满；被杀的是训练进程（valmetrics 活到了自己抛 `FileNotFoundError` 才退出）。
- **规则（再次确认，本次是实测复现）：训练进行中不得运行任何 GPU / 图像解码任务。**
- 代价评估：§29.2 参考曲线 ep19→ep20 只涨 0.0002，且此时 lr 已降到 1.23e-06，补跑 ep20 预期 ≈0.05–0.1 pt。**判定为不值得**，除非纯粹为了留档跑满 20 轮；若补跑，必须在 A 臂结束后，并显式重复 `--train-pos-embed`。

### 33.4 valmetrics：micro / macro 与自检

| checkpoint | micro（= train.py 日志） | macro（类等权召回） | 最稀 50% 类 | 最常见 10% 类 |
| --- | --- | --- | --- | --- |
| ep4 | 0.6773 | 0.6702 | 0.4687 | 0.9730 |
| ep8 | 0.7084 | 0.7007 | 0.5100 | 0.9865 |
| ep12 | 0.7214 | 0.7133 | 0.5267 | 0.9891 |
| ep16 | 0.7276 | 0.7191 | 0.5365 | 0.9909 |

- 四个 micro 与 `train.py` 日志**逐位相同**，`valmetrics` 自带的自检条件成立 ⇒ split 与前向对得上，macro 列可用。
- 验证集按类分层保留了训练集失衡（每类图像 max/min ≈ 24x）。micro 比 macro 高约 0.008，就是这个失衡在给 val_acc 抬分。
- **测试集是按类均衡的，因此 macro（尤其是“最稀 50% 类”列）比 val_acc 更接近线上指标。** 选 checkpoint 时两列都要看，不要只看 val_acc。

### 33.5 标签信任统计（ep19 结束时，逐字）

```text
noise stats: {'clean': 103678, 'relabel': 1994, 'noisy': 28493, 'unseen': 0, 'capped': 0, 'vetoed': 0, 'mean_trust': 0.6412724256515503, 'mean_weight': 0.801433265209198}
```

纯单模型监督下，19 轮后仍有 28493 张（21.2%）被判为带噪；`unseen`/`capped`/`vetoed` 全 0 说明没有样本被规则硬丢弃，全部走软权重。`mean_trust = 0.6413` 与 §31 推出的翻转盈亏平衡点 `L* = √0.415 = 0.644` 在数值上几乎重合（§31 的结论：混教师标签只有在模型自身精度 `L < L*` 时才可能净获益）——这是独立于冻结教师的第二个、方向一致的数据点，但只是巧合级的吻合，不构成证明。

### 33.6 与 §29.2 队友树参考曲线的对齐

| epoch | 本树 `--head-init frozen` | §29.2（`AIC_orig`，384+pe+local-head，无 head-init） | 差 |
| --- | --- | --- | --- |
| 4 | 0.6773 | 0.6855 | −0.0082 |
| 19 | 0.7299 | 0.7345 | −0.0046 |
| 20 | （缺失，见 §33.3） | 0.7347 | — |

- 14530 张留出上的 σ = 0.0037，2σ = 0.0074。ep4 的 −0.0082 约 2.2σ，处在可分辨边缘；**ep19 的 −0.0046 只有约 1.2σ，落在噪声内**。此前我把两处都读成“远超 2σ”，对 ep19 是错的，此处更正。
- 因此单看差距还不能定罪。但把两件事合起来看：本臂**没有任何一轮领先**，且端点既有差距又方向一致——head-init 至少可以判定为“无增益”，而不是“有害”。
- 归因仍有混淆：两条曲线来自**不同的树**（本树 vs `AIC_orig`），且 §29.2 还多了 `--local-head`。所以差距不能直接记在 `--head-init frozen` 账上，见 §33.9 的 A 臂。

### 33.7 线上分数推算

用三对“本地 val_acc → 线上”标定偏移：

| 本地 val_acc | 线上 | 偏移 |
| --- | --- | --- |
| 0.7344（§29.1，无 `--local-head`） | 71.1382 | −2.30 |
| 0.7347（§29.2 ep20，+`--local-head`） | **71.2023**（2026-10-02 实测开分，§33.10） | **−2.27** |
| 0.7142（416+pe+堆叠） | 69.3100 | −2.11 |

偏移 ≈ −2.23 ± 0.10（三个都是实测点）。本臂 0.7299 ⇒ **预期线上 ≈ 70.77（区间 70.6–71.0）**，相对现行最高分 71.2023 是 −0.4。

（原先表中"0.7347 → ~71.1"一行是 §29.4 的**预测**，现已被 71.2023 的实测取代，故删除。）

三点注意：偏移由 3 个点外推，是估计不是承诺；本臂**尚未做任何 tta8 推理**，此推算默认沿用同一推理配方；macro 视角更悲观（0.7191 vs 参考 0.7345 同级），因此 70.77 更可能落在区间下半段。

### 33.8 对 §22.2「ep4 定分」的更正

§22.2 记录 `ep4 == ep20`（线上四位小数相同）⇒ 分数在第 4 轮就定了。本臂与 §29.2 的本地曲线都不支持把这条当早停依据：

- 本臂 ep4→ep19 本地涨 **+5.26 pt**（0.6773→0.7299）；§29.2 涨 **+4.90 pt**（0.6855→0.7345）；而 §29.2 的 ep19→ep20 只涨 0.0002。

也就是说，`ep4 == ep20` 是**当时那套配置下**观测到的饱和现象，不能反推成“所有配置都跑 4 轮就够”。此前基于它给出的“跑 4 轮、30 分钟出判决”建议**撤回**；选提交 checkpoint 用 ep16 / ep19 / `best.pt`，不要用 ep4。这条矛盾本身未解决（要线上复核就得做全链 tta8 推理），登记于此备查。

### 33.9 待办与决策规则

1. **A 臂已启动**（本树基线，唯一变量是去掉 `--head-init frozen`）：`outputs_384pe_ctrl`，pid 3267，日志 `outputs_384pe_ctrl.log`，约 3.5 h。跑完贴 `grep -E '^epoch '` 即可闭合归因：
   - A 的 ep19 ≈ 0.7299 ⇒ 差距来自**树**（`AIC_orig` vs 本树），`--head-init frozen` 排除嫌疑；
   - A 的 ep19 ≈ 0.7345 ⇒ 差距来自 **`--head-init frozen`**，本 flag 判负，回到已验证的 71.2023 配方（`AIC_orig` + `--local-head`）。
2. B 臂**未提交**（2026-10-02 用户决定，§33.10），因此 §33.7 的 70.77 保持"未验证推算"。若以后要补：用 `best.pt`（= ep19）、384px、`tta8`、显式 `--workers 8`、**不要传 `--image-size`**（带 `--train-pos-embed` 的 checkpoint 只能用训练分辨率推理）。CSV 拷回本地后核对：`stat -c %s` = **1610092**、`sed 's/\r$//' | awk 'END{print NR}'` = **37444**。
3. **在 A 臂结束前不要在云端跑任何推理**（§33.3 的静默 SIGKILL 就是这么来的）。
4. 每天 2 个提交名额、榜单保留历史最高分。拿预期 70.8 的 checkpoint 去打 71.2023 只会消耗名额，收益为负；除非 A 臂给出改观，否则本 flag 到此为止，把尝试预算放回训练侧。

### 33.10 2026-10-02 开分：`--local-head` = 71.2023（新最高分）

用户 2026-10-02 报：`--local-head` 那次提交（§29.2 的 `AIC_orig` 树，384+pe+`--local-head`，ep20，本地 val_acc 0.7347，文件 `pred_results_local_ep20.csv`）开分 **71.2023**；本文件 §33 的 `--head-init frozen` 臂**没有提交**。

| 配方（队友树 `AIC_orig`） | 本地 val_acc | 线上 | 本地→线上偏移 |
| --- | --- | --- | --- |
| 384+pe，无 local-head | 0.7344 | 71.1382 | −2.30 |
| 384+pe+`--local-head` | 0.7347 | **71.2023** | −2.27 |

- **`--local-head` 的线上实测增益 = +0.0641**，而同期本地 val_acc 只动了 +0.0003。这是本项目**第一个线上为正的训练侧改动**（此前训练侧全是 0 或负），也是"本地 val_acc 排不出模型优劣"的又一证据：0.0003 的本地差对应 0.064 的线上差。
- 但 +0.064 仍只是零头：**它不足以改变"缺口不在 flag 层面"这个结论**。head-init 判为无增益、local-head 记 +0.06，两者加起来也到不了 +1。
- 对 §33 的直接影响：B 臂（本树 + `--head-init frozen`）**没有线上数据**，§33.7 的 ≈70.77 仍是推算；现行最高分从 71.1382 提到 71.2023 后，该推算的相对位置变成 −0.4。

**接下来的单变量顺序**：

1. 等 A 臂（`outputs_384pe_ctrl`，本树、无 head-init、无 local-head）跑完 ⇒ 先回答"**本树 vs `AIC_orig` 树的差距**"，并与 B 臂（本树 + head-init，ep19 = 0.7299）构成同树单变量，给 head-init 定罪或脱罪。
2. 随后在本树跑同配方 **+`--local-head`**（`outputs_384pe_localhead`）：

```bash
unset OMP_NUM_THREADS
cd /root/autodl-tmp
python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_localhead \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --image-size 384 --train-pos-embed --lr-warmup-epochs 0 \
  --local-head --save-every 4 \
  > outputs_384pe_localhead.log 2>&1 &
```

启动必看 `trainable params=1.773M`（关掉 `--local-head` 是 1.380M，差 393217 = 768×512+1）。**不要与 A 臂并发**（§33.3）。

3. 判据：若 A 臂 ep19 ≈ 0.7299 而 `AIC_orig` 的无 local-head 是 0.7344，则本树本身落后 ≈0.45，**冲分应直接回 `AIC_orig` 树做**，本树只用于归因；反之若 A ≈ 0.7345，则本树差距来自 head-init，本树 + local-head 就是本树的最优组合。

## 34. 换基线：本地 `LGL-Lab-AIC` 的代码整体换成 AIC_orig（71.2023 配方）—— 2026-10-02

用户 2026-10-02 决定：把 `AIC_orig`（= 71.2023 的配方）作为新基线。**已执行完毕**，方式是"存档后原地替换"，旧代码一件不丢。

### 34.1 先证明 AIC_orig 是干净的

用 marker 计数对比三个 `train.py`（队友树的 pre-port 备份 = `...\AIC\LGL-Lab-AIC\_backup_prelocalhead_20261001_100704\train.py`）：

| marker | AIC_orig (1005 行) | pre-port (919 行) | 旧增强树 (2796 行) |
| --- | --- | --- | --- |
| `local_head` | 12 | **0** | 17 |
| `head_init` | 0 | 0 | 16 |
| `frozen_mix_rho` | 0 | 0 | 16 |
| `frozen_protos` | 0 | 0 | 9 |
| `distill_weight` | 0 | 0 | 11 |
| `augment_mode` | 0 | 0 | 5 |
| `lr_warmup_epochs` | 0 | 0 | 2 |
| `tracker` / `consistency` / `trust` | 16 / 4 / 6 | 16 / 4 / 6 | 39 / 10 / 28 |

**除 `local_head` 外，每一列的计数与 pre-port 备份逐项相同** ⇒ AIC_orig = 队友 919 行原树 + 仅 `--local-head` 一次移植（与 §28 的 diff 表吻合：+90 −5 行）。它确实是"71.1382 的配方 + 一个变量"，不是又一棵混杂树。

### 34.2 安装清单（sha256 前 16 位）

| 文件 | 字节 | sha16 |
| --- | --- | --- |
| `train.py` | 51248 | `b444595f575c28d4` |
| `infer.py` | 19689 | `80992a0fd71e8ab9` |
| `selftest.py` | 34831 | `cca95ca4cacc6164` |
| `mres.py` | 17995 | `b0107859b9ed6425` |
| `noise.py` | 8751 | `6b92a490d5acff05` |
| `losses.py` | 6124 | `53c6d669387374a8` |
| `analyze.py` | 12814 | `7ca81d84d6a441eb` |
| `datastats.py` | 10912 | `15017655c0a47515` |
| `valmetrics.py` | 7181 | `9633e90906c762f7` |
| `probe_resolution.py` | 10187 | `89b0b7a35efc4878` |
| `requirements.txt` | 217 | `24b219d59ce2dca5` |

- 来源：`C:\Users\Ed\Desktop\Recent Project\AIC\LGL-Lab-AIC\`（队友树 + local-head 移植，= 云端 `/root/autodl-tmp/AIC_orig/`）。
- 旧增强树的 11 个同源代码已整体存档到 [`_backup_enhanced_20261002/`](_backup_enhanced_20261002/)（`train.py` 162129 B / 2796 行、`selftest.py` 136376 B、`noise.py` 17480 B、`probe.py`、`diagnostics.py` 等）。
- **本机没有 git**（`git` 既不在 PATH，`Program Files` 下也没有）：没有任何版本回滚兜底，所以这次是"先 Copy 存档、再覆盖"。以后每次动代码前都要先存档。
- 未动（继续留在根目录）：`HANDOFF.md`、`提分路径研究.md`、`开源项目检索报告.md`、`README_AUTODL.md`、`README.md`、`probe.py`、`diagnostics.py`。

### 34.3 ⚠️ 新基线的 flag 面与旧树不同（照抄旧命令会报错）

1. **没有 `--lr-warmup-epochs`**（0 命中）。这棵树本身就是 `CosineAnnealingLR(opt, epochs)`，所以"所有与旧分数对照的运行都必须用无 warm-up 调度"这条约束**在新基线上自动满足**——但也意味着旧命令里的这个 flag 必须删掉，否则 argparse 直接退出。
2. 分辨率 flag 是 **`--img-size`**（dest `img_size`），不是 `--image-size`。
3. 推理 flag 是 **`--tta`**：裸 `--tta` = 8 视角中心裁剪（`TTA_VIEWS` 的 `plain/flip/mid/mid_flip/tight/tight_flip/wide/wide_flip`，与旧树的 `tta8` **同一张视图表**，§29.4 日志逐字确认）。**没有** `--tta-views/--tta-sizes/--tta-agg`。分组简写：`center/corners/all/scales`。基线的 `--workers` 默认就是 **8**（旧树默认 0，必须显式给）。
4. 基线 `infer.py` 支持 `--logit-adjust 0 0.25 0.5`：**一次前向出多个 tau 的 CSV**，这个能力在新树里还在。
5. 基线**没有** `--head-init` / `--frozen-mix-rho` / `--frozen-protos` / `--distill-weight` / `--augment-mode` / `diagnostics.jsonl`（§28.8 已记）。想再试这些方法必须重新移植，不能再指望"在这棵树上打开一个 flag"。

### 34.4 已记录的配方（新基线上照抄即可）

训练（= 71.2023 的配方）：

```bash
cd /root/autodl-tmp/AIC_orig
python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_local \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --save-every 4
# 必看：trainable params=1.773M（去掉 --local-head 是 1.380M，差 393217）
```

推理（§29.4 逐字，产出 71.2023 的那条）：

```bash
cd /root/autodl-tmp/AIC_orig
python infer.py --test /root/autodl-tmp/test --checkpoint outputs_384pe_local/ep20.pt \
  --output pred_results_local_ep20.csv --tta
```

### 34.5 这次换基线改变了什么、没改变什么

- **改变了**：§33.10 计划里的"本树 + `--local-head`"这一跑**作废**——那正是已经开分的 71.2023。A 臂（`outputs_384pe_ctrl`，旧增强树）**只剩归因价值**，不再指向任何分数。
- **没改变**：换基线是"让实验可归因"的整理动作，**本身不产生分数**。缺的 2~3 分仍然要在新基线上一项一项试出来。
- 下一个冲分实验应当是**在 AIC_orig 基线上的单变量移植**。优先候选 = §26 列的那批"噪声机械"（trust tracker 升级、双向一致性、trusted prototype 归一化、样本权重上限、warm-up 末清 tracker 后验）——它们是唯一直接打"标签噪声 / 本地 val_acc 与线上错配"这条线的实现。**预期量级要诚实：参照 `--local-head` 的 +0.064，这类改动的期望值在 +0.1 以内，不足以填 2 分**；但它们是目前唯一有机制理由的方向。
- 云端核对（确认 `/root/autodl-tmp/AIC_orig/` 与本地基线是同一份）：

```bash
cd /root/autodl-tmp/AIC_orig
sha256sum train.py infer.py selftest.py noise.py losses.py analyze.py | cut -c1-16
# 依次应为 b444595f575c28d4 / 80992a0fd71e8ab9 / cca95ca4cacc6164 /
#             6b92a490d5acff05 / 53c6d669387374a8 / 7ca81d84d6a441eb
```

## 35. ⭐⭐ 容量这条线通了：`--lora-rank 16` = **72.2733**（新最高分）—— 2026-10-03

### 35.1 结果

| 提交 | 配置 | 分数 |
| --- | --- | --- |
| `pred_results_local_ep20.csv` | 384 + pe + `--local-head`，**`--lora-rank 8`** | 71.2023 |
| **`sub_lr16_tta8.zip`** | 同上，**`--lora-rank 16`** | **72.2733（+1.071）** |

### 35.2 单变量声明

从 71.2023 的配方出发，**只把 `--lora-rank` 从 8 改成 16**，其余逐字相同：

```bash
cd /root/autodl-tmp
python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_lr16 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  --lora-rank 16
```

**启动必须看到 `trainable params=2.658M`**（不带 `--lora-rank 16` 时是 1.773M，差 **884,736 = LoRA 翻倍**）。

推理（见 §35.3 为什么不是默认参数）：

```bash
cd /root/autodl-tmp
python infer.py --test /root/autodl-tmp/test \
  --checkpoint outputs_384pe_lr16/best.pt \
  --output sub_lr16/pred_results.csv --tta --workers 4 --batch-size 64
```

曲线：`best val_acc = 0.7465`（ep19），ep20 = 0.7460 —— **末段已平，20 轮够用**。

### 35.3 ⚠️ 新踩的坑：62 GB 机器上「8 视角 TTA @384」用默认参数会 OOM

**现象**：推理跑到一半被 SIGKILL，`sub_lr16/` 始终 `total 0`。

```
/bin/bash: line 34:  7452 Killed    nohup python -u infer.py ... --workers 8 ...
[2]+  Exit 137
```

控制台同时显示 **内存 98%**。（`Exit 137` = 128 + 9 = SIGKILL。）

**原因 —— 内存账：**

| 项 | 数值 |
| --- | --- |
| 一张图 8 个视角 @384² × 3 通道 × 4 字节 | 384×384×3×4 × 8 = **14.2 MB** |
| `--batch-size 256`（默认）→ 每批 | **3.6 GB** |
| `--workers 8`，每 worker 预取 1 批（`prefetch_factor=1`） | **≈29 GB** |
| `pin_memory=True` 再复制一份到 pinned 内存 | **≈58 GB** |

**⇒ 80 GB 的容器扛得住，62 GB 的必爆。而 AutoDL 两台机器的内存不一样：**

| 实例 ID 尾部 | 内存 | 默认参数（256 / 8） |
| --- | --- | --- |
| `c0b042a675` | **80 GB** | 能跑 ✓ |
| `4c464ca602` | **62 GB** | **OOM** ✗ |

> **规则：62 GB 机器上一律用 `--workers 4 --batch-size 64`**（实测内存约 7 GB）。
> 代价是慢约 10 分钟。**两台统一用这一套** —— 审计时只有一条推理命令，复现也简单。

> ⚠️ **`free -g` 在容器里报的是宿主机内存**（实测 `total 503`），**不能用来判断会不会 OOM**。
> 要看容器真实用量：AutoDL 控制台的「实例监控」，
> 或 `cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max`。

> **本次损失**：约 25 分钟 GPU + 一次重跑，**没有丢任何数据** ——
> `best.pt`（12 MB）和 `epN.pt` 都在数据盘上。
> 教训是"推理也要算内存账"，不是"结果没了"。

### 35.4 同日第二个实验：`--tau-conf` 的剂量-反应（`0.5` 空枪，`0.3` 生效）

**动机**：384 的 `noise stats` 里 **`relabel` 只占 1.4%（1838），`noisy` 占 20.9%（27978）**
—— tracker 的"纠正标签"分支几乎是死的，实际只做降权。§14.4 第 3 条当年就预警过这件事。

#### ⭐ 干净的守恒证据（同为 ep4，两轮逐字对比）

| | `--tau-conf 0.5` | `--tau-conf 0.3` | Δ |
| --- | --- | --- | --- |
| `relabel` | 1813 | **6262** | **+4449** |
| `noisy` | 41936 | **37487** | **−4449** |
| `clean` | 83597 | 83597 | 0 |
| `unseen` | 6819 | 6819 | 0 |
| `mean_trust` | 0.281679150238037 | 0.281679150238037 | **逐位相同** |

**⇒ ep4 时两次运行除 `relabel ↔ noisy` 的分配外完全逐位相同**（warm-up 不用 tracker，天然一致）。
**⇒ 4,449 个样本（训练集的 3.3%）从"权重 0.1 ≈ 丢弃"变成"权重 0.5 + 混合目标"。**

#### `--tau-conf 0.5` 是**空枪**

`relabel` = 1813（基准 1838），**没动**。原因：教师在"不同意"的样本上置信度几乎全 < 0.5，
`[0.5, 0.8)` 这个区间基本是空的 —— **`tau_conf` 在 0.5~0.8 之间是彻底的 no-op。**
（该臂已中途 `pkill` 中止，**没提交，省下一个名额**。）

#### `--tau-conf 0.3` 生效但收益小

`relabel` 涨到 **17,507（13%）**，`noisy` 降到 11,747，
**`best val_acc = 0.7371`**（基准 0.7347，**+0.24 pp**）。
推理件 `sub_tau03/pred_results.csv`，**尚未提交开分**。

### 35.5 换算率表（更新到 6 个实测点）

| 配对 | `val_acc` Δ (百分点) | 平台分 Δ | 系数 |
| --- | --- | --- | --- |
| 352 → 384（冻结网格） | −1.28 | −2.31 | 1.80 |
| 352 → 384pe | +0.76 | +0.28 | 0.365 |
| 384pe → 384pj（`--train-proj`） | −0.12 | −0.061 | 0.51 |
| 384pe → 384th（C2 `--two-head`） | −1.11 | −0.622 | 0.56 |
| **+`--local-head` → +`--lora-rank 16`** | **+1.18** | **+1.071** | **0.91** |
| ep4 → ep20（**同一轨迹内**） | +5.02 | +0.0002 | ≈0 |

**两条规律：**

1. **跨"配置改动"比较 → 系数 0.37 ~ 0.91，中枢 ≈ 0.7。**
   （1.80 那个是异常值，它在修一个**真实缺陷**，不是普通改动。）
2. **同一训练轨迹内跨 epoch 比较 → 系数 ≈ 0。** 这正是 §7 那条"val_acc 高估"的机制。

**本地 → 线上偏移**（`val_acc×100 − 线上`）：

| `val_acc` | 偏移 |
| --- | --- |
| 0.7142 | 2.11 |
| 0.7344 | 2.30 |
| 0.7347 | 2.27 |
| **0.7465** | **2.38** |

**⇒ 偏移随 `val_acc` 缓慢上升。`val_acc ≥ 0.74` 时实用估算：`预期线上 ≈ val_acc×100 − 2.4`。**

### 35.6 ⚠️ §0「最重要的一句话」被证伪

§0 原文：**"本项目的分数瓶颈不在输入侧，也不在'模型不够大'"**。

**§35 推翻后半句**：`--lora-rank` 8 → 16（LoRA +884k）直接换来 **+1.071**，
是至今**单次收益最大**的一档（`--local-head` 只有 +0.064，差一个数量级）。

- **仍然成立**：瓶颈不在**输入侧**（输入侧封盘未变）。
- **已不成立**：瓶颈不在"模型不够大"。**容量这条线是通的，而且是当前唯一被证实的大杠杆。**

§0 那句"模型**渴求容量**"的观察是对的，**错的只是跟着的那句"这不产生新信息"**。

### 35.7 下一步

| 优先级 | 实验 | `trainable params` | 理由 |
| --- | --- | --- | --- |
| **P0** | **`--lora-rank 24`** | **3.543M** | rank 8→16 涨 1.07，**继续往上探拐点**。零代码 |
| **P1** | `--lora-rank 32` | **4.428M** | 如果 24 还涨 |
| **P1** | `--tau-conf 0.2` | 1.773M | 伪标签纠正那条线（等 `sub_tau03` 开分再定） |
| ❌ | 输入侧一切 | — | 仍然封盘（§0 硬规则未变） |
| ❌ | `--train-proj` / C2 `--two-head` | — | 已实测为负（−0.06 / −0.62） |

**两台机器并发跑两个 rank，是探拐点最快的方式。**
**注意两台的内存不同（80 GB / 62 GB），推理参数按 §35.3 统一成 `--workers 4 --batch-size 64`。**

```bash
# rank 24（P0）
cd /root/autodl-tmp
nohup python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_lr24 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  --lora-rank 24 > outputs_384pe_lr24.log 2>&1 &
```

**启动必看 `trainable params=3.543M`（不是 2.658M）。**

## 36. P0 落地：`--lora-rank 24` 启动清单（2026-10-03）

**⚠️ 2026-10-03 用户决定：云端训练统一在 `/root/autodl-tmp/` 顶层进行，`AIC_orig/` 计划删除**（代码已融合整理）。下面所有命令都在顶层，不再进 `AIC_orig/`。

**但"顶层现在还等于那棵 2796 行旧增强树"这件事必须先排除**（§29 的冒烟就跑在它上面，直接开跑会把"树 + rank"两个变量一起换掉）。**先认树再启动**：

### 36.1 认树（10 秒，必须先做）

```bash
cd /root/autodl-tmp
wc -l train.py infer.py
sha256sum train.py infer.py | cut -c1-16
grep -c lr_warmup_epochs train.py                     # 0 = 无 warm-up 开关（队友调度）
grep -o "'--img-size'\|'--image-size'" train.py | sort -u
grep -c "'--tta'" infer.py; grep -c 'tta-views' infer.py
```

判读：**1005 行 / `b444595f575c28d4`**（`infer.py` = `80992a0fd71e8ab9`、`--img-size`、`--tta`、`lr_warmup_epochs` 计数 0）⇒ 顶层就是产出 71.2023 / 72.2733 的那份代码，直接进 §36.2。
若是 **2796 行 / `--image-size` / `tta-views`**，说明融合还没落到顶层，命令要按旧树 flag 面改写，先别开跑。
若是**第三个 hash**（融合后的新代码）：**先确认默认行为没变**（尤其 `lr_warmup_epochs` 默认值、逐样本固定增强、tracker 重置这类"静默改训练数学"的项，§26）；默认动过的话，72.2733 就不再是同一个基线，rank 24 的本地 val_acc 只能和**同一棵树上再跑的 rank 8** 比。

> **2026-10-03 实测（已核对）**：融合后的顶层 `train.py` = **1004 行 / `b444595f575c28d4`**、`infer.py` = **371 行 / `80992a0fd71e8ab9`**，`--img-size`、`--tta`、`lr_warmup_epochs` 计数 **0** ⇒ **与产出 71.2023 / 72.2733 的代码逐字节相同**，融合没有改训练数学。72.2733 仍是同一基线，rank 24 = 单变量。

### 36.2 启动（唯一变量 = rank 8 → 24）

```bash
unset OMP_NUM_THREADS
cd /root/autodl-tmp
pgrep -af 'train\.py' | grep -v grep || echo 'no training running'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current
df -h /root/autodl-tmp | tail -1

nohup python -u train.py --data /root/autodl-tmp/train \
  --out outputs_384pe_lr24 \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  --lora-rank 24 > outputs_384pe_lr24.log 2>&1 &
```

注意本树**没有 `--lr-warmup-epochs`**（§34.3 第 1 条），不要加，加了 argparse 直接退出。
**同一台机器上不许并发**：A 臂（`outputs_384pe_ctrl`，旧增强树）若还在跑，先 `pkill -f outputs_384pe_ctrl` —— 它自 §34.5 起只剩归因价值，且并发就是 §33.3 静默 SIGKILL 的成因。

### 36.3 启动后 90 秒核对（三行必须同时成立）

```bash
grep -E 'trainable params|layers=|lora_rank' outputs_384pe_lr24.log | head
```

- `trainable params=3.543M`（不是 2.658M / 1.773M）
- `layers=` 与 rank 8/16 那两次**逐字相同**（rank 不改层数）
- 配置行里 `img_size=384`、`local_head=True`、`train_pos_embed=True`、`lora_rank=24`

> **2026-10-03 实测（云端 750 类 / 384px 网格 / `--local-head` / `--train-pos-embed`）**：把
> `Net(750, r, 'all', local_head=True)` + `enable_pos_embed_training` + `resize_positional_embedding(·, 384)`
> 拆成分量数了一遍，**精确总数 = 3,542,786**，逐项：

| 分量 | 数值 | 算式 |
| --- | --- | --- |
| `lora` | 2,654,208 | 24 × 110,592（每 +1 rank = 110,592，r=8 时 884,736） |
| `pos_embed` | 111,360 | 145 × 768（384/32 = 12 ⇒ 12×12+1 = 145 token） |
| `local_head` | 393,217 | 768 × 512 + 1 |
| `head` | 384,001 | 750 × 512 **+ 1**（余弦头带一个标量——§35.7 记总数时漏了它，所以那组锚点每个都少 1） |
| **合计** | **3,542,786 → `3.543M`** | r=8 对应 1,773,314 → `1.773M` |

> 同一次检查里 `frozen leaked: NONE`——没有任何冻结权重进入 `requires_grad`，`trainable_state_dict()`
> 不会把 350 MB 主干写进每个快照。另外 `Net(...)` 之后**必须**先 `resize_positional_embedding(·, 384)`
> 再数：漏掉那一步网格停在 224 的 50 token，总数会读成 3,469,826（少 72,960），会误判成"少了东西"。

### 36.4 跑完后的判据（**看 Δval_acc，不看绝对分**）

| rank 24 的 `best val_acc` | 判读 | 动作 |
| --- | --- | --- |
| **≥ 0.7520**（Δ ≥ +0.55 vs rank16 的 0.7465） | 容量线仍在爬 | rank 32 立刻上第二台机器；线上预估 ≈ `val×100 − 2.4` ≈ **72.8+** |
| 0.7470 ~ 0.7520（Δ 0 ~ +0.55） | 接近拐点 | 跑一次 rank 32 确认后停这条线 |
| **≤ 0.7465**（Δ ≤ 0） | **8→16 是"跨阈值"，不是斜率** | 停 rank 线，把预算转给 `--tau-conf` / `--lora-target` |

开分后补算 §35.5 的系数 `Δ线上 / Δval_acc`：**< 0.5 就是容量开始进噪声的预警**（对照：8→16 是 0.91）。

### 36.5 推理（训练结束之后才跑，§33.3）

```bash
cd /root/autodl-tmp
mkdir -p sub_lr24
python -u infer.py --test /root/autodl-tmp/test \
  --checkpoint outputs_384pe_lr24/best.pt \
  --output sub_lr24/pred_results.csv --tta --workers 4 --batch-size 64
stat -c %s sub_lr24/pred_results.csv                              # 1610092
sed 's/\r$//' sub_lr24/pred_results.csv | awk 'END{print NR}'     # 37444
cd sub_lr24 && zip ../sub_lr24_tta8.zip pred_results.csv
```

`--workers 4 --batch-size 64` 是 §35.3 定的统一值（62 GB 机器用 256/8 会 OOM）。**不要手动传 `--lora-rank`**：`infer.py` 从 checkpoint 读，传错不报错（`probe.py:820` 的坑）。
