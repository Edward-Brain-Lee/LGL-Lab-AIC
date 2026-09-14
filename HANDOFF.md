# HANDOFF — LGL-Lab AIC 算法大赛（噪声标签细粒度识别）

> 交接文档。最后更新：2026-09-14。
> 面向：接手继续推进的工程师。假设你会 PyTorch、用过 Linux 命令行，但不了解这个赛题和之前发生过什么。

---

## 0. 30 秒现状

| 项目 | 状态 |
| --- | --- |
| 云端训练环境 | AutoDL 租用 **RTX 4090 24GB**，已跑通，数据/权重/依赖都已就位 |
| 代码位置（云端） | `/root/autodl-tmp/` |
| 代码位置（本地） | `C:\Users\Ed\Desktop\Recent Project\LGL-Lab-AIC\` |
| 已完成 | 一版**有 bug 的** 20 epoch 基线训练，已生成提交 CSV |
| 平台实际得分 | **68.02**（Top-1 准确率，即 0.6802） |
| 基线自测指标 | `val_acc = 0.7305`（**注意：这个数是乐观的，见 §7**） |
| 代码修复 | **已改完 3 个 bug，但还没在真实数据上跑过**（只过了 `selftest.py`） |
| 下一步 | 上传修复后的代码 → 跑 selftest → 重新训练 20 epoch → 用排行榜选轮次 → 重新提交 |

**最重要的一句话**：上一轮训练是有 bug 的，修复已经写好但**未经验证**。当前第一优先级是把修复后的版本跑起来，而不是继续加新东西。

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
| `train.py` | ~634 | 训练主程序：模型定义（LoRA/余弦头/原型头）、数据、训练循环、评估、存档 |
| `infer.py` | ~107 | 用 checkpoint 跑测试集，生成官方格式 `pred_results.csv` |
| `losses.py` | ~107 | 鲁棒损失：`ce` / `gce` / `nce` / `rce` / `apl` |
| `noise.py` | ~132 | `LabelTrustTracker`：逐样本标签可信度跟踪、噪声过滤、伪标签纠正 |
| `selftest.py` | ~370 | **改代码后必须先跑这个**：约 10 秒，不需要数据/GPU/CLIP 权重 |
| `README.md` | — | 赛题原文（不要改） |
| `README_AUTODL.md` | — | AutoDL 平台操作手册 + 方法说明 + 参数表（中文，给操作者看的） |
| `HANDOFF.md` | — | 本文件 |

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
5. y_used = tracker.label[idx]                 # 可能被纠正过的标签
   w      = tracker.weight[idx] * max_p^2      # 逐样本权重
   w      = w / w.mean()                       # 归一化，稳定有效步长
6. loss = 0.5*(w*CE(out,y_used) + w*CE(out2,y_used))          # 标注项
       + robust_weight * w * APL(out/out2, y_used)            # 鲁棒项 0.5
       + consistency_weight * MSE(z, z2.detach())             # 一致性 0.05
       + anchor_weight * (1 - cos(z, anchor))                 # 锚定 0.1
       + proto_weight * w * CE(proto.logits(z), y_used)       # 原型对比 0.5
7. 反向 + 梯度裁剪(1.0) + AdamW
8. EMA 更新教师权重: tp = 0.995*tp + 0.005*sp   （只更新 requires_grad 的参数）
9. model.proto.update(tz, y_used, trusted)     # 只用被信任样本
```

每个 epoch 开始时（非 warm-up）调用一次 `tracker.refresh()` 重算标签和权重。

优化器 `AdamW(lr=2e-4, weight_decay=0.05)` + `CosineAnnealingLR`（周期 = `--epochs`）。
AMP 默认 **bf16**（4090 是 Ada 架构，比 fp16 + GradScaler 更稳更快，也省掉梯度缩放的随机性）。

### 3.4 噪声处理（`noise.py`）——本项目最核心的算法

`LabelTrustTracker` 维护**每个训练样本**的教师后验 EMA（`momentum=0.9`）。规则：

| 条件 | 标签 | 权重 |
| --- | --- | --- |
| `argmax p == 给定标签`（教师**同意**） | 保持给定 `y` | `1.0` |
| 不同意 且 `max(p) >= tau_conf`(0.8) | **改成** `argmax p`（伪标签） | `w_relabel` = 0.5 |
| 不同意 且 `max(p) < 0.8` | 保持给定 `y` | `w_noise` = 0.1 |
| 采样器还没抽到过 | 保持给定 `y` | `1.0` |

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

wc -l pred_results.csv        # 初赛必须 = 24967
head -3 pred_results.csv      # 形如 00012f3f....jpg,0083
```

### Step 8 — 打包提交

```bash
zip submission.zip pred_results.csv
```

**提交前清单**：

- [ ] 行数 = 24967
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

**改动还没有上传到云端。** 云端 `/root/autodl-tmp/` 里现在是**旧的、有 bug 的版本**。

---

## 9. 现在要做什么（立即下一步）

### 9.1 上传修复后的代码

用 FileZilla 把本地这 5 个文件覆盖到 `/root/autodl-tmp/`：

```
train.py    noise.py    losses.py    selftest.py    README_AUTODL.md
```

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
- [ ] 本地 `HANDOFF.md` / 代码是否已同步到云端（**云端现在是旧版本**）

**第一件事：跑 §9 的流程。**
