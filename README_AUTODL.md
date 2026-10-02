# LGL-Lab-AIC 训练说明（AutoDL / 4090）

严格使用 **OpenAI CLIP ViT-B/32** 作为唯一骨干，**单模型、单推理流程**，不使用任何额外数据、不使用测试集训练、不做模型集成。

## 0. 文件

| 文件 | 作用 |
| --- | --- |
| `train.py` | 训练主程序（LoRA + 噪声筛选/伪标签纠正 + 鲁棒损失 + 原型对比 + CLIP 锚定） |
| `infer.py` | 生成官方格式的 `pred_results.csv` |
| `losses.py` | 鲁棒损失：CE / GCE / NCE / RCE / APL |
| `noise.py` | 逐样本标签可信度跟踪、噪声过滤与伪标签纠正 |
| `analyze.py` | 训练后的错误分析报告（哪些类坏了、混淆是不是单向的）。`selftest.py` 会 import 它 |
| `selftest.py` | **先跑这个**：10 秒自检，不需要数据 / GPU / CLIP 权重 |
| `valmetrics.py` | 对**已有 checkpoint** 算 micro / macro 准确率并排对比，零训练成本。用来判断"排行榜分数比 `val_acc` 低"是不是度量口径造成的。**脚本会自检：`val_acc` 必须等于训练日志里的值** |
| `probe.py` | **只读诊断，先跑这个**：一次冻结 CLIP 前向，测**标签噪声率 ρ**、构造干净代理集 `V*`、量测 train/test 域偏移、搜索 TTA/分辨率配方（`--tta`）、输出**噪声网格图**供人工确认噪声主线。`--calib` 是**离线排行榜代理**：给已有 checkpoint × 视图集在 `acc@V*` 上打分，**0 个提交名额**，且**必须先过校准门**（见 §4.5）。不训练、不提交 |
| `perclass.py` | 对**已有 checkpoint** 打召回率分布直方图 + 每个坏类被误判成了什么。诊断"哪些类卡死了"。**注：至 2026-09-24 尚未创建**（此前的文档记载有误） |

## 1. 方法

1. **LoRA**（Hu et al., ICLR 2022）：只训练 CLIP 视觉塔的低秩适配器 + 余弦分类头，约百万级参数，预训练先验不会被噪声带偏；
2. **类别均衡采样**（`1/sqrt(freq)`）：为复赛/半决赛的长尾阶段准备；
3. **纯 CE warm-up**（默认 3 epoch）。warm-up 结束后由 **EMA 教师**驱动：
   * **噪声过滤 + 伪标签纠正**（DivideMix / co-teaching 思路），判据是**教师的首选类别**而不是它的绝对置信度：
     教师首选 == 给定标签 → 保留原标签、权重 1；首选 ≠ 给定标签且 `max(p) ≥ 0.8` → **改标注**为教师预测、权重 0.5；首选 ≠ 给定标签且不确定 → 权重降到 0.1（不删除样本，采样计划始终有效）；
   * **置信度重加权**：教师置信度 `^2` 作为额外权重；
4. **APL 鲁棒损失**（Ma et al., ICML 2020）：`NCE + RCE`。NCE 在 `p > k` 时替换为 `-log p` 在 `p=k` 处的切线，梯度不会像 CE 那样消失，因此**记住错误标签的样本无法主导更新**；
5. **原型对比损失**（MoPro / Sel-CL 思路）：维护每个类别的 EMA 特征原型（用 warm-up 期间冻结 CLIP 特征的类均值初始化，之后只用**可信样本**的教师特征更新），对学生特征做余弦 NCA；
6. **两视图一致性 + 冻结 CLIP 锚定**：抑制表征漂移与灾难性遗忘。

### 相对原版代码修掉的问题

* **anchor 不再 deepcopy**：用 `lora_disabled` 复用同一份冻结权重，省掉一整份视觉塔（约 350 MB）；`best.pt` 只保存可训练参数，从 ~350 MB 降到 **几 MB**；
* **验证集标签本身也是噪声**：改为同时打印 `val_acc`（全部）与 `val_acc_hi`（高置信子集，更接近真实精度）；
* **推理不再丢图**：原来 `except: pass` 会让读不出的图片在 CSV 里**缺行**（提交直接判无效）；现在用灰图兜底，保证行数与测试集图片数一致；
* **可复现**：默认关闭 `cudnn.benchmark`、DataLoader 显式传 `generator` + `worker_init_fn`，checkpoint 里保存 `args` 与 RNG 状态，支持 `--resume` 断点续训；
* **增强的随机性改成按样本 index 派生，不再依赖 `--workers`**：原先图像增强是从*该 worker 自己的*
  torch/python RNG 取随机数的，于是"哪张图被哪个 worker 取到"决定了它被怎么增强，而这个映射随
  `--workers` 变化。**每种 `--workers` 都能正常跑、正常收敛，只是收敛到不同的点**，故障因此完全
  不可见——唯一的症状是"提交的代码复现不出提交的分数"，而赛题明确要求可复现。
  现在 `augmented(img, i, view)` 由 `(seed, index, view)` 派生种子（`fork_rng` 隔离，
  先归一化到 2^63-1 空间，149k×2 个取值零碰撞），增强成为**纯函数**，与 worker 数无关；
  `selftest.py` 新增断言：同一 index 在 `--workers 0/1/2/3` 下逐位相同，且同一张图的两个视图不相同
  （否则双视图一致性损失是拿一张图和自己比）；
* 默认 `bf16`（4090 是 Ada，比 fp16 + GradScaler 更稳、更快，并省掉梯度缩放的随机性）；
* **LoRA 层必须是 `nn.Linear` 的完全替身**：`nn.MultiheadAttention` 不走 `out_proj(x)` 调用，
  而是直接读 `out_proj.weight` / `.bias` 交给底层函数。原来的包装只实现了 `forward`，
  于是 `AttributeError: 'LoRALinear' object has no attribute 'weight'`；现在 `LoRALinear`
  暴露 `weight`（返回 `base.weight + scale·B@A` 的**合并权重**）/ `bias` / `in_features` /
  `out_features`，注意力里的 `out_proj` 也能正常适配（不再静默失效）。
  `selftest.py` 的桩模型里放了真的 `nn.MultiheadAttention` 来守住这条回归；
* **学习率不再第一轮就给峰值**：`CosineAnnealingLR` 起步即 `--lr`，而那一刻 `head.weight`
  是完全随机初始化的——750 类余弦头要在它一辈子最大的步长下、顶着错标率不低的标签，
  同时找到自己的尺度和 750 个方向。现在 `--lr-warmup-epochs`（默认 0，匹配 71.13 队友配方）
  可显式设为 1 先线性爬升，
  再余弦退火；`--lr-warmup-epochs 0` **逐位还原**旧调度，所以加这个改动之前的 run 依然可复现。
  （调度器是自写的 `WarmupCosineLR`，不用 `SequentialLR`/`LambdaLR`：前者的交接语义随 torch
  版本变，后者的 `state_dict` 会把 lambda 清空，两者都会让 `--resume` 出问题。）
* 骨干用 `ViT-B-32-quickgelu`，与 OpenAI 权重的 QuickGELU 激活一致（见 Step 3）；
* **噪声筛选不再用绝对置信度当门槛**：原判据 `p[y] ≥ 0.8` 要求教师"既同意又自信到 0.8"才肯相信，
  但 500 类的余弦头 logit scale 是慢慢长起来的，实测**只有 16.6% 的样本达标**，其余（包括教师
  *明确同意*、只是 p[y]≈0.5 的样本）全被压到 0.1 权重，`noisy` 数量连续每轮**正好卡在 40% 上限**。
  改为看教师的**首选类别**：同意即全权重，只有"首选 ≠ 给定标签"才算反对意见；
* **统计口径自洽**：`clean/relabel/noisy/unseen` 现在严格划分整个训练集（原来被上限救回的那批样本
  不计入任何一类，四个数加起来只有 53125，而训练集是 93102，从日志上根本看不出问题）；
* **`rce` 改回 APL 论文定义**：原实现返回的是预测**熵**，而最小化熵是在*鼓励*过度自信，和它
  自己文档里写的"damps over-confidence"正好相反。现为 `-log(1e-4)·(1-p_y)`，即论文里的 MAE 型项。

## 2. AutoDL 运行步骤

**Step 0（省钱）** 用**无卡模式**开机，传数据 / 装依赖 / 下权重；全部就绪后再关机切换到 GPU 模式训练。

**Step 1 上传** FileZilla → 主机填实例的 `region.autodl.com`，**端口填 SSH 登录指令里 `-p` 后面的那个数字**（不是 22），协议 `SFTP`，用户 `root`，密码为实例密码。

* 代码：`train.py` / `infer.py` / `losses.py` / `noise.py` / `analyze.py` / `valmetrics.py` /
  `datastats.py` / `selftest.py` / `probe.py` / `requirements.txt` → `/root/autodl-tmp/`。
  **必须放在同一层目录**——`train.py` import `losses`/`noise`，`selftest.py` import 其余全部，
  `probe.py` import `train`（拿它的常量与数据集类，因此也会连带 import `losses`/`noise`）。
  别传 `__pycache__/`
* 数据：几万张小文件用 FileZilla 传极慢，**先在本地打成 zip**，上传后 `unzip train.zip -d /root/autodl-tmp/`
  （Windows 下用 `tar -a -c -f train.zip train` 或右键压缩；AutoDL 上 `apt install unzip -y` 若缺）

**Step 2 环境**
```bash
cd /root/autodl-tmp
nvidia-smi                       # 应看到 RTX 4090 24GB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Step 3 CLIP 权重**（国内最容易卡的一步）

`open_clip` 3.3.0 已改为从 **HuggingFace** 拉 OpenAI 权重（不再是 Azure CDN），国内直连会
`Network is unreachable`。先设镜像，**每次开新终端都要设**（写进 `~/.bashrc` 一劳永逸）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```
设好后第一次运行会自动下载（`timm/vit_base_patch32_clip_224.openai`，605 MB，落在
`~/.cache/huggingface/`），之后走缓存、不再联网。

也可以手动下载官方 `.pt` 再用 `--pretrained` 指过去（同样的权重，只是走 Azure CDN）：
```bash
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18e219bf2a342f68e27b3960b7620019/ViT-B-32.pt -O /root/autodl-tmp/ViT-B-32.pt
```

> **必须是 `ViT-B-32-quickgelu`**（默认值）。OpenAI 官方 ViT-B/32 用的是 QuickGELU 激活，
> 如果建成普通的 `ViT-B-32`（GELU），open_clip 会打印
> `QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'openai'`
> ——那不是"无害警告"：模型会拿 GELU 去跑按 QuickGELU 训练的权重，**精度会掉**。

**Step 4 自检（一定要先跑）**
```bash
python selftest.py
```
输出 `ALL CHECKS PASSED` 才继续。它会用桩模型跑完一次完整的 4 轮训练 + 推理 + CSV 校验。

**Step 5 冒烟测试（先跑 1 个 epoch 的 20 步）**

用**和 Step 6 完全一样的 batch-size / workers**，否则测不出正式运行时会不会爆显存：
```bash
python train.py --data /root/autodl-tmp/train --out ./outputs \
  --epochs 1 --warmup-epochs 1 --limit-batches 20 --batch-size 128 --workers 12
```
另开终端 `watch -n 1 nvidia-smi` 看显存与 GPU 利用率。**如果 GPU 利用率长期低于 60%，说明是数据加载瓶颈**，把 `--workers` 加到 16~20。

冒烟测试看三件事，**不看准确率**（只跑了 20 步，`noise stats` 必然是乱的）：

1. **不报错**：`CUDA out of memory` / `DataLoader worker ... killed` 都要在这一步暴露，别留到两小时的正式训练；
2. **显存**：应明显低于 24 GB，留出余量；
3. **速度**：日志里 `time=xx.xs` 是这 20 步的耗时，`20 步 × (训练集/128) ÷ 步速` 就是一轮的真实时间，
   乘 20 轮再乘 2（两个分支）就是总预算。先算清楚再开跑。

**Step 6 正式训练**
```bash
python train.py --data /root/autodl-tmp/train --out ./outputs \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 12
```

> **不要加 `--pretrained`。** 默认值 `openai` 就是对的：Step 3 走镜像下好的权重已经在
> `~/.cache/huggingface/` 里，`open_clip` 直接命中缓存。只有当你走的是 Step 3 下半段那条
> `wget ...ViT-B-32.pt` 的路子时，才需要 `--pretrained /root/autodl-tmp/ViT-B-32.pt`
> ——那个文件不存在的话这一步会直接报错；而且 `train.py` 看到 `--pretrained` 不是 `openai`
> 会打一条"可能不符合规则"的警告（规则 五.1），属于误报，别被它吓到，也别因此改成别的值。
中断了就 `--resume ./outputs/last.pt` 续训（记得带上其余参数）。

> **当前已知最好的训练配置是再加一个 `--image-size 320`**（同一套配方，只改分辨率：
> 320+tta4 = **69.968** > 288+tta4 = 69.218 > 224+tta4 = 66.2456）。显存约 14.2 GB、
> 442~454 s/epoch，224 是 235 s/epoch。代价与收益的算法见 §4 第 8 项，
> **别再往上试 352**——那是第一条要上采样的尺寸，先验 ≲+0.3 点。

**Step 7 推理 + 提交**
```bash
unzip test.zip -d /root/autodl-tmp/                 # 得到 /root/autodl-tmp/test/*.jpg
python infer.py --test /root/autodl-tmp/test --checkpoint outputs/best.pt \
  --tta-views tta4 --workers 8 \
  --output pred_results.csv --logit-adjust 0 0.25 0.5
wc -l pred_results.csv                              # 初赛 24967 / 复赛 37444
head -3 pred_results.csv                            # 形如 00012f3f....jpg,0007
zip submission.zip pred_results.csv
```

> `--tta-views tta4` 就是拿过 66.2456 的那 4 视图配方（4 次前向），**默认不动 `--tta-agg`**
> （`logit`：逐视角过同一个头再平均 logit，已拿分的都是这么测的）。
> `--workers 8` 只影响**多久跑完**，不影响结果（逐字节相同），测试集 3.7 万张时别省这个参数。
> 单视图对照 / 其它配方的 CSV 见 §4.5.4。**换配方等于换一份提交，每天只有 2 个名额。**

> `--logit-adjust` 一次前向出多个文件（`pred_results.csv` / `..._tau025.csv` / `..._tau050.csv`），
> **每个 tau 都是一份合法提交**，可以分别上排行榜试。它只在"训练集不均衡、测试集均衡"时才有用——
> 复赛训练集实测近乎均衡（见 HANDOFF §14.1），所以**预期收益很小**，当 tie-breaker 用就好。

## 3. 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--warmup-epochs` | 3 | 纯 CE 预热轮数，之后才启用筛选/纠正（**注意与下面的 LR warmup 是两回事**） |
| `--lr-warmup-epochs` | 0 | **学习率**线性爬升轮数：从 `--lr × 0.1` 升到 `--lr`，之后余弦退火。`0` = 71.13 队友配方的峰值首轮调度；设为 `1` 才启用 warm-up |
| `--model` | ViT-B-32-quickgelu | 别改成 `ViT-B-32`，见 Step 3 |
| `--tau-conf` | 0.8 | 教师**否定**给定标签时，要多自信才敢改标注 |
| `--w-noise` / `--w-relabel` | 0.1 / 0.5 | 不可信样本 / 被改标注样本的损失权重 |
| `--max-noise-frac` | 0.4 | 最多判定多少比例的样本为不可信（防止教师未收敛时误杀） |
| `--relabel-mix` | 0.5 | 改标注时有多少目标质量移到教师的选择上。`1.0` = 硬覆盖（旧行为） |
| `--label-smooth` | 0.05 | CE 项的标签平滑——抗"记忆结构化噪声"最便宜的一条 |
| `--robust-loss` | apl | `apl`（NCE+RCE）/ `gce` / `nce` / `rce` / `ce`，建议做消融 |
| `--robust-weight` | 0.5 | 鲁棒损失权重 |
| `--apl-k` / `--apl-rce` | 0.2 / 1.0 | NCE 拐点；RCE 缩放（调小更温和） |
| `--proto-weight` | 0.5 | 原型对比损失权重 |
| `--anchor-weight` | 0.1 | 冻结 CLIP 锚定权重 |
| `--lora-rank` / `--lora-target` | 8 / all | `mlp` 只加在 MLP 上，更保守 |
| `--val-ratio` | 0.1 | 验证集比例；定稿冲分时可降到 0.05 |
| `--image-size` | 224 | 训练分辨率。**必须是 32 的倍数**（256/288/320/352/384），不要 336，见 §5 |
| `--noise-judge` | 关 | 用**冻结 CLIP 的独立判断**否决“教师同意⇒标签干净”这条分支；被判为 suspect 的样本只降权、不会改标。默认关：先让 `probe.py` 量出它的误杀率再开 |
| `--judge-margin` | 0.02 | 另一个类的质心要近多少（余弦）才判定"可疑"。0 = 平手也算；越大越保守 |
| `--select` | val_acc | 选最佳权重依据，可换 `val_acc_hi` |
| `--save-every` | 4 | 每 N 轮额外存一个 `epN.pt`（0 = 只留 best/last）。**这是拿真实排行榜选轮次的唯一手段**，见下文 |

推理端（`infer.py`，都不需要重训）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--tta-crops` | center | 要平均进来的裁剪策略，可选 `center` / `full` / `pad`。**最安全的一条 TTA**（不碰位置编码），见 §4 第 7 项 |
| `--tta-ratios` | plain | 中心裁剪的**短边比例**（只对 `center` 有效，`full`/`pad` 自己就定了短边）：`wide` 1.0 / `plain` 1.143（训练默认）/ `mid` 1.286 / `tight` 1.429。**这就是拿过 66.2456 的那条轴**（缩放轴），单品 `wide` 单用掉过 2.16 分，赢的是"平均"不是"某一档"，见 §4.5.4 |
| `--tta-sizes` | 空 | 额外分辨率，如 `288 320`。**只对在该尺寸上训练过的 checkpoint 才有效**——在没训过的尺寸上推理属于"只插值"，是唯一有掉分记载的情形，见 §4 第 8 项 |
| `--tta-flip` | 关 | 水平翻转。便宜且安全，但预期收益很小：CLIP 特征在翻转下余弦相似度约 0.99，两个视角近乎冗余 |
| `--tta-views` | 空 | **直接用 `train.VIEW_SETS` 里的具名视图集**，如 `--tta-views tta4`。排行榜验证过的那 4 视图**用乘积式参数拼不出来**（`--tta-flip` 会把每个视角都翻，得到的是 6 视图超集），所以复现已拿分的配方必须走这条路。多个名字取并集。**与乘积式参数（`--tta-sizes/--tta-crops/--tta-ratios/--tta-flip`）同时给会直接报错**，不是"以谁为准"——否则 CSV 看起来像那个具名配方却不是它。可选名字见 `train.VIEW_SETS` |
| `--tta-agg` | **logit** | `logit` = 逐视角过分类头后平均 logit（**默认**，本工程拿到的每一个分数 64.16 / 66.2456 / 69.218 / 69.968 都是这么测的）；`feat` = 各视角特征先 L2 归一化、求平均、再归一化一次，分类头只作用一次。两者都只用**一个** checkpoint 和一个头，不构成多模型融合 |
| `--workers` | 0 | 变换（解码 + 缩放）的并行进程数。**这是多视角跑得慢的原因**：8 视图 @288、3.7 万张测试图，单线程约 2 小时。变换在图片之间**完全可并行**，而输出**与进程数无关（逐字节相同）**——变换是确定性的、loader 按序产出，`selftest.py` 断言的是这件事而不是假设它 |
| `--amp` | none | **提交路径保持 fp32**（已测过的分数都是 fp32 下拿的）。`bf16` 只用于多视角扫参，别直接用于最终提交 |

> `noise stats` 里的四个数（`clean` / `relabel` / `noisy` / `unseen`）**加起来必须等于训练集大小**。
> 如果 `noisy` 恰好等于 `max_noise_frac × N`，说明它长期撞在上限上——阈值对这个模型不成立，需要调。
> 另外 `mean_weight` 是平均样本权重，跌破 0.35 时会自动打 `!! 警告`（等于大部分数据不产生梯度）。
> `vetoed` 是 `--noise-judge` 判为可疑（被降权、但**没有被改标注**）的样本数。
>
> **预热结束会 `reset` 一次 tracker 的后验**（日志里会打印）。原因是后验从第 0 个 batch 就在累积，
> 而那时教师还是一个随机初始化的头；又因为**样本的第一次观测是原样写入**（不是平均进去的），
> 这些噪声不会被后续更新冲掉。reset 之后下一个 batch 的教师后验成为"第一次观测"。
> 代价是 reset 后的那一轮 `unseen == N`（等于不筛选），再下一轮才恢复筛选。

> **`val_acc` 会高估真实水平，不要只信它。** 验证集是从训练集里切出来的，**带着同一套结构化噪声**。
> 设真实准确率为 `a`、噪声比例 `η`：如果噪声是随机的、模型又完全抵抗住了，那么
> `val_acc ≈ (1-η)·a`，**必然小于 `a`**。实测却相反：`val_acc = 0.7305` 而平台真实得分
> **0.6802**，代进去 `1-η = 0.7305/0.6802 = 1.074 > 1`，是不可能的。
> 唯一解释是模型把给定标签里的**错误**也学了过去，而且这种学习**泛化到了没见过的验证集图片**——
> 随机噪声做不到这一点，所以噪声是**类间一致的**（README 简报第 58 行的"弱相关标注"）。
> 日志里 `clean`（`p[y]≥0.8`）从 8797 一路涨到 43859、`mean_trust` 从 0.297 涨到 0.628 都没饱和，
> 就是它在持续吃噪声的证据。
>
> 结论：**`val_acc` 奖励的正是我们要避免的行为**。所以每 4 轮存一个快照，训完用 `infer.py`
> 各生成一份 CSV 提交，**看排行榜选轮次**——如果某个较早的快照分数更高，就说明后面的轮次
> 纯粹在记噪声。

> 注意：APL 里的 NCE 是“主动”损失，**warm-up 之后打印的 loss 可能为负**（`k=0.2` 时下界约 `-2.39`），这是设计如此，不是发散。

## 4. 建议的消融顺序（每项单独跑，别一次全改）

**复赛最该先做的对照**是"抗记忆三件套"整体开关，因为它是本轮唯一的算法改动方向：

0. **新默认 vs 旧行为** → 默认（`--label-smooth 0.05 --relabel-mix 0.5`）
   vs `--out ./outputs_old --label-smooth 0 --relabel-mix 1.0`。
   **两边同一轮次各提交一次比排行榜**——别比 `val_acc`，`val_acc` 奖励的正是我们要避免的行为。

   > **实测修正（2026-09-23，见 HANDOFF §15.2）**：这个 A/B **并不干净**。
   > `relabel` 在两个分支里都只占 **1.7%** 的样本，所以 `--relabel-mix` 0.5 vs 1.0
   > 几乎没有区别——**这一对实际上只在比 `--label-smooth` 0.05 vs 0**。
   > 想单独测伪标签纠正，得先把 `--tau-conf` 从 0.8 调低（750 类下 argmax 置信度
   > 到不了 0.8，这套机制等于没开）。
   >
   > 另外：`val_loss` **不能跨分支比**（两边损失函数定义不同），跨分支唯一可比的是 `val_acc`。
   > 而 `val_acc` 在这个数据集上会高估 6.5 点（HANDOFF §15.4），所以**只有排行榜能裁决**。

然后：

1. 原版 GCE vs 新的 **APL** → `--robust-loss gce` vs `apl`；
2. 关掉噪声筛选/纠正 → `--w-noise 1 --w-relabel 1`（等价于只做置信度重加权）；
3. 关掉原型对比 → `--proto-weight 0`；
4. 关掉锚定 → `--anchor-weight 0`；
5. LoRA 位置 → `--lora-target mlp`；
6. **冻结 CLIP 独立判据** → `--noise-judge`（可配 `--judge-margin`）。
   这是本方案唯一不依赖类名的语义纠错手段，也是"自动、无需人工清洗"的落点。
   **先跑 `probe.py` 看它的误杀率再开**：`probe.py` 报的 `s_other > s_self` 比例就是它的
   触发率，而 `V*` 上的准确率就是它的精度；精度不够时调大 `--judge-margin`。
   注意它是**降权、不改标注**（`vetoed` 计数会出现在 `noise stats` 里）。
7. **裁剪策略** → `infer.py --tta-crops center full pad`（**不需要重训**，纯粹是推理端）。
   默认的 `center` 并不是"整张图"：实测 3:4 的图只保留 **57.5%** 的画面，2:3 的只剩
   **51.0%**，而**正方形也只有 76.6%** —— 因为 `Resize(size*256/224)` + `CenterCrop(size)`
   本来就砍掉两个方向各 1/8。细粒度判据小、又常常不在正中，被裁掉的可能正是决定性的那块。
   三种策略都留在训练时的 patch 网格里，**不需要位置编码插值**，所以这条比第 8 项安全得多，
   **建议先测这条**。哪些策略该进最终提交由 `probe.py --tta`（冻结 CLIP，快）粗筛，
   再由 `probe.py --calib`（真 checkpoint，`acc@V*`，过了校准门才算数）定夺。
   同一根轴上还有更细的一档 `--tta-ratios`（`wide`/`plain`/`mid`/`tight`），见 §4.5.4。

   > ⚠️ 但"不碰位置编码"**不等于**"没有风险"。三种策略的**表观物体尺寸**差别很大
   > （相对 `full`=1.000，以 3:4 图为例）：训练 RRC 是 1.000~1.348（均值 1.148）、
   > `center` 1.320、`full` 1.000、`pad` 0.866。也就是说 `full`/`pad` 让物体比训练均值**更小**，
   > 正是 FixRes（Touvron 2019）警告的那种失配；`center` 则偏**大**。
   > 它和"`center` 裁掉画面"这两条效应**方向相反**，会部分抵消 ——
   > 所以既不能推断 `center` 赢，也不能推断 `full` 赢，**只能实测**。
   > 另外 `pad` 会在画布上引入合成灰边（2:3 竖图约 34% 的像素是模型没见过的纯灰），
   > 它是三种里唯一把**输入分布**推到训练分布之外的，风险最高。
8. **分辨率** → `--image-size 288` / `--image-size 320`（**不要 336**，见 §5）。
   这条与噪声轴**不冲突**，可以和第 6 项同时上。**但关键在于：
   有价值的是"在目标分辨率上训练"，不是"推理期换分辨率"。**

   这两件事常被混为一谈，而证据指向**相反**的两个结论：

   * **只插值、不在该尺寸上训练过** → 会掉分（这是唯一有记载的不利情形）：
     DeiT-tiny 插值位置编码、不微调，224: 72.2 → 384: 71.2 → 448: 68.8 → 512: 65.9。
     常有引用的"CLIP-B/32 到 320 微调后仍不如 224"（open_clip Discussion #987）
     是**单条、未复现**的报告，训练配置未公布、线程无结论，**大概率测的就是这种情形**。
   * **在目标尺寸上训练** → 同骨干、同配方、只改微调分辨率，ViT-B/32 在
     **384 上是 +2.1、448 上是 +2.5**（timm/Cherti, ImageNet-1k）。

   ⇒ 所以 `train.py --image-size 320` 是有希望的那条路；而拿一个 **224 训出来的** checkpoint
   去 `infer.py --tta-sizes 320`，正是**已知会掉分**的那条路。`infer.py` 的 `base` 读的是
   checkpoint 里的 `image_size`，`--tta-sizes` 应当只在**该 checkpoint 训练过**的尺寸上使用。

   **已落地，而且兑现了**：同一套配方只改训练分辨率，平台的实测阶梯是
   64.16（224 单视图）→ **+2.09** → 66.2456（224+tta4）→ **+2.97** → 69.218（288+tta4）
   → **+0.75** → **69.968**（320+tta4）。注意最后一步：320 上 `val_acc` 只动了 0.0003
   而平台涨了 0.75 —— **涨的来自"输入表示变好"，不是"模型学得更多"**。
   同样地，`val_acc` 与平台的分差随表示变好单调收窄（6.54 → 4.45 → 3.34 → **2.62** 点）。
   再往上（352）会开始**上采样**（所需短边 402 > 测试集中位数 375），先验只有 ≲+0.3 点，
   而且 320 已经把源图自己的像素用掉 0.98 倍 —— **这条轴接近用完了**，不值得再花那 3 小时。
   真正该用哪个尺寸由 `probe.py --tta` 实测决定
   （它会在冻结 CLIP 上先做分辨率门控）。另见 §4.5：`--img-size` 是 `--image-size` 的别名
   （队友那边的拼法），checkpoint 里的分辨率两个键都认，**走别名会打日志**。

## 4.5 离线排行榜代理（0 个提交名额）

**为什么有这一节**：全队**每天只有 2 个提交名额**，比 GPU 时间紧得多。在这之前，任何一个关于
"该用哪套推理视图"的判断都要**烧掉一个名额**。这一节买的是一份本地信号。

### 4.5.1 怎么跑

```bash
python selftest.py                     # 必须 ALL CHECKS PASSED
# ⓪ 冒烟（一分钟上下）：只验"能不能加载、塔建得对不对"，数字全部无意义。
#    --calib-views 不能少于 plain wide tta4：门就定义在这三个上，少给一个会被
#    **在冻结 judge 跑完之后**直接拒绝（时间白花，什么都没得到）。
#    **必须单独给 --out**：缓存文件名里只有视图，没有图片列表。现在读回时会比对
#    图片列表、对不上就重算并打印 NOT reused，但分开 --out 是不必依赖那条保护的做法。
#    --workers 默认 12，每个 worker 都带着一份模型：小内存实例会被 OOM 杀掉，
#    冒烟给 4 就够。
python probe.py --calib --checkpoint outputs/ep20.pt --limit 16 \
  --calib-views plain wide tta4 --workers 4 --out ./probe_smoke
# ① 先校准（必须有：没校准过的代理不是证据）。**必须传同一轮的两个 epoch**——
#    门要比较 ep 之间是否相等，只给一个 checkpoint 门会直接判 FAIL 并停手
python probe.py --calib --checkpoint outputs/ep4.pt outputs/ep20.pt
# ② 过了门，再做全扫描（9 个单视图 + 36 个两两组合 + train.VIEW_SETS_SWEPT 的 10 个多视图集 = 55 个；
#    其中在本次分辨率上退化的会被自动跳过并说明原因，例如 320 的 checkpoint 上 tta4s320 就不必跑
#    ——它的 5 个视图里有两个是同一批像素。单独 s320 不跳过：它等于 plain，跑出来和 plain 同数）
python probe.py --calib --calib-sweep --checkpoint outputs/ep4.pt outputs/ep20.pt
# ③ 聚合方式的裁决：同一条扫描换一种聚合再跑一遍（缓存按 agg 分文件，重复跑是免费的）
python probe.py --calib --calib-sweep --checkpoint outputs/ep4.pt outputs/ep20.pt \
  --calib-agg feat --out ./probe_feat
```

> 冒烟那步会打印 `self-check ... SKIPPED -- --limit ...`，**这是预期的**：`--limit` 只跑了前
> 16 张图，它的 `val_acc` 必然对不上训练时记录的全量值，所以这一步显式跳过而不是让断言误报。
> 只有那一步会这样；① ② ③ 都是全量，自检必须通过（打印 `val_acc reproduces training`）。
> 另外，如果某个 `--out` 里留着上一次不同参数的缓存，会看到 `[cache] ... NOT reused`：
> 那是正常的（说明它没被读错），不必重跑，也不影响后面任何数字。

> `--calib-agg` **两种都要跑**：`logit` 是 `infer.py` 的默认、也是拿到过分的那个；
> `feat` 只是有理由的替代品，**从未上过排行榜**。两者缓存不共用
> （缓存名 `calib_{stem}_{view}_{agg}_{hash8}`），所以第二次跑只花多出来的时间，不会重算第一遍。

> **扫描比看上去贵，而且大部分成本在 CPU。** 视图集是作用在 val 子集（`--val-ratio 0.1`；
> 初赛 val 是 10116 张，复赛按同比例更大）上的，每个视图集都要把每张图重新解一遍、缩一遍。
> `probe.py --workers` **默认就是 12**
> （`infer.py` 默认是 0，两者默认值不同，别记混），并行是默认开着的；真嫌慢就把它加上去，
> 输出与 worker 数无关（变换确定性 + loader 按序产出）。
> 带分辨率的名字（`s256`/`s288`/`s320`/`s352` 及其组合）还会**额外建一座 CLIP 塔**
> （位置编码按该尺寸重插），选得越多建得越多 —— 先把单视图 + 多视图集扫完，
> 分辨率组合按需再补，不必一次全开。

`--calib` 读的输入是**队友那棵树产出的 checkpoint**（`args['img_size']` 键），
我这边写的是 `args['image_size']`；兼容层两边都认，**走别名时会在日志里显式说明**
（见 `train.ck_image_size`）。分辨率一律**从 checkpoint 里读**，不从 `--sizes` 读——
拿 288 的 checkpoint 按 224 跑**不会报错，只是分数更低**，这正是最需要防的静默错误。

### 4.5.2 校准门：代理必须先答对已经知道答案的题

| # | 排行榜已实测 | 代理必须 |
| --- | --- | --- |
| 1 | ep4 ≈ ep20（差 **0.0002**） | `acc@V*` 的差 **≤ 0.5 点** |
| 2 | `wide` 单视图 < `plain`（**−2.16**） | 排序正确、幅度 ≥ 0.5 |
| 3 | `tta4` > `plain`（**+2.09**） | 排序正确、幅度 ≥ 0.5 |
| 负对照 | `val_acc` 在 #1 上是 **+5.02** | 必须**判不过** |

**负对照是这一节的全部意义**：如果 `val_acc` 也能过门，那代理什么也没买到。
判据在 `probe.calib_gate()`，是纯算术，`selftest.py` 会喂一张**故意做错**的表进去，
确认它确实会判 FAIL——**一个恒返回 PASS 的门等于没有门**。

> 这个自检抓到过一个真 bug：门里的度量是**分数**（0~1），而容差和排行榜数字是**百分点**。
> 不换算的话 `abs(0.0002) <= 0.5` 会因为完全错误的理由成立，整张表全 PASS 而什么都没测。

**门没过就停手**，如实报告哪一条没过、为什么。**不要拿一个没校准的代理去做决策。**

### 4.5.3 度量的定义

* `V*` = val 里**冻结 CLIP 的最近类均值（质心取自不相交的 train split）同意其给定标签**的子集。
  质心来自 train ⇒ 没有任何 val 标签影响过这个判据，它的同意是独立证据。
* `acc@V*` = 在该子集上的准确率。**关键：V\* 是给"训练过的模型"打分的**；
  冻结 CLIP 自己在 V* 上必然接近 100%，它自己那个数字没有意义。
* `acc@V*-hard` = V* 里**冻结判据最不确定的那一半**（按 margin 中位数切）。
  V* 选的是"判据同意"，也就是**典型样本**，所以 `acc@V*` 可能饱和到 1.0 而排不动任何东西；
  排序信号如果存在，只可能在低 margin 那一半里。门对两个度量**都**跑，谁过用谁。

### 4.5.4 视图集：`--tta-views`（与 `--tta-ratios` / `--tta-crops` 的关系）

`infer.py` 的视图从 `(size, crop, flip)` 扩成 `(size, crop, ratio, flip)`。
`ratio` 只对 `center` 有效（`full`/`pad` 自己就定了短边），取值见 `train.VIEW_RATIOS`：
`wide` 1.0 / `plain` 1.143（训练默认）/ `mid` 1.286 / `tight` 1.429。

**复现已拿分的配方用 `--tta-views`，不要用乘积式参数拼**：排行榜那 4 视图
（`plain` / `plain+flip` / `wide` / `tight`）在乘积式参数下**表达不出来**——`--tta-flip` 会把每个视角都翻，
拼出来的是 6 视图超集，看着像那个配方而**不是**它：

```bash
# 排行榜验证过的配方本身（4 视图，66.2456）——只有名字能表达
python infer.py --test ... --checkpoint outputs/best.pt --tta-views tta4
# 6 视图：上面 4 视图的乘积式超集，多出 wide+flip 与 tight+flip 两次前向
python infer.py --test ... --checkpoint outputs/best.pt --tta-ratios plain wide tight --tta-flip
# 同样 4 次前向，改花在裁剪轴上（= train.VIEW_SETS['axis4']，本项目的核心假设）
python infer.py --test ... --checkpoint outputs/best.pt --tta-views axis4
# 同为 6 次前向：tta4 + 两个裁剪轴视角，与 tta8 成本相同、只差钱花在哪条轴（= mix6）
python infer.py --test ... --checkpoint outputs/best.pt --tta-views mix6
# 多分辨率那两条按 checkpoint 的训练分辨率选：288 的模型用 tta4s320，320 的模型用 tta4s288
# （反过来配会被判退化并拒绝：对 320 的模型，`s320` 与 `None` 是同一批像素）
python infer.py --test ... --checkpoint outputs_320/best.pt --tta-views tta4s288
```

两条硬规则（都是**报错**，不是"以谁为准"）：`--tta-views` 与乘积式参数同时给直接拒绝；
名字在该 checkpoint 上**退化**——即集合里有两个视图落到同一批像素（320 的 checkpoint 配 `tta4s320`：
它的 `None` 尺寸视图和字面 320 视图都是 320；288 的 checkpoint 配 `s288+plain` 同理）也拒绝——
提交一份"看起来像具名配方却不是它"的 CSV 比不提交更糟。
注意**单个**尺寸视图恰好等于训练分辨率（320 的 checkpoint 配 `s320`）**不**拒绝：它就是 `plain`，
名字如实描述了像素；拒绝它就得连 `plain` 一起拒绝，而 `plain` 正是 64.16 那条配方。
这种冗余只在扫描表里可见（两行数字相同），不构成误提交。可选名字见 `train.VIEW_SETS`，
其中预注册给扫描用的多视图集见 `train.VIEW_SETS_SWEPT`。

**核心假设**：队友已验证的收益来自**错误去相关**（4 视图平均比最好的单视图还高 2.09），
而他们 8 个视图**全在缩放这一条轴上**（= 本工程的 `tta8`）。那 `sub_tta8` 花在 `mid`/`mid_flip`
上的两次前向，换成 `full`/`pad` 买到的去相关性更多，**前向成本相同**（= `mix6` / `crops6`）。
这条由 `probe.py --calib --calib-sweep` 离线裁决，不烧名额。

### 4.5.5 暂时不动的（等 TTA 视图轴的裁决落地再说）

288 与 320 都已落地（69.218 / **69.968**），而**训练侧的证据一致指向天花板**：224 上 ep4 == ep20
（差 0.0002），320 上 `val_acc` 只涨了 0.0003 而平台涨了 0.75 —— 也就是说**同一个 checkpoint 换个输入表示
就能拿分，多训没有**。所以此刻不要去改训练数学，改它只会混淆唯一还有信号的那条轴（视图集）。
以下是队友那棵树里我这边已修、但**现在不动的**真实缺陷：

* `head.logit_scale` 吃 weight decay（我这边是 D1）；
* 无 LR warm-up（D6）；
* 增强随机性来自 worker RNG 而非 `(seed, index, view)`（D12，**这是合规问题**）；
* `--select` 只有 `val_acc` / `val_acc_hi`。

## 5. 规则符合性

* 骨干：仅 `open_clip` 的 `ViT-B/32` + OpenAI 官方公开权重；未使用任何其他视觉基础模型或 API；
* 数据：只用官方训练集（`--data` 目录）；测试集仅在 `infer.py` 中做前向，未参与训练；
* **单模型、单一训练权重**：不做模型集成、不做多模型投票，**也不对多个 checkpoint 做平均**；
* **TTA 与测试时分辨率按赛事方确认可用**（2026-09-23 询问后确认）：允许测试时增强（如水平翻转、多尺度），
  也允许改变输入分辨率（由 `open_clip` 的 `force_image_size` 对位置编码做插值），**因为骨干网络与训练权重都不变**。
  判据是**同一份训练权重、同一条推理路径**：翻转/多尺度取平均属于 TTA，是对**同一模型**的输入做增强；
  而把多个**不同 checkpoint** 的输出平均属于集成，**不在获批范围内，不要做**。
  已落地：`infer.py --tta-views tta4`（或 `--tta-ratios plain wide tight --tta-flip` 那个 6 视图超集）
  —— 逐视角过**同一个头**再平均 logit（`--tta-agg logit`，默认）就是单条决策路径、就是单模型；
  `--tta-agg feat`（各视角特征先 L2 归一化、再求平均、再归一化一次，分类头只作用一次）同样只用**一个**
  checkpoint 和一个头，也可以，但它**没有上过排行榜**，用之前先过 `probe.py --calib`）；
  `train.py --image-size N`
  训练高分辨率；`infer.py --tta-crops center full pad` 换裁剪策略（**连位置编码都不碰**，
  是最安全的一条 TTA）。**"训 3 个种子再平均"不可用。**

  > ⚠️ **分辨率请用 32 的倍数：256 / 288 / 320 / 352 / 384，不要用 336。**
  > 336 是 ViT-**L**/14@336 的数字（336/14 = 24 整除）；ViT-**B**/32 的 patch 是 32，
  > 336/32 = 10.5 不是整数，patch 卷积只能产出 10×10 的网格 —— 等价于 320 再加 16px
  > **永远没有 patch 看到的**死边。`train.build_clip` 会对这种尺寸打印警告，
  > `patch_grid()` 会算出真实网格（`selftest.py` 已锁死 224/288/320/352/384 干净、336 不干净）。
  > 具体选哪个由 `probe.py --tta` 在**本数据集上实测**决定，不要照搬文献里的 336。
* 训练过程可复现：`selftest.py` 自检 + 固定种子 + checkpoint 内保存完整 `args`；噪声筛选（`noise.py`）是自动算法，不是人工清洗。
  `--image-size` 会随 `args` 进 checkpoint，`infer.py` / `valmetrics.py` / `analyze.py` 自动读回，
  所以"训练 320、推理 224"这种错配不会静默发生。
