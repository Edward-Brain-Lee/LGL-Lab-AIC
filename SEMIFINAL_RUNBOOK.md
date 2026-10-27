# 半决赛基线复现 Runbook（2026-10-26，含云端实测）

目标：在半决赛 **500 类**数据上，从 OpenAI 官方权重**重训**复赛拿到 **74.249** 的那一版配方，作为半决赛基线。
不引入任何新方法，不改任何超参；只换数据阶段。

---

## 0. 改了什么、没改什么

| 部分 | 状态 | 说明 |
|---|---|---|
| `train.py` / `infer.py` / `noise.py` / `losses.py` | **一行没改** | 配方与类别数无关；500 类只是 `nclass` 变 500（头 500×512） |
| 复赛 run 脚本（`run_lr64qkv.sh` 等） | 保留作历史 | 写死了 `N_ROWS=37444` / `10.325M`。**不要用它们跑半决赛**——smoke 会在参数闸门就拒绝 |
| `--save-every` | **4 → 2** | 唯一一处改动，且只影响落盘频率：多存 ep2/6/10/14/18，**不改变任何一步优化**。目的是让"轮次饱和"这条曲线以后能零成本量出来 |
| 新增 `sf_gate.py` / `sf_run.py` / `sf_tower_gate.py` / `test_sf_gate.py` / `run_sf.sh` | 校验与编排 | 断言可在笔记本上跑（`test_sf_gate.py` 5 组） |

## 1. 配方（复赛 74.249，逐字不变）

```bash
python train.py --data <train> --out outputs_sf_base \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --save-every 2 \
  --lora-rank 64 --lora-qkv
```

推理（同一个 `best.pt`，16 视角）：

```bash
python infer.py --test <test> --checkpoint outputs_sf_base/best.pt \
  --output sub_sf_base_scales/pred_results.csv --tta scales --workers 4 --batch-size 64
```

## 2. 云端实测到的环境事实（2026-10-26）

| 事实 | 影响 |
|---|---|
| `python` **不在**非交互 PATH（只有 `/root/miniconda3/bin/python`） | 直接 `ssh 'python ...'` 会 `command not found`；`run_sf.sh` 已做解释器发现 |
| 镜像里**没有 open_clip** | 要 `pip install open_clip_torch==3.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple`（已装好） |
| 首次用 `openai` 权重会下载 ~600 MB（`HF_ENDPOINT=https://hf-mirror.com`） | 实测 452 s；之后走 `~/.cache/huggingface` |
| **无卡模式 CPU 配额 = 0.5 核**（`cpu.max = 50000 100000`） | 解包 18 GB 要 ~50 min，`selftest.py` 要 ~10 min，**任何基准测试都不可信**。重量级预检请等挂上 GPU 再做 |
| `/root/autodl-tmp` 50 GB，解包后 34 GB 已用 | 压缩包还占 17 GB，确认数据无误后可以删掉换空间 |
| torch 2.3.0+cu121 / torchvision 0.18 / open_clip 3.3.0 / PIL 10.3 | 与复赛不同机但版本兼容；真塔闸门会验证 qkv 包装在这个组合下仍然成立 |

## 3. 数据画像（在云端对解包后的文件实测）

| 项 | 值 | 对配方的含义 |
|---|---|---|
| 类别 | 500（`0000`~`0499` 连续） | 头上 500×512，预算 10.198M |
| 训练图 | 90,197 | 每类 min 10 / 中位 183 / max 221（**22.1x**），≤50 张仅 7 类 |
| 测试图 | 49,857 | 每类约 100 张（=500×100−143），官方声明均衡 |
| 图像尺寸 | **训练/测试全部 500×500，RGB，0 张不可读**（抽样 3000/3000 全解码） | **384 的 eval resize 是 439 → 下采样 ×0.88**；复赛是 375 → **上采样 ×1.17** |
| 重复图 | 抽样 500+500 全零重复；zip CRC 全域比对也是 0 | 去重/来源分组这条路在本阶段无效 |
| 留出集 | **train 81,402 / val 8,795**（`int(n*0.1)` 截断，不是 9,020） | 635 步/轮 @batch 128 |
| 单张 val 的类 | 仅 2 个 | 逐类召回对这两个类只能是 0 或 1，不影响总分 |

> **最重要的一条**：复赛的测试图短边只有 375，所以 384 推理时每张图都被插值放大了 1.17 倍，
> 320 就已经用满原生像素；而半决赛测试图是 500，384 是**缩小**，`--tta scales` 的 16 个视角
> 也不再过度插值。**同样的配方在半决赛数据上应当比它在复赛的分数更"舒服"**——这是"努力复现"的
> 主要依据，也是我不建议在这一轮动分辨率的原因。

## 4. 每一步应该看到什么数字

| 阶段 | 必须成立 | 不成立的含义 |
|---|---|---|
| `prep` | 500 个 `0000`~`0499` 连续目录 / 90,197 / 49,857 / 名字全 32 位 hex | 目录有缺口 → `sorted()` 会把类别 id 整体错位 |
| `prep` | `trainable params = 10,197,762 (10.198M)` | 不是这个数 ⇒ 类别数或 flag 不对（750 类是 10.326M） |
| `smoke` gate 0 | `module closure present (10 files)` + `ALL CHECKS PASSED` | 少了 `build_semantic_reference.py` 之类会在这里一秒拦住 |
| `smoke` gate 2 | `layers=48`、`adapter params exact: 9437184`、`init max|dz|<1e-5`、qkv 梯度 12+12 | 冻结塔被破坏 / qkv 适配器没接上 |
| `smoke` gate 3 | 日志 9 项 + ckpt 24 个 qkv 张量 + 位置网格 `(145,768)` | 配置漂移 |
| `train` | 同一组 9 项 + ckpt 校验；`classes=500 train=81402 val=8795` | 同上 |
| `infer` | `49,857 rows / 2,143,851 bytes / 500 labels / 每个测试名一次` | 缺行 → 提交直接无效 |

参数预算由模块形状**推导**并用 750 类的历史观测自校准：

```
adapters 9,437,184 + head 256,001 + local 393,217 + pos 111,360 = 10,197,762 → 10.198M
```

## 5. 操作顺序

```bash
cd /root/autodl-tmp
export PATH=/root/miniconda3/bin:$PATH          # 必须；否则 python 找不到
export HF_ENDPOINT=https://hf-mirror.com        # 首次拉权重用

bash run_sf.sh plan  --root /root/autodl-tmp    # 干跑：路径/命令/预算，不执行
bash run_sf.sh prep  --root /root/autodl-tmp    # 已解包就只校验（0.5 核下解包要 ~50 min）
bash run_sf.sh smoke --root /root/autodl-tmp    # 挂上 GPU 之后再跑；最后两步要独占显卡
nohup bash run_sf.sh train --root /root/autodl-tmp > sf_train.log 2>&1 &
bash run_sf.sh infer --root /root/autodl-tmp
```

时间预算（**挂上 4090 之后**）：train 81,402 图 × 20 轮 ≈ **1.9 h**（复赛 4.16 ms/图实测折算）；
8 视角推理 ≈ 35 min、16 视角 ≈ 70 min ⇒ 整链约 4 h。**不要用 0.5 核的无卡实例估这些数。**

`--workers`：复赛 20 vCPU / 8 workers 实测 240 img/s（够用）。半决赛图 500 px（比 480 大 8%）
且总图数少 40%，8 个仍应够；如果 `nvidia-smi` 显示 GPU 利用率长期 < 70%，再加到 12~16
（**这条要在 GPU 模式下重测**，0.5 核测出来的数没有意义）。

## 6. 合规（README 十一）

- 骨干只能是官方 CLIP ViT-B/32：`check_backbone()` 在训练与推理两处硬检查；
- 权重从 `pretrained='openai'` 重建，**不加载任何复赛 checkpoint**（十一(二).2）；
- 数据只有半决赛官方 zip；测试集只出现在 `infer.py` 的 `no_grad` 前向里，不参与训练/选型/阈值；
- 单模型单推理流程：一个 `best.pt` + 一套确定性 16 视角；两个视角集合**绝不平均**；
- 噪声处理全自动（`noise.py`），符合五.6；种子/args/RNG 进 checkpoint，符合五.5。

## 7. 提交前必须先确认

赛题文档写半决赛测试集 **24,912** 张，而交付的 `test.zip` 里是 **49,857** 张（=500×100−143）。
流水线按目录实际张数出 CSV。**若平台只认 24,912，这份提交会直接无效**——`prep`/`infer` 都会打印
49,857，请先与公告核对再花名额。

## 8. 基线落地后（0 名额）

```bash
# 用留出集给 ep2 起的所有快照排序，量"轮次饱和"——决定以后每个臂能不能只跑 8~10 轮
python checkpoint_search.py --data /root/autodl-tmp/train \
  --checkpoints outputs_sf_base/ep4.pt outputs_sf_base/ep8.pt outputs_sf_base/ep12.pt \
    outputs_sf_base/ep16.pt outputs_sf_base/ep20.pt \
  --recipes scales --taus 0 --workers 4 --batch-size 64 --out analysis_sf_ckpts
```

`val_acc` 历史上三次把排序搞反过；只看留出集 + 平台分数。同配置换 seed 的噪声地板是 **0.03 分**。

## 9. 排错

| 症状 | 原因 |
|---|---|
| `python: command not found` | 非交互 PATH 没有 conda；`export PATH=/root/miniconda3/bin:$PATH` 或用 `run_sf.sh` |
| `ModuleNotFoundError: open_clip` | 镜像里没有，`pip install open_clip_torch==3.3.0` |
| `missing local module(s) [...]` | 上传不全；`sf_run.py` 的 `NEEDED_MODULES` 就是清单 |
| `prep` 报 "could not find train (dir or .zip)" | 数据不在 `/root/autodl-tmp` 下两层内，用 `--train/--test` |
| `smoke` 报 `trainable params` 不符 | 500/750 弄混，或缺 `--local-head` / `--train-pos-embed` |
| `train` 报 `smoke gate is STALE` | 闸门通过后又改了 `train.py`/`infer.py`，重跑 `smoke` |
| 命令跑得异常慢 | 先看 `cat /sys/fs/cgroup/cpu.max`：无卡模式是 `50000 100000`（0.5 核） |
