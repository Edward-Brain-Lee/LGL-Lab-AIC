# cand_74.43 — 416+SE + local-attn + gate-init 0.02 + 颜色擦除增强

**榜分 74.43**（前值 74.2132，**+0.2168**）。完整记录见 `HANDOFF.md` §54。

## 这个目录是什么

`run_cand.sh` / `run_post.sh` 是打出 74.43 的那条链的**原始入口脚本**，保留作存档与复现依据。
它们依赖**队友的两阶段优化版包**（`AIC训练代码_74.2132优化版_两阶段鲁棒微调`），
那份代码在本仓库之外，与这里 `train.py` 的差别是多了三个开关：

| 开关 | 本仓库 | 队友包 |
| --- | --- | --- |
| `--local-gate-init` | ✗ | ✓ |
| `--augment-extra` | ✗ | ✓ |
| `--clean-finetune-epochs` 等 | ✗ | ✓ |

⇒ **在本仓库直接跑 `run_cand.sh` 会 argparse 报错。** 要复现，先拿到那份包，把它作为工作目录，
再把本目录的两个脚本放进去。

## 实际执行的配方（18/20 轮，两阶段未执行）

```bash
python -u train.py --data /root/autodl-tmp/train --out outputs_cand \
  --epochs 20 --warmup-epochs 3 --batch-size 72 --workers 8 --img-size 416 \
  --train-pos-embed --local-head --local-attn --local-se --local-gate-init 0.02 \
  --lora-rank 64 --lora-qkv --augment-extra clip_color_erasing \
  --clean-finetune-epochs 2 --save-every 2 --nan-guard
```

- `--batch-size 72`：该机 416 在 bs≥80 静默 NaN（HANDOFF §52.2）。
- 实例在 ep18 后被关机，**ep19/20 与两阶段均未执行**；`best.pt` = ep18。
- 可训参数 **10,390,581**；`tanh(local_gate)` 收敛到 **+0.371**（74.2132 那轮是 **−0.284**）。

## 提交件的产生方式（判据与 §53.4 不同之处）

```bash
# 1) 16 视角留出集评估，按 micro 选 checkpoint（而不是 best.pt 的 val_acc）
python validate_tta.py --data train --checkpoint outputs_cand/best.pt \
  --views scales --tau 0 1.0 --batch-size 16 --workers 4 --amp bf16 \
  --output-json outputs_cand/holdout/best.json

# 2) 一次前向出 5 档 tau，并落盘 logits
python infer.py --test test --checkpoint outputs_cand/best.pt \
  --output sub_cand/pred_results.csv --workers 4 --batch-size 64 --tta scales \
  --logit-adjust 0 0.75 1.0 1.25 1.5 --save-logits outputs_cand_test_logits.npz

# 3) 按边际违反率选 tau（而不是留出集 micro/macro）
python select_tau.py --test test --csv sub_cand/pred_results*.csv
```

最终提交 `pred_results_tau075.csv`（违反率 18.58%，与 τ=1.0 的 18.59% 是平局，两档只差 545 行）。

## 关键产物（不在本仓库，见桌面打包）

| 文件 | 用途 |
| --- | --- |
| `outputs_cand/best.pt` | ep18 权重，sha256 `9b255f67e6107c99` |
| `sub_cand.zip` | 提交件，MD5 `05f3c7d3956e7438ae6766d5937891be`，1,074,141 B |
| `outputs_cand/holdout/*.json` | 三个候选的 16 视角 micro/macro |
| `outputs_cand/train_vs_val_curve.csv` | 实测 train_acc vs val_acc（见根目录同名文件与图） |

## 两条要记住的教训

1. **`val_acc` 与 16 视角 micro 都给了 ≈0 的读数，榜分却是 +0.2168。** 本地指标只能排掉明显坏的，
   不能批准明显好的（§54.4-2）。
2. **`train loss` 与 `val_loss` 不可比**（前者是混合目标、NCE 项可到 −2.39；后者是纯 CE）。
   `train_acc` 必须实测，见根目录 `train_acc_probe.py`（§54.4-1 更正了 §42.3 的「37 倍背题」）。
