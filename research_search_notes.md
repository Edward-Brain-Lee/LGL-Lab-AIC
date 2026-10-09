# 搜索设计研究：在 74.249 基线上节省 GPU 时间（2026-10-08）

本报告区分三件事：历史重构测试代理能诊断失败模式；官方训练留出集能合法选择参数；平台是实际分数的最终验证。现有代理吻合得很准，不等于据其标签逐类拟合偏置、正则、路由或反复搜索后仍然合规。README 十一（二）要求测试仅用于测试，且禁止额外标注。搜索代码只读本阶段官方训练目录；最终保留单个 checkpoint、统一确定性推理配方。

## 代码和历史证据

- `checkpoint_search.py` 原来每次命令都重新构造 CLIP、执行完整视图前向。tau 在一次命令内共用 logits，但改 tau 重跑没有跨命令缓存。
- `experiment_search.py` 每臂 20 轮、约 3.4 小时历史成本，加 5 个 checkpoint 的 16 视图评估；三个臂绝非三小时总成本。
- 历史 `--seed` 同时控制训练随机数和数据分割，因此换种子不能直接用各自留出分数公平排名。新增 `--split-seed 3407` 应在所有候选中固定；训练 seed 才作为重复验证变量。
- `HANDOFF.md` 既有“ep4==ep20”的旧记录，也有 §33.8 明确撤回据其普遍早停的结论。当前 qkv 轨迹不能以早期四轮定终局。
- 官方留出标签含噪，已有 `val_acc` 与平台方向相反的案例。因此宏召回是与均衡测试更一致的聚合方式，但没有把标签变干净。不能发明“macro 提高多少必定兑换平台多少”的公式。
- §42 关于 74.1/75 的硬天花板曾从跨配置偏移外推，与已验证 74.249 不相容；不能继续作为结构上限。

## 已实现：精确 logits 缓存

仅修改 `checkpoint_search.py`，不改变默认预测数学：缓存每个单 checkpoint / 完整有序视图配方的 float32 平均 logits。再次搜索同一数据/环境/配方的 tau 时，不构造模型、不执行 GPU 前向。

缓存键包含完整 checkpoint SHA256、固定留出集指纹、留出图像内容 SHA256、路径顺序/官方标签/训练计数、代码文件 SHA256、模型名、预训练权重标识（本地权重还计算内容 SHA256）、有序视图及变换参数、分辨率、运行库版本、CPU/GPU 身份、batch size、CUDA/cuDNN 数值设置。载入校验 metadata 摘要、logits 摘要、顺序、标签、计数、dtype、维度与有限值；损坏文件报错。缓存只收官方训练留出 logits，不含测试标签。

用 `--cache-dir analysis_shared_logits` 可跨不同输出目录复用。默认缓存到输出目录的 `logits_cache`。tau 和排序指标不属于键，因为它们不改变前向。CPU 与 GPU 有不同键，避免静默混用数值路径。首次内容校验增加磁盘读取，之后每次也校验实际内容；这是用确定性 provenance 换取安全复用。约 14,500 × 750 的 float32 aggregate logits 每份约 44 MB；逐视图全缓存约 16 倍，当前先保留更省空间、保持精确旧数值的方案。

```bash
python checkpoint_search.py --data /root/autodl-tmp/train \
  --checkpoints outputs_384pe_lr64qkv/ep16.pt outputs_384pe_lr64qkv/ep20.pt \
  --recipes scales --taus 0 0.1 0.25 0.5 \
  --cache-dir analysis_shared_logits --out analysis_search_coarse

# 仅当首轮显示小范围有意义，再局部细化；复用相同缓存，不重做前向。
python checkpoint_search.py --data /root/autodl-tmp/train \
  --checkpoints outputs_384pe_lr64qkv/ep16.pt outputs_384pe_lr64qkv/ep20.pt \
  --recipes scales --taus 0 0.05 0.1 0.15 \
  --cache-dir analysis_shared_logits --out analysis_search_local
```

首次不全扫五轮 × 三配方，优先当前最强 ep20/scales 与 ep16/scales；只有末段明显有差异再增 ep12。tau 只用训练频次产生全局先验修正；当前频次近均衡，所以它是低成本小变量，而非 1.75 分主线。不同 recipe 是各自独立候选，不按类别合成或按图片人工挑答案。

## 训练搜索：少量有解释的候选优于大网格

首轮只选一条单变量方向与 control 比，预注册参数，不把十个模块一起放进 Optuna。建议顺序为训练样本覆盖/优化器合理分组等机制明确的实现，再 LR、增强强度、anchor、低权重一致性。原 cosine consistency=0.05 相对旧 MSE 数值约放大 256 倍，不宜把它当安全修正；先看 0.002～0.01 小幅候选的损失和梯度占比。每臂全程同一 split_seed、同一 20 轮 cosine、同一 warm-up、训练/推理尺寸、TTA；训练种子先相同便于归因。

只有有价值的一个维度已确认，再做局部对数 LR 扫描，例如 1e-4、2e-4、3e-4。范围是假设候选，尚无实测收益。三个全训练点要约十小时，预算不足时控制在两个，不能承诺短训排序保持到后期。

### 为什么现在不盲接 ASHA/Hyperband

ASHA 是成熟早停搜索算法，但早期排名必须有信息量；本工程 warm-up 三轮、第四轮刚换损失机制，而且带噪 val 曾指错方向。`--epochs 4` 的 `CosineAnnealingLR(T_max=4)` 与 `--epochs 20` 的第 4 轮 LR 完全不同；把二者当同一个 rung 会比较不同训练算法。若未来用逐步资源分配，所有 trial 必须从一开始声明同一 20 轮日程，以 ep8/12/20 作为观察点；不能把不同 epochs 参数直接当保真度。

此外当前 resume 会从 student 重新复制 EMA teacher，未完整持久化 teacher、原型、sampler generator 和 numpy RNG；仅保存 opt/sched/tracker 不足以保证恢复轨迹与不中断训练一致。因此正式实现 SHA resume 前需要独立验证完整恢复，现阶段用不中断 trial 的监控/早停。GPU 只有一张且完整 trial 数个位数时，多 bracket Hyperband/TPE 的学习优势不值得复杂化；先做固定候选、单变量与缓存评估。

## 可信排序与验收

先在固定 split 比较：macro 为主、micro 为诊断；弱类变化采用逐类 support 与平滑召回，不能把只有一两张留出的 0/100% 当稳定结论。可附加固定官方训练 V*（由官方冻结 CLIP 构造）诊断，必须在所有候选前生成同一 mask，不按候选重做。V* 偏向冻结模型容易识别的样本，改善该子集也不保证整体泛化。

报告每候选相对 control 的新胜/旧胜、净正确数、宏召回差、尾类变化与置信度；配对 bootstrap 按类分层并固定随机种子，仅表达对现有带噪标签的采样不确定性，不能覆盖标签偏差。若大量候选共享同一留出，选择分数会乐观；最终候选先固定，再用新训练随机种子与相同 split 重复，必要时用官方训练的第二预留确认集（需新训练从始至终不触碰它）。不能直接把其他 checkpoint 已经训练过的图片称为验证集。

验收必须满足：1）默认基线复现且保存的 checkpoint 哈希/代码哈希可追溯；2）单变量主要指标有稳定正向、弱类没有大范围崩塌；3）固定单模型/统一配方后才做测试推理；4）平台高于 74.249 才称真实提分。若 0.1 分左右的改善只来自挑许多候选或一个种子，就仍是候选。已有一次 seed 差 0.03 分只是一次观测，不是已估计的噪声分布。

## 核实来源

以下两篇的原始页面于 2026-10-08 用 Python urllib 直接读取成功，正文快照保存在 `analysis_search_sources/2.txt` 与 `3.txt`：

1. [Li et al., A System for Massively Parallel Hyperparameter Tuning / ASHA](https://arxiv.org/abs/1810.05934)。作者描述其利用异步并行和激进早停；这不证明本工程带噪早期指标适合剪枝。
2. [Bergstra & Bengio, Random Search for Hyper-Parameter Optimization, JMLR 2012](https://jmlr.org/papers/v13/bergstra12a.html)。原文强调少数重要超参时随机搜索比网格更有效；可作为大范围探索基线，但这里最先需要修复选择指标与预算分配。

Optuna 的 SuccessiveHalvingPruner/HyperbandPruner 官方 docs 访问 403，GitHub raw 超时；本轮没有把未读页面的细节当核实事实。scikit-learn nested-CV 官方示例读取 SSL 失败，未作为已核实引用。上述针对工程 cosine/恢复/指标的判断来自本地代码审阅，不依赖这些未读链接。
