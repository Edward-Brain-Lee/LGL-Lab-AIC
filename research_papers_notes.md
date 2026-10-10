# 2026-10-08 开放研究：论文主源核验及可迁移机制

这份记录只区分已核实的论文机制与本项目尚待验证的实验假设，没有新增实测分数。检索使用 arXiv API、作者 GitHub README 与微软官方 USB 代码；部分 arXiv 请求超时，未取得资料的细节不作为已核实事实。README 允许官方 CLIP ViT-B/32 上的 PEFT、鲁棒损失、样本筛选和表征约束；仅可使用当前阶段官方数据，测试图不得参与任何训练，最终不得多模型集成。

## 主源与机制

| 方法 | 已核实主源 | 核心机制 | 本项目适用性 |
| --- | --- | --- | --- |
| OwMatch，NeurIPS 2024 | [论文](https://arxiv.org/abs/2411.01833) / [作者源码](https://github.com/niusj03/OwMatch) | conditional self-labeling 与 open-world hierarchical thresholding；面向未标注数据包含未知类的开放世界 SSL | 750 类固定分类不存在对应的未知类任务；仅参考训练内自适应阈值和一致性思想。源码使用 SimCLR 预训练，不能迁移其骨干或权重。 |
| SoftMatch，ICLR 2023 | [论文](https://arxiv.org/abs/2301.10921) / [微软官方实现](https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/softmatch/utils.py) | 用截断高斯根据置信度软加权，兼顾伪标签数量和质量；另有 uniform alignment | 可把当前置信度幂权重替换为训练内统计驱动的软权重。不要照搬均匀分布假设，更不能基于测试预测拟合统计。 |
| FreeMatch，ICLR 2023 | [论文](https://arxiv.org/abs/2205.07246) / [微软官方实现](https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/freematch/utils.py) | self-adaptive thresholding；全局阈值随模型学习状态更新，类别阈值按预测类别概率调节 | 可探索训练内类别自适应筛选，必须有阈值上下限与最低证据量，避免易错类因预测质量低而降门槛、反向放大错误重标。 |
| NLPrompt，CVPR 2025 Highlight | [论文](https://arxiv.org/abs/2412.01256) / [作者源码](https://github.com/qunovo/NLPrompt) | PromptMAE：prompt learning 下采用 MAE；PromptOT：文本特征作原型，通过 OT 分 clean/noisy，对二者用 CE/MAE | 数字类别没有官方物种名，无法直接构造真实类别文本原型；视觉原型替代只能称受启发的新变体，不能宣称复现论文收益。纯 MAE 不一定适用于本题 LoRA。 |
| ProGrad，ICCV 2023 | [论文](https://arxiv.org/abs/2205.14865) / [作者源码](https://github.com/BeierZhu/Prompt-align) | 防遗忘：任务梯度若与固定 prompt 预测的 KL 梯度冲突，则沿一般知识方向投影 | 匿名类别无法直接获取该零样本文本教师。用冻结 CLIP 训练视觉原型替代 KL 参考分布，是类比方案而非原论文复现；梯度投影增加计算和归因成本，应晚于轻量蒸馏消融。 |
| JoAPR，CVPR 2024 | [CVF 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.pdf) / [作者源码](https://github.com/yunncheng/JoAPR) | 已核实为噪声标签 prompt learning 研究，源码提供噪声率与数据集相关超参数 | 主源已找到，尚未精读完整 PDF，不以本记录认定其具体算法算子。默认 RN50 配置与外部数据准备不能直接复制，骨干必须改为官方 ViT-B/32。 |

用户暂称的 OutMatch 没有确定标题；检索最相关候选是 OwMatch，不应把二者说成已确认同一论文。TrustCLIP 搜索命中 2026 年隐私表征同名论文，并不是 README 中 2025 年噪声标签论文；未核实的同名项目不进入推荐依据。

## 两种训练内置信度方法的精确机制

SoftMatch 官方源码权重为 `exp(-min(pmax-mu,0)^2 / (2*var/n_sigma^2))`，均值、方差由训练预测维护 EMA。超过均值的样本权重为 1，不会只偏重最高置信样本。源码带 `per_class` 选项，但每类稀疏统计必须谨慎处理；对本工程首次实现应从全局版本开始，记录每类有效样本量与保留率。

FreeMatch 官方源码 SAT 为 `tau_c = EMA(mean(pmax)) * EMA(p_c) / max_j EMA(p_j)`。这能给学习弱的类更低阈值，但在有错标与结构化混淆时，也可能放大被弱类吸入的误标。此处必须区分“降低监督权重”与“硬重标”：第一轮只改权重，不放宽当前重标条件，更容易保证收益归因。

当前训练已经有 tracker 权重、`conf_gamma=2` 置信度幂、APL/RCE 鲁棒损失，多种抑制难样本机制相乘可能导致难的干净样本长期缺梯度。先做 `conf_gamma:2→1`，观察训练 ESS 和稳定近类识别；它是比完整复刻 SSL 框架更清楚的单变量。不要同时改筛选阈值、损失和类别采样。

## 关于“语义标签输入 CLIP、失败语义惩罚”的判断

若官方数据附带真实物种名称，可直接用官方冻结 CLIP 文本塔编码固定模板，构成分类头初始化或训练期语义蒸馏，最终仍一个视觉分类头。但当前工程只有 `0000..0749`，对 `a photo of class 0007` 编码不会自动带出物种含义；外部反查物种名、LLM 写物种说明会引入额外监督，不能作为本项目第一路线。学习匿名 pseudo-token 可以重新参数化类别原型，却不能被解释为获得新的真实语义知识。

可立即验证的替代是：用当前官方训练划分上的官方冻结 CLIP 特征，自动构造经过稳健修剪、归一化的视觉类别原型 `mu_c`；定义 `D(y,c)=1-cos(mu_y,mu_c)`，仅在可信训练样本上使用 `L_sem=Σ_c p_c D(y,c)`。原型、cost 与阈值不能由测试代理逐类真值或测试混淆表反推。这是训练视觉几何正则，不是文本语义分类，也不增加最终预测融合。

但这种期望距离目标对相似近类惩罚更小，会容忍最难的细粒度误判，降低语义距离不等于提高 Top-1。建议保持原分类损失，将权重作为小幅单变量，并检查最常见近类对的 logit margin 是否下降。若目标是细粒度区分，可另设独立实验：基于同一训练原型定义 top-k 相似负类，添加 gated near-class margin，让可信正类与相似负类拉开边界。两者含义不同，不同时叠加。

额外风险是原型错标污染与尾类估计方差；必须保存每类有效样本量、修剪比例和版本指纹，最小支持不足时退回零正则，禁止把人工逐类识别作为训练必要条件。

## 初步优先级

1. checkpoint 邻近 epoch 和推理标量搜索。只选择一个最终 checkpoint，不融合；代理如果来自测试真值重构，不能仅因算法可复现而宣称调参完全合规，需与训练留出/平台独立验证区分。
2. 当前优化器、学习率、权重衰减、`conf_gamma` 单变量。这些先测再讨论论文式改造，不从论文成绩预报本题收益。
3. 上述训练视觉 cost 作为低成本明确新方向；一次只验证一种 cost 与一个小权重。
4. SoftMatch式软加权；FreeMatch式类阈值更高风险，在充分训练诊断后探索。
5. NLPrompt式视觉 OT、ProGrad式训练原型梯度保护作为后续研究，不能复制外部权重/文本类别名或测试期适配。

任何方法现在都没有本项目增益证据，也不承诺达到 76。应记录净修正与净损伤、按类覆盖和相邻 checkpoint 稳定性；把论文机制、代码通过、实际提分分开。
