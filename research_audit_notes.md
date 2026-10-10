# 开放探索代码审阅：训练信号与低成本候选（2026-10-08）

本文件是本轮独立子 agent 的静态审阅与一项 CPU 数学验证。当前最高平台分仍为 **74.249**；本轮没有训练数据、可访问云端 checkpoint 或新平台结果，不能为候选填写实际收益。固定 anchor／余弦一致性已有独立实现与说明，这里不重复把它们列为新发现。

## 1. 规则结论

`README.md` 五、十一要求 CLIP ViT-B/32、OpenAI 官方初始权重、当前阶段官方数据、单模型／单推理流程；明确列举 LoRA、Prompt、Adapter、鲁棒损失、伪标签、表征约束，并鼓励保留预训练先验。不是“任何骨干参数都不可更新”：基于官方权重训练 PE／LayerNorm／LoRA 并不自动违反骨干限制。不能换成其他 CLIP 预训练来源、DINO/ConvNeXt 或其他基础模型。测试数据不得参与训练，重构测试代理不能生成类别权重、纠错标签或损失输入。

单一 checkpoint 的单一流程允许进行候选比较的方向与多模型输出融合不同；旧文档将同模型两种 TTA 的任何平均一概称为“多模型融合”，表述过宽。不过本轮无需触碰这个解释边界：选一套固定 TTA 即可，已验证的 scales 优先保留。

## 2. 最值得实际验证的遗漏

### 2.1 原有 uniform 仍然有放回，并不遍历全部样本

原 `build_loaders` 对 balanced 与 uniform 都用 `WeightedRandomSampler(..., replacement=True)`；uniform 仅把每个样本权重变为 1。长度 N 的有放回采样每轮期望覆盖率为 `1-(1-1/N)^N≈63.2%`，其余部分不进入当轮训练。约 134,165 图时每轮约 49,355 图未抽到；三轮后理论仍有约 5% 未被观察，旧日志 ep4 前 `unseen=6819` 与量级一致。

这是**每轮**覆盖问题，不是整个训练永久丢弃 37% 图。20 轮随机采样几乎可遍历主体，但 warm-up prototype、早期 teacher/tracker 都依赖前三轮曝光。有放回还令极端尾类反复看到相同图。主体 p10/p90=159/239，balanced 的主体修正不大，有理由用真正 shuffle 检查更均匀的早期曝光。

主 agent 已加入 opt-in `--sampler shuffle`（`RandomSampler(replacement=False)`），保留旧默认。但 `balanced→shuffle` 同时移除频次均衡与重复曝光，**是一个配置因素，包含两个机制**。要归因“是否有放回”，需再比较 `uniform→shuffle`；要归因“频次校正”，比較 `balanced→uniform`。不能只凭 balanced/shuffle 一次结果声称证明采样重复是主因。shuffle 配合 drop_last=True 每轮最多漏最后不足128图，仍应记录每类实际曝光。

### 2.2 confidence² 可能让难类缺少训练信号

post-warm-up 当前权重 `tracker.weight × max(teacher_prob)^conf_gamma`，默认 gamma=2，再按 batch 均值归一。归一不改变样本间比例：同为 tracker trusted、置信度0.2/0.8的图，权重比为1:16；gamma=1则1:4。平方权重倾向容易类与容易取景，不保证对应“错标”。750类任务的置信度还受 head 温度影响。

`--conf-gamma 1` 是高价值、低工程复杂度单因素候选，已加入 `conf_gamma1` arm。应输出每类 ESS、confidence 分位数、降权比例；确认增益来自难类恢复，还是引入噪声使总体下降。不能直接推断关闭所有降权更好。若只跑一个新训练臂，我会优先在 shuffle 与 gamma1 中按曝光／ESS日志决定，而不是先扩大正则力度。

### 2.3 teacher 的判噪视图本身很强

`TwoView` 两个视图均为 RandomResizedCrop(crop_min=.55)+flip+RandAugment(n=2,m=9)；训练 teacher forward 直接读取 x1。tracker 的 disagreement 与即时 confidence 降权因此可能在辨识线索被裁掉／颜色被增强时判为噪声。增加增强并不只有学生泛化收益，也会改变筛选目标。这使“RandAugment幅度更大肯定针对取景敏感”不成立。

后续值得做**弱增强 teacher 视图**：same teacher 仍做一次 forward，dataset 额外提供 Resize/适度 crop/flip、无 RandAugment 的训练图给它；学生两图不改。无额外基础模型，无测试数据。其工程改动比 gamma1 大，先保证 control 的 tracker/数据链复现，再做单臂；不要同时改 tau/conf_gamma/强增强。

## 3. 优化器与超参数

旧 AdamW 对 LoRA A/B、分类头、logit_scale、local gate、PE统一使用 lr=2e-4/decay=.05。已加入 `no_decay_small` 参数组：维度<2的参数、PE免衰减，矩阵层仍保留原衰减，LR不变。其效果无法由常见惯例保证，但比同时修改多组LR更好归因。约1048 step/epoch、20轮cosine，单看直接衰减累计可约使这些参数乘以 `exp(-.11)≈.896`，并非严格为零的小效应；训练梯度当然同时改变它们。

新 arm `alpha64` 将 rank64 的实际 alpha从128降至64，即 LoRA scale 2→1，是 LoRA有效更新幅度候选。它与把全部 LR 2e-4→1e-4不同：head/PE/local不降低LR，LoRA A/B的梯度和优化轨迹也不是简单同倍关系。分别训练，不把二者组合成第一次对照。`lr010`、`lr030` 先按统一日程和同一 split筛选；不要用4轮独立cosine训练替代20轮中的ep4。

`warmup_epochs=3` 是损失／筛选 warm-up，当前 scheduler为20轮直接cosine，**不是 LR warm-up**。加LR预热也可探索，但应明确它会同时改变早期 teacher 与 warm-up prototype，当前没有日志证据显示初期不稳定，所以优先级低于已有覆盖/权重候选。

## 4. EMA、tracker 与prototype的实际行为

teacher EMA=.995 的半衰期约138.3 optimizer step，在约1048 step/epoch下仅约0.13轮；它不等于很多轮前的独立干净模型。tracker EMA=.9 的半衰期约6.58**每样本观测**，均匀有放回时约6.58轮，尾类／头类曝光量还不同。即时 tprob 用于 confidence，较慢的 tracker.prob 用于 epoch级 relabel；它们时间尺度差很大。随意加大EMA可能增大滞后，随意降低tracker momentum又可能让强增强噪声更直接进入target。

prototype update的 trusted 实际是 `tracker.weight>=proto_min_weight(.5)`：既含clean，也含relabel(weight=.5)，还可能含全局cap救回的disagreement(weight=1)。因此代码注释“噪声不可能拖动prototype”“始终冻结CLIP特征空间”不是严格成立：warm-up始于anchor，后续更新来自 adapted teacher tz。当前proto loss对所有样本使用既有w，并不只对trusted做loss，这是既有历史基线，不应悄悄修成新配方。

CPU实证 `LabelTrustTracker.update` 对重复idx的advanced indexing assignment只留下同batch中最后一项，不会顺序折入多次观测。输入idx=[0,0]、prob两行[.8,.1,.1]/[.1,.8,.1]，实际存第二行；下一次重复batch仍只从旧值折入最后行。均匀N134k/B128时每batch重复对期望约.06，主体影响很小，极端尾类oversampling更明显。这是可靠性问题，不宣称修复可提分。shuffle同时避免这一问题。

`resume`仍重置teacher、未保存boot_sum/boot_cnt、未恢复loader generator／persistent worker状态；当前可做到同配置重新从头复现，不能声称断点续训逐步无损。对照从头跑，若必须resume，需单独补完整状态，不应把从旧checkpoint恢复当成无变量的继续。

## 5. 冻结视觉语义惩罚的支持与限制

本轮已增加训练only raw frozen CLIP class mean与有界cost `D(y,c)=(1-cos(proto_y,proto_c))/2`，训练附加 `Σ_c p_c D(y,c)`。官方当前训练split生成、native224、split/class/count fingerprint核对；student输出与推理均不融合prototype logits，CE／现有target不变。这与旧frozen-mix将大量target质量改成教师分布不同，工程和合规路径更清楚。

历史 §31 的32.3%目录一致率来自单medoid教师，不能直接拿来证明raw均值cost一定坏；反之，§33 frozen head-init最终未提交，不能把它当“已证明无收益的平台负例”。它们只说明**冻结原型很难直接替代强模型监督**，不要借新名称重新假设正增益。

重要机制限制：语义距离越近，错误代价越小。它优先抑制远类误判，**不会自动加大容易混淆的细粒度邻居margin**；如果最新剩余误差主要在近邻，它可能几乎没有收益，甚至让近邻prob更易吸收远类prob。固定anchor也已提供CLIP先验约束，额外cost可能重复。CLIP anisotropy可能让类均值互相很近，cost分布窄；统计cost分位数／每类concentration／符合gate的比例与加权项规模，再决定0.05是否有实际梯度。EMA同意且confidence>=.5的gate在750类可能很稀疏，不要把未生效候选当失败方法。

语义标签输入text tower仅在官方提供类名／描述或代码可复现的已有官方元数据映射时才有可靠输入。目录0000…0749不是物种名；凭测试代理／外网图片找名称或人工补充标注会改变数据来源，不能直接当合规文本监督。通用自然语言prompt本身与新增图像集不同，但没有正确的类ID→语义映射无法产生750类有效text classifier。本轮视觉语义cost避免猜物种名，不过它应称“视觉类原型几何”，不应包装成已利用准确物种语义。

## 6. 建议执行优先级与淘汰标准

1. **已有checkpoint搜索**：rank64+qkv学生固定快照，缓存hold-out logits后搜索全局tau，单模型/单固定TTA；无需重训，不平均checkpoint。训练hold-out有噪声，排名只是预筛。逐轮保存可补ep16～20，不增加训练前向成本。
2. **shuffle 或 gamma1**：优先有明确覆盖／梯度机制的简单臂，同control、相同20轮日程；先查看每类曝光与ESS。如果仅能跑约3小时，不一次启动三个训练臂。
3. **优化器 no_decay_small／alpha64／LR**：各自单臂，日志记录head scale、gate、PE范数／LoRA范数、train loss与逐类验证。不同时动LR、alpha和decay。
4. **semantic005 或弱teacher view**：机制不同，分别确认gate/语义cost梯度与筛样结构，然后决定完整训练投入。
5. **全训练数据最终重训**：获得稳定配置和epoch后，把原10%留出加入最终训练，可能有比小正则更直接的有效样本增益。但目前`val_ratio=0`仍按max(1,...)留至少1张/class，不能用这个参数冒充full-data；需要明确full-data模式、固定epoch、不用新holdout重新调参，属于后续独立实现。

短冒烟用于运行正确性／loss与gate是否生效，不用于准确率结论。没有量化真实结果时不承诺任何候选正收益；最佳、最差类先与官方训练hold-out独立确认，特别不能将预测次数少直接当训练长尾。目标76需要净多约656张图，仅7个极端尾类理论最多约0.93分，而且实际更小，故不能只靠尾类惩罚承担整个目标。
