---
title: 第 11 课：优先经验回放
---

# 第 11 课：优先经验回放

<div class="lesson-lead">
均匀经验回放把“早已会做的简单题”和“仍然错得离谱的关键题”用相同概率抽出。本课让高 TD 误差经验获得更多复习机会，同时修正由此引入的偏差。
</div>

## 本课主线

> 优先经验回放提高学习效率，但非均匀抽样会改变数据分布，因此必须用重要性采样权重补账。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解优先级、抽样概率、α、β、重要性采样和优先级更新。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>DQN 回放池、TD 目标和损失函数。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>均匀抽样把大量训练机会花在低信息、低误差样本上。</p></article>
  <article class="lesson-card"><h3>工程代价</h3><p>需要快速按优先级抽样、更新优先级，并管理数值稳定性。</p></article>
</div>

## 1. 什么经验更值得复习？

如果一条经验的 TD 误差很大，说明：

- 网络对它预测很错；或
- 它包含意外奖励、终止、危险边界等重要信息；或
- 当前模型还没理解这一类状态。

**优先经验回放（Prioritized Experience Replay, PER）**把 `|TD 误差|` 当成“错题重要度”的主要依据。

但大误差也可能来自奖励噪声或异常值，所以不是永远只抽最大误差样本，而是提高它的概率。

## 2. 优先级怎样变成概率？

常用公式：

<div class="formula-box">
  <span class="formula">pᵢ=(|δᵢ|+ε)^α</span>
  <span class="formula">P(i)=pᵢ / Σⱼpⱼ</span>
  ε 防止零误差样本永远抽不到；α 控制偏向优先样本的程度。
</div>

- `α=0`：所有优先级都变成 1，退化为均匀回放。
- `α` 越大：越偏爱高误差经验。
- `α=1`：概率与绝对 TD 误差近似成正比。

## 3. 非均匀抽样为什么会带来偏差？

原本希望优化整个回放池上的平均损失。现在高误差样本被重复看很多次，训练目标变成“优先样本分布下的平均损失”。

类比民意调查：如果你为了研究某类意见，故意多采访了一个群体，最后统计总体观点时必须按抽样比例重新加权，否则结论会偏。

因此使用**重要性采样（Importance Sampling）**权重：

`wᵢ = [N·P(i)]^-β`

再把权重归一化，让最大值为 1。

- 高频样本权重更小；
- 低频样本权重更大；
- `β=1` 时进行完整修正；
- 训练早期常用较小 β，后期逐渐增加到 1。


<MechanismDiagram kind="per-cycle" />
<AlgorithmLab lesson="11" />

## 4. 一条新经验刚进入回放池时优先级是多少？

它还没有经过网络更新，暂时没有 TD 误差。常见做法是赋予当前最大优先级，确保至少尽快被抽到一次；计算出真实 TD 误差后，再更新它的优先级。

如果新样本初始优先级设得很低，它可能长期抽不到，就永远没有机会证明自己重要。

## 5. 优先级什么时候更新？

一次训练批次的顺序：

1. 按 P(i) 抽取索引；
2. 读取经验和重要性权重；
3. 计算新的 TD 误差；
4. 用 `wᵢ × lossᵢ` 更新网络；
5. 用新的 `|δᵢ|` 回写这些索引的优先级。

注意优先级会过时。网络学会一条经验后，误差下降，它的复习频率也应下降。

## 6. 高效数据结构为什么常用 SumTree？

直接每次对几十万条优先级求和并抽样，成本高。**和树（Sum Tree）**在树节点保存子树优先级总和：

- 抽样：生成 `[0,总优先级)` 随机数，沿树向下查找，复杂度约 `O(log N)`；
- 更新：修改叶子后沿父节点更新总和，也是 `O(log N)`。

小型教学项目可以先用 NumPy 概率抽样，理解正确后再优化结构。

## 7. 对应的可运行代码

核心位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
def prioritized_replay_probabilities(td_errors, priority_exponent, epsilon=1e-6):
    priorities = (np.abs(td_errors) + epsilon) ** priority_exponent
    return priorities / priorities.sum()


def importance_sampling_weights(probabilities, correction_exponent):
    weights = (len(probabilities) * probabilities) ** (-correction_exponent)
    # 归一化避免权重整体过大，最大样本梯度尺度保持为 1。
    return weights / weights.max()
```

生成抽样 5000 次的可重复实验：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/generate_advanced_lessons.py
```

## 8. 看抽样次数实验

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson11.sample_counts"
  title="不同 TD 误差经验在 5000 次抽样中的出现次数"
  subtitle="六个点依次对应 |δ|=0.1、0.4、1、2、6、10；α=0.6。"
  color="#d97706"
  badge="随机抽样实验"
/>

这不是按时间变化的训练曲线，而是六类经验的抽中次数。越靠后的高误差样本出现越多，但低误差样本仍保留非零机会。

## 9. 什么时候 PER 可能有害？

- 奖励或观测噪声很大，大误差主要是不可预测噪声；
- 异常样本长期占据最高优先级；
- β 修正不足，网络过拟合少数样本；
- 优先级长时间不更新；
- 回放池很小，均匀抽样已经足够频繁看到所有经验。

## 10. 常见误区

- **只保留高误差经验。** PER 改变抽样概率，不等于删除低误差经验。
- **α 和 β 作用相同。** α 控制抽样偏向，β 控制损失修正。
- **TD 误差越大越有价值。** 大误差也可能是噪声或错误奖励。
- **重要性权重乘在优先级上。** 它通常乘在每个样本的损失上。

## 11. 动手练习

1. 分别用 `α=0、0.6、1` 抽样 5000 次并比较直方图。
2. 固定概率，比较 `β=0、0.4、1` 的权重。
3. 给 TD 误差加一个极端异常值 100，观察是否垄断抽样。

<div class="checkpoint"><strong>学会标准：</strong>你能从一组 TD 误差算出优先级和抽样概率，并解释为什么高概率样本反而要降低损失权重。</div>

## 12. 快速自测

<ChapterQuiz lesson="11" />

## 13. 本课只需要记住这些

- PER 更常复习高 TD 误差经验。
- α 决定优先程度，β 决定偏差修正程度。
- 重要性采样权重用于修正非均匀抽样。
- 优先级必须随网络学习结果持续更新。
