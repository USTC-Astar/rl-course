---
title: 第 6 课：连续动作与高斯策略
---

# 第 6 课：连续动作与高斯策略

<div class="lesson-lead">
CartPole 只有“向左”和“向右”两个按钮。真实机器人更常需要决定推力、转矩、速度或舵角的具体大小。本课把动作从按钮升级成油门。
</div>

## 本课主线

> 连续策略不是在无限多个动作中逐个打分，而是输出一个概率分布，再从分布中采样合法动作。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解高斯策略、均值、标准差、重参数采样、tanh 压缩和雅可比修正。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 3 课的概率策略和第 5 课的 PPO 概率比。</p></article>
  <article class="lesson-card"><h3>训练环境</h3><p>MuJoCo InvertedPendulum-v5，动作是 -3～3 的连续水平推力。</p></article>
  <article class="lesson-card"><h3>本课产物</h3><p>一个输出高斯均值和标准差、再压缩成合法动作的 PPO 策略。</p></article>
</div>

## 1. 离散动作和连续动作有什么本质差别？

离散动作像电灯开关：开或关。连续动作像水龙头：可以开 10%、35.7% 或 92%。

DQN 可以为每个离散动作输出一个 Q 值；但连续区间里有无限多个实数，不可能让网络输出无限长列表。

策略梯度则自然得多：网络输出一个动作分布，分布可以覆盖连续数轴。

## 2. 高斯策略的两个输出分别在说什么？

常见连续策略使用**高斯分布（Gaussian Distribution）**：

- 均值 `μ(s)`：当前最想执行的动作中心；
- 标准差 `σ(s)`：愿意在中心附近探索多远。

例如：

- `μ=1.2, σ=0.1`：大概率轻微偏离 1.2，策略比较确定；
- `μ=1.2, σ=1.0`：可能采样到很宽范围，策略仍在探索。

网络通常输出 `log σ`，再通过指数得到正数标准差。这样不用担心优化器把标准差直接更新成负数。

## 3. 重参数技巧为什么把随机性单独拿出来？

采样可写成：

<div class="formula-box">
  <span class="formula">u = μ(s) + σ(s) · ξ， ξ ~ N(0,1)</span>
  网络负责可导的 μ 和 σ，随机性放进独立标准噪声 ξ，这叫重参数技巧（Reparameterization Trick）。
</div>

它让梯度能够沿着 `u → μ,σ → 网络参数` 传播。SAC 的行动者更新尤其依赖这种可微采样。

## 4. 为什么还要经过 tanh？

高斯分布覆盖整个实数轴，理论上可能采到 100 或 -50；环境只允许 -3～3。

项目先做：

`z = tanh(u)`，把动作压到 `(-1,1)`；

再做：

`a = 3 × z`，缩放到 `(-3,3)`。

与直接硬裁剪相比，tanh 是平滑函数。硬裁剪把所有超过边界的值都压成同一个动作，边界外梯度信息很差；tanh 会逐渐饱和。


<NetworkDiagram kind="gaussian-tanh" />
<AlgorithmLab lesson="06" />

## 5. “精确抽到 1.2 的概率”为什么是 0？

离散分布可以说“抽到红桃 A 的概率是 1/52”。连续分布中，单个点没有宽度，概率要通过一个区间面积计算。

例如人的身高：

- 问“身高精确等于无限小数 175.0000… 厘米”的概率，数学上是 0；
- 问“身高在 174.5～175.5 厘米之间”的概率，才有实际意义。

连续策略代码里的 `log_prob` 严格说是**概率密度（Probability Density）**的对数，不是单点事件概率。密度可以大于 1，只要整个数轴积分为 1 即可。

## 6. 雅可比修正在修正什么账？

tanh 会把数轴拉伸和压缩：中间区域变化较均匀，靠近 ±1 时大量原始数值被挤进很窄区间。

如果还沿用原始高斯密度，就像地图缩放后仍拿旧比例尺计算面积。需要**雅可比修正（Jacobian Correction）**：

`log π(a|s) = log N(u;μ,σ) - log(1 - tanh(u)²)`

实际代码还会加入很小的 `ε`，防止 `tanh(u)` 接近 ±1 时对 0 取对数。

::: warning 动作硬裁剪不能直接沿用原始 log_prob
如果训练时计算的是未裁剪高斯动作概率，环境执行的却是硬裁剪动作，概率账本和实际行为不一致，PPO 概率比会失真。
:::

## 7. 训练动作和评估动作为什么不同？

- 训练时：从分布采样，保留探索。
- 评估时：常使用压缩后的均值动作，减少随机波动。

评估动作 `3×tanh(μ)` 严格来说不一定等于变换后分布的数学期望，但它稳定、直观，工程中很常见。

## 8. 对应的可运行代码

核心位于 `src/rl_learning_lab/continuous_ppo.py`：

```python
means, log_standard_deviations, values = network(observations)
standard_deviations = log_standard_deviations.exp()
noise = torch.randn_like(means)

# 把随机性显式写成标准噪声，梯度可以回到均值和标准差。
raw_actions = means + standard_deviations * noise
squashed_actions = torch.tanh(raw_actions)
actions = squashed_actions * action_scale + action_bias

normal = torch.distributions.Normal(means, standard_deviations)
log_probabilities = normal.log_prob(raw_actions)
# tanh 改变了密度尺度，必须修正后才能用于 PPO 概率比。
log_probabilities -= torch.log(1.0 - squashed_actions.pow(2) + 1e-6)
log_probabilities = log_probabilities.sum(dim=-1)
```

运行本课：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_continuous_ppo.py --steps 200000
```

## 9. 看真实训练曲线

<TrainingCurve
  file="continuous_ppo.json"
  path="episode_returns"
  title="MuJoCo 连续动作 PPO 回报"
  subtitle="倒立摆每保持一步得到 1 分；连续策略要学会推力方向和大小。"
  color="#d97706"
  :target="950"
/>

## 10. 什么情况下不适合简单高斯策略？

单峰高斯适合“一个主要动作中心 + 附近探索”。如果同一状态下有两个相隔很远且都很好的动作，例如绕障碍物可以从最左或最右通过，中间反而危险，单个高斯峰可能把均值放在危险中间。

可考虑：混合高斯、流模型策略、离散高层决策加连续低层控制，或让状态包含更多信息使动作选择不再多峰。

## 11. 常见误区

- **标准差越大探索越好。** 太大时动作频繁撞边界，策略难以精细控制。
- **均值就是最终每次执行动作。** 训练时动作还包含随机噪声和 tanh 压缩。
- **概率密度必须小于 1。** 连续密度可以大于 1，真正概率是区间积分。
- **tanh 只负责裁剪，无需改概率。** 非线性变换后必须做密度修正。

## 12. 动手练习

1. 把初始 `log_std` 调低，观察策略是否探索不足。
2. 移除雅可比修正，只做短训练并监控 PPO 概率比。
3. 把评估动作改为随机采样，比较多次评估方差。

<div class="checkpoint"><strong>学会标准：</strong>你能从 μ、σ 和一次噪声 ξ，算出原始动作、tanh 后动作和环境缩放动作。</div>

## 13. 快速自测

<ChapterQuiz lesson="06" />

## 14. 本课只需要记住这些

- 连续策略输出分布参数，而不是枚举无限多个动作。
- 均值表示主要动作，标准差表示探索宽度。
- tanh 把无界高斯动作平滑压进合法范围。
- 变换后的概率密度必须进行雅可比修正。
