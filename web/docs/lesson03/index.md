---
title: 第 3 课：REINFORCE 策略梯度
---

# 第 3 课：REINFORCE 策略梯度

<div class="lesson-lead">
DQN 先估计每个动作值多少分，再选最高分动作。本课换一条路线：让网络直接输出动作概率，并根据整局结果调整这些概率。
</div>

## 本课主线

> 策略梯度不问“这个动作值多少分”，而问“怎样改变动作概率，才能让高回报轨迹更常发生”。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解概率策略、轨迹、折扣回报、对数概率和 REINFORCE 损失。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>知道神经网络输出、概率分布和第 1 课的折扣奖励。</p></article>
  <article class="lesson-card"><h3>训练环境</h3><p>仍然是 CartPole-v1，方便只比较“大脑”变化。</p></article>
  <article class="lesson-card"><h3>本课产物</h3><p>一个直接输出左右动作概率的策略网络。</p></article>
</div>

## 1. 从“评分后选择”变成“直接学习选择习惯”

DQN 像先给每道菜打分，再点最高分菜；**策略梯度（Policy Gradient）**像直接形成点菜习惯：

```text
DQN：状态 → [左动作价值, 右动作价值] → 选较大值
策略：状态 → [左动作概率, 右动作概率] → 按概率抽样
```

输出动作概率的函数叫**策略（Policy）**，常写作 `π(a|s)`，意思是“在状态 s 下选择动作 a 的概率”。

概率策略保留随机性并不是缺点：训练早期如果“向右 51%，向左 49%”，它仍会尝试两边；如果直接只选较大值，微小的初始误差就可能被过早固化。

## 2. 环境为什么没有逐帧标准答案？

CartPole 在某一帧向左推，可能是在救杆子，也可能让几秒后更糟。环境只在每一步给存活奖励，无法像分类数据那样告诉你“这一帧标准动作就是左”。

因此 REINFORCE 收集一整条**轨迹（Trajectory）**：

`s₀,a₀,r₀,s₁,a₁,r₁,...,s_T`

回合结束后再回看：这条轨迹总成绩好不好？哪些早期动作应得到更多或更少鼓励？

## 3. 折扣回报怎样给不同时间的动作分配结果？

时间 `t` 的**折扣回报（Discounted Return）**是：

<div class="formula-box">
  <span class="formula">Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...</span>
  越靠近当前动作的奖励权重越大，越远的奖励乘更多次 γ。
</div>

假设未来三步奖励是 `[1, 1, 1]`，`γ=0.9`：

- 第一步的回报：`1 + 0.9×1 + 0.9²×1 = 2.71`；
- 第二步的回报：`1 + 0.9×1 = 1.9`；
- 第三步的回报：`1`。

越早的动作影响了更多后续结果，因此它看到的累计回报更长。

## 4. REINFORCE 损失为什么有一个负号？

核心损失：

<div class="formula-box">
  <span class="formula">L = - log π(aₜ|sₜ) · Gₜ</span>
  优化器默认最小化损失。高回报 Gₜ 为正时，负号会推动动作对数概率增大；负回报则推动概率减小。
</div>

对数并不是为了“让公式显得高级”。它有两个实用作用：

1. 一条轨迹的概率是许多动作概率相乘，取对数后乘法变加法，数值更稳定。
2. `log π` 的梯度形式适合直接推导“提高采样动作概率”的方向。

<AlgorithmLab lesson="03" />


## 4.5 加餐：策略梯度定理为什么成立？

第 4 节直接使用了"对 log 概率求梯度"。这一框把中间省略的推导补上，只需要一条恒等式。

<div class="derivation-box">

**策略梯度定理的推导（离散版直觉）**

目标：最大化期望回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$，轨迹 $\tau$ 由策略 $\pi_\theta$ 自己采样产生。对 $\theta$ 求梯度，只需四步：

1. 轨迹概率是各步概率的连乘：$\pi_\theta(\tau) = \prod_t \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)$。
1. 对数微分恒等式：$\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau)$——分母的 $\pi$ 被挪进了期望的权重，而期望本身按 $\pi$ 采样，正好抵消。
1. 环境的转移项 $P(s_{t+1} \mid s_t, a_t)$ 与 $\theta$ 无关，求梯度后消失——**不需要知道环境模型**。
1. 剩下 $\nabla_\theta J = \mathbb{E}_{\tau}\left[ \left(\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right) G(\tau) \right]$。乘上回报 $G$：回报高的轨迹，其每步对数概率都被推高；回报低的被压低。

**数字代入**：一条轨迹回报 $G=+2$，其中某步 $\pi_\theta(a\mid s)=0.4$。梯度方向是 $+2 \cdot \nabla \log 0.4$——增大这一步的概率；若 $G=-2$，方向反转为压低。这正是上一节损失 $-\log \pi \cdot G$ 的来源。

</div>

## 5. 为什么训练曲线比 DQN 更晃？

REINFORCE 使用整局回报。一次回合的好坏可能同时受到：

- 动作采样随机性；
- 初始状态差异；
- 前面一个小错误引发的连锁反应；
- 很多动作共同造成的最终结果。

它像比赛结束后只告诉全队“今天得了 82 分”，却没有逐回合教练评分。反馈无偏但噪声很大。

常见改进包括：

- 对回报做标准化；
- 加入基线减少方差；
- 同时训练多个环境收集更多轨迹；
- 使用评论家提供更及时评价，这就是下一课。

## 6. 熵奖励为什么能防止策略过早僵化？

**熵（Entropy）**衡量动作分布有多分散：

- `[0.5, 0.5]` 熵高，两个动作都愿意尝试；
- `[0.999, 0.001]` 熵低，策略几乎已经锁死。

训练早期给一点熵奖励，相当于提醒策略：“先别因为几次偶然成功就认定唯一答案。”但熵系数太大，策略会一直犹豫，难以稳定执行好动作。

## 7. 对应的可运行代码

核心位于 `src/rl_learning_lab/policy_gradient.py`：

```python
logits = policy_network(observations)
distribution = torch.distributions.Categorical(logits=logits)
actions = distribution.sample()

# 回合结束后，把每个时刻的未来奖励变成可比较的训练信号。
returns = normalized_returns(rewards, config.discount_factor)
log_probabilities = distribution.log_prob(actions)

policy_loss = -(log_probabilities * returns).mean()
entropy = distribution.entropy().mean()
loss = policy_loss - config.entropy_coefficient * entropy
```

运行本课：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_policy_gradient.py --episodes 600
```

## 8. 看真实训练曲线

<TrainingCurve
  file="policy_gradient.json"
  path="episode_returns"
  title="REINFORCE 每回合回报"
  subtitle="整局回报共同参与更新，因此曲线通常比 DQN 更有波动。"
  color="#7c3aed"
  :target="475"
/>

## 9. 使用前后有什么差别？

<div class="compare-grid">
  <article class="compare-card"><h3>价值方法</h3><p>先学习动作价值，再从价值间接得到策略；离散动作中选择方便。</p></article>
  <article class="compare-card"><h3>策略方法</h3><p>直接优化动作分布，天然支持随机策略和连续动作，但梯度方差常更大。</p></article>
</div>

## 10. 常见误区

- **概率最大动作就是每次必选。** 训练时通常按分布采样，评估时才常用最大概率动作。
- **回报高就把所有动作概率都提高。** 概率总和必须为 1，提高采样动作会相对压低其他动作。
- **熵越大越好。** 熵只是探索手段，最终目标仍是高累计回报。
- **REINFORCE 每一步都立刻更新。** 经典形式需要先得到后续回报，通常在回合结束后更新。

## 11. 动手练习

1. 关闭回报标准化，比较不同回合长度导致的梯度尺度。
2. 把熵系数改为 0，观察策略是否更早变成接近 0/1。
3. 把 `γ` 调低，解释早期动作为什么更难得到长期信用。

<div class="checkpoint"><strong>学会标准：</strong>你能解释“高回报为什么提高当时动作概率”，而不是只说“因为公式有负号”。</div>

## 12. 快速自测

<ChapterQuiz lesson="03" />

## 13. 本课只需要记住这些

- 策略网络直接输出动作概率。
- REINFORCE 用完整轨迹的折扣回报调整采样动作概率。
- 对数概率让轨迹概率计算和梯度更稳定。
- 整局反馈带来高方差，下一课用评论家提供更及时的评价。
