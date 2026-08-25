---
title: 第 4 课：行动者—评论家
---

# 第 4 课：行动者—评论家

<div class="lesson-lead">
REINFORCE 像比赛结束后才复盘整场。本课增加一位“场边教练”：行动者负责选择动作，评论家随时估计局面价值，让反馈更及时。
</div>

## 本课主线

> 行动者—评论家把“直接学策略”和“学习价值”组合起来：策略负责行动，价值负责降低策略梯度的噪声。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>分清 V、Q、A 和 TD 误差；理解两个损失如何协同。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 2 课的价值学习和第 3 课的概率策略。</p></article>
  <article class="lesson-card"><h3>训练环境</h3><p>16 个并行 CartPole 环境，提高采样效率和数据多样性。</p></article>
  <article class="lesson-card"><h3>本课产物</h3><p>共享特征层，以及行动者输出头和评论家输出头。</p></article>
</div>

## 1. 两个角色分别解决什么问题？

**行动者—评论家（Actor-Critic）**包含：

- **行动者（Actor）**：输出策略 `π(a|s)`，决定怎么做。
- **评论家（Critic）**：估计状态价值 `V(s)` 或动作价值 `Q(s,a)`，判断局面有多好。

类比足球训练：

- 运动员在场上做传球、射门选择；
- 教练不必等 90 分钟结束才说“整场还行”，而能在一次配合后指出“这次比预期更好”。

## 2. V、Q、A 到底有什么区别？

| 名称 | 写法 | 通俗问题 |
| --- | --- | --- |
| 状态价值（State Value） | `V(s)` | 站在这个局面，整体前景有多好？ |
| 动作价值（Action Value） | `Q(s,a)` | 在这个局面采取这个动作，前景有多好？ |
| 优势函数（Advantage Function） | `A(s,a)=Q(s,a)-V(s)` | 这个动作比当前局面的平均选择好多少？ |

如果某个状态本身已经非常好，两个动作的 Q 值都可能很高；优势值会把“状态本来就好”扣掉，只保留动作额外贡献。

## 3. 时序差分误差怎样提供即时评价？

**时序差分误差（Temporal-Difference Error, TD Error）**：

<div class="formula-box">
  <span class="formula">δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)</span>
  它比较“原来预测的局面价值”和“行动后看到的新目标”。在行动者更新中，δ 常被当作优势的近似。
</div>

数字例子：

- 当前评论家估计 `V(s)=4`；
- 走一步得到 `r=1`；
- 下一状态估计 `V(s′)=5`；
- `γ=0.9`。

新目标是 `1+0.9×5=5.5`，TD 误差是 `5.5-4=1.5`。这是正惊喜，应提高刚才动作的概率。


<NetworkDiagram kind="actor-critic" />
<AlgorithmLab lesson="04" />

## 4. 行动者和评论家各自优化什么？

行动者损失：

`L_actor = -log π(a|s) × A`

评论家损失：

`L_critic = [V(s) - 回报目标]²`

总损失通常还包含熵奖励：

`L = L_actor + c_v L_critic - c_e H(π)`

三个部分分工明确：

1. 行动者提高好动作概率、降低坏动作概率；
2. 评论家让价值预测更接近真实回报；
3. 熵项防止策略过早失去探索。

::: warning 优势进入行动者损失时通常要停止梯度
行动者只需要把优势当评分，不应通过行动者损失反向修改评论家。因此代码常写 `advantages.detach()`。
:::

## 5. 为什么向前看 5 步，而不是只看一步？

项目使用**n 步回报（N-step Return）**和**广义优势估计（Generalized Advantage Estimation, GAE）**。

- 只看 1 步：大量依赖评论家估计，比较稳定但可能有偏。
- 看完整回合：更多使用真实奖励，偏差小但波动大。
- 看 5 步：在二者之间折中。

GAE 再用参数 `λ` 把不同距离的 TD 误差加权混合。第 8 课会把这部分完整展开，本课先记住：它是一个“远近反馈混合器”。

## 6. 为什么并行训练 16 个环境？

单个 CartPole 连续轨迹中的状态非常相似。并行环境让同一个批次同时包含 16 辆处于不同阶段的小车：有的刚开始，有的接近失败，有的正在恢复。

这不是 16 个模型，而是：

- 1 个共享模型；
- 16 个环境副本；
- 收集后合成一个更丰富的训练批次。

## 7. 对应的可运行代码

核心位于 `src/rl_learning_lab/actor_critic.py`：

```python
logits, state_values = network(observations)
distribution = torch.distributions.Categorical(logits=logits)
actions = distribution.sample()

advantages, return_targets = calculate_gae(
    rewards=rewards,
    values=state_values,
    next_value=next_value,
    dones=dones,
    discount_factor=config.discount_factor,
    gae_lambda=config.gae_lambda,
)

# 优势只是给行动者的评分，不能让这条梯度偷偷改动评论家。
actor_loss = -(distribution.log_prob(actions) * advantages.detach()).mean()
critic_loss = torch.nn.functional.mse_loss(state_values, return_targets)
entropy = distribution.entropy().mean()
loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
```

运行本课：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_actor_critic.py --steps 300000
```

## 8. 看真实训练曲线

<TrainingCurve
  file="actor_critic.json"
  path="episode_returns"
  title="行动者—评论家每回合回报"
  subtitle="评论家提供更及时的优势估计，通常比纯 REINFORCE 更高效。"
  color="#059669"
  :target="475"
/>

## 9. 评论家会不会把行动者带偏？

会。评论家只是估计器，不是真正裁判。如果它还没学准，错误优势就可能鼓励错误动作。因此训练需要：

- 让评论家损失稳定下降；
- 对优势标准化；
- 限制梯度大小；
- 不让策略一次变化太大；
- 定期用真实回合回报评估，而不是只信 V 值。

这正是 PPO 要继续解决的问题。

## 10. 常见误区

- **评论家等于环境奖励。** 环境给即时奖励，评论家预测长期价值。
- **行动者和评论家必须是两张完全独立网络。** 它们可以共享前面的特征层，再分成两个输出头。
- **价值损失越低，策略一定越好。** 评论家可能准确预测一个很差的策略；最终仍要看真实回报。
- **并行环境等于多智能体。** 这里是多个独立副本为同一个策略采样，不是多个会互相影响的智能体。

## 11. 动手练习

1. 把熵系数设为 0，观察动作概率是否更快接近 0 或 1。
2. 把 rollout 步数从 5 改为 1，再比较评论家损失和训练速度。
3. 移除 `detach()`，查看评论家梯度如何被行动者损失干扰。

<div class="checkpoint"><strong>学会标准：</strong>看到 `δ=r+γV(s′)-V(s)` 时，你能说出正负号分别在告诉行动者什么。</div>

## 12. 快速自测

<ChapterQuiz lesson="04" />

## 13. 本课只需要记住这些

- 行动者负责选择，评论家负责评价。
- 优势值衡量具体动作相对状态平均水平的好坏。
- TD 误差把等待整局结束的反馈变成更及时的反馈。
- 评论家降低方差，但错误评论家也会误导行动者。
