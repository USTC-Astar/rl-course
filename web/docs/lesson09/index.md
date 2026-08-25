---
title: 第 9 课：Double DQN
---

# 第 9 课：Double DQN

<div class="lesson-lead">
DQN 的 max 操作既负责挑出“最高分动作”，又直接相信这个最高分。只要估计里有噪声，偶然被高估的动作就更容易获胜。本课把“选择”和“评价”拆开。
</div>

## 本课主线

> Double DQN 不增加第三张网络，而是重新分配在线网络和目标网络的职责：在线网络选动作，目标网络评动作。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解最大化偏差、高估形成过程和 Double DQN 目标。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 2 课 DQN 的在线网络、目标网络和 TD 目标。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>带噪 Q 估计经过 max 后产生系统性乐观偏差。</p></article>
  <article class="lesson-card"><h3>改造成本</h3><p>几乎不增加参数，只修改下一状态目标值的计算方式。</p></article>
</div>

## 1. 为什么 max 容易挑中“虚高选手”？

假设两个动作真实价值都等于 5，但网络估计有误差：

- 动作左：估计 6.5；
- 动作右：估计 4.2。

`max` 会选择 6.5。下一次误差可能换方向，但每次 max 都更容易选择正噪声较大的那个值。长期看，即使单个估计误差平均为 0，最大值的误差平均也可能大于 0。

类比招聘：两位能力相同的候选人各参加一次带运气成分的测试。公司永远录取分数较高者，并把测试最高分当成真实能力，录取者能力就容易被高估。

这种现象叫**最大化偏差（Maximization Bias）**。

## 2. 普通 DQN 的目标哪里混合了两项职责？

普通 DQN：

`y = r + γ maxₐ Q_target(s′,a)`

目标网络同时做了：

1. 在所有动作里选谁最大；
2. 用这个最大值评价未来。

如果某个动作在目标网络中偶然虚高，它既赢得选择，又把虚高数值写入训练目标，在线网络随后会向这个虚高目标靠近。

## 3. Double DQN 怎样拆开？

**双重深度 Q 网络（Double Deep Q-Network, Double DQN）**：

<div class="formula-box">
  <span class="formula">a* = argmaxₐ Q_online(s′,a)</span>
  <span class="formula">y = r + γQ_target(s′,a*)</span>
  在线网络只说“我选哪个动作”，目标网络只说“这个指定动作值多少”。
</div>

关键不是两张网络永远独立——目标网络仍定期来自在线网络——而是同一次目标计算中，选择噪声和评价噪声不完全相同。

<AlgorithmLab lesson="09" />
<OverestimateLab />


## 4. 用具体数字比较一次

在线网络给出：`[左=8, 右=6]`，所以选择左。

目标网络给出：`[左=5, 右=7]`。

- 普通 DQN：目标网络自己选最大值 7，选择右。
- Double DQN：在线网络已经指定左，只读取目标网络左动作值 5。

普通 DQN 在这次样本上得到 7，Double DQN 得到 5。后者不一定总是更小，但不会让评价网络临时改选它自己偶然高估的动作。

## 5. Double DQN 会不会变成系统性低估？

它的目标是降低高估偏差，不是保证所有估值准确，也不保证永远小于普通 DQN。

仍可能存在：

- 两个网络误差高度相关；
- 数据覆盖不足；
- 函数逼近能力不足；
- 奖励噪声；
- 优化器没有收敛。

因此应比较独立评估回报、Q 值均值和真实蒙特卡洛回报，而不是只看“Q 值更小了”。

## 6. 对应的可运行代码

核心位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
def calculate_double_dqn_targets(
    rewards,
    dones,
    next_online_q_values,
    next_target_q_values,
    discount_factor,
):
    # 在线网络只负责选择，避免目标网络既当选手又当裁判。
    selected_actions = next_online_q_values.argmax(dim=1, keepdim=True)
    selected_values = next_target_q_values.gather(1, selected_actions).squeeze(1)
    return rewards + discount_factor * (1.0 - dones) * selected_values
```

把它接入现有 DQN 时，只需在优化函数中额外计算在线网络的下一状态 Q 值，并替换原来的 `target.max(...)`。

## 7. 看 200 次带噪估计实验

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson09.standard_estimates"
  title="普通 DQN 的带噪最大值估计"
  subtitle="两个动作真实价值都为 5，每次叠加零均值噪声后取最大值。"
  color="#d97706"
  :target="5"
  badge="蒙特卡洛模拟"
/>

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson09.double_estimates"
  title="Double DQN 的选择—评价解耦估计"
  subtitle="在线噪声用于选动作，另一组目标网络噪声用于评价被选动作。"
  color="#2563eb"
  :target="5"
  badge="蒙特卡洛模拟"
/>

单次样本可能更高或更低，真正要比较的是大量样本平均偏差。

## 8. 使用前后有什么差别？

| 项目 | 普通 DQN | Double DQN |
| --- | --- | --- |
| 选动作 | 目标网络 max | 在线网络 argmax |
| 评价动作 | 同一个目标网络最大值 | 目标网络评价指定动作 |
| 参数量 | 在线 + 目标 | 在线 + 目标 |
| 主要收益 | 实现简单 | 缓解 Q 值高估 |

## 9. 常见误区

- **Double DQN 需要四张网络。** 实际仍常是在线网络和目标网络两张，只改变职责。
- **Q 值降低就说明性能提高。** 更小不等于更准，应与真实回报比较。
- **Double DQN 解决所有不稳定。** 回放相关性、目标漂移和分布偏移仍存在。
- **在线网络选动作时需要梯度。** 目标计算通常放在 `no_grad` 中，不反向传播。

## 10. 动手练习

1. 把模拟噪声标准差从 1.5 改成 0.2，比较两者平均差距。
2. 让在线和目标噪声完全相同，观察解耦收益为何减弱。
3. 在 CartPole DQN 中切换普通和 Double 目标，固定随机种子比较。

<div class="checkpoint"><strong>学会标准：</strong>你能用一句话准确说出哪张网络选动作、哪张网络评动作，并写出 gather 的索引来源。</div>

## 11. 快速自测

<ChapterQuiz lesson="09" />

## 12. 本课只需要记住这些

- max 会偏爱被噪声偶然抬高的动作。
- 普通 DQN 让同一组估计同时选择和评价最大动作。
- Double DQN 用在线网络选择、目标网络评价。
- 它降低高估偏差，但不保证没有估计误差。
