---
title: 第 13 课：TD3 连续控制
---

# 第 13 课：TD3 连续控制

<div class="lesson-lead">
连续动作的行动者会主动寻找评论家评分最高的位置。只要 Q 网络出现一个窄而虚假的高峰，策略就可能钻进这个漏洞。TD3 用三道稳定器专门处理这种问题。
</div>

## 本课主线

> 双延迟深度确定性策略梯度使用双评论家、目标策略平滑和延迟策略更新，让确定性连续策略不那么容易追逐错误 Q 峰值。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解确定性策略、双评论家、目标动作平滑和延迟更新。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>SAC 的回放池、目标网络、连续动作和双 Q 思想。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>连续动作评论家高估、动作单点尖峰和行动者追逐尚未学准的 Q。</p></article>
  <article class="lesson-card"><h3>适用场景</h3><p>低维连续控制、仿真机器人、希望稳定使用确定性评估动作。</p></article>
</div>

## 1. 什么是确定性策略？

SAC 的随机策略输出分布，每次可以采样不同动作。**确定性策略（Deterministic Policy）**直接输出一个动作：

`a = μ(s)`

训练时仍会在执行动作上加入探索噪声，但策略网络本身给定状态后只有一个中心动作。

**深度确定性策略梯度（Deep Deterministic Policy Gradient, DDPG）**是早期常见算法。它使用行动者、评论家、经验回放和目标网络，但容易受 Q 值高估影响。

**双延迟深度确定性策略梯度（Twin Delayed Deep Deterministic Policy Gradient, TD3）**可理解为“DDPG + 三个关键修正”。

## 2. 第一重稳定器：双评论家取较小值

TD3 训练 `Q₁` 和 `Q₂`，目标使用：

`min(Q₁′(s′,a′), Q₂′(s′,a′))`

这与 SAC 的双 Q 保守估计相似。行动者只要发现一个评论家虚高还不够，因为目标会参考两者中更低的那一个。

但行动者更新通常只最大化 `Q₁(s,π(s))`，这是经典实现中的效率选择；双评论家的主要保守作用发生在目标计算。

## 3. 第二重稳定器：目标策略平滑

假设 Q 网络在动作 `a=0.731` 处出现一个很窄的错误尖峰，旁边 `0.72` 和 `0.74` 都很普通。行动者可能精确钻到 0.731。

TD3 在目标动作附近加入截断噪声：

<div class="formula-box">
  <span class="formula">ã = clip(π_target(s′) + clip(ε,-c,c), a_low, a_high)</span>
  <span class="formula">ε ~ N(0,σ²)</span>
  评论家目标必须在邻近一小片动作区域都表现合理，不能只靠单点尖峰。
</div>

类比餐馆评分：不只检查厨师“盐正好 2.731 克”的一道菜，而是在附近多个盐量随机抽检。真正稳健的配方应在小扰动下仍然好吃。

## 4. 第三重稳定器：延迟策略更新

评论家每拿到一批数据就更新，但行动者通常每 2 次评论家更新才更新 1 次。

原因是行动者会主动利用评论家的错误。如果评论家刚学一点、判断还摇摆，行动者立刻追上去，二者会互相放大误差。先让评论家多走几步，再让行动者行动，目标更稳定。


<NetworkDiagram kind="td3-structure" />
<AlgorithmLab lesson="13" />

## 5. TD3 一次训练更新的顺序

<ol class="step-list">
  <li>从回放池随机抽取状态、动作、奖励和下一状态。</li>
  <li>目标行动者输出下一动作，并加入截断平滑噪声。</li>
  <li>两个目标评论家评价下一动作，取较小值构造 TD 目标。</li>
  <li>两个在线评论家分别拟合同一个保守目标。</li>
  <li>到达延迟周期时，更新行动者以最大化 Q₁。</li>
  <li>软更新行动者和评论家的目标网络。</li>
</ol>

## 6. TD3 与 SAC 有什么差别？

| 维度 | TD3 | SAC |
| --- | --- | --- |
| 策略 | 确定性中心策略 | 随机高斯策略 |
| 探索 | 执行动作外加噪声 | 策略分布本身采样 |
| 熵目标 | 没有 | 有最大熵和温度参数 |
| 双 Q | 有 | 有 |
| 目标动作平滑 | 显式加入截断噪声 | 随机下一动作天然带分布采样 |
| 行动者更新 | 延迟 | 通常每个训练步更新 |

二者都适合连续控制，没有永远的赢家。任务随机性、奖励尺度、调参预算和探索需求都会影响选择。

## 7. 对应的可运行核心代码

核心位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
def smooth_td3_target_actions(
    target_actions,
    standard_noise,
    noise_standard_deviation,
    noise_clip,
    action_low,
    action_high,
):
    # 先截断噪声，再限制动作，避免目标动作越过环境边界。
    noise = (standard_noise * noise_standard_deviation).clamp(-noise_clip, noise_clip)
    return (target_actions + noise).clamp(action_low, action_high)


def calculate_td3_targets(rewards, dones, target_q_one, target_q_two, gamma):
    conservative_next_value = torch.minimum(target_q_one, target_q_two)
    return rewards + gamma * (1.0 - dones) * conservative_next_value
```

生成目标动作平滑实验：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/generate_advanced_lessons.py
```

## 8. 看目标动作平滑结果

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson13.smoothed_actions"
  title="TD3 目标动作加入截断噪声后的结果"
  subtitle="基础动作从 -1 均匀变化到 1，噪声标准差 0.2、截断 0.5，并再次限制在合法范围。"
  color="#0891b2"
  badge="可重复噪声实验"
/>

曲线中的局部抖动就是目标策略平滑。它只用于目标计算，不等于评估时让机器人持续抖动。

## 9. 常见误区

- **Twin 表示两个行动者。** 经典 TD3 的 twin 主要指两个评论家。
- **目标噪声就是环境探索噪声。** 一个用于构造平滑训练目标，一个用于收集多样化经验。
- **行动者延迟越久越好。** 过度延迟会让策略跟不上已经改进的评论家。
- **取最小 Q 完全消除高估。** 它缓解高估，也可能带来一定保守偏差。

## 10. 动手练习

1. 把目标噪声标准差设为 0，观察平滑消失。
2. 把噪声截断范围设得很大，比较边界动作堆积。
3. 复用 SAC 的回放池与 Q 网络，尝试实现确定性行动者和延迟更新计数器。

<div class="checkpoint"><strong>学会标准：</strong>你能分别说明“双、平滑、延迟”各自针对哪一种不稳定。</div>

## 11. 快速自测

<ChapterQuiz lesson="13" />

## 12. 本课只需要记住这些

- TD3 是面向连续动作的确定性异策略算法。
- 双评论家取较小目标，缓解高估。
- 目标策略平滑抑制动作空间中的虚假尖峰。
- 延迟行动者更新，让评论家先稳定一些。
