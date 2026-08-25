---
title: 第 7 课：SAC 与最大熵学习
---

# 第 7 课：SAC 与最大熵学习

<div class="lesson-lead">
PPO 像“当堂练习”：数据由当前策略产生，用几轮后就丢弃。软行动者—评论家把经验放入长期错题本，隔很久仍能随机抽出来学习，同时保留连续动作探索。
</div>

## 本课主线

> SAC 用经验回放提高样本利用率，用双 Q 网络降低高估，用最大熵目标让策略在追求回报时不过早失去多样性。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解异策略学习、回放池、双 Q、软目标网络、熵温度和 SAC 三类更新。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>DQN 的回放池、行动者—评论家、高斯连续策略。</p></article>
  <article class="lesson-card"><h3>训练环境</h3><p>Pendulum-v1：用 -2～2 的连续转矩把摆锤甩起并稳定在上方。</p></article>
  <article class="lesson-card"><h3>本课产物</h3><p>随机行动者、两个 Q 网络、两个慢速目标 Q 网络和一个温度参数。</p></article>
</div>

## 1. 同策略和异策略到底差在哪里？

**同策略学习（On-Policy Learning）**要求训练数据来自当前或非常接近当前的策略；**异策略学习（Off-Policy Learning）**允许使用其他旧策略产生的数据。

类比：

- PPO 像教练只分析运动员最近这一套战术录像，因为战术变化后旧录像代表性下降。
- SAC 像长期错题本：题目来自过去不同阶段，只要记录了题目、选择、结果和下一题状态，现在仍可拿来学习价值关系。

异策略方法通常样本效率更高，代价是目标计算和稳定性处理更复杂。

## 2. Pendulum 的状态为什么用 sin 和 cos？

摆锤观测是 `[cos θ, sin θ, 角速度]`，而不是直接只给角度 θ。

因为角度有环绕问题：`-π` 和 `π` 在数值上相差很大，物理方向却几乎相同；`sin/cos` 把圆周连续表示出来。

动作是连续转矩：

- 负值向一个方向施力；
- 正值向另一个方向施力；
- 绝对值表示力度。

## 3. 经验回放池里存什么？

SAC 的**经验回放池（Replay Buffer）**仍保存：

`(s, a, r, s′, done)`

训练流程不是“走一步只学这一步”，而是：

1. 与环境交互并追加新经验；
2. 从整个回放池随机抽一批；
3. 更新两个评论家；
4. 更新行动者；
5. 更新熵温度；
6. 让目标评论家慢慢跟随。

随机抽样降低连续轨迹相关性，也让失败、荡起和接近稳定等不同阶段混在同一批次中。

## 4. 为什么要两个 Q 网络？

连续策略会主动寻找 Q 网络输出最高的动作。如果 Q 网络某处因为数据少产生虚假尖峰，行动者会像发现“评分漏洞”一样钻进去。

SAC 训练两个独立评论家 `Q₁` 和 `Q₂`，目标中取较小值：

`min(Q₁(s,a), Q₂(s,a))`

类比两位房屋估价师：一个估价 82 万，一个估价 67 万。为了避免贷款建立在过度乐观报价上，先采用较保守的 67 万。

它不能保证完全没有误差，但能显著缓解系统性高估。

## 5. 最大熵目标到底在奖励什么？

**最大熵强化学习（Maximum-Entropy Reinforcement Learning）**的目标不只是累计奖励，还加入策略熵：

<div class="formula-box">
  <span class="formula">目标 ≈ E[累计奖励 + α × 策略熵]</span>
  温度系数 α 决定随机性值多少钱。α 大，更愿意探索；α 小，更专注当前高价值动作。
</div>

它不是让动作永远随机。高回报仍是主目标；熵只是在多个差不多的动作间，鼓励暂时保留选择余地。


<NetworkDiagram kind="sac-structure" />
<AlgorithmLab lesson="07" />


<div class="derivation-box">

**最大熵目标的形式与温度的作用**

SAC 把"回报"与"随机性"放进同一个目标：

$$J(\theta) = \mathbb{E}\left[\sum_t \gamma^t \big(\, r_t + \alpha\, H(\pi_\theta(\cdot \mid s_t)) \,\big)\right]$$

对它做四步拆解：

1. $H(\pi) = -\sum_a \pi(a)\log \pi(a)$ 是策略熵：动作越均匀熵越大（2 个动作各 0.5 时 $H=\ln 2 \approx 0.69$；全押一个动作时 $H=0$）。
1. 对 $\theta$ 求梯度后，策略更新同时收到两个方向的拉力：**提高预测回报**（第一项）与**保持动作分布平摊**（第二项）。
1. $\alpha$ 是两者汇率：$\alpha$ 大 → 略牺牲回报也要多探索；$\alpha \to 0$ → 退化成普通回报最大化。SAC 的做法是把 $\alpha$ 也当成可学习参数，锁定目标熵（通常 $-\dim(\mathcal{A})$），自动升降温。
1. 温度自动调节的直觉：策略太确定（熵低于目标）→ 损失逼 $\alpha$ 变大压回随机；探索已足够（熵高于目标）→ $\alpha$ 变小，专注回报。

</div>

## 6. SAC 三个主要更新分别在做什么？

### 更新评论家

目标大致是：

`y = r + γ[min(Q₁′,Q₂′) - α log π(a′|s′)]`

目标价值既看保守 Q，也考虑下一动作的熵收益。两个在线评论家分别拟合这个目标。

### 更新行动者

行动者最小化：

`L_actor = α log π(a|s) - min(Q₁,Q₂)`

第一项惩罚过于确定，第二项鼓励高 Q 动作。最小化后自然形成“高价值且保留适度随机”的策略。

### 更新温度

温度参数可以自动学习，使实际熵靠近**目标熵（Target Entropy）**。如果策略太确定，就提高随机性价值；如果过于随机，就降低温度。

## 7. 软目标网络为什么不是定期整份复制？

DQN 常每隔一段时间硬复制。SAC 常使用**Polyak 平均（Polyak Averaging）**进行软更新：

`θ_target ← (1-τ)θ_target + τθ_online`

例如 `τ=0.005`，目标网络每次只吸收在线网络 0.5% 的新参数。它像一位保守编辑，不会因为作者刚改一版就全部替换出版稿。

## 8. 对应的可运行代码

核心位于 `src/rl_learning_lab/sac.py`：

```python
states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

with torch.no_grad():
    next_actions, next_log_probabilities = actor.sample(next_states)
    next_q = torch.minimum(
        target_q_one(next_states, next_actions),
        target_q_two(next_states, next_actions),
    )
    target = rewards + gamma * (1.0 - dones) * (
        next_q - temperature * next_log_probabilities
    )

critic_loss = torch.nn.functional.mse_loss(q_one(states, actions), target)
critic_loss += torch.nn.functional.mse_loss(q_two(states, actions), target)

new_actions, log_probabilities = actor.sample(states)
minimum_q = torch.minimum(q_one(states, new_actions), q_two(states, new_actions))
actor_loss = (temperature * log_probabilities - minimum_q).mean()
```

运行本课：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_sac.py --steps 100000
```

## 9. 看真实训练曲线

<TrainingCurve
  file="sac.json"
  path="episode_returns"
  title="Pendulum SAC 每回合回报"
  subtitle="Pendulum 回报通常为负数，越接近 0 越好；早期需要先学会荡起。"
  color="#dc2626"
  :target="-200"
/>

::: tip 为什么“最好回报”不是正数？
Pendulum 每步会根据角度偏差、角速度和动作能耗扣分。最理想情况是每步接近 0 分，因此总回报越不负越好。
:::

## 10. PPO 和 SAC 应怎样选择？

| 维度 | PPO | SAC |
| --- | --- | --- |
| 数据使用 | 当前批次有限复用 | 回放池长期复用 |
| 类型 | 同策略 | 异策略 |
| 动作 | 离散和连续都常用 | 主要用于连续动作 |
| 样本效率 | 通常较低 | 通常较高 |
| 实现复杂度 | 中等 | 较高 |
| 适合场景 | 大量并行仿真、追求稳定 | 真实交互昂贵、连续控制 |

## 11. 常见误区

- **最大熵等于动作完全随机。** 熵是与回报权衡的辅助目标。
- **两个 Q 网络输出取平均。** 经典 SAC 目标通常取较小值。
- **回放池任何旧数据都永远有用。** 环境变化或数据质量极差时，旧数据也可能有害。
- **目标网络 τ 越小越好。** 太小会让目标严重滞后，太大又失去平滑作用。

## 12. 动手练习

1. 把回放池缩小到 2000，观察是否更容易遗忘早期荡起经验。
2. 把软更新率改成 1，比较 Q 值分歧和损失波动。
3. 固定很小的温度，观察策略是否过早变得确定。

<div class="checkpoint"><strong>学会标准：</strong>你能说明 SAC 的行动者、两个评论家、两个目标评论家、回放池和温度参数各自解决什么问题。</div>

## 13. 快速自测

<ChapterQuiz lesson="07" />

## 14. 本课只需要记住这些

- SAC 是可长期复用旧经验的异策略连续控制算法。
- 双 Q 取较小值，用来缓解行动者钻 Q 网络高估漏洞。
- 最大熵目标同时追求回报和适度随机性。
- 目标网络通过软更新缓慢跟随在线评论家。
