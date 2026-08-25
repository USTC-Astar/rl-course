---
title: 第 2 课：CartPole 与 DQN
---

# 第 2 课：CartPole 与深度 Q 网络

<div class="lesson-lead">
网格世界只有 36 个位置，可以直接存 Q 表。CartPole 的位置、速度、角度和角速度都是连续数值，几乎不可能把每种情况逐格列完。本课让神经网络代替查表。
</div>

## 本课主线

> 深度 Q 网络没有改变 Q 学习的目标，只是把“查一张完整表”改成“让神经网络近似 Q 函数”。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解函数逼近、经验回放、目标网络和 DQN 的一次更新。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>理解第 1 课的 Q 值和 Q 学习目标；知道神经网络能拟合输入到输出。</p></article>
  <article class="lesson-card"><h3>训练环境</h3><p>CartPole-v1：左右推小车，让杆子保持竖直。</p></article>
  <article class="lesson-card"><h3>本课产物</h3><p>一个输入 4 个状态量、输出左右动作 Q 值的神经网络。</p></article>
</div>

## 1. Q 表为什么装不下 CartPole？

CartPole 的观测包含：

1. 小车位置；
2. 小车速度；
3. 杆子角度；
4. 杆子角速度。

即使把每个量粗糙切成 100 档，也会得到 `100⁴=1 亿` 个状态；每个状态还要存两个动作的 Q 值。更麻烦的是，真实数值不一定刚好落在离散格点上。

**深度 Q 网络（Deep Q-Network, DQN）**改为：输入当前 4 个数，输出“向左”和“向右”的两个 Q 值。

这像从“背完所有题目的答案表”变成“学会解题规律”：看到没见过但相似的题，也能估计答案。

## 2. DQN 的网络到底输出什么？

网络不是直接输出动作名称，也不是输出动作概率，而是输出每个离散动作的价值：

```text
输入状态 [位置, 速度, 角度, 角速度]
              ↓
          Q 网络
              ↓
输出 [Q(向左), Q(向右)] = [8.2, 10.5]
```

执行时通常选择 Q 值更大的“向右”，训练时仍保留 ε-贪心探索。

::: tip DQN 适合什么动作空间？
DQN 一次输出所有动作的 Q 值，因此适合动作数量有限的**离散动作空间（Discrete Action Space）**。如果动作是任意大小的油门或转矩，无法列出无限多个输出，后面需要 PPO、SAC 或 TD3。
:::

## 3. 经验回放：为什么不能只学刚发生的一步？

**经验回放（Experience Replay）**把每次经历保存成：

`(状态 s, 动作 a, 奖励 r, 下一状态 s′, 是否结束 done)`

训练时随机抽一批旧经历，而不是按发生顺序只学最新一步。

生活类比：如果你只按时间顺序复习错题，连续十道都是同一题型，大脑容易“刚学什么就只会什么”；把不同章节错题混合抽取，训练更均衡。

它解决三个问题：

- 打散相邻状态之间的强相关性；
- 一条昂贵环境经验可以使用多次；
- 训练批次同时包含成功、失败和不同阶段的数据。

## 4. 目标网络：为什么要保留一份慢速副本？

DQN 的目标仍是：

<div class="formula-box">
  <span class="formula">y = r + γ maxₐ Q_target(s′,a)</span>
  在线网络 Q_online 负责学习；目标网络 Q_target 负责提供暂时稳定的下一步答案。
</div>

如果只有一个网络，它刚根据目标修改参数，目标本身也立刻跟着变化，就像老师一边让学生答题，一边每秒修改标准答案。

目标网络的做法是：

1. 在线网络每个训练批次更新；
2. 目标网络保持不动；
3. 每隔若干步，把在线网络参数复制给目标网络。


<NetworkDiagram kind="dqn-structure" />
<AlgorithmLab lesson="02" />

## 5. 一条经验怎样变成损失？

假设某条经验中：

- 执行动作的当前估计 `Q_online(s,a)=4.0`；
- 奖励 `r=1`；
- 下一状态目标网络最大值为 `5.0`；
- `γ=0.99`。

目标是 `1 + 0.99×5 = 5.95`，误差是 `5.95-4=1.95`。网络通过反向传播，让当前动作的输出向 5.95 靠近。

项目使用**平滑 L1 损失（Smooth L1 Loss）**，它在误差较小时像均方误差，误差很大时又不至于产生过大的梯度。

## 6. 对应的可运行代码

核心位于 `src/rl_learning_lab/cartpole_dqn.py`：

```python
with torch.no_grad():
    # 目标网络只提供标签，不参与本次反向传播。
    next_values = target_network(next_states).max(dim=1).values
    targets = rewards + discount_factor * next_values * (1.0 - dones)

# 在线网络只取当时真正执行动作对应的 Q 值。
current_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
loss = torch.nn.functional.smooth_l1_loss(current_values, targets)

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
optimizer.step()
```

运行本课：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_cartpole.py --steps 150000
```

## 7. 看真实训练曲线

<TrainingCurve
  file="cartpole.json"
  path="episode_returns"
  title="CartPole DQN 每回合回报"
  subtitle="每坚持一步得到 1 分；曲线越接近 500，说明平衡时间越长。"
  color="#0891b2"
  :target="475"
/>

单个回合突然掉分不一定代表网络退步，可能是 ε 探索动作刚好在临界时刻把杆子推倒。判断模型是否学会，应关闭探索做多回合独立评估。

## 8. DQN 使用前后有什么差别？

<div class="compare-grid">
  <article class="compare-card"><h3>Q 表</h3><p>每个状态单独存答案，精确但不能自然推广到未见连续状态。</p></article>
  <article class="compare-card"><h3>DQN</h3><p>用共享参数学习规律，能泛化，但会引入函数逼近误差和训练不稳定。</p></article>
</div>

DQN 不是“全面优于 Q 表”。在小型、离散、可枚举环境中，Q 表更透明、更稳定；状态很大或连续时，网络近似才有价值。

## 9. 常见误区

- **把 DQN 输出当概率。** Q 值可以大于 1、可以为负，不要求总和为 1。
- **回放池越大越好。** 太大的回放池可能保留大量过时或低价值经验，也会占更多内存。
- **目标网络永远不更新。** 它需要慢速同步，否则目标会长期落后。
- **终止和截断完全相同。** 真正终止通常不 bootstrap；时间上限截断是否 bootstrap 要根据任务语义判断。

## 10. 动手练习

1. 把目标网络同步间隔从 2000 改成 50，观察损失是否更抖。
2. 把回放池容量缩小到 1000，观察是否更容易遗忘旧情况。
3. 暂时取消 ε 探索，比较不同随机种子的稳定性。

<div class="checkpoint"><strong>学会标准：</strong>你能画出“环境 → 回放池 → 在线网络 → 目标网络”的数据流，并说明两个网络为什么不能每步完全同步。</div>

## 11. 快速自测

<ChapterQuiz lesson="02" />

## 12. 本课只需要记住这些

- DQN 用神经网络近似 Q 函数，适合大状态空间和离散动作。
- 经验回放负责打乱并复用数据。
- 目标网络负责提供相对稳定的学习目标。
- 神经网络带来泛化能力，也带来训练不稳定和估计误差。
