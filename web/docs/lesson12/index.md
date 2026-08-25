---
title: 第 12 课：Rainbow DQN
---

# 第 12 课：Rainbow DQN

<div class="lesson-lead">
前几课像分别升级发动机、刹车、轮胎和导航。Rainbow DQN 的重点不是再发明一个单独技巧，而是把多个互补改进装进同一辆车，并验证它们能否协同。
</div>

## 本课主线

> Rainbow 是一套组合工程：每个组件解决不同薄弱点，真正难处在接口一致、目标兼容和消融验证，而不是把六个名字写在一起。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>认识经典六组件、分布式价值和噪声网络，理解组合与消融。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 8—11 课以及 DQN 的目标网络和回放池。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>DQN 的高估、表示、采样、奖励传播、回报不确定性和探索效率。</p></article>
  <article class="lesson-card"><h3>工程重点</h3><p>先单独验证模块，再逐步组合；不要一次写完后只看最终分数。</p></article>
</div>

## 1. 经典 Rainbow 包含哪六块？

| 组件 | 解决的问题 | 课程位置 |
| --- | --- | --- |
| Double DQN | max 高估 | 第 9 课 |
| Dueling DQN | 状态价值与动作差异表示效率 | 第 10 课 |
| 优先经验回放（PER） | 均匀抽样浪费训练机会 | 第 11 课 |
| 多步回报（N-step Return） | 延迟奖励传播慢 | 第 8 课 |
| C51 | 只学平均回报丢失分布信息 | 本课 |
| 噪声网络（Noisy Network, NoisyNet） | 手工 ε 探索不够自适应 | 本课 |


<NetworkDiagram kind="rainbow-map" />
<AlgorithmLab lesson="12" />

## 2. C51 为什么不只预测一个 Q 均值？

两个动作平均回报都为 5：

- 动作 A 每次都得到 5；
- 动作 B 一半得到 0，一半得到 10。

只看期望，两者相同；但回报风险和不确定性明显不同。

**分布式强化学习（Distributional Reinforcement Learning）**学习随机回报 `Z(s,a)` 的分布，Q 值只是它的期望。

**分类分布算法（Categorical Distributional Algorithm, C51）**在固定最小值和最大值之间放 51 个“原子”：

`z₁, z₂, ..., z₅₁`

网络为每个动作输出落在这些原子上的概率。最终 Q 值可通过 `Σ pᵢzᵢ` 计算。

## 3. 为什么叫“投影回支撑集”？

Bellman 更新后的目标原子 `r+γzᵢ` 通常不会刚好落在固定 51 个位置上，甚至可能超出最小/最大范围。

处理过程：

1. 把目标值裁到 `[V_min,V_max]`；
2. 找到左右相邻原子；
3. 按距离把概率质量分给两边；
4. 确保总概率仍为 1。

类比把连续身高数据放入固定直方图区间：一个值落在两个中心之间时，按距离把权重分摊，而不是凭空增加或丢失人数。

## 4. NoisyNet 与 ε-贪心有什么差别？

ε-贪心在动作层面随机：以固定概率忽略 Q 值，随机选动作。

**噪声网络（Noisy Network, NoisyNet）**在网络参数中加入可学习噪声：

`w = μ_w + σ_w ⊙ ε_w`

同一次参数噪声会在一段状态上产生一致的探索倾向。例如智能体可能连续尝试“更偏右的策略”，而不是每一步独立乱按按钮。

噪声尺度 `σ_w` 也可学习：某些状态需要探索时保持较大，确定后可以减小。

## 5. 多个组件怎样相互连接？

一次 Rainbow 更新大致是：

<ol class="step-list">
  <li>用 NoisyNet 策略选择动作并收集经验。</li>
  <li>把连续经验组织成 n 步转移，写入优先回放池。</li>
  <li>按 PER 抽样，并计算重要性权重。</li>
  <li>在线 Dueling 网络用 Double 规则选择下一动作。</li>
  <li>目标网络给出该动作的 C51 回报分布并投影。</li>
  <li>用分布交叉熵更新网络，再用新误差更新优先级。</li>
</ol>

注意这里不再只拟合一个标量 Q 目标，而是拟合整个类别概率分布。

## 6. 对应的 C51 投影代码

核心位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
target_atoms = rewards[:, None] + gamma * (1.0 - dones[:, None]) * support
target_atoms = target_atoms.clamp(v_min, v_max)
positions = (target_atoms - v_min) / atom_delta
lower = positions.floor().long()
upper = positions.ceil().long()

projected = torch.zeros_like(next_probabilities)
for batch_index in range(batch_size):
    for atom_index in range(atom_count):
        probability = next_probabilities[batch_index, atom_index]
        # 目标恰好落在原子上时只能加一次，否则会重复计算概率质量。
        if lower[batch_index, atom_index] == upper[batch_index, atom_index]:
            projected[batch_index, lower[batch_index, atom_index]] += probability
        else:
            # 按目标位置到左右原子的距离线性分摊概率。
            ...
```

项目测试会检查投影后概率质量仍等于 1。

## 7. 为什么必须做消融实验？

**消融实验（Ablation Study）**是逐个移除组件，再比较性能变化：

- 完整 Rainbow；
- 去掉 PER；
- 去掉 N-step；
- 去掉 NoisyNet；
- 只保留基础 DQN。

如果只比较“基础 DQN vs 六件套”，你不知道提升来自哪个组件，也不知道某个模块是否在当前环境反而有害。

::: warning 组件越多不等于任务越简单
更多模块意味着更多超参数、更多张量形状、更多边界条件和更难定位的错误。CartPole 这类简单环境往往不需要完整 Rainbow。
:::

## 8. 结构复杂度示意

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson12.relative_complexity"
  title="六个组件的相对接入复杂度示意"
  subtitle="依次为 Double、Dueling、PER、N-step、C51、NoisyNet；这是教学刻度，不是性能基准。"
  color="#7c3aed"
  badge="结构比较"
/>

C51 和 PER 往往需要更多数据结构与形状测试；Double 和 Dueling 的局部改动相对小。真实复杂度仍取决于现有代码架构。

## 9. 推荐的实现顺序

1. 先把基础 DQN 测试稳定。
2. 加 Double，只改目标计算。
3. 加 Dueling，只改网络输出头。
4. 加 n 步回报，先验证终止边界。
5. 加 PER，验证概率和权重。
6. 加 C51，重点测试投影概率守恒。
7. 最后用 NoisyNet 替换 ε 探索。

每一步都保留开关，方便对照和消融。

## 10. 常见误区

- **Rainbow 是单一新公式。** 它是多个改进的组合系统。
- **C51 表示有 51 个动作。** 51 是回报分布的原子数。
- **NoisyNet 只是给观测加噪声。** 它通常给网络参数加可学习噪声。
- **完整组合一定优于任意子集。** 不同环境、预算和实现下贡献会变化。

## 11. 动手练习

1. 用 5 个原子手算一次投影，检查概率和是否为 1。
2. 设计一个开关配置对象，能独立启用 Double、Dueling 和 PER。
3. 为组合实验写表格，至少记录随机种子、训练步数、平均回报和运行时间。

<div class="checkpoint"><strong>学会标准：</strong>你能分别说出六个组件解决的问题，而不是只会背 Rainbow 名字。</div>

## 12. 快速自测

<ChapterQuiz lesson="12" />

## 13. 本课只需要记住这些

- Rainbow 组合多个互补的 DQN 改进。
- C51 学习回报分布，Q 值只是分布期望。
- NoisyNet 用参数噪声形成更一致、可学习的探索。
- 组合前应单测每个组件，组合后必须做消融实验。
