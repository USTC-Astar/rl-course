---
title: 第 10 课：Dueling DQN
---

# 第 10 课：Dueling DQN

<div class="lesson-lead">
在很多状态中，“这里整体很危险”比“左比右究竟高 0.1 分还是 0.2 分”更容易先学会。Dueling DQN 把状态本身的好坏和动作相对差异分开表示。
</div>

## 本课主线

> Dueling 是网络结构改造，不是新的 Bellman 目标：共享特征后分成价值支路 V 和优势支路 A，再合成为 Q。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解状态价值、动作优势、可辨识性和 Dueling 网络结构。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 4 课的 V、Q、A 区别和第 2 课的 DQN 网络。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>每个动作都单独学 Q，难以快速共享“这个状态整体好或坏”的信息。</p></article>
  <article class="lesson-card"><h3>适用场景</h3><p>许多动作价值接近、状态质量比精确动作差异更容易判断的离散控制任务。</p></article>
</div>

## 1. 普通 DQN 的输出头有什么局限？

普通 DQN 最后一层直接输出 `[Q左,Q右,...]`。如果某个状态是“车辆即将撞墙”，所有动作都很差，网络仍要从每个动作样本中分别学会这些 Q 值很低。

但人类常先判断：“这个局面本身危险。”然后再比较：“急刹车虽然也不好，但比继续加速稍好。”

**对决式深度 Q 网络（Dueling Deep Q-Network, Dueling DQN）**让网络显式学习这两层判断。

## 2. 两条支路分别输出什么？

共享特征层之后分成：

```text
状态 s → 共享特征
           ├─ 价值支路 → V(s)
           └─ 优势支路 → A(s,每个动作)
```

- `V(s)`：状态整体前景。
- `A(s,a)`：动作相对平均水平的额外好坏。

最后合成为 Q 值。

## 3. 为什么不能简单写 Q=V+A？

如果 `Q=V+A`，分解不唯一：

- `V=5, A=[2,-1]`；
- 把 V 加 10、所有 A 减 10；
- 合成的 Q 完全不变。

网络不知道哪部分应该由 V 解释，哪部分应该由 A 解释。这叫**可辨识性问题（Identifiability Problem）**。

经典处理：

<div class="formula-box">
  <span class="formula">Q(s,a)=V(s)+A(s,a)-meanₐ A(s,a)</span>
  先让优势平均值为 0，V 就等于所有动作 Q 值的平均水平，分工更清晰。
</div>

## 4. 用具体数字合成 Q 值

假设：

- `V(s)=5`；
- 左动作优势 `2`；
- 右动作优势 `-1`。

优势平均值是 `(2-1)/2=0.5`。

- `Q左=5+2-0.5=6.5`；
- `Q右=5-1-0.5=3.5`。

两个 Q 的平均值正好是 5，与状态价值一致。


<NetworkDiagram kind="dueling-structure" />
<AlgorithmLab lesson="10" />

## 5. 它为什么能提高样本利用率？

如果一个状态下很多动作差不多，普通 DQN 每看到一个动作样本，只直接修正这个动作输出。Dueling 结构中，共享的 V 支路会从任意动作经验中学习“这个状态整体如何”，其他动作也间接受益。

类比酒店评价：即使你只体验了早餐，也能更新“酒店整体环境不错”的判断；至于早餐比健身房好多少，再由优势部分表示。

## 6. Dueling 和 Double 解决的是同一问题吗？

不是：

- Double DQN 修改目标计算，解决 max 高估。
- Dueling DQN 修改网络表示，解决状态价值和动作差异学习效率。

两者可以同时使用，而且经典 Rainbow 就会组合它们。

## 7. 对应的可运行代码

合成函数位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
def combine_dueling_values(state_values, advantages):
    # 中心化让优势平均为 0，避免 V 与 A 任意挪动常数。
    centered_advantages = advantages - advantages.mean(dim=-1, keepdim=True)
    return state_values + centered_advantages
```

一个完整输出头可以写成：

```python
class DuelingHead(torch.nn.Module):
    def __init__(self, hidden_size: int, action_count: int) -> None:
        super().__init__()
        self.value = torch.nn.Linear(hidden_size, 1)
        self.advantage = torch.nn.Linear(hidden_size, action_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.value(features)
        advantages = self.advantage(features)
        return combine_dueling_values(values, advantages)
```

## 8. 什么情况下收益可能不明显？

- 动作数量很少，而且每个动作差异始终明显；
- 环境很简单，普通 DQN 已快速收敛；
- 共享特征层太弱，V 和 A 两个头都学不好；
- 数据、目标或探索问题才是真正瓶颈。

架构改进不是魔法。应做相同随机种子、相同训练步数的对照实验。

## 9. 常见误区

- **Dueling 指两个网络互相对抗。** 名字指价值与优势两条估计流，不是生成对抗网络。
- **优势就是 Q 值。** 优势是相对状态平均水平的差值。
- **V 支路为每个动作输出一个值。** V 通常只输出每个状态一个标量。
- **用了 Dueling 就不需要 Double。** 两者解决不同问题，可以组合。

## 10. 动手练习

1. 随机生成一组 V 和 A，验证合成 Q 的动作平均等于 V。
2. 把 `mean(A)` 改成 `max(A)`，比较两种可辨识约束。
3. 在现有 CartPole QNetwork 中把最后一层替换为双支路。

<div class="checkpoint"><strong>学会标准：</strong>你能解释为什么要减去优势平均值，并能由 V 和 A 手算所有动作 Q 值。</div>

## 11. 快速自测

<ChapterQuiz lesson="10" />

## 12. 本课只需要记住这些

- Dueling 把状态整体价值 V 和动作相对优势 A 分开学习。
- 合成时减去优势平均值，解决分解不唯一。
- 它是网络结构改造，不改变 DQN 的基本目标。
- 它与 Double DQN 可以互补组合。
