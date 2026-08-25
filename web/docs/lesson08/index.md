---
title: 第 8 课：多步回报与 TD(λ)
---

# 第 8 课：多步回报与 TD(λ)

<div class="lesson-lead">
第 4 课用一步 TD 误差快速评价动作，但遥远奖励只能一步一步往前传。本课学习怎样一次向前看多步，以及怎样把不同观察距离平滑混合。
</div>

## 本课主线

> 一步回报更依赖价值估计，完整回报更依赖真实轨迹；n 步回报和 TD(λ) 在偏差与方差之间选择折中点。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解 bootstrap、n 步回报、偏差—方差权衡、资格迹和 λ 回报。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 1 课的折扣因子和第 4 课的 TD 误差、价值函数。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>稀疏或延迟奖励传播太慢，一步目标过度依赖尚未学准的评论家。</p></article>
  <article class="lesson-card"><h3>适用场景</h3><p>行动者—评论家、PPO、A3C、价值学习和长时间信用分配。</p></article>
</div>

## 1. 一步 TD 为什么有时太短视？

一步目标是：

`Gₜ⁽¹⁾ = rₜ + γV(sₜ₊₁)`

它只使用一个真实奖励，后面全部相信价值网络。优点是很快就能更新，缺点是评论家早期不准时，错误会直接进入目标。

想象一条 20 步迷宫，只有终点奖励 10，前 19 步奖励都是 0。一步 TD 需要先让第 19 步知道终点好，再让第 18 步知道第 19 步好，如同接力传话，奖励传播很慢。

## 2. n 步回报怎样让奖励一次走得更远？

**n 步回报（N-step Return）**：

<div class="formula-box">
  <span class="formula">Gₜ⁽ⁿ⁾ = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿV(sₜ₊ₙ)</span>
  前 n 步使用真实奖励，到了第 n 步再接上价值估计，这个“接上估计”的动作叫 bootstrap。
</div>

假设未来三步奖励都是 1，`γ=0.9`，第 3 步后的价值估计是 5：

`G⁽³⁾ = 1 + 0.9×1 + 0.9²×1 + 0.9³×5 = 6.355`

与一步目标相比，它直接看到了三步真实反馈；与完整回报相比，它又不用等整局结束。

## 3. n 越大越好吗？

不一定。这里出现强化学习中常见的**偏差—方差权衡（Bias-Variance Trade-off）**：

| 观察距离 | 主要依赖 | 偏差 | 方差 | 特点 |
| --- | --- | --- | --- | --- |
| 1 步 | 价值估计 | 较高 | 较低 | 更新快、目标稳，但容易继承评论家错误 |
| 中等 n | 真实奖励 + 价值估计 | 中等 | 中等 | 常用折中 |
| 完整回合 | 真实轨迹 | 较低 | 较高 | 无需末尾估计，但随机波动大、等待久 |

类比天气预测：只看明天并大量依赖模型，结果稳定但可能有系统偏差；记录整个月真实天气再评价决策，事实更多，却受到大量偶然因素影响。


<MechanismDiagram kind="eligibility-trace" />
<AlgorithmLab lesson="08" />

## 4. TD(λ) 为什么不只选一个 n？

**时序差分 λ 方法（Temporal-Difference Lambda, TD(λ)）**不押注单一观察距离，而是把 1 步、2 步、3 步……回报按权重混合。

直觉上：

- `λ=0`：只相信 1 步回报；
- `λ` 接近 1：更多相信长回报；
- 中间值：短回报权重大，长回报仍提供补充。

一种递推写法是：

`Gₜ^λ = rₜ + γ[(1-λ)V(sₜ₊₁) + λGₜ₊₁^λ]`

它像把“明天模型预测”和“继续追踪更长真实结果”混在一起，`λ` 决定继续往远处看的比例。

## 5. 资格迹是什么？

**资格迹（Eligibility Trace）**是 TD(λ) 的另一种实现视角：最近访问过的状态—动作保留一条逐渐衰减的“责任痕迹”。新奖励出现时，不只更新最后一步，也沿着痕迹向前分配信用。

类比团队项目：成果发布后，不只奖励最后提交文件的人；最近几周参与过的成员都有贡献记录，但越久以前、关联越弱的贡献权重越小。

前向视角把多个 n 步回报加权；后向视角维护资格迹。在线性表格设置下，两者可以等价理解。

## 6. GAE 和 TD(λ) 有什么关系？

**广义优势估计（Generalized Advantage Estimation, GAE）**把不同距离的 TD 误差按 `γλ` 衰减相加：

`Aₜ^GAE = δₜ + γλδₜ₊₁ + (γλ)²δₜ₊₂ + ...`

它本质上沿用了 TD(λ) 的远近混合思想，只是目标变成估计优势，特别适合行动者—评论家和 PPO。


<div class="derivation-box">

**GAE：从"n 步混合"到"TD 误差加权和"的一步推导**

上一节给了 GAE 公式 $\hat A_t = \sum_k (\gamma\lambda)^k \delta_{t+k}$。它为什么等价于混合各种长度的优势估计？关键是一个"伸缩求和"：

1. 定义 TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
1. 把 n 步优势写成伸缩和：$A_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}$——中间的 $V$ 项两两相消，只剩真实奖励和首尾两个 $V$（代入两三项即可验证）。
1. λ-回报优势 = 各长度按 $(1-\lambda)\lambda^{n-1}$ 加权：$\hat A_t = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} A_t^{(n)}$。
1. 把第 2 步代入第 3 步，交换求和次序：系数合并后每个 $\delta_{t+k}$ 前恰好剩 $(\gamma\lambda)^k$——得到 $\hat A_t = \sum_k (\gamma\lambda)^k \delta_{t+k}$。

**数字感受**：$\gamma\lambda = 0.9$ 时，$\delta_{t+5}$ 的权重只剩 $0.9^5 \approx 0.59$，10 步外剩 $0.35$——既记得住远处，又不被远处噪声主导。这就是 PPO 训练脚本里 `advantages` 一行代码背后的全部来历。

</div>

## 7. 对应的可运行代码

核心位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
def calculate_n_step_return(rewards, next_value, discount_factor):
    result = float(next_value)
    # 反向递推自然形成 r_t + γ(r_t+1 + γ(...))。
    for reward in reversed(list(rewards)):
        result = float(reward) + discount_factor * result
    return result


def calculate_lambda_returns(rewards, values, discount_factor, trace_decay):
    returns = np.empty_like(rewards, dtype=np.float64)
    running_return = float(values[-1])
    for index in range(len(rewards) - 1, -1, -1):
        one_step = (1.0 - trace_decay) * values[index + 1]
        longer = trace_decay * running_return
        running_return = rewards[index] + discount_factor * (one_step + longer)
        returns[index] = running_return
    return returns
```

生成本课实验数据：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/generate_advanced_lessons.py
```

## 8. 看可重复计算实验

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson08.n_step_targets"
  title="观察步数 n 对回报目标的影响"
  subtitle="实验假设每步奖励为 1、末尾价值为 6；横轴依次对应 n=1～8。"
  color="#7c3aed"
  badge="可重复计算实验"
/>

这条曲线不是“训练分数”，而是同一奖励序列在不同 n 下计算出的目标。n 变化会改变真实奖励和末尾估计在目标中的占比。

## 9. 常见误区

- **n 越大一定越准确。** 长轨迹包含更多真实奖励，也包含更多环境随机性。
- **bootstrap 是伪造奖励。** 它不是奖励，而是对尚未观察到未来回报的价值估计。
- **终止状态仍接 V(s_T)。** 真正终止后没有未来，应把末尾 bootstrap 置零。
- **λ=1 一定等于纯蒙特卡洛。** 有限截断轨迹末尾如果仍接价值估计，就不完全等同于完整终局回报。

## 10. 动手练习

1. 把奖励序列改成 `[0,0,0,10]`，比较 n=1 和 n=4。
2. 固定 n=5，分别使用 `γ=0.5` 和 `γ=0.99`。
3. 画出 `λ=0、0.5、0.95、1` 的第一个回报目标。

<div class="checkpoint"><strong>学会标准：</strong>你能解释 n 和 λ 各自在控制什么，并说明长回报为什么“偏差小但方差大”。</div>

## 11. 快速自测

<ChapterQuiz lesson="08" />

## 12. 本课只需要记住这些

- n 步回报在真实奖励后接一个价值估计。
- n 控制看多远，λ 控制混合多少种观察距离。
- 短回报稳定但更依赖估计，长回报事实更多但波动更大。
- GAE 是把 TD(λ) 思想用于优势估计。
