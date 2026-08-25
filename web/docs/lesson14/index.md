---
title: 第 14 课：Dyna-Q 学习与规划
---

# 第 14 课：Dyna-Q 学习与规划

<div class="lesson-lead">
前面的无模型算法只从真实或仿真环境采到的经验中学习。Dyna-Q 会顺便记住“做了什么后世界怎样变化”，然后在脑内重复模拟，减少真实试错次数。
</div>

## 本课主线

> Dyna-Q 把直接强化学习、环境模型学习和规划放进同一个循环：一次真实经历既更新价值，也成为以后想象练习的素材。

<div class="lesson-grid">
  <article class="lesson-card"><h3>学习目标</h3><p>理解环境模型、规划更新、真实经验和模拟经验的协同。</p></article>
  <article class="lesson-card"><h3>前置知识</h3><p>第 1 课的表格 Q 学习。</p></article>
  <article class="lesson-card"><h3>解决的问题</h3><p>真实交互昂贵，而一条经验在普通 Q 学习中只直接更新一次。</p></article>
  <article class="lesson-card"><h3>适用场景</h3><p>转移规律可学习、真实试错昂贵、模型误差可以控制的环境。</p></article>
</div>

## 1. 什么叫“环境模型”？

这里的模型不是策略网络，而是对环境规律的预测：

`模型(state, action) → next_state, reward`

在确定性网格世界中，可以直接记忆：

`(位置 17, 向右) → (位置 18, 奖励 -0.1)`

在复杂连续环境中，模型可能是神经网络，预测下一状态分布和奖励。

有模型后，智能体不必每次真的走到位置 17 才练习“向右”；它可以从记忆中抽出这条转移，再做一次 Q 更新。

## 2. Dyna-Q 每个真实步骤做三件事

**Dyna-Q** 的循环：

1. **直接学习（Direct Learning）**：用真实转移更新 Q。
2. **模型学习（Model Learning）**：记录或拟合这条环境转移。
3. **规划（Planning）**：从模型中抽取旧转移，再做若干次模拟 Q 更新。

类比学骑自行车：真实骑行提供身体经验；回家后在脑中复盘“当时车头向左偏，如果提前回正会怎样”；下次真实骑行前，你已经多练了几遍决策。

## 3. 规划更新和经验回放一样吗？

相似但不完全相同：

- 经验回放存储真实 `(s,a,r,s′)`，抽出来原样学习。
- 模型规划可以根据学到的模型生成转移，甚至组合出没有原样存储的状态—动作后果。

表格 Dyna-Q 中模型常只是已见转移字典，因此看起来很像经验回放；在神经网络世界模型中，两者差别更明显。

## 4. 规划次数越多越好吗？

如果模型准确，更多规划能用更少真实步数传播奖励；如果模型错误，反复规划会把错误放大。

想象地图把一座桥误标成可通行：你在脑内规划 1000 次，可能越来越确信“过桥是最短路线”；真正出发才发现桥已断。想象得越多，错得越坚定。


<MechanismDiagram kind="dyna-loop" />
<AlgorithmLab lesson="14" />

## 5. 用数字看有效更新量

假设每个真实步骤做 20 次规划：

- 10 个真实步骤：10 次直接 Q 更新；
- 每步 20 次规划：额外 200 次模拟更新；
- 总共 210 次价值更新。

但“更新次数多”不等于信息多。200 次规划都来自已有模型，无法替代探索未知状态。

## 6. 对应的完整表格实现

核心位于 `src/rl_learning_lab/advanced_lessons.py`：

```python
def learn_from_real_step(self, state, action, next_state, reward, done):
    # 第一部分：真实经验立刻更新 Q。
    self._update_q(state, action, next_state, reward, done)

    # 第二部分：把真实见过的转移写入环境模型。
    self.model[state, action] = (next_state, reward, done)

    # 第三部分：从模型记忆中随机抽样，进行额外规划更新。
    model_keys = tuple(self.model)
    for _ in range(self.config.planning_steps):
        simulated_state, simulated_action = random_choice(model_keys)
        simulated_next_state, simulated_reward, simulated_done = self.model[
            simulated_state, simulated_action
        ]
        self._update_q(
            simulated_state,
            simulated_action,
            simulated_next_state,
            simulated_reward,
            simulated_done,
        )
```

运行可重复网格实验：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/generate_advanced_lessons.py
```

## 7. 比较无规划与 20 次规划

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson14.planning_0_steps"
  title="普通 Q 学习：每回合路线步数"
  subtitle="规划次数为 0；纵轴越低，表示越快到达目标。"
  color="#d97706"
  badge="真实网格交互"
/>

<TrainingCurve
  file="advanced_lessons.json"
  path="lesson14.planning_20_steps"
  title="Dyna-Q：每个真实步骤规划 20 次"
  subtitle="使用相同网格和随机种子，比较奖励传播与路线收敛速度。"
  color="#059669"
  badge="真实交互 + 模型规划"
/>

## 8. 环境突然变化时怎么办？

如果墙壁或奖励改变，旧模型会过时。经典扩展 **Dyna-Q+** 会给长时间未尝试的动作探索奖励，鼓励重新检查可能变化的路线。

工程上还可以：

- 给模型经验加时间戳；
- 检测预测误差突然升高；
- 降低旧模型样本权重；
- 保留真实交互比例，不能只在模型里闭门训练。

## 9. 常见误区

- **有模型就不需要真实环境。** 模型必须用真实数据校准，未知区域也需要探索。
- **规划等于搜索最短路。** Dyna-Q 这里是从模型转移中做价值更新，规划形式可以很多。
- **模型越复杂越好。** 小环境字典模型更准确透明，神经模型反而可能引入不必要误差。
- **模拟经验和真实经验同样可信。** 应持续监控模型预测误差。

## 10. 动手练习

1. 比较规划次数 `0、5、20、50` 的前 20 回合平均步数。
2. 训练一半后移动一堵墙，观察旧模型怎样误导规划。
3. 给模型条目增加最后访问时间，设计 Dyna-Q+ 探索奖励。

<div class="checkpoint"><strong>学会标准：</strong>你能画出“真实环境、Q 表、环境模型、规划更新”四者之间的数据流。</div>

## 11. 快速自测

<ChapterQuiz lesson="14" />

## 12. 本课只需要记住这些

- Dyna-Q 同时做直接学习、模型学习和规划。
- 一条真实经验可被模型反复用于模拟更新。
- 准确模型提高样本效率，错误模型会放大偏差。
- 规划不能代替探索未知世界。
