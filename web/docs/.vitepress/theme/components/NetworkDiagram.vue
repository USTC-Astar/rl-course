<script setup lang="ts">
// 网络结构图解：展示各算法中“谁读什么、谁更新谁”的数据流。
// 蓝色实线=主数据流，灰色虚线=冻结/低频同步，红色=偏差来源，绿色=改进点。
defineProps<{
  kind:
    | 'dqn-structure'
    | 'actor-critic'
    | 'gaussian-tanh'
    | 'sac-structure'
    | 'dueling-structure'
    | 'rainbow-map'
    | 'td3-structure'
}>()
</script>

<template>
  <!-- DQN：回放池 + 双网络（第 2 课） -->
  <div v-if="kind === 'dqn-structure'" class="concept-diagram">
    <p class="diagram-title">图解：DQN 的四个部件与数据流</p>
    <svg viewBox="0 0 680 360" role="img" aria-label="经验回放池、在线网络、目标网络之间的数据流图">
      <defs>
        <marker id="nd-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="30" y="30" width="170" height="80" rx="12" stroke-width="2" />
      <text class="diagram-label" x="115" y="62" text-anchor="middle" font-size="14" font-weight="700">经验回放池</text>
      <text class="diagram-label-small" x="115" y="84" text-anchor="middle" font-size="12">存 (s, a, r, s′, done)</text>

      <rect class="diagram-node-accent" x="250" y="30" width="180" height="80" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="56" text-anchor="middle" font-size="14" font-weight="700">在线网络 Q(s,a;θ)</text>
      <text class="diagram-label-small" x="340" y="76" text-anchor="middle" font-size="12">每步都更新</text>
      <text class="diagram-label-small" x="340" y="94" text-anchor="middle" font-size="12">负责选动作和学</text>

      <rect class="diagram-node-muted" x="480" y="30" width="170" height="80" rx="12" stroke-width="2" />
      <text class="diagram-label" x="565" y="56" text-anchor="middle" font-size="14" font-weight="700">目标网络 Q_target</text>
      <text class="diagram-label-small" x="565" y="76" text-anchor="middle" font-size="12">参数 θ⁻ 冻结</text>
      <text class="diagram-label-small" x="565" y="94" text-anchor="middle" font-size="12">只算目标里的 max Q</text>

      <path class="diagram-edge-accent" d="M 200 70 H 248" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow)" />
      <text class="diagram-label-small" x="224" y="60" text-anchor="middle" font-size="11.5">随机抽样</text>
      <path class="diagram-edge" d="M 430 70 H 478" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow)" />
      <text class="diagram-label-small" x="454" y="60" text-anchor="middle" font-size="11.5">同步 θ⁻ ← θ</text>
      <text class="diagram-label-small" x="454" y="112" text-anchor="middle" font-size="11.5">每 N 步一次</text>

      <path class="diagram-edge" d="M 200 92 C 240 120 460 120 500 96" fill="none" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#nd-arrow)" />

      <rect class="diagram-node" x="150" y="220" width="380" height="120" rx="14" stroke-width="2" />
      <text class="diagram-label" x="340" y="248" text-anchor="middle" font-size="14.5" font-weight="700">每批样本的回归目标</text>
      <text class="diagram-label-small" x="340" y="274" text-anchor="middle" font-size="13.5">y = r + γ · maxₐ Q_target(s′, a; θ⁻)</text>
      <text class="diagram-label-small" x="340" y="298" text-anchor="middle" font-size="12">y 在几步内基本不变 → 网络不是在追一个移动靶</text>
      <text class="diagram-label-small" x="340" y="322" text-anchor="middle" font-size="12">损失 = (Q(s,a;θ) − y)²，只对 θ 求梯度</text>
      <path class="diagram-edge-accent" d="M 340 112 V 216" fill="none" stroke-width="2" marker-end="url(#nd-arrow)" />
      <path class="diagram-edge" d="M 565 112 C 565 170 500 190 440 218" fill="none" stroke-width="2" marker-end="url(#nd-arrow)" />
    </svg>
    <p class="diagram-note">读图结论：回放池负责打乱样本（打破时间相关性），目标网络负责稳住回归目标（打破“自己追自己”的循环）。第 9—12 课的改进都在修改这张图的局部。</p>
  </div>

  <!-- Actor-Critic 双角色（第 4 课） -->
  <div v-else-if="kind === 'actor-critic'" class="concept-diagram">
    <p class="diagram-title">图解：行动者与评论家怎样互相配合</p>
    <svg viewBox="0 0 680 320" role="img" aria-label="行动者输出动作、评论家估计价值、TD 误差同时驱动两者更新的结构图">
      <defs>
        <marker id="nd-arrow2" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="30" y="130" width="140" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="100" y="156" text-anchor="middle" font-size="14" font-weight="700">状态 s</text>
      <text class="diagram-label-small" x="100" y="176" text-anchor="middle" font-size="12">4 个观测值</text>

      <rect class="diagram-node-accent" x="250" y="30" width="200" height="76" rx="12" stroke-width="2" />
      <text class="diagram-label" x="350" y="56" text-anchor="middle" font-size="14.5" font-weight="700">行动者 Actor</text>
      <text class="diagram-label-small" x="350" y="78" text-anchor="middle" font-size="12">策略 π(a|s)：输出各动作概率</text>

      <rect class="diagram-node" x="250" y="214" width="200" height="76" rx="12" stroke-width="2" />
      <text class="diagram-label" x="350" y="240" text-anchor="middle" font-size="14.5" font-weight="700">评论家 Critic</text>
      <text class="diagram-label-small" x="350" y="262" text-anchor="middle" font-size="12">价值 V(s)：估计这个处境有多好</text>

      <path class="diagram-edge-accent" d="M 170 148 C 210 148 220 70 248 68" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow2)" />
      <path class="diagram-edge-accent" d="M 170 172 C 210 172 220 250 248 252" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow2)" />
      <path class="diagram-edge-accent" d="M 450 68 C 520 68 560 120 560 160" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow2)" />
      <text class="diagram-label" x="582" y="150" text-anchor="middle" font-size="13" font-weight="700">动作 a</text>
      <text class="diagram-label-small" x="582" y="170" text-anchor="middle" font-size="11.5">交给环境执行</text>

      <rect class="diagram-node-good" x="490" y="214" width="160" height="76" rx="12" stroke-width="2" />
      <text class="diagram-label" x="570" y="240" text-anchor="middle" font-size="14" font-weight="700">TD 误差 δ</text>
      <text class="diagram-label-small" x="570" y="262" text-anchor="middle" font-size="12">δ = r + γV(s′) − V(s)</text>
      <path class="diagram-edge" d="M 450 252 H 488" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow2)" />
      <path class="diagram-edge-accent" d="M 570 212 C 540 170 480 130 452 100" fill="none" stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#nd-arrow2)" />
      <text class="diagram-label-small" x="560" y="196" text-anchor="middle" font-size="11.5">δ&gt;0：刚才的动作值得加强</text>
      <text class="diagram-label-small" x="560" y="186" text-anchor="middle" font-size="11.5"></text>
    </svg>
    <p class="diagram-note">读图结论：评论家用 δ 给行动者的每个动作打“相对分”（比预期好还是差），同一个 δ 也用来修正评论家自己。这样行动者不必等整局结束，方差比 REINFORCE 小得多。</p>
  </div>

  <!-- 高斯策略 + tanh 压缩（第 6 课） -->
  <div v-else-if="kind === 'gaussian-tanh'" class="concept-diagram">
    <p class="diagram-title">图解：连续动作从网络到合法区间的流水线</p>
    <svg viewBox="0 0 680 240" role="img" aria-label="状态输入网络输出均值方差、正态采样、tanh 压缩到合法区间的流程图">
      <defs>
        <marker id="nd-arrow3" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="26" y="90" width="110" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="81" y="116" text-anchor="middle" font-size="13.5" font-weight="700">状态 s</text>
      <text class="diagram-label-small" x="81" y="136" text-anchor="middle" font-size="11.5">角度、角速度…</text>

      <rect class="diagram-node-accent" x="176" y="90" width="130" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="241" y="110" text-anchor="middle" font-size="13.5" font-weight="700">策略网络</text>
      <text class="diagram-label-small" x="241" y="130" text-anchor="middle" font-size="11.5">输出两个数</text>

      <rect class="diagram-node" x="346" y="42" width="120" height="46" rx="12" stroke-width="2" />
      <text class="diagram-label" x="406" y="70" text-anchor="middle" font-size="13" font-weight="700">均值 μ(s)</text>
      <rect class="diagram-node" x="346" y="152" width="120" height="46" rx="12" stroke-width="2" />
      <text class="diagram-label" x="406" y="174" text-anchor="middle" font-size="13" font-weight="700">标准差 σ(s)</text>
      <text class="diagram-label-small" x="406" y="192" text-anchor="middle" font-size="11">探索范围</text>

      <rect class="diagram-node-accent" x="506" y="90" width="140" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="576" y="110" text-anchor="middle" font-size="13" font-weight="700">u ~ N(μ, σ²)</text>
      <text class="diagram-label-small" x="576" y="130" text-anchor="middle" font-size="11.5">无界，可能是 ±37</text>

      <path class="diagram-edge-accent" d="M 136 120 H 174" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow3)" />
      <path class="diagram-edge" d="M 306 110 C 326 100 326 68 344 66" fill="none" stroke-width="2" marker-end="url(#nd-arrow3)" />
      <path class="diagram-edge" d="M 306 130 C 326 140 326 170 344 174" fill="none" stroke-width="2" marker-end="url(#nd-arrow3)" />
      <path class="diagram-edge-accent" d="M 466 66 C 490 76 490 106 504 116" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow3)" />
      <path class="diagram-edge-accent" d="M 466 174 C 490 164 490 134 504 124" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow3)" />

      <rect class="diagram-node-good" x="506" y="176" width="140" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="576" y="197" text-anchor="middle" font-size="13" font-weight="700">a = tanh(u)</text>
      <text class="diagram-label-small" x="576" y="216" text-anchor="middle" font-size="11.5">严格落在 [−1, 1]</text>
      <path class="diagram-edge-accent" d="M 576 150 V 174" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow3)" />
      <text class="diagram-label-small" x="240" y="30" text-anchor="middle" font-size="12">对数概率 log π(u|s) 一路跟着走，tanh 后要加雅可比修正</text>
    </svg>
    <p class="diagram-note">读图结论：均值是“主要意图”，标准差是“探索幅度”，tanh 负责把数学上无界的采样压进环境合法区间。概率账本必须同步修正，否则梯度方向是错的。</p>
  </div>

  <!-- SAC 最大熵结构（第 7 课） -->
  <div v-else-if="kind === 'sac-structure'" class="concept-diagram">
    <p class="diagram-title">图解：SAC 的四个部件</p>
    <svg viewBox="0 0 680 330" role="img" aria-label="SAC 策略网络、双 Q 网络、温度系数与熵奖励的结构图">
      <defs>
        <marker id="nd-arrow4" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="40" y="40" width="180" height="70" rx="12" stroke-width="2" />
      <text class="diagram-label" x="130" y="66" text-anchor="middle" font-size="14" font-weight="700">策略网络 π(a|s)</text>
      <text class="diagram-label-small" x="130" y="88" text-anchor="middle" font-size="12">随机策略：输出分布</text>

      <rect class="diagram-node" x="460" y="26" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="550" y="48" text-anchor="middle" font-size="13.5" font-weight="700">评论家 Q₁(s,a)</text>
      <rect class="diagram-node" x="460" y="90" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="550" y="112" text-anchor="middle" font-size="13.5" font-weight="700">评论家 Q₂(s,a)</text>
      <rect class="diagram-node-good" x="460" y="160" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="550" y="182" text-anchor="middle" font-size="13.5" font-weight="700">目标取 min(Q₁,Q₂)</text>
      <text class="diagram-label-small" x="550" y="200" text-anchor="middle" font-size="11.5">压住高估（第 9 课同款思想）</text>

      <path class="diagram-edge" d="M 220 60 C 320 40 380 44 458 50" fill="none" stroke-width="2" marker-end="url(#nd-arrow4)" />
      <path class="diagram-edge" d="M 220 92 C 320 110 380 108 458 112" fill="none" stroke-width="2" marker-end="url(#nd-arrow4)" />
      <text class="diagram-label-small" x="340" y="80" text-anchor="middle" font-size="11.5">采样动作 a 交给环境</text>

      <rect class="diagram-node-warn" x="40" y="170" width="180" height="70" rx="12" stroke-width="2" />
      <text class="diagram-label" x="130" y="194" text-anchor="middle" font-size="13.5" font-weight="700">熵 H(π(·|s))</text>
      <text class="diagram-label-small" x="130" y="216" text-anchor="middle" font-size="11.5">策略的随机程度</text>
      <rect class="diagram-node-warn" x="40" y="262" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="130" y="284" text-anchor="middle" font-size="13.5" font-weight="700">温度系数 α</text>
      <text class="diagram-label-small" x="130" y="302" text-anchor="middle" font-size="11.5">熵值多少“奖励分”</text>
      <path class="diagram-edge" d="M 130 240 V 260" fill="none" stroke-width="2" marker-end="url(#nd-arrow4)" />

      <path class="diagram-edge-accent" d="M 222 200 C 340 230 420 210 458 190" fill="none" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#nd-arrow4)" />
      <text class="diagram-label-small" x="336" y="248" text-anchor="middle" font-size="11.5">策略目标 = min Q + α·H</text>
      <text class="diagram-label-small" x="336" y="266" text-anchor="middle" font-size="11.5">回报和高随机性一起最大化</text>
    </svg>
    <p class="diagram-note">读图结论：SAC = 异策略（有回放池）+ 双评论家取小（防高估）+ 熵奖励（保持探索）。三个部件分别解决“数据复用”“估计偏差”“过早收敛”三个问题。</p>
  </div>

  <!-- Dueling 双支路（第 10 课） -->
  <div v-else-if="kind === 'dueling-structure'" class="concept-diagram">
    <p class="diagram-title">图解：Dueling 网络把价值拆成两条支路</p>
    <svg viewBox="0 0 680 300" role="img" aria-label="共享骨干分别连接状态价值支路和优势支路再合并的对比图">
      <defs>
        <marker id="nd-arrow5" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="30" y="120" width="130" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="95" y="146" text-anchor="middle" font-size="13.5" font-weight="700">输入状态 s</text>
      <text class="diagram-label-small" x="95" y="166" text-anchor="middle" font-size="11.5">4 个数</text>

      <rect class="diagram-node-accent" x="200" y="120" width="130" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="265" y="146" text-anchor="middle" font-size="13.5" font-weight="700">共享骨干</text>
      <text class="diagram-label-small" x="265" y="166" text-anchor="middle" font-size="11.5">提特征</text>

      <rect class="diagram-node-good" x="390" y="40" width="200" height="58" rx="12" stroke-width="2" />
      <text class="diagram-label" x="490" y="62" text-anchor="middle" font-size="13.5" font-weight="700">价值支路 V(s)</text>
      <text class="diagram-label-small" x="490" y="84" text-anchor="middle" font-size="11.5">输出 1 个数：这状态本身多好</text>

      <rect class="diagram-node-warn" x="390" y="200" width="200" height="58" rx="12" stroke-width="2" />
      <text class="diagram-label" x="490" y="222" text-anchor="middle" font-size="13.5" font-weight="700">优势支路 A(s,a)</text>
      <text class="diagram-label-small" x="490" y="244" text-anchor="middle" font-size="11.5">输出 |A| 个数：各动作比平均好多少</text>

      <path class="diagram-edge-accent" d="M 160 140 C 180 140 180 70 388 68" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow5)" />
      <path class="diagram-edge-accent" d="M 330 150 C 356 160 360 220 388 228" fill="none" stroke-width="2.2" marker-end="url(#nd-arrow5)" />
      <path class="diagram-edge" d="M 160 150 C 180 150 180 70 388 68" fill="none" stroke-width="0" />

      <rect class="diagram-node" x="200" y="255" width="130" height="40" rx="10" stroke-width="2" />
      <text class="diagram-label" x="265" y="280" text-anchor="middle" font-size="12.5" font-weight="700">合并：Q = V + A − mean(A)</text>
      <path class="diagram-edge" d="M 490 98 C 460 150 400 200 336 250" fill="none" stroke-width="2" marker-end="url(#nd-arrow5)" />
      <path class="diagram-edge" d="M 490 200 C 460 190 400 240 334 262" fill="none" stroke-width="2" marker-end="url(#nd-arrow5)" />
      <text class="diagram-label-small" x="490" y="130" text-anchor="middle" font-size="11.5">减 mean(A) 让分解唯一：</text>
      <text class="diagram-label-small" x="490" y="148" text-anchor="middle" font-size="11.5">V 加常数、A 减常数时 Q 不变，得锁住</text>
    </svg>
    <p class="diagram-note">读图结论：很多状态下“选哪个动作差别不大”（比如局势已定），Dueling 让 V 支路直接学状态好坏，不必等每个动作都被充分试过，样本效率更高。</p>
  </div>

  <!-- Rainbow 组件地图（第 12 课） -->
  <div v-else-if="kind === 'rainbow-map'" class="concept-diagram">
    <p class="diagram-title">图解：Rainbow = 基础 DQN + 六个针对性改进</p>
    <svg viewBox="0 0 680 420" role="img" aria-label="Rainbow DQN 六个组件各自解决什么问题的地图">
      <defs>
        <marker id="nd-arrow5b" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="210" y="170" width="260" height="80" rx="14" stroke-width="2.5" />
      <text class="diagram-label" x="340" y="202" text-anchor="middle" font-size="15" font-weight="700">基础 DQN（第 2 课）</text>
      <text class="diagram-label-small" x="340" y="226" text-anchor="middle" font-size="12">回放池 + 目标网络</text>

      <rect class="diagram-node" x="30" y="20" width="190" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="125" y="44" text-anchor="middle" font-size="13" font-weight="700">Double（第 9 课）</text>
      <text class="diagram-label-small" x="125" y="66" text-anchor="middle" font-size="11.5">拆开“选动作”和“打分”，治高估</text>

      <rect class="diagram-node" x="245" y="20" width="190" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="44" text-anchor="middle" font-size="13" font-weight="700">Dueling（第 10 课）</text>
      <text class="diagram-label-small" x="340" y="66" text-anchor="middle" font-size="11.5">V 与 A 分开学，状态好学时更快</text>

      <rect class="diagram-node" x="460" y="20" width="190" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="555" y="44" text-anchor="middle" font-size="13" font-weight="700">PER（第 11 课）</text>
      <text class="diagram-label-small" x="555" y="66" text-anchor="middle" font-size="11.5">按惊讶程度抽经验复习</text>

      <rect class="diagram-node" x="30" y="330" width="190" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="125" y="354" text-anchor="middle" font-size="13" font-weight="700">多步回报（第 8 课）</text>
      <text class="diagram-label-small" x="125" y="376" text-anchor="middle" font-size="11.5">目标多看几步真实奖励</text>

      <rect class="diagram-node" x="245" y="330" width="190" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="354" text-anchor="middle" font-size="13" font-weight="700">分布式 C51</text>
      <text class="diagram-label-small" x="340" y="376" text-anchor="middle" font-size="11.5">学回报的分布，不只学均值</text>

      <rect class="diagram-node" x="460" y="330" width="190" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="555" y="354" text-anchor="middle" font-size="13" font-weight="700">噪声网络 NoisyNet</text>
      <text class="diagram-label-small" x="555" y="376" text-anchor="middle" font-size="11.5">把 ε 探索改进为参数噪声探索</text>

      <path class="diagram-edge" d="M 125 84 C 170 130 230 150 270 172" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow5b)" />
      <path class="diagram-edge" d="M 340 84 V 168" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow5b)" />
      <path class="diagram-edge" d="M 555 84 C 510 130 450 150 410 172" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow5b)" />
      <path class="diagram-edge" d="M 125 330 C 170 290 230 270 270 250" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow5b)" />
      <path class="diagram-edge" d="M 340 330 V 252" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow5b)" />
      <path class="diagram-edge" d="M 555 330 C 510 290 450 270 410 250" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow5b)" />
    </svg>
    <p class="diagram-note">读图结论：Rainbow 不是新发明，而是把六个“各治一种病”的组件组合起来。消融实验证明不是每个组件在所有任务上都有贡献——组合不等于免费提升。</p>
  </div>

  <!-- TD3 三个机制（第 13 课） -->
  <div v-else-if="kind === 'td3-structure'" class="concept-diagram">
    <p class="diagram-title">图解：TD3 对 DDPG 的三处手术</p>
    <svg viewBox="0 0 680 330" role="img" aria-label="TD3 双评论家取小值、延迟更新策略、目标策略平滑三个机制的示意图">
      <defs>
        <marker id="nd-arrow6" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="30" y="136" width="170" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="115" y="162" text-anchor="middle" font-size="14" font-weight="700">确定性行动者</text>
      <text class="diagram-label-small" x="115" y="184" text-anchor="middle" font-size="11.5">μ(s)：直接输出动作数值</text>

      <rect class="diagram-node" x="470" y="100" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="560" y="122" text-anchor="middle" font-size="13.5" font-weight="700">评论家 Q₁(s,a)</text>
      <rect class="diagram-node" x="470" y="168" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="560" y="190" text-anchor="middle" font-size="13.5" font-weight="700">评论家 Q₂(s,a)</text>
      <path class="diagram-edge" d="M 200 152 C 320 130 380 122 468 124" fill="none" stroke-width="2" marker-end="url(#nd-arrow6)" />
      <path class="diagram-edge" d="M 200 184 C 320 200 380 196 468 192" fill="none" stroke-width="2" marker-end="url(#nd-arrow6)" />
      <rect class="diagram-node-good" x="470" y="236" width="180" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="560" y="258" text-anchor="middle" font-size="13" font-weight="700">目标 = min(Q₁,Q₂)</text>
      <text class="diagram-label-small" x="560" y="277" text-anchor="middle" font-size="11.5">手术一：防高估（同 SAC）</text>
      <path class="diagram-edge" d="M 560 220 V 234" fill="none" stroke-width="2" marker-end="url(#nd-arrow6)" />

      <rect class="diagram-node-warn" x="120" y="30" width="220" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="230" y="52" text-anchor="middle" font-size="13" font-weight="700">手术二：延迟更新行动者</text>
      <text class="diagram-label-small" x="230" y="71" text-anchor="middle" font-size="11.5">评论家每 2 次更新，行动者才 1 次</text>

      <rect class="diagram-node-warn" x="120" y="250" width="260" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="250" y="272" text-anchor="middle" font-size="13" font-weight="700">手术三：目标策略平滑</text>
      <text class="diagram-label-small" x="250" y="292" text-anchor="middle" font-size="11.5">目标动作 = μ(s′) + 截断噪声，防尖峰</text>
      <path class="diagram-edge" d="M 230 82 C 210 100 180 110 130 134" fill="none" stroke-width="1.8" marker-end="url(#nd-arrow6)" />
      <path class="diagram-edge" d="M 250 250 C 330 240 400 260 468 250" fill="none" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#nd-arrow6)" />
    </svg>
    <p class="diagram-note">读图结论：确定性策略会主动寻找 Q 网络的错误并加以利用；TD3 的三处手术都在阻止这件事——估计保守一点（取 min）、行动者慢一点（延迟）、目标钝一点（加噪声）。</p>
  </div>
</template>
