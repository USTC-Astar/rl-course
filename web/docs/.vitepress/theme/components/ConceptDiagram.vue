<script setup lang="ts">
// 基础概念图解：无状态的静态 SVG 结构图，按 kind 分发。
// 所有颜色走 custom.css 中的语义类，保证浅色/深色主题一致。
defineProps<{ kind: 'agent-env-loop' | 'learning-routes' | 'backup-diagram' | 'mc-vs-td' }>()
</script>

<template>
  <!-- 图 1：智能体与环境交互循环（第 0、1 课共用） -->
  <div v-if="kind === 'agent-env-loop'" class="concept-diagram">
    <p class="diagram-title">图解：强化学习的基本循环</p>
    <svg viewBox="0 0 640 330" role="img" aria-label="智能体与环境之间的状态、动作、奖励循环图">
      <defs>
        <marker id="cd-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="190" y="24" width="260" height="76" rx="14" stroke-width="2" />
      <text class="diagram-label" x="320" y="56" text-anchor="middle" font-size="17" font-weight="700">智能体 Agent</text>
      <text class="diagram-label-small" x="320" y="80" text-anchor="middle" font-size="13">按策略 π(a|s) 决定怎么走</text>

      <rect class="diagram-node" x="190" y="230" width="260" height="76" rx="14" stroke-width="2" />
      <text class="diagram-label" x="320" y="262" text-anchor="middle" font-size="17" font-weight="700">环境 Environment</text>
      <text class="diagram-label-small" x="320" y="286" text-anchor="middle" font-size="13">转移状态并发放奖励</text>

      <path class="diagram-edge-accent" d="M 452 96 C 540 96 540 234 452 234" fill="none" stroke-width="2.2" marker-end="url(#cd-arrow)" />
      <text class="diagram-label" x="548" y="158" text-anchor="middle" font-size="14" font-weight="700">动作 aₜ</text>
      <text class="diagram-label-small" x="548" y="178" text-anchor="middle" font-size="12">左 / 右 / 前 / 后</text>

      <path class="diagram-edge" d="M 188 234 C 100 234 100 96 188 96" fill="none" stroke-width="2.2" marker-end="url(#cd-arrow)" />
      <text class="diagram-label" x="96" y="146" text-anchor="middle" font-size="14" font-weight="700">新状态 sₜ₊₁</text>
      <text class="diagram-label" x="96" y="166" text-anchor="middle" font-size="14" font-weight="700">奖励 rₜ₊₁</text>
      <text class="diagram-label-small" x="96" y="186" text-anchor="middle" font-size="12">−0.1 / +10 / −10</text>

      <text class="diagram-label-small" x="320" y="175" text-anchor="middle" font-size="13">循环 t = 0, 1, 2, …</text>
    </svg>
    <p class="diagram-note">读图结论：学习只发生在智能体一侧；环境从不告诉智能体“正确动作”，只在每步回传新状态和奖励。第 1 课的 Q 表、第 3 课的策略网络，都是这个循环里“怎么决定动作”的不同实现。</p>
  </div>

  <!-- 图 2：MDP 已知/未知两条学习路线（第 0 课） -->
  <div v-else-if="kind === 'learning-routes'" class="concept-diagram">
    <p class="diagram-title">图解：从同一个 MDP 出发的三条求解路线</p>
    <svg viewBox="0 0 680 400" role="img" aria-label="动态规划、蒙特卡洛、时序差分三条学习路线的分支图">
      <defs>
        <marker id="cd-arrow2" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="140" y="16" width="400" height="58" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="40" text-anchor="middle" font-size="15" font-weight="700">一个马尔可夫决策过程 MDP</text>
      <text class="diagram-label-small" x="340" y="60" text-anchor="middle" font-size="12.5">状态 S、动作 A、转移 P、奖励 R、折扣 γ</text>

      <path class="diagram-edge" d="M 340 74 V 104" fill="none" stroke-width="2" marker-end="url(#cd-arrow2)" />
      <rect class="diagram-node" x="190" y="104" width="300" height="46" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="132" text-anchor="middle" font-size="14" font-weight="700">转移概率 P 和奖励 R 是否已知？</text>

      <path class="diagram-edge" d="M 246 150 C 200 180 180 200 180 224" fill="none" stroke-width="2" marker-end="url(#cd-arrow2)" />
      <path class="diagram-edge" d="M 434 150 C 480 180 500 200 500 224" fill="none" stroke-width="2" marker-end="url(#cd-arrow2)" />
      <text class="diagram-label-small" x="176" y="186" text-anchor="end" font-size="12.5">已知（模型在手）</text>
      <text class="diagram-label-small" x="504" y="186" text-anchor="start" font-size="12.5">未知（只能试错）</text>

      <rect class="diagram-node-good" x="60" y="224" width="240" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="180" y="248" text-anchor="middle" font-size="14.5" font-weight="700">动态规划 DP</text>
      <text class="diagram-label-small" x="180" y="268" text-anchor="middle" font-size="12">策略迭代 / 价值迭代，不解方程不采样</text>

      <rect class="diagram-node-accent" x="380" y="224" width="240" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="500" y="248" text-anchor="middle" font-size="14.5" font-weight="700">强化学习</text>
      <text class="diagram-label-small" x="500" y="268" text-anchor="middle" font-size="12">靠与环境交互采样，本课程主线</text>

      <path class="diagram-edge" d="M 420 284 C 380 310 360 320 340 336" fill="none" stroke-width="2" marker-end="url(#cd-arrow2)" />
      <path class="diagram-edge" d="M 580 284 C 620 310 640 320 660 336" fill="none" stroke-width="2" marker-end="url(#cd-arrow2)" />

      <rect class="diagram-node" x="200" y="336" width="230" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="315" y="357" text-anchor="middle" font-size="13.5" font-weight="700">蒙特卡洛 MC</text>
      <text class="diagram-label-small" x="315" y="376" text-anchor="middle" font-size="11.5">等整局结束，用真实回报（第 18 课）</text>

      <rect class="diagram-node" x="470" y="336" width="200" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="570" y="357" text-anchor="middle" font-size="13.5" font-weight="700">时序差分 TD</text>
      <text class="diagram-label-small" x="570" y="376" text-anchor="middle" font-size="11.5">走一步就学（第 1 课 Q 学习）</text>
    </svg>
    <p class="diagram-note">读图结论：动态规划是“模型已知时的教科书解”，强化学习是“模型未知时的现实解”。MC 与 TD 的分岔在第 8 课和第 18 课还会以“偏差—方差权衡”的形式再次出现。</p>
  </div>

  <!-- 图 3：贝尔曼一步备份树（第 0 课） -->
  <div v-else-if="kind === 'backup-diagram'" class="concept-diagram">
    <p class="diagram-title">图解：价值从“后继一步”回传——贝尔曼备份</p>
    <svg viewBox="0 0 680 330" role="img" aria-label="状态价值等于所有动作的即时奖励加折扣后继价值的平均的树状图">
      <defs>
        <marker id="cd-arrow3" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="270" y="16" width="140" height="56" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="40" text-anchor="middle" font-size="14.5" font-weight="700">状态 s 的价值</text>
      <text class="diagram-label-small" x="340" y="60" text-anchor="middle" font-size="12.5">V(s)：从这里出发的长期期望回报</text>

      <path class="diagram-edge" d="M 300 72 C 220 100 200 112 190 138" fill="none" stroke-width="1.8" marker-end="url(#cd-arrow3)" />
      <path class="diagram-edge" d="M 340 72 V 138" fill="none" stroke-width="1.8" marker-end="url(#cd-arrow3)" />
      <path class="diagram-edge" d="M 380 72 C 460 100 480 112 490 138" fill="none" stroke-width="1.8" marker-end="url(#cd-arrow3)" />
      <text class="diagram-label-small" x="150" y="112" text-anchor="middle" font-size="11.5">以 π(a|s) 选动作</text>
      <text class="diagram-label-small" x="530" y="112" text-anchor="middle" font-size="11.5">策略也是输入</text>

      <rect class="diagram-node" x="120" y="138" width="140" height="48" rx="12" stroke-width="2" />
      <text class="diagram-label" x="190" y="158" text-anchor="middle" font-size="13.5" font-weight="700">动作 a₁</text>
      <text class="diagram-label-small" x="190" y="176" text-anchor="middle" font-size="11.5">概率 π(a₁|s)</text>
      <rect class="diagram-node" x="270" y="138" width="140" height="48" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="158" text-anchor="middle" font-size="13.5" font-weight="700">动作 a₂</text>
      <text class="diagram-label-small" x="340" y="176" text-anchor="middle" font-size="11.5">概率 π(a₂|s)</text>
      <rect class="diagram-node" x="420" y="138" width="140" height="48" rx="12" stroke-width="2" />
      <text class="diagram-label" x="490" y="158" text-anchor="middle" font-size="13.5" font-weight="700">动作 a₃</text>
      <text class="diagram-label-small" x="490" y="176" text-anchor="middle" font-size="11.5">概率 π(a₃|s)</text>

      <path class="diagram-edge" d="M 190 186 C 190 210 190 220 190 240" fill="none" stroke-width="1.8" marker-end="url(#cd-arrow3)" />
      <path class="diagram-edge" d="M 340 186 V 240" fill="none" stroke-width="1.8" marker-end="url(#cd-arrow3)" />
      <path class="diagram-edge" d="M 490 186 C 490 210 490 220 490 240" fill="none" stroke-width="1.8" marker-end="url(#cd-arrow3)" />

      <rect class="diagram-node-muted" x="100" y="240" width="180" height="58" rx="12" stroke-width="2" />
      <text class="diagram-label" x="190" y="263" text-anchor="middle" font-size="13" font-weight="700">拿到奖励 r(s,a₁)</text>
      <text class="diagram-label-small" x="190" y="284" text-anchor="middle" font-size="12">再进入后继 s′，价值 γV(s′)</text>
      <rect class="diagram-node-muted" x="250" y="240" width="180" height="58" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="263" text-anchor="middle" font-size="13" font-weight="700">r(s,a₂) + γV(s′)</text>
      <text class="diagram-label-small" x="340" y="284" text-anchor="middle" font-size="12">虚线框：估计值还在学习中</text>
      <rect class="diagram-node-muted" x="400" y="240" width="180" height="58" rx="12" stroke-width="2" />
      <text class="diagram-label" x="490" y="263" text-anchor="middle" font-size="13" font-weight="700">r(s,a₃) + γV(s′)</text>
      <text class="diagram-label-small" x="490" y="284" text-anchor="middle" font-size="12">折扣 γ 缩远期贡献</text>

      <path class="diagram-edge-accent" d="M 96 269 C 40 269 30 44 268 44" fill="none" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#cd-arrow3)" />
      <text class="diagram-label" x="60" y="150" text-anchor="middle" font-size="12.5" font-weight="700" transform="rotate(-90 60 150)">回传：V(s) = Σₐ π(a|s)[r + γV(s′)]</text>
    </svg>
    <p class="diagram-note">读图结论：贝尔曼方程说“一个状态的价值 = 所有动作分支的（即时奖励 + 折扣后的后继价值）按策略加权平均”。第 1 课的 Q 学习更新，就是把这条方程右边采样出来、去修正左边的旧估计。</p>
  </div>

  <!-- 图 4：MC 与 TD 两条更新路线对比（第 18 课） -->
  <div v-else-if="kind === 'mc-vs-td'" class="concept-diagram">
    <p class="diagram-title">图解：同一条轨迹，两种更新时机</p>
    <svg viewBox="0 0 680 340" role="img" aria-label="蒙特卡洛等待整局回报与时序差分一步更新两种方式的对比图">
      <defs>
        <marker id="cd-arrow4" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="30" y="16" width="290" height="42" rx="12" stroke-width="2" />
      <text class="diagram-label" x="175" y="42" text-anchor="middle" font-size="14.5" font-weight="700">蒙特卡洛 MC：等整局结束</text>
      <rect class="diagram-node-accent" x="360" y="16" width="290" height="42" rx="12" stroke-width="2" />
      <text class="diagram-label" x="505" y="42" text-anchor="middle" font-size="14.5" font-weight="700">时序差分 TD：走一步就更新</text>

      <text class="diagram-label" x="175" y="104" text-anchor="middle" font-size="13.5" font-weight="700">s₀ → s₁ → s₂ → … → s_T（结局）</text>
      <circle class="diagram-node" cx="60" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="60" y="133" text-anchor="middle" font-size="11.5">s₀</text>
      <circle class="diagram-node" cx="130" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="130" y="133" text-anchor="middle" font-size="11.5">s₁</text>
      <circle class="diagram-node" cx="200" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="200" y="133" text-anchor="middle" font-size="11.5">s₂</text>
      <text class="diagram-label-small" x="248" y="133" text-anchor="middle" font-size="13">…</text>
      <circle class="diagram-node-good" cx="300" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="300" y="133" text-anchor="middle" font-size="11.5">s_T</text>
      <path class="diagram-edge-accent" d="M 300 110 C 260 60 140 60 62 108" fill="none" stroke-width="2" marker-end="url(#cd-arrow4)" />
      <text class="diagram-label-small" x="178" y="76" text-anchor="middle" font-size="12">必须等到结局，用真实 G₀ 修正 V(s₀)</text>

      <text class="diagram-label" x="505" y="104" text-anchor="middle" font-size="13.5" font-weight="700">s₀ → s₁ → s₂ → …（不必等结局）</text>
      <circle class="diagram-node" cx="390" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="390" y="133" text-anchor="middle" font-size="11.5">s₀</text>
      <circle class="diagram-node" cx="460" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="460" y="133" text-anchor="middle" font-size="11.5">s₁</text>
      <circle class="diagram-node" cx="530" cy="128" r="17" stroke-width="2" />
      <text class="diagram-label" x="530" y="133" text-anchor="middle" font-size="11.5">s₂</text>
      <text class="diagram-label-small" x="578" y="133" text-anchor="middle" font-size="13">…</text>
      <path class="diagram-edge-accent" d="M 460 110 C 440 88 420 88 398 108" fill="none" stroke-width="2" marker-end="url(#cd-arrow4)" />
      <text class="diagram-label-small" x="430" y="80" text-anchor="middle" font-size="12">用 r₀ + γV(s₁) 立刻修正 V(s₀)</text>

      <rect class="diagram-node" x="30" y="196" width="290" height="112" rx="12" stroke-width="2" />
      <text class="diagram-label" x="175" y="222" text-anchor="middle" font-size="13" font-weight="700">更新目标：真实完整回报 G₀</text>
      <text class="diagram-label-small" x="175" y="246" text-anchor="middle" font-size="12">✓ 不依赖任何估计，无偏</text>
      <text class="diagram-label-small" x="175" y="268" text-anchor="middle" font-size="12">✗ 整条轨迹的随机性都进目标，方差大</text>
      <text class="diagram-label-small" x="175" y="290" text-anchor="middle" font-size="12">✗ 必须能等到回合结束</text>

      <rect class="diagram-node" x="360" y="196" width="290" height="112" rx="12" stroke-width="2" />
      <text class="diagram-label" x="505" y="222" text-anchor="middle" font-size="13" font-weight="700">更新目标：r₀ + γV(s₁)</text>
      <text class="diagram-label-small" x="505" y="246" text-anchor="middle" font-size="12">✓ 每步都能学，在线、快</text>
      <text class="diagram-label-small" x="505" y="268" text-anchor="middle" font-size="12">✓ 目标波动小，方差低</text>
      <text class="diagram-label-small" x="505" y="290" text-anchor="middle" font-size="12">✗ V(s₁) 不准时偏差会传给 V(s₀)</text>
    </svg>
    <p class="diagram-note">读图结论：MC 与 TD 的本质区别是“更新目标里有多少真实、多少估计”。第 8 课的 n 步回报与 TD(λ) 就是在这两端之间连续取折中。</p>
  </div>
</template>
