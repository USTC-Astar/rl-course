<script setup lang="ts">
// 机制与流程图解：展示“循环、流水线、分布、信用分配”等动态机制。
defineProps<{
  kind:
    | 'eligibility-trace'
    | 'per-cycle'
    | 'dyna-loop'
    | 'ctde'
    | 'offline-shift'
    | 'rlhf-pipeline'
    | 'exploration-map'
    | 'sim2real-bridge'
}>()
</script>

<template>
  <!-- 资格迹衰减（第 8 课） -->
  <div v-if="kind === 'eligibility-trace'" class="concept-diagram">
    <p class="diagram-title">图解：资格迹——最近走过的步骤留下衰减的责任分</p>
    <svg viewBox="0 0 680 260" role="img" aria-label="资格迹随时间步指数衰减的条形示意图">
      <text class="diagram-label-small" x="30" y="30" font-size="12.5">设 λ=0.8、γ=1，每走一步，之前所有痕迹乘 0.8，当前步痕迹置 1：</text>
      <g v-for="(bar, index) in [
        { label: 't−5', value: 0.33, note: '0.8⁵' },
        { label: 't−4', value: 0.41, note: '0.8⁴' },
        { label: 't−3', value: 0.51, note: '0.8³' },
        { label: 't−2', value: 0.64, note: '0.8²' },
        { label: 't−1', value: 0.8, note: '0.8¹' },
        { label: 't（当前）', value: 1.0, note: '1' },
      ]" :key="index">
        <rect class="diagram-node-accent" :x="60 + index * 100" :y="200 - bar.value * 140" width="64" :height="bar.value * 140" rx="6" stroke-width="1.5" />
        <text class="diagram-label" :x="92 + index * 100" y="222" text-anchor="middle" font-size="12" font-weight="700">{{ bar.label }}</text>
        <text class="diagram-label-small" :x="92 + index * 100" :y="192 - bar.value * 140" text-anchor="middle" font-size="11.5">{{ bar.note }}</text>
      </g>
      <path class="diagram-edge-accent" d="M 70 190 C 200 60 480 60 630 56" fill="none" stroke-width="2" stroke-dasharray="6 5" />
      <text class="diagram-label-small" x="350" y="46" text-anchor="middle" font-size="12">奖励到来时，按当前痕迹比例分配给每一步</text>
    </svg>
    <p class="diagram-note">读图结论：一步 TD 只更新当前步；资格迹让“最近走过的一串步骤”都分到信用，越近分得越多。这等价于把 1 步、2 步……n 步回报按 (1−λ)λⁿ⁻¹ 加权混合。</p>
  </div>

  <!-- 优先经验回放循环（第 11 课） -->
  <div v-else-if="kind === 'per-cycle'" class="concept-diagram">
    <p class="diagram-title">图解：优先经验回放的工作循环</p>
    <svg viewBox="0 0 680 300" role="img" aria-label="经验按 TD 误差大小获得抽样优先级、训练后更新优先级的循环图">
      <defs>
        <marker id="md-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="40" y="110" width="170" height="70" rx="12" stroke-width="2" />
      <text class="diagram-label" x="125" y="138" text-anchor="middle" font-size="13.5" font-weight="700">经验进回放池</text>
      <text class="diagram-label-small" x="125" y="160" text-anchor="middle" font-size="11.5">初始优先级 = 当前最大</text>

      <rect class="diagram-node-accent" x="270" y="110" width="180" height="70" rx="12" stroke-width="2" />
      <text class="diagram-label" x="360" y="132" text-anchor="middle" font-size="13.5" font-weight="700">按优先级抽样</text>
      <text class="diagram-label-small" x="360" y="152" text-anchor="middle" font-size="11.5">p ∝ (|TD 误差|+ε)^α</text>
      <text class="diagram-label-small" x="360" y="170" text-anchor="middle" font-size="11.5">α=0 均匀，α=1 完全按大小</text>

      <rect class="diagram-node" x="510" y="110" width="150" height="70" rx="12" stroke-width="2" />
      <text class="diagram-label" x="585" y="132" text-anchor="middle" font-size="13.5" font-weight="700">训练这批经验</text>
      <text class="diagram-label-small" x="585" y="152" text-anchor="middle" font-size="11.5">损失乘重要性</text>
      <text class="diagram-label-small" x="585" y="170" text-anchor="middle" font-size="11.5">权重 w=(1/Np)^β</text>

      <path class="diagram-edge-accent" d="M 210 145 H 268" fill="none" stroke-width="2.2" marker-end="url(#md-arrow)" />
      <path class="diagram-edge-accent" d="M 450 145 H 508" fill="none" stroke-width="2.2" marker-end="url(#md-arrow)" />
      <path class="diagram-edge-accent" d="M 585 182 C 585 250 340 250 120 214" fill="none" stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#md-arrow)" />
      <text class="diagram-label-small" x="350" y="270" text-anchor="middle" font-size="11.5">用新的 TD 误差更新优先级：学会了的降到低优先级</text>

      <rect class="diagram-node-warn" x="40" y="20" width="260" height="56" rx="12" stroke-width="2" />
      <text class="diagram-label" x="170" y="42" text-anchor="middle" font-size="12.5" font-weight="700">两个旋钮要一起记</text>
      <text class="diagram-label-small" x="170" y="62" text-anchor="middle" font-size="11.5">α 管偏多少，β 管纠回多少偏差</text>
    </svg>
    <p class="diagram-note">读图结论：PER 把“均匀轮询所有经验”改成“错得最离谱的经验优先复习”。非均匀抽样引入偏差，所以要用重要性权重在损失里纠正回来。</p>
  </div>

  <!-- Dyna 学习与规划循环（第 14 课） -->
  <div v-else-if="kind === 'dyna-loop'" class="concept-diagram">
    <p class="diagram-title">图解：Dyna-Q——真实经验与规划经验共用一个 Q 表</p>
    <svg viewBox="0 0 680 340" role="img" aria-label="真实经验直接学习并写入模型、模型生成模拟经验反复规划的循环图">
      <defs>
        <marker id="md-arrow2" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="40" y="40" width="160" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="120" y="64" text-anchor="middle" font-size="13.5" font-weight="700">真实环境</text>
      <text class="diagram-label-small" x="120" y="86" text-anchor="middle" font-size="11.5">每步 (s,a,r,s′)</text>

      <rect class="diagram-node" x="300" y="40" width="150" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="375" y="64" text-anchor="middle" font-size="13.5" font-weight="700">Q 学习更新</text>
      <text class="diagram-label-small" x="375" y="86" text-anchor="middle" font-size="11.5">直接强化</text>

      <rect class="diagram-node-warn" x="510" y="40" width="150" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="585" y="64" text-anchor="middle" font-size="13.5" font-weight="700">环境模型</text>
      <text class="diagram-label-small" x="585" y="86" text-anchor="middle" font-size="11.5">记录 (s,a)→(r,s′)</text>

      <rect class="diagram-node-muted" x="410" y="220" width="230" height="64" rx="12" stroke-width="2" />
      <text class="diagram-label" x="525" y="244" text-anchor="middle" font-size="13.5" font-weight="700">模拟经验（规划）</text>
      <text class="diagram-label-small" x="525" y="266" text-anchor="middle" font-size="11.5">每步真实交互后，模型里再演练 k 次</text>

      <path class="diagram-edge-accent" d="M 200 66 C 240 66 240 68 298 70" fill="none" stroke-width="2.2" marker-end="url(#md-arrow2)" />
      <path class="diagram-edge" d="M 200 90 C 300 130 440 90 508 70" fill="none" stroke-width="2" marker-end="url(#md-arrow2)" />
      <text class="diagram-label-small" x="350" y="122" text-anchor="middle" font-size="11.5">顺手写入模型</text>

      <path class="diagram-edge-accent" d="M 585 104 C 585 160 560 180 530 218" fill="none" stroke-width="2.2" marker-end="url(#md-arrow2)" />
      <path class="diagram-edge-accent" d="M 410 252 C 300 252 260 160 380 106" fill="none" stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#md-arrow2)" />
      <text class="diagram-label-small" x="250" y="216" text-anchor="middle" font-size="11.5">模拟经验同样做 Q 更新</text>

      <text class="diagram-label-small" x="120" y="290" text-anchor="middle" font-size="12">真实 1 步 + 模型 k 步</text>
      <text class="diagram-label-small" x="120" y="310" text-anchor="middle" font-size="12">= 同样的更新规则，多倍的经验量</text>
    </svg>
    <p class="diagram-note">读图结论：Dyna-Q 的“规划”不是搜索树，而是拿学到的模型凭空生成额外经验，喂给同一个 Q 更新。模型不准时，规划会把错误也放大——第 14 课的实验会展示这一点。</p>
  </div>

  <!-- 集中训练分散执行（第 15 课） -->
  <div v-else-if="kind === 'ctde'" class="concept-diagram">
    <p class="diagram-title">图解：集中训练、分散执行（CTDE）</p>
    <svg viewBox="0 0 680 340" role="img" aria-label="训练时评论家可看全局信息，执行时每个智能体只用本地观测的对比图">
      <defs>
        <marker id="md-arrow3" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node-accent" x="30" y="24" width="280" height="46" rx="12" stroke-width="2" />
      <text class="diagram-label" x="170" y="52" text-anchor="middle" font-size="14" font-weight="700">训练阶段：信息可以共享</text>
      <rect class="diagram-node-accent" x="380" y="24" width="280" height="46" rx="12" stroke-width="2" />
      <text class="diagram-label" x="520" y="52" text-anchor="middle" font-size="14" font-weight="700">执行阶段：只靠本地观测</text>

      <rect class="diagram-node" x="60" y="110" width="100" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="110" y="132" text-anchor="middle" font-size="12.5" font-weight="700">智能体 1</text>
      <text class="diagram-label-small" x="110" y="150" text-anchor="middle" font-size="11">本地观测 o₁</text>
      <rect class="diagram-node" x="60" y="210" width="100" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="110" y="232" text-anchor="middle" font-size="12.5" font-weight="700">智能体 2</text>
      <text class="diagram-label-small" x="110" y="250" text-anchor="middle" font-size="11">本地观测 o₂</text>

      <rect class="diagram-node-good" x="240" y="160" width="190" height="60" rx="12" stroke-width="2" />
      <text class="diagram-label" x="335" y="184" text-anchor="middle" font-size="12.5" font-weight="700">集中式评论家</text>
      <text class="diagram-label-small" x="335" y="204" text-anchor="middle" font-size="11">看 (o₁, o₂, a₁, a₂) 全局信息</text>
      <path class="diagram-edge-accent" d="M 160 136 C 200 150 200 168 238 176" fill="none" stroke-width="2" marker-end="url(#md-arrow3)" />
      <path class="diagram-edge-accent" d="M 160 236 C 200 222 200 204 238 196" fill="none" stroke-width="2" marker-end="url(#md-arrow3)" />

      <rect class="diagram-node" x="490" y="110" width="150" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="565" y="132" text-anchor="middle" font-size="12.5" font-weight="700">策略 π₁(a₁|o₁)</text>
      <text class="diagram-label-small" x="565" y="150" text-anchor="middle" font-size="11">部署后独立运行</text>
      <rect class="diagram-node" x="490" y="210" width="150" height="52" rx="12" stroke-width="2" />
      <text class="diagram-label" x="565" y="232" text-anchor="middle" font-size="12.5" font-weight="700">策略 π₂(a₂|o₂)</text>
      <text class="diagram-label-small" x="565" y="250" text-anchor="middle" font-size="11">不需要通信</text>
      <path class="diagram-edge" d="M 430 176 C 460 160 460 138 488 134" fill="none" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#md-arrow3)" />
      <path class="diagram-edge" d="M 430 204 C 460 220 460 236 488 238" fill="none" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#md-arrow3)" />
      <text class="diagram-label-small" x="335" y="286" text-anchor="middle" font-size="11.5">评论家只在训练时存在；部署时丢掉评论家，只带走各自的行动者</text>
    </svg>
    <p class="diagram-note">读图结论：训练时用全局信息把“谁的锅”算清楚（缓解信用分配与非平稳），执行时每个智能体仍只用本地观测，满足真实的通信与传感器限制。</p>
  </div>

  <!-- 离线 RL 的分布偏移（第 16 课） -->
  <div v-else-if="kind === 'offline-shift'" class="concept-diagram">
    <p class="diagram-title">图解：为什么离线 RL 最怕“数据外动作”</p>
    <svg viewBox="0 0 680 300" role="img" aria-label="数据集只覆盖部分状态动作空间，策略外推到未覆盖区域产生虚假高值的示意图">
      <defs>
        <marker id="md-arrow3b" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="40" y="40" width="280" height="200" rx="14" stroke-width="2" />
      <text class="diagram-label" x="180" y="66" text-anchor="middle" font-size="13.5" font-weight="700">全部可能的 (状态, 动作) 组合</text>
      <ellipse cx="180" cy="150" rx="95" ry="70" class="diagram-node-good" stroke-width="2" />
      <text class="diagram-label" x="180" y="146" text-anchor="middle" font-size="12.5" font-weight="700">数据集覆盖区域</text>
      <text class="diagram-label-small" x="180" y="166" text-anchor="middle" font-size="11">人类演示实际走过的部分</text>
      <text class="diagram-label-small" x="280" y="106" text-anchor="middle" font-size="11">从未见过</text>
      <text class="diagram-label-small" x="280" y="124" text-anchor="middle" font-size="11">的区域</text>

      <path class="diagram-edge-accent" d="M 270 190 C 380 210 440 180 470 130" fill="none" stroke-width="2.2" marker-end="url(#md-arrow3b)" />
      <text class="diagram-label-small" x="400" y="216" text-anchor="middle" font-size="11.5">最大化 Q 的优化方向</text>

      <rect class="diagram-node-bad" x="480" y="90" width="180" height="76" rx="12" stroke-width="2" />
      <text class="diagram-label" x="570" y="116" text-anchor="middle" font-size="12.5" font-weight="700">策略跑向数据外动作</text>
      <text class="diagram-label-small" x="570" y="136" text-anchor="middle" font-size="11">Q 网络在那里自由外推</text>
      <text class="diagram-label-small" x="570" y="154" text-anchor="middle" font-size="11">可能凭空出现很高的假 Q 值</text>

      <text class="diagram-label-small" x="340" y="272" text-anchor="middle" font-size="11.5">保守 Q 学习（CQL）的对策：把数据外动作的 Q 压低，宁可错过、不可轻信</text>
    </svg>
    <p class="diagram-note">读图结论：在线算法犯错能靠新数据纠正自己；离线训练时数据集固定，错误外推永远等不到被纠正的机会，所以必须主动压低数据外动作的估值。</p>
  </div>

  <!-- RLHF 三阶段流水线（第 20 课） -->
  <div v-else-if="kind === 'rlhf-pipeline'" class="concept-diagram">
    <p class="diagram-title">图解：从预训练模型到对齐模型的三段流水线</p>
    <svg viewBox="0 0 680 320" role="img" aria-label="监督微调、奖励模型训练、强化学习优化三个阶段的流水线图">
      <defs>
        <marker id="md-arrow4" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="20" y="90" width="180" height="120" rx="14" stroke-width="2" />
      <text class="diagram-label" x="110" y="114" text-anchor="middle" font-size="13.5" font-weight="700">阶段一：监督微调 SFT</text>
      <text class="diagram-label-small" x="110" y="138" text-anchor="middle" font-size="11.5">人工示范“问题→好回答”</text>
      <text class="diagram-label-small" x="110" y="158" text-anchor="middle" font-size="11.5">普通监督学习，学会格式</text>
      <text class="diagram-label-small" x="110" y="186" text-anchor="middle" font-size="11.5">≈ 第 20 课的行为克隆</text>

      <rect class="diagram-node" x="250" y="90" width="180" height="120" rx="14" stroke-width="2" />
      <text class="diagram-label" x="340" y="114" text-anchor="middle" font-size="13.5" font-weight="700">阶段二：奖励模型 RM</text>
      <text class="diagram-label-small" x="340" y="138" text-anchor="middle" font-size="11.5">人工对回答排序 A&gt;B&gt;C</text>
      <text class="diagram-label-small" x="340" y="158" text-anchor="middle" font-size="11.5">学一个打分器 r(x, y)</text>
      <text class="diagram-label-small" x="340" y="186" text-anchor="middle" font-size="11.5">代替人类持续在线打分</text>

      <rect class="diagram-node" x="480" y="90" width="180" height="120" rx="14" stroke-width="2" />
      <text class="diagram-label" x="570" y="114" text-anchor="middle" font-size="13.5" font-weight="700">阶段三：RL 优化 PPO</text>
      <text class="diagram-label-small" x="570" y="138" text-anchor="middle" font-size="11.5">生成回答，RM 打分当奖励</text>
      <text class="diagram-label-small" x="570" y="158" text-anchor="middle" font-size="11.5">加 KL 惩罚防止跑偏</text>
      <text class="diagram-label-small" x="570" y="186" text-anchor="middle" font-size="11.5">策略梯度是第 3—5 课内容</text>

      <path class="diagram-edge-accent" d="M 200 150 H 248" fill="none" stroke-width="2.4" marker-end="url(#md-arrow4)" />
      <path class="diagram-edge-accent" d="M 430 150 H 478" fill="none" stroke-width="2.4" marker-end="url(#md-arrow4)" />
      <path class="diagram-edge" d="M 570 210 C 570 262 340 262 340 212" fill="none" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#md-arrow4)" />
      <text class="diagram-label-small" x="455" y="284" text-anchor="middle" font-size="11.5">KL 约束：别为讨好打分器而忘了怎么好好说话</text>
    </svg>
    <p class="diagram-note">读图结论：RLHF 用的不是新 RL 算法，而是“把人的偏好变成奖励函数”的工程。你在这门课学的策略梯度、PPO、KL 散度，在阶段三原样出现。</p>
  </div>

  <!-- 探索方法地图（第 19 课） -->
  <div v-else-if="kind === 'exploration-map'" class="concept-diagram">
    <p class="diagram-title">图解：四种探索策略按“依据什么决定试哪里”排布</p>
    <svg viewBox="0 0 680 340" role="img" aria-label="epsilon 贪心、UCB、汤普森采样、好奇心驱动四种探索方法的关系图">
      <rect class="diagram-node" x="30" y="30" width="620" height="44" rx="12" stroke-width="2" />
      <text class="diagram-label" x="340" y="57" text-anchor="middle" font-size="13.5" font-weight="700">核心问题：下一步试哪个从来（或很少）试过的动作？</text>

      <rect class="diagram-node" x="30" y="110" width="145" height="130" rx="12" stroke-width="2" />
      <text class="diagram-label" x="102" y="134" text-anchor="middle" font-size="13" font-weight="700">ε-贪心</text>
      <text class="diagram-label-small" x="102" y="156" text-anchor="middle" font-size="11">不挑地方</text>
      <text class="diagram-label-small" x="102" y="176" text-anchor="middle" font-size="11">概率 ε 纯随机</text>
      <text class="diagram-label-small" x="102" y="200" text-anchor="middle" font-size="11">简单、最常用</text>
      <text class="diagram-label-small" x="102" y="220" text-anchor="middle" font-size="11">浪费在差动作上</text>

      <rect class="diagram-node" x="196" y="110" width="145" height="130" rx="12" stroke-width="2" />
      <text class="diagram-label" x="268" y="134" text-anchor="middle" font-size="13" font-weight="700">UCB</text>
      <text class="diagram-label-small" x="268" y="156" text-anchor="middle" font-size="11">按不确定性挑</text>
      <text class="diagram-label-small" x="268" y="176" text-anchor="middle" font-size="11">均值 + √(log t / N(a))</text>
      <text class="diagram-label-small" x="268" y="200" text-anchor="middle" font-size="11">没试过的优先</text>
      <text class="diagram-label-small" x="268" y="220" text-anchor="middle" font-size="11">需要动作计数</text>

      <rect class="diagram-node" x="362" y="110" width="145" height="130" rx="12" stroke-width="2" />
      <text class="diagram-label" x="434" y="134" text-anchor="middle" font-size="13" font-weight="700">汤普森采样</text>
      <text class="diagram-label-small" x="434" y="156" text-anchor="middle" font-size="11">按信念抽样挑</text>
      <text class="diagram-label-small" x="434" y="176" text-anchor="middle" font-size="11">从后验采一套参数</text>
      <text class="diagram-label-small" x="434" y="200" text-anchor="middle" font-size="11">选它眼里的最优</text>
      <text class="diagram-label-small" x="434" y="220" text-anchor="middle" font-size="11">理论性质好</text>

      <rect class="diagram-node" x="528" y="110" width="145" height="130" rx="12" stroke-width="2" />
      <text class="diagram-label" x="600" y="134" text-anchor="middle" font-size="13" font-weight="700">好奇心驱动</text>
      <text class="diagram-label-small" x="600" y="156" text-anchor="middle" font-size="11">按“新鲜感”挑</text>
      <text class="diagram-label-small" x="600" y="176" text-anchor="middle" font-size="11">预测误差当内在奖励</text>
      <text class="diagram-label-small" x="600" y="200" text-anchor="middle" font-size="11">适用于状态空间大</text>
      <text class="diagram-label-small" x="600" y="220" text-anchor="middle" font-size="11">可能被噪声吸引</text>

      <path class="diagram-edge-accent" d="M 102 82 V 108" fill="none" stroke-width="2" />
      <path class="diagram-edge-accent" d="M 268 82 V 108" fill="none" stroke-width="2" />
      <path class="diagram-edge-accent" d="M 434 82 V 108" fill="none" stroke-width="2" />
      <path class="diagram-edge-accent" d="M 600 82 V 108" fill="none" stroke-width="2" />

      <text class="diagram-label-small" x="340" y="284" text-anchor="middle" font-size="12">从左到右：探索依据从“完全不看信息”到“利用不确定性/新鲜度”越来越聪明，</text>
      <text class="diagram-label-small" x="340" y="304" text-anchor="middle" font-size="12">代价是需要维护的额外统计量也越来越多</text>
    </svg>
    <p class="diagram-note">读图结论：没有免费的探索。ε-贪心是默认起点；动作数少且可枚举时用 UCB/汤普森；状态空间巨大、奖励稀疏时才考虑好奇心这类内在奖励。</p>
  </div>

  <!-- 域随机化桥梁（第 17 课） -->
  <div v-else-if="kind === 'sim2real-bridge'" class="concept-diagram">
    <p class="diagram-title">图解：域随机化——用一族仿真世界包住真实世界</p>
    <svg viewBox="0 0 680 300" role="img" aria-label="域随机化让策略在一族环境上训练，覆盖真实参数范围的示意图">
      <defs>
        <marker id="md-arrow4b" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path class="diagram-arrowhead" d="M0,0 L8,4.5 L0,9 z" />
        </marker>
      </defs>
      <rect class="diagram-node" x="30" y="30" width="200" height="240" rx="14" stroke-width="2" />
      <text class="diagram-label" x="130" y="56" text-anchor="middle" font-size="13.5" font-weight="700">单个标称仿真</text>
      <circle cx="130" cy="160" r="46" class="diagram-node-accent" stroke-width="2" />
      <text class="diagram-label-small" x="130" y="156" text-anchor="middle" font-size="11.5">质量=1.0</text>
      <text class="diagram-label-small" x="130" y="174" text-anchor="middle" font-size="11.5">摩擦=1.0</text>
      <text class="diagram-label-small" x="130" y="244" text-anchor="middle" font-size="11">真实世界多半不在圆心</text>

      <rect class="diagram-node" x="290" y="30" width="360" height="240" rx="14" stroke-width="2" />
      <text class="diagram-label" x="470" y="56" text-anchor="middle" font-size="13.5" font-weight="700">域随机化：训练时每回合换一个世界</text>
      <ellipse cx="470" cy="160" rx="150" ry="80" class="diagram-node-good" stroke-width="2" />
      <circle cx="430" cy="140" r="7" class="diagram-node-accent" stroke-width="1.5" />
      <circle cx="500" cy="120" r="7" class="diagram-node-accent" stroke-width="1.5" />
      <circle cx="530" cy="180" r="7" class="diagram-node-accent" stroke-width="1.5" />
      <circle cx="400" cy="190" r="7" class="diagram-node-accent" stroke-width="1.5" />
      <text class="diagram-label-small" x="470" y="160" text-anchor="middle" font-size="11.5">质量 0.7~1.3，摩擦 0.5~1.5</text>
      <text class="diagram-label-small" x="470" y="180" text-anchor="middle" font-size="11.5">延迟 0~3 帧，观测加噪</text>
      <text class="diagram-label-small" x="470" y="250" text-anchor="middle" font-size="11">范围包住真实参数 → 差距变成“训练内扰动”</text>

      <path class="diagram-edge-accent" d="M 232 150 H 288" fill="none" stroke-width="2.4" marker-end="url(#md-arrow4b)" />
      <text class="diagram-label-small" x="260" y="136" text-anchor="middle" font-size="11">扩大覆盖</text>
    </svg>
    <p class="diagram-note">读图结论：域随机化不追求把仿真器调得和真实一模一样（那是系统辨识的思路），而是让策略对“世界参数本来就会变”免疫。范围要来自实测，不是越宽越好。</p>
  </div>
</template>
