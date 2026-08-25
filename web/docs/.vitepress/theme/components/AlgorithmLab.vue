<script setup lang="ts">
import { computed, reactive, watch } from 'vue'

type Control = {
  key: string
  label: string
  min?: number
  max?: number
  step?: number
  default: number
  unit?: string
  kind?: 'range' | 'toggle'
}

type LabResult = {
  headline: string
  explanation: string
  formula: string
  bars: Array<{ label: string; value: number; display: string; color?: string }>
  steps: string[]
}

type LabConfig = {
  title: string
  intro: string
  controls: Control[]
  calculate: (values: Record<string, number>) => LabResult
}

const props = defineProps<{ lesson: string }>()
const percentage = (value: number) => `${Math.round(value * 100)}%`
const fixed = (value: number, digits = 2) => value.toFixed(digits)
const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value))

const labs: Record<string, LabConfig> = {
  '01': {
    title: '探索与利用拨杆',
    intro: '拖动探索率，观察机器人“试新路”和“走当前最好路线”的比例怎样变化。',
    controls: [
      { key: 'epsilon', label: '探索率 ε', min: 0, max: 1, step: 0.05, default: 0.3 },
      { key: 'quality', label: '当前 Q 表可靠度', min: 0, max: 1, step: 0.05, default: 0.7 },
    ],
    calculate: ({ epsilon, quality }) => {
      const explore = epsilon
      const exploit = 1 - epsilon
      const useful = explore * (1 - quality) + exploit * quality
      return {
        headline: useful > 0.65 ? '目前比较均衡' : epsilon > 0.6 ? '探索偏多' : '可能过早自信',
        explanation: `Q 表可靠度约为 ${percentage(quality)}。探索能在经验不足时发现新路线，利用则把已有经验变成稳定表现。`,
        formula: `P(随机探索)=ε=${fixed(epsilon)}；P(选择当前最佳动作)=1-ε=${fixed(exploit)}`,
        bars: [
          { label: '随机探索', value: explore, display: percentage(explore), color: '#d97706' },
          { label: '利用经验', value: exploit, display: percentage(exploit), color: '#2563eb' },
          { label: '预计有效决策', value: useful, display: percentage(useful), color: '#059669' },
        ],
        steps: ['先根据 ε 决定是否探索', '探索时随机选动作，利用时查 Q 表', '得到奖励后更新对应的 Q 值'],
      }
    },
  },
  '02': {
    title: 'DQN 稳定性实验',
    intro: '经验回放越充分、目标网络同步越慢，训练通常越稳定，但反馈也会更迟。',
    controls: [
      { key: 'buffer', label: '回放池多样性', min: 0.05, max: 1, step: 0.05, default: 0.75 },
      { key: 'delay', label: '目标网络滞后程度', min: 0, max: 1, step: 0.05, default: 0.55 },
    ],
    calculate: ({ buffer, delay }) => {
      const correlation = 1 - buffer
      const stability = clamp(buffer * 0.65 + delay * 0.35, 0, 1)
      const freshness = 1 - delay * 0.75
      return {
        headline: stability > 0.7 ? '目标比较稳定' : '训练目标容易摇晃',
        explanation: '经验回放打乱连续样本，目标网络暂时冻结下一步答案。两者共同避免网络一边出题、一边立刻改答案。',
        formula: 'y = r + γ · maxₐ Q_target(s′, a)',
        bars: [
          { label: '样本相关性', value: correlation, display: percentage(correlation), color: '#dc2626' },
          { label: '目标稳定性', value: stability, display: percentage(stability), color: '#059669' },
          { label: '目标新鲜度', value: freshness, display: percentage(freshness), color: '#2563eb' },
        ],
        steps: ['把经历放入回放池', '随机抽取一批旧经历', '在线网络学习，目标网络隔一段时间再同步'],
      }
    },
  },
  '03': {
    title: '策略梯度信用分配实验',
    intro: '同一个动作，如果最终回报不同，更新方向也会不同。',
    controls: [
      { key: 'probability', label: '当时选中动作的概率', min: 0.05, max: 0.95, step: 0.05, default: 0.4 },
      { key: 'returnValue', label: '归一化回报 Gₜ', min: -2, max: 2, step: 0.1, default: 1.2 },
    ],
    calculate: ({ probability, returnValue }) => {
      const loss = -Math.log(probability) * returnValue
      const direction = returnValue >= 0 ? '提高这个动作的概率' : '降低这个动作的概率'
      return {
        headline: direction,
        explanation: `当时动作概率是 ${percentage(probability)}，事后回报为 ${fixed(returnValue, 1)}。REINFORCE 用最终结果倒推：这次选择是否值得鼓励。`,
        formula: `L_policy = -log π(a|s) · Gₜ = ${fixed(loss)}`,
        bars: [
          { label: '动作概率', value: probability, display: percentage(probability), color: '#2563eb' },
          { label: '正向鼓励', value: clamp(returnValue / 2, 0, 1), display: fixed(Math.max(returnValue, 0), 1), color: '#059669' },
          { label: '反向抑制', value: clamp(-returnValue / 2, 0, 1), display: fixed(Math.max(-returnValue, 0), 1), color: '#dc2626' },
        ],
        steps: ['按策略概率采样动作', '完成回合并计算折扣回报', '用回报放大或反转对数概率梯度'],
      }
    },
  },
  '04': {
    title: '评论家打分实验',
    intro: '评论家不等整局结束，而是比较“眼前估计”和“走一步后的新估计”。',
    controls: [
      { key: 'reward', label: '即时奖励 r', min: -2, max: 2, step: 0.1, default: 1 },
      { key: 'current', label: '当前价值 V(s)', min: 0, max: 10, step: 0.5, default: 4 },
      { key: 'next', label: '下一状态价值 V(s′)', min: 0, max: 10, step: 0.5, default: 5 },
      { key: 'gamma', label: '折扣因子 γ', min: 0, max: 1, step: 0.05, default: 0.9 },
    ],
    calculate: ({ reward, current, next, gamma }) => {
      const target = reward + gamma * next
      const delta = target - current
      return {
        headline: delta >= 0 ? '这一步比预期更好' : '这一步比预期更差',
        explanation: `评论家原来估计 ${fixed(current, 1)}，走一步后得到的新目标是 ${fixed(target, 1)}，差值就是给行动者的即时反馈。`,
        formula: `δ = r + γV(s′) - V(s) = ${fixed(delta)}`,
        bars: [
          { label: '原估计 V(s)', value: current / 12, display: fixed(current, 1), color: '#2563eb' },
          { label: '新目标', value: target / 12, display: fixed(target, 1), color: '#7c3aed' },
          { label: '正优势', value: clamp(delta / 5, 0, 1), display: fixed(Math.max(delta, 0)), color: '#059669' },
          { label: '负优势', value: clamp(-delta / 5, 0, 1), display: fixed(Math.max(-delta, 0)), color: '#dc2626' },
        ],
        steps: ['行动者先采样动作', '环境返回奖励和下一状态', '评论家计算 TD 误差并指导行动者'],
      }
    },
  },
  '05': {
    title: 'PPO 裁剪实验',
    intro: '改变新旧策略概率比，直接观察“限速器”什么时候开始工作。',
    controls: [
      { key: 'ratio', label: '新旧概率比 rₜ', min: 0.4, max: 1.6, step: 0.02, default: 1.25 },
      { key: 'advantage', label: '优势 Aₜ', min: -2, max: 2, step: 0.1, default: 1 },
      { key: 'clip', label: '裁剪范围 ε', min: 0.05, max: 0.4, step: 0.05, default: 0.2 },
    ],
    calculate: ({ ratio, advantage, clip }) => {
      const clippedRatio = clamp(ratio, 1 - clip, 1 + clip)
      const raw = ratio * advantage
      const clipped = clippedRatio * advantage
      const objective = Math.min(raw, clipped)
      const active = Math.abs(ratio - 1) > clip
      return {
        headline: active ? '限速器已经介入' : '仍在允许更新范围内',
        explanation: `旧策略概率比基准是 1。当前允许区间为 ${fixed(1 - clip)}～${fixed(1 + clip)}，超出后收益不再继续增加。`,
        formula: `min(rₜAₜ, clip(rₜ)Aₜ) = ${fixed(objective)}`,
        bars: [
          { label: '原始目标', value: clamp((raw + 3) / 6, 0, 1), display: fixed(raw), color: '#d97706' },
          { label: '裁剪目标', value: clamp((clipped + 3) / 6, 0, 1), display: fixed(clipped), color: '#2563eb' },
          { label: '最终采用', value: clamp((objective + 3) / 6, 0, 1), display: fixed(objective), color: '#059669' },
        ],
        steps: ['用旧策略采集一批经验', '计算新旧策略对同一动作的概率比', '裁剪过大的策略变化并重复小步更新'],
      }
    },
  },
  '06': {
    title: '高斯策略动作实验',
    intro: '均值决定主要意图，标准差决定探索范围，tanh 负责把动作压进合法区间。',
    controls: [
      { key: 'mean', label: '高斯均值 μ', min: -2, max: 2, step: 0.1, default: 0.6 },
      { key: 'std', label: '标准差 σ', min: 0.05, max: 1.5, step: 0.05, default: 0.45 },
      { key: 'noise', label: '本次标准噪声 ξ', min: -2, max: 2, step: 0.1, default: 0.5 },
    ],
    calculate: ({ mean, std, noise }) => {
      const raw = mean + std * noise
      const action = 3 * Math.tanh(raw)
      return {
        headline: Math.abs(action) > 2.4 ? '动作接近边界' : '动作仍有调节余量',
        explanation: `策略先在无限范围内采样 ${fixed(raw)}，再用 tanh 压缩到 -1～1，最后缩放到环境的 -3～3。`,
        formula: `a = 3 · tanh(μ + σξ) = ${fixed(action)}`,
        bars: [
          { label: '向左推力', value: clamp(-action / 3, 0, 1), display: action < 0 ? fixed(-action) : '0.00', color: '#7c3aed' },
          { label: '向右推力', value: clamp(action / 3, 0, 1), display: action > 0 ? fixed(action) : '0.00', color: '#2563eb' },
          { label: '探索宽度 σ', value: std / 1.5, display: fixed(std), color: '#d97706' },
        ],
        steps: ['网络输出 μ 和 log σ', '从高斯分布采样原始动作', '用 tanh 压缩并进行概率密度修正'],
      }
    },
  },
  '07': {
    title: 'SAC 回报—熵权衡实验',
    intro: '温度系数 α 越大，策略越愿意保留随机性；越小，越专注当前高 Q 动作。',
    controls: [
      { key: 'alpha', label: '温度系数 α', min: 0, max: 1, step: 0.05, default: 0.2 },
      { key: 'q', label: '动作价值 Q(s,a)', min: -5, max: 5, step: 0.25, default: 3 },
      { key: 'entropy', label: '动作熵 H', min: 0, max: 2, step: 0.1, default: 0.8 },
    ],
    calculate: ({ alpha, q, entropy }) => {
      const entropyBonus = alpha * entropy
      const softValue = q + entropyBonus
      return {
        headline: alpha > 0.55 ? '策略更重视探索' : '策略更重视当前回报',
        explanation: `普通目标只看 Q=${fixed(q)}；SAC 额外加入 ${fixed(entropyBonus)} 的熵奖励，让多个还不错的动作都有机会被尝试。`,
        formula: `soft value = Q + αH = ${fixed(softValue)}`,
        bars: [
          { label: '回报贡献', value: clamp((q + 5) / 10, 0, 1), display: fixed(q), color: '#2563eb' },
          { label: '熵奖励', value: entropyBonus / 2, display: fixed(entropyBonus), color: '#d97706' },
          { label: '软价值', value: clamp((softValue + 5) / 12, 0, 1), display: fixed(softValue), color: '#059669' },
        ],
        steps: ['从回放池抽取旧经验', '双 Q 网络采用较小估计', '策略同时追求高 Q 和适度随机'],
      }
    },
  },
  '08': {
    title: '多步回报传播实验',
    intro: 'n 越大，奖励能更快传回远处，但同时会引入更多随机波动。',
    controls: [
      { key: 'n', label: '向前观察步数 n', min: 1, max: 8, step: 1, default: 3 },
      { key: 'gamma', label: '折扣因子 γ', min: 0.5, max: 1, step: 0.05, default: 0.9 },
      { key: 'bootstrap', label: '第 n 步价值 V(sₜ₊ₙ)', min: 0, max: 10, step: 0.5, default: 6 },
    ],
    calculate: ({ n, gamma, bootstrap }) => {
      let rewards = 0
      for (let index = 0; index < n; index += 1) rewards += gamma ** index
      const bootstrapPart = gamma ** n * bootstrap
      const target = rewards + bootstrapPart
      const variance = clamp(n / 8, 0, 1)
      return {
        headline: n <= 2 ? '偏差较大，但比较稳' : n >= 6 ? '看得远，但波动更大' : '偏差与方差较均衡',
        explanation: `假设前 ${n} 步每步奖励都是 1。目标由真实奖励 ${fixed(rewards)} 和第 n 步估计 ${fixed(bootstrapPart)} 两部分组成。`,
        formula: `Gₜ⁽${n}⁾ = Σγᵏrₜ₊ₖ + γⁿV(sₜ₊ₙ) = ${fixed(target)}`,
        bars: [
          { label: '真实奖励占比', value: rewards / Math.max(target, 1), display: fixed(rewards), color: '#059669' },
          { label: '估计价值占比', value: bootstrapPart / Math.max(target, 1), display: fixed(bootstrapPart), color: '#2563eb' },
          { label: '波动风险', value: variance, display: percentage(variance), color: '#d97706' },
        ],
        steps: [`收集连续 ${n} 步奖励`, '对越远的奖励乘更高次的 γ', '末尾再接上价值网络估计'],
      }
    },
  },
  '09': {
    title: 'Double DQN 选动作—估价值实验',
    intro: '普通 DQN 用同一套估计既挑赢家又给赢家打分，Double DQN 刻意把两件事拆开。',
    controls: [
      { key: 'onlineLeft', label: '在线网络：左动作', min: -2, max: 10, step: 0.5, default: 8 },
      { key: 'onlineRight', label: '在线网络：右动作', min: -2, max: 10, step: 0.5, default: 6 },
      { key: 'targetLeft', label: '目标网络：左动作', min: -2, max: 10, step: 0.5, default: 5 },
      { key: 'targetRight', label: '目标网络：右动作', min: -2, max: 10, step: 0.5, default: 7 },
    ],
    calculate: ({ onlineLeft, onlineRight, targetLeft, targetRight }) => {
      const onlineChoice = onlineLeft >= onlineRight ? '左' : '右'
      const standard = Math.max(targetLeft, targetRight)
      const doubleValue = onlineChoice === '左' ? targetLeft : targetRight
      const overestimate = standard - doubleValue
      return {
        headline: overestimate > 0 ? `普通 DQN 多估了 ${fixed(overestimate)}` : '两种目标这次相同',
        explanation: `在线网络选择“${onlineChoice}”，Double DQN 只让目标网络评价这个动作，不允许目标网络临时换成自己更喜欢的动作。`,
        formula: `y_DDQN = r + γ Q_target(s′, argmax Q_online) = r + γ·${fixed(doubleValue)}`,
        bars: [
          { label: '普通 DQN 目标值', value: clamp((standard + 2) / 12, 0, 1), display: fixed(standard), color: '#d97706' },
          { label: 'Double DQN 目标值', value: clamp((doubleValue + 2) / 12, 0, 1), display: fixed(doubleValue), color: '#2563eb' },
          { label: '高估差值', value: clamp(overestimate / 8, 0, 1), display: fixed(overestimate), color: '#dc2626' },
        ],
        steps: ['在线网络负责 argmax 选择动作', '目标网络只评价被选动作', '降低“噪声最大者被误当成最好”的偏差'],
      }
    },
  },
  '10': {
    title: 'Dueling DQN 拆分实验',
    intro: '先判断“这个状态整体好不好”，再判断“各动作比平均水平好多少”。',
    controls: [
      { key: 'value', label: '状态价值 V(s)', min: -5, max: 10, step: 0.5, default: 5 },
      { key: 'left', label: '左动作优势 A左', min: -5, max: 5, step: 0.5, default: 2 },
      { key: 'right', label: '右动作优势 A右', min: -5, max: 5, step: 0.5, default: -1 },
    ],
    calculate: ({ value, left, right }) => {
      const mean = (left + right) / 2
      const qLeft = value + left - mean
      const qRight = value + right - mean
      return {
        headline: qLeft >= qRight ? '左动作更优' : '右动作更优',
        explanation: `两个优势先减去平均值 ${fixed(mean)}，这样 V 和 A 的分工才不会互相“抢着解释同一部分”。`,
        formula: `Q(s,a)=V(s)+A(s,a)-mean(A)；Q左=${fixed(qLeft)}，Q右=${fixed(qRight)}`,
        bars: [
          { label: '状态价值 V', value: clamp((value + 5) / 15, 0, 1), display: fixed(value), color: '#7c3aed' },
          { label: 'Q(左)', value: clamp((qLeft + 5) / 15, 0, 1), display: fixed(qLeft), color: '#2563eb' },
          { label: 'Q(右)', value: clamp((qRight + 5) / 15, 0, 1), display: fixed(qRight), color: '#0891b2' },
        ],
        steps: ['共享特征提取层', '价值支路输出 V，优势支路输出 A', '中心化优势后合成为每个动作的 Q'],
      }
    },
  },
  '11': {
    title: '优先经验回放抽样实验',
    intro: 'TD 误差越大的经验越像“高价值错题”，但抽得太偏又需要重要性采样来纠正。',
    controls: [
      { key: 'error', label: '当前样本 |TD 误差|', min: 0.01, max: 10, step: 0.1, default: 6 },
      { key: 'alpha', label: '优先程度 α', min: 0, max: 1, step: 0.05, default: 0.6 },
      { key: 'beta', label: '偏差修正 β', min: 0, max: 1, step: 0.05, default: 0.4 },
    ],
    calculate: ({ error, alpha, beta }) => {
      const priority = (error + 0.01) ** alpha
      const relativeProbability = priority / (priority + 4)
      const correction = (1 / Math.max(relativeProbability * 10, 0.01)) ** beta
      return {
        headline: alpha < 0.1 ? '几乎等同均匀回放' : beta > 0.8 ? '强优先、强修正' : '重点复习高误差样本',
        explanation: `α 决定“偏爱错题”的程度，β 决定训练时对这种偏爱修正多少。β 通常随训练逐渐增加到 1。`,
        formula: `pᵢ=(|δᵢ|+ε)^α=${fixed(priority)}；wᵢ∝(N·P(i))^-β`,
        bars: [
          { label: '优先级', value: clamp(priority / 5, 0, 1), display: fixed(priority), color: '#d97706' },
          { label: '相对抽中概率', value: relativeProbability, display: percentage(relativeProbability), color: '#2563eb' },
          { label: '损失修正权重', value: clamp(correction / 3, 0, 1), display: fixed(correction), color: '#059669' },
        ],
        steps: ['用 TD 误差生成优先级', '按优先级概率抽样', '用重要性采样权重修正有偏损失'],
      }
    },
  },
  '12': {
    title: 'Rainbow 组件组合台',
    intro: 'Rainbow 不是一种全新更新公式，而是把多个互补改进组合到同一个 DQN 智能体中。',
    controls: [
      { key: 'double', label: 'Double DQN', default: 1, kind: 'toggle' },
      { key: 'dueling', label: 'Dueling 网络', default: 1, kind: 'toggle' },
      { key: 'per', label: '优先经验回放', default: 1, kind: 'toggle' },
      { key: 'nstep', label: '多步回报', default: 1, kind: 'toggle' },
      { key: 'distributional', label: '分布式价值 C51', default: 1, kind: 'toggle' },
      { key: 'noisy', label: '噪声网络', default: 1, kind: 'toggle' },
    ],
    calculate: (values) => {
      const enabled = Object.values(values).reduce((sum, value) => sum + Number(Boolean(value)), 0)
      const stability = (values.double + values.dueling + values.distributional) / 3
      const efficiency = (values.per + values.nstep) / 2
      const exploration = values.noisy
      return {
        headline: enabled === 6 ? '完整 Rainbow 组合' : `当前启用 ${enabled}/6 个组件`,
        explanation: '这些模块解决的问题不同：高估、表示效率、采样效率、奖励传播、回报不确定性和探索。组件越多不代表任何任务都一定更好，还要看实现和调参成本。',
        formula: `Rainbow = Double + Dueling + PER + N-step + C51 + NoisyNet`,
        bars: [
          { label: '估值稳定性', value: stability, display: percentage(stability), color: '#2563eb' },
          { label: '样本效率', value: efficiency, display: percentage(efficiency), color: '#059669' },
          { label: '参数化探索', value: exploration, display: exploration ? '已启用' : '未启用', color: '#d97706' },
        ],
        steps: ['先用单独实验验证每个组件', '确认接口和目标计算能兼容', '再做消融实验判断真正贡献'],
      }
    },
  },
  '13': {
    title: 'TD3 三重稳定器实验',
    intro: '双评论家、目标动作平滑、延迟策略更新共同抑制连续动作中的 Q 值尖峰。',
    controls: [
      { key: 'q1', label: '目标评论家 Q₁', min: -10, max: 10, step: 0.5, default: 7 },
      { key: 'q2', label: '目标评论家 Q₂', min: -10, max: 10, step: 0.5, default: 4 },
      { key: 'noise', label: '目标动作平滑噪声', min: 0, max: 0.5, step: 0.05, default: 0.2 },
      { key: 'delay', label: '策略延迟步数', min: 1, max: 5, step: 1, default: 2 },
    ],
    calculate: ({ q1, q2, noise, delay }) => {
      const conservative = Math.min(q1, q2)
      const smoothPenalty = noise * Math.abs(q1 - q2)
      const target = conservative - smoothPenalty
      return {
        headline: q1 === q2 ? '两位评论家意见一致' : '采用更保守的评论家估计',
        explanation: `TD3 采用 min(Q₁,Q₂)=${fixed(conservative)}，再用邻近动作平滑削弱尖锐假高值；行动者每 ${delay} 次评论家更新才更新一次。`,
        formula: `y = r + γ min(Q₁′,Q₂′)(s′, π′(s′)+ε)`,
        bars: [
          { label: '较高估计', value: clamp((Math.max(q1, q2) + 10) / 20, 0, 1), display: fixed(Math.max(q1, q2)), color: '#d97706' },
          { label: '保守估计', value: clamp((conservative + 10) / 20, 0, 1), display: fixed(conservative), color: '#2563eb' },
          { label: '平滑后目标', value: clamp((target + 10) / 20, 0, 1), display: fixed(target), color: '#059669' },
        ],
        steps: ['为目标动作加入截断噪声', '两个目标评论家取较小值', `评论家更新 ${delay} 次后再更新行动者`],
      }
    },
  },
  '14': {
    title: 'Dyna-Q 规划次数实验',
    intro: '一次真实经历既能直接学习，也能被模型“在脑内重放”多次。',
    controls: [
      { key: 'realSteps', label: '真实环境步数', min: 1, max: 50, step: 1, default: 10 },
      { key: 'planning', label: '每步规划次数', min: 0, max: 50, step: 1, default: 20 },
      { key: 'accuracy', label: '环境模型准确度', min: 0.5, max: 1, step: 0.05, default: 0.9 },
    ],
    calculate: ({ realSteps, planning, accuracy }) => {
      const simulatedUpdates = realSteps * planning
      const effective = realSteps + simulatedUpdates * accuracy
      const modelRisk = (1 - accuracy) * planning / 50
      return {
        headline: planning === 0 ? '退化为普通 Q 学习' : modelRisk > 0.2 ? '规划很多，但模型误差会被放大' : '真实学习与脑内规划协同',
        explanation: `${realSteps} 次真实交互产生 ${simulatedUpdates} 次规划更新。模型并不完美，所以有效收益要乘准确度，同时警惕错误想象反复强化。`,
        formula: `有效更新量 ≈ real + accuracy × real × planning = ${fixed(effective, 0)}`,
        bars: [
          { label: '真实更新', value: realSteps / 50, display: fixed(realSteps, 0), color: '#2563eb' },
          { label: '规划更新', value: clamp(simulatedUpdates / 1000, 0, 1), display: fixed(simulatedUpdates, 0), color: '#7c3aed' },
          { label: '模型误差风险', value: clamp(modelRisk, 0, 1), display: percentage(modelRisk), color: '#dc2626' },
        ],
        steps: ['真实行动并更新 Q', '把状态转移写入模型', '随机抽取模型经验进行额外规划更新'],
      }
    },
  },
  '15': {
    title: '多智能体非平稳性实验',
    intro: '你正在学习时，别的智能体也在改变策略，所以你眼中的“环境规则”会移动。',
    controls: [
      { key: 'cooperate', label: '自己的合作概率', min: 0, max: 1, step: 0.05, default: 0.7 },
      { key: 'partner', label: '队友的合作概率', min: 0, max: 1, step: 0.05, default: 0.6 },
      { key: 'change', label: '队友策略变化速度', min: 0, max: 1, step: 0.05, default: 0.4 },
    ],
    calculate: ({ cooperate, partner, change }) => {
      const jointSuccess = cooperate * partner
      const expectedReward = jointSuccess * 3 + cooperate * (1 - partner) * -1 + (1 - cooperate) * partner
      const stationarity = 1 - change
      return {
        headline: change > 0.65 ? '环境在快速移动' : jointSuccess > 0.5 ? '合作较容易形成' : '联合成功率偏低',
        explanation: `只有双方都合作时才拿到团队大奖励。队友变化越快，昨天学到的 Q 值越容易过期。`,
        formula: `P(共同合作)=P(我合作)×P(队友合作)=${percentage(jointSuccess)}`,
        bars: [
          { label: '联合成功率', value: jointSuccess, display: percentage(jointSuccess), color: '#059669' },
          { label: '期望个人回报', value: clamp((expectedReward + 1) / 4, 0, 1), display: fixed(expectedReward), color: '#2563eb' },
          { label: '环境稳定度', value: stationarity, display: percentage(stationarity), color: '#d97706' },
        ],
        steps: ['每个智能体观察局部信息', '同时选择动作并获得联合奖励', '集中训练时可让评论家看到全局信息'],
      }
    },
  },
  '16': {
    title: '离线数据覆盖实验',
    intro: '离线强化学习不能随时试错；数据里没见过的动作，Q 网络可能会凭空乐观。',
    controls: [
      { key: 'coverage', label: '数据动作覆盖率', min: 0.05, max: 1, step: 0.05, default: 0.45 },
      { key: 'shift', label: '新策略偏离数据程度', min: 0, max: 1, step: 0.05, default: 0.5 },
      { key: 'alpha', label: '保守惩罚强度 α', min: 0, max: 2, step: 0.1, default: 0.8 },
    ],
    calculate: ({ coverage, shift, alpha }) => {
      const rawRisk = (1 - coverage) * shift
      const correctedRisk = rawRisk * Math.exp(-alpha)
      const freedom = coverage * (1 - alpha / 3)
      return {
        headline: correctedRisk > 0.3 ? '分布外动作风险较高' : alpha > 1.5 ? '很保守，可能学不到改进' : '风险和改进空间较均衡',
        explanation: `覆盖率越低、策略偏离越大，分布外动作越危险。保守 Q 学习会压低未充分出现动作的价值，但惩罚过强也会变成只会模仿。`,
        formula: `分布外风险≈(1-coverage)×shift×e^-α=${fixed(correctedRisk)}`,
        bars: [
          { label: '原始分布外风险', value: rawRisk, display: percentage(rawRisk), color: '#dc2626' },
          { label: '保守修正后风险', value: correctedRisk, display: percentage(correctedRisk), color: '#d97706' },
          { label: '策略改进自由度', value: clamp(freedom, 0, 1), display: percentage(clamp(freedom, 0, 1)), color: '#059669' },
        ],
        steps: ['固定数据集，不再与环境交互', '估计数据内动作价值', '对数据外高 Q 动作施加保守惩罚'],
      }
    },
  },
  '17': {
    title: '仿真到现实鲁棒性实验',
    intro: '训练时主动随机化质量、摩擦和传感器噪声，让策略不要只会适应“一个完美世界”。',
    controls: [
      { key: 'randomization', label: '训练随机化宽度', min: 0, max: 1, step: 0.05, default: 0.6 },
      { key: 'realityGap', label: '现实与标称仿真差距', min: 0, max: 1, step: 0.05, default: 0.5 },
      { key: 'noise', label: '现实传感器噪声', min: 0, max: 1, step: 0.05, default: 0.25 },
    ],
    calculate: ({ randomization, realityGap, noise }) => {
      const coveredGap = Math.min(randomization, realityGap)
      const uncovered = Math.max(0, realityGap - randomization)
      const robustness = clamp(1 - uncovered * 0.9 - noise * (1 - randomization * 0.5), 0, 1)
      const trainingDifficulty = clamp(randomization * 0.75 + noise * 0.25, 0, 1)
      return {
        headline: robustness > 0.75 ? '有较好的迁移希望' : randomization > 0.85 ? '训练世界可能随机得过头' : '仍存在明显现实差距',
        explanation: `训练随机化覆盖了约 ${percentage(coveredGap)} 的差距，未覆盖部分为 ${percentage(uncovered)}。随机化能换来鲁棒性，但会让训练任务更难。`,
        formula: `部署成功 ≠ 仿真高分；还要通过扰动评估、限幅和真实安全检查`,
        bars: [
          { label: '差距覆盖', value: coveredGap, display: percentage(coveredGap), color: '#2563eb' },
          { label: '预计鲁棒性', value: robustness, display: percentage(robustness), color: '#059669' },
          { label: '训练难度', value: trainingDifficulty, display: percentage(trainingDifficulty), color: '#d97706' },
        ],
        steps: ['随机化物理参数和观测噪声', '在未见扰动上做鲁棒性评估', '真实部署前加入动作限幅、急停和逐级放权'],
      }
    },
  },
}

const config = computed(() => labs[props.lesson] ?? labs['01'])
const values = reactive<Record<string, number>>({})

function resetValues() {
  for (const key of Object.keys(values)) delete values[key]
  for (const control of config.value.controls) values[control.key] = control.default
}

watch(config, resetValues, { immediate: true })
const result = computed(() => config.value.calculate(values))
</script>

<template>
  <section class="algorithm-lab">
    <header>
      <div>
        <span>可交互实验</span>
        <h3>{{ config.title }}</h3>
        <p>{{ config.intro }}</p>
      </div>
      <button type="button" @click="resetValues">恢复默认值</button>
    </header>

    <div class="lab-layout">
      <div class="controls">
        <label v-for="control in config.controls" :key="control.key" :class="{ toggle: control.kind === 'toggle' }">
          <template v-if="control.kind === 'toggle'">
            <input v-model="values[control.key]" type="checkbox" :true-value="1" :false-value="0">
            <span>{{ control.label }}</span>
          </template>
          <template v-else>
            <span>{{ control.label }}</span>
            <strong>{{ Number(values[control.key]).toFixed(control.step && control.step < 0.1 ? 2 : 1) }}{{ control.unit ?? '' }}</strong>
            <input
              v-model.number="values[control.key]"
              type="range"
              :min="control.min"
              :max="control.max"
              :step="control.step"
            >
          </template>
        </label>
      </div>

      <div class="result-panel">
        <small>当前判断</small>
        <h4>{{ result.headline }}</h4>
        <p>{{ result.explanation }}</p>
        <code>{{ result.formula }}</code>
        <div class="bars">
          <article v-for="bar in result.bars" :key="bar.label">
            <div><span>{{ bar.label }}</span><strong>{{ bar.display }}</strong></div>
            <i><b :style="{ width: `${Math.max(2, Math.min(100, bar.value * 100))}%`, background: bar.color }" /></i>
          </article>
        </div>
      </div>
    </div>

    <ol>
      <li v-for="step in result.steps" :key="step">{{ step }}</li>
    </ol>
  </section>
</template>

<style scoped>
.algorithm-lab {
  margin: 24px 0;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 18px;
  background: linear-gradient(145deg, var(--vp-c-bg-soft), var(--vp-c-bg));
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
}

header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

header span {
  color: var(--vp-c-brand-1);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
}

h3,
h4,
p {
  margin: 0;
}

h3 {
  margin-top: 3px;
  font-size: 1.2rem;
}

header p {
  margin-top: 6px;
  color: var(--vp-c-text-2);
  font-size: 0.92rem;
}

button {
  flex: none;
  min-height: 36px;
  padding: 0 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  cursor: pointer;
}

.lab-layout {
  display: grid;
  grid-template-columns: minmax(230px, 0.85fr) minmax(300px, 1.35fr);
  gap: 16px;
  margin-top: 18px;
}

.controls,
.result-panel {
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg);
}

.controls {
  display: grid;
  align-content: start;
  gap: 15px;
}

.controls label:not(.toggle) {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 7px 12px;
  color: var(--vp-c-text-2);
  font-size: 0.86rem;
}

.controls label strong {
  color: var(--vp-c-text-1);
}

.controls input[type='range'] {
  grid-column: 1 / -1;
  width: 100%;
  accent-color: var(--vp-c-brand-1);
}

.toggle {
  display: flex;
  gap: 9px;
  align-items: center;
  min-height: 36px;
  padding: 7px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  cursor: pointer;
}

.toggle input {
  accent-color: var(--vp-c-brand-1);
}

.result-panel > small {
  color: var(--vp-c-text-2);
}

h4 {
  margin-top: 2px;
  color: var(--vp-c-brand-1);
  font-size: 1.22rem;
}

.result-panel > p {
  margin-top: 7px;
  color: var(--vp-c-text-2);
  line-height: 1.7;
}

.result-panel > code {
  display: block;
  margin-top: 11px;
  padding: 10px 12px;
  overflow-x: auto;
  border-radius: 9px;
  color: var(--vp-c-text-1);
  background: var(--vp-code-block-bg);
  white-space: nowrap;
}

.bars {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.bars article div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 0.83rem;
}

.bars article span {
  color: var(--vp-c-text-2);
}

.bars i {
  display: block;
  height: 8px;
  margin-top: 5px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--vp-c-bg-soft);
}

.bars b {
  display: block;
  height: 100%;
  border-radius: inherit;
  transition: width 180ms ease;
}

ol {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin: 16px 0 0;
  padding: 0;
  list-style: none;
  counter-reset: process;
}

ol li {
  position: relative;
  padding: 12px 11px 12px 40px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  color: var(--vp-c-text-2);
  font-size: 0.83rem;
  line-height: 1.55;
  counter-increment: process;
}

ol li::before {
  position: absolute;
  top: 12px;
  left: 11px;
  display: grid;
  width: 21px;
  height: 21px;
  place-items: center;
  border-radius: 50%;
  color: white;
  background: var(--vp-c-brand-1);
  content: counter(process);
  font-size: 0.7rem;
  font-weight: 800;
}

@media (max-width: 760px) {
  header {
    display: block;
  }

  button {
    margin-top: 10px;
  }

  .lab-layout {
    grid-template-columns: 1fr;
  }

  ol {
    grid-template-columns: 1fr;
  }
}
</style>
