<script setup lang="ts">
import { computed, reactive, ref } from 'vue'

// 浏览器内真实运行的 Q 学习实验台：环境规则与 src/rl_learning_lab/gridworld.py
// 完全一致（布局、奖励、max_steps），所有数字都由真实计算得到，无预设脚本。

const ROWS = 6
const COLS = 6
const START: [number, number] = [5, 0]
const GOAL: [number, number] = [0, 5]
const WALLS = new Set(['1,1', '2,1', '3,1', '1,3', '2,3', '4,4'])
const TRAPS = new Set(['3,3', '4,2'])
const STEP_REWARD = -0.1
const WALL_REWARD = -1.0
const TRAP_REWARD = -10.0
const GOAL_REWARD = 10.0
const MAX_STEPS = 100
const ACTIONS = ['上', '右', '下', '左'] as const
const DELTAS: Array<[number, number]> = [[-1, 0], [0, 1], [1, 0], [0, -1]]

// 可复现随机数：mulberry32，与课程“固定种子”的要求一致。
function makeRng(seed: number) {
  let a = seed >>> 0
  return () => {
    a += 0x6d2b79f5
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const epsilon = ref(0.2)
const alpha = ref(0.15)
const gamma = ref(0.95)
const showAllQ = ref(false)

const qTable = reactive<number[][]>(
  Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0]),
)
const agent = ref<number>(START[0] * COLS + START[1])
const episode = ref(0)
const stepsInEpisode = ref(0)
const rewardInEpisode = ref(0)
const exploresInEpisode = ref(0)
const history = ref<number[]>([])
const successWindow = ref<boolean[]>([])
const rng = ref(makeRng(42))

type StepTrace = {
  step: number
  from: string
  action: string
  explored: boolean
  reward: number
  target: number
  oldQ: number
  newQ: number
  done: string
} | null

const lastTrace = ref<StepTrace>(null)

const toPos = (state: number): [number, number] => [Math.floor(state / COLS), state % COLS]
const posKey = (r: number, c: number) => `${r},${c}`
const cellText = (r: number, c: number) => {
  if (r === GOAL[0] && c === GOAL[1]) return '终点'
  if (posKey(r, c) === posKey(START[0], START[1])) return '起点'
  if (WALLS.has(posKey(r, c))) return '墙'
  if (TRAPS.has(posKey(r, c))) return '陷阱'
  return ''
}

function walkable(r: number, c: number) {
  return r >= 0 && r < ROWS && c >= 0 && c < COLS && !WALLS.has(posKey(r, c))
}

function chooseAction(state: number, eps: number): { action: number; explored: boolean } {
  if (rng.value() < eps) {
    return { action: Math.floor(rng.value() * 4), explored: true }
  }
  const values = qTable[state]
  let best = values[0]
  for (const v of values) if (v > best) best = v
  const ties: number[] = []
  values.forEach((v, i) => {
    if (Math.abs(v - best) < 1e-9) ties.push(i)
  })
  return { action: ties[Math.floor(rng.value() * ties.length)], explored: false }
}

function envStep(state: number, action: number): { next: number; reward: number; done: boolean; hit: 'goal' | 'trap' | 'wall' | 'normal' | null } {
  const [dr, dc] = DELTAS[action]
  const [r, c] = toPos(state)
  const nr = r + dr
  const nc = c + dc
  if (!walkable(nr, nc)) {
    return { next: state, reward: WALL_REWARD, done: stepsInEpisode.value >= MAX_STEPS, hit: 'wall' }
  }
  const next = nr * COLS + nc
  if (nr === GOAL[0] && nc === GOAL[1]) return { next, reward: GOAL_REWARD, done: true, hit: 'goal' }
  if (TRAPS.has(posKey(nr, nc))) return { next, reward: TRAP_REWARD, done: true, hit: 'trap' }
  return { next, reward: STEP_REWARD, done: stepsInEpisode.value >= MAX_STEPS, hit: null }
}

function qLearningStep() {
  const { action, explored } = chooseAction(agent.value, epsilon.value)
  const before = qTable[agent.value][action]
  // 与 Python 版一致：先计入步数，再判断是否超过回合上限。
  stepsInEpisode.value += 1
  const outcome = envStep(agent.value, action)

  rewardInEpisode.value += outcome.reward
  if (explored) exploresInEpisode.value += 1

  // 终点/陷阱之后没有未来价值，bootstrap 必须为 0。
  const nextBest = outcome.done ? 0 : Math.max(...qTable[outcome.next])
  const target = outcome.reward + gamma.value * nextBest
  qTable[agent.value][action] = before + alpha.value * (target - before)

  const [fr, fc] = toPos(agent.value)
  lastTrace.value = {
    step: stepsInEpisode.value,
    from: `(${fr},${fc})`,
    action: ACTIONS[action],
    explored,
    reward: outcome.reward,
    target,
    oldQ: before,
    newQ: qTable[agent.value][action],
    done: outcome.hit === 'goal' ? '到达终点' : outcome.hit === 'trap' ? '掉进陷阱' : outcome.done ? '超过步数上限' : '继续',
  }

  agent.value = outcome.next
  if (outcome.done) endEpisode(outcome.hit === 'goal')
}

function endEpisode(success: boolean) {
  episode.value += 1
  history.value.push(rewardInEpisode.value)
  if (history.value.length > 400) history.value.shift()
  successWindow.value.push(success)
  if (successWindow.value.length > 30) successWindow.value.shift()
  agent.value = START[0] * COLS + START[1]
  stepsInEpisode.value = 0
  rewardInEpisode.value = 0
  exploresInEpisode.value = 0
}

function runEpisode() {
  const guard = MAX_STEPS + 5
  for (let i = 0; i < guard; i += 1) {
    const before = stepsInEpisode.value
    qLearningStep()
    if (stepsInEpisode.value < before || stepsInEpisode.value === 0) break
  }
}

function runEpisodes(count: number) {
  for (let i = 0; i < count; i += 1) runEpisode()
}

function resetAll() {
  for (const row of qTable) row.fill(0)
  agent.value = START[0] * COLS + START[1]
  episode.value = 0
  stepsInEpisode.value = 0
  rewardInEpisode.value = 0
  exploresInEpisode.value = 0
  history.value = []
  successWindow.value = []
  rng.value = makeRng(42)
  lastTrace.value = null
}

const successRate = computed(() =>
  successWindow.value.length === 0
    ? '—'
    : `${Math.round((successWindow.value.filter(Boolean).length / successWindow.value.length) * 100)}%`,
)

const greedyRoute = computed(() => {
  // 纯贪心（ε=0）走一遍，展示当前学到的路线；最多 40 步防死循环。
  let state = START[0] * COLS + START[1]
  const path: number[] = [state]
  for (let i = 0; i < 40; i += 1) {
    const values = qTable[state]
    let best = 0
    values.forEach((v, a) => { if (v > values[best]) best = a })
    const savedSteps = stepsInEpisode.value
    stepsInEpisode.value = 0
    const outcome = envStep(state, best)
    stepsInEpisode.value = savedSteps
    state = outcome.next
    if (path.includes(state)) break
    path.push(state)
    if (outcome.hit === 'goal' || outcome.hit === 'trap') break
  }
  return new Set(path)
})

const cells = computed(() =>
  Array.from({ length: ROWS }, (_, r) =>
    Array.from({ length: COLS }, (_, c) => {
      const state = r * COLS + c
      const values = qTable[state]
      const maxQ = Math.max(...values)
      const norm = Math.max(-1, Math.min(1, maxQ / 10))
      const color = norm >= 0
        ? `rgba(5, 150, 105, ${0.08 + norm * 0.45})`
        : `rgba(220, 38, 38, ${0.08 + (-norm) * 0.35})`
      let arrow = ''
      if (!cellText(r, c)) {
        let best = 0
        values.forEach((v, a) => { if (v > values[best]) best = a })
        const any = values.some(v => v !== 0)
        arrow = any ? ['↑', '→', '↓', '←'][best] : '·'
      }
      return {
        state,
        label: cellText(r, c),
        arrow,
        maxQ: maxQ === 0 ? '' : maxQ.toFixed(1),
        color,
        isAgent: agent.value === state,
        onRoute: greedyRoute.value.has(state),
        values: values.map(v => (v === 0 ? '0' : v.toFixed(1))),
      }
    }),
  ),
)

const curve = computed(() => {
  const data = history.value
  if (data.length < 2) return null
  const w = 600
  const h = 150
  const lo = Math.min(-12, ...data)
  const hi = Math.max(11, ...data)
  const x = (i: number) => (i / (data.length - 1)) * (w - 20) + 10
  const y = (v: number) => h - 12 - ((v - lo) / (hi - lo)) * (h - 24)
  const raw = data.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const avg = data
    .map((_, i) => {
      const from = Math.max(0, i - 19)
      const slice = data.slice(from, i + 1)
      return slice.reduce((a, b) => a + b, 0) / slice.length
    })
    .map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
    .join(' ')
  return { raw, avg, lo, hi, count: data.length }
})

const fixed1 = (v: number) => v.toFixed(1)
</script>

<template>
  <section class="course-lab q-lab" aria-labelledby="qlab-title">
    <div class="lab-heading">
      <div>
        <h3 id="qlab-title">实验台：亲手运行 Q 学习（真实算法，非动画演示）</h3>
        <p>
          任务：先点几次「单步执行」看清每一步的 Q 更新数字，再点「训练 100 回合」，
          观察箭头路线怎样从乱走变成绕开陷阱的最短路线。
        </p>
      </div>
    </div>

    <div class="lab-body">
      <div class="lab-grid-panel" aria-label="网格世界">
        <div class="grid" :style="{ gridTemplateColumns: `repeat(${COLS}, minmax(44px, 1fr))` }">
          <div
            v-for="(row, r) in cells"
            :key="r"
            class="cell-row"
          >
            <div
              v-for="cell in row"
              :key="cell.state"
              class="cell"
              :class="{
                wall: cell.label === '墙',
                trap: cell.label === '陷阱',
                goal: cell.label === '终点',
                start: cell.label === '起点',
                agent: cell.isAgent,
                route: cell.onRoute,
              }"
              :style="{ background: cell.label === '墙' || cell.label === '终点' || cell.label === '陷阱' ? undefined : cell.color }"
            >
              <span v-if="cell.label" class="cell-tag">{{ cell.label }}</span>
              <template v-else>
                <span class="arrow">{{ cell.arrow }}</span>
                <span class="q-value">{{ cell.maxQ }}</span>
                <span v-if="showAllQ" class="all-q">{{ cell.values.join(' ') }}</span>
              </template>
            </div>
          </div>
        </div>
        <p class="legend">
          背景色 = 该格的最大 Q 值（绿正红负）；箭头 = 当前贪心动作；绿框 = 纯贪心路线。
          <label class="toggle">
            <input v-model="showAllQ" type="checkbox">
            显示每格 4 个动作的 Q 值
          </label>
        </p>
      </div>

      <div class="lab-controls">
        <label>
          探索率 ε = {{ epsilon.toFixed(2) }}
          <input v-model.number="epsilon" type="range" min="0" max="1" step="0.05">
        </label>
        <label>
          学习率 α = {{ alpha.toFixed(2) }}
          <input v-model.number="alpha" type="range" min="0.05" max="1" step="0.05">
        </label>
        <label>
          折扣 γ = {{ gamma.toFixed(2) }}
          <input v-model.number="gamma" type="range" min="0.5" max="0.99" step="0.05">
        </label>
        <div class="buttons">
          <button type="button" class="primary" @click="qLearningStep">单步执行</button>
          <button type="button" @click="runEpisode">跑完本回合</button>
          <button type="button" @click="runEpisodes(100)">训练 100 回合</button>
          <button type="button" @click="resetAll">全部重置</button>
        </div>

        <dl class="stats" aria-live="polite">
          <div><dt>已完成回合</dt><dd>{{ episode }}</dd></div>
          <div><dt>本回合步数</dt><dd>{{ stepsInEpisode }}</dd></div>
          <div><dt>本回合奖励</dt><dd>{{ fixed1(rewardInEpisode) }}</dd></div>
          <div><dt>本回合探索次数</dt><dd>{{ exploresInEpisode }}</dd></div>
          <div><dt>近 30 回合成功率</dt><dd>{{ successRate }}</dd></div>
        </dl>

        <div v-if="lastTrace" class="trace" aria-live="polite">
          <strong>最近一步的 Q 更新</strong>
          <p>
            第 {{ lastTrace.step }} 步：状态 {{ lastTrace.from }}，动作「{{ lastTrace.action }}」<span class="muted">（{{ lastTrace.explored ? '探索：随机选的' : '利用：按 Q 表选的' }}）</span><br>
            奖励 r = {{ lastTrace.reward.toFixed(1) }}，目标 = r + γ·max Q(s′) = {{ lastTrace.target.toFixed(2) }}<br>
            Q：{{ lastTrace.oldQ.toFixed(2) }} → {{ lastTrace.newQ.toFixed(2) }}（{{ lastTrace.done }}）
          </p>
        </div>
      </div>
    </div>

    <div v-if="curve" class="curve-panel">
      <strong>每回合总奖励（真实训练记录）</strong>
      <svg viewBox="0 0 600 150" role="img" aria-label="每回合总奖励曲线">
        <line x1="10" y1="138" x2="590" y2="138" class="axis" />
        <text x="588" y="148" text-anchor="end" class="axis-label">{{ curve.lo }}</text>
        <text x="12" y="12" class="axis-label">{{ curve.hi }}</text>
        <polyline :points="curve.raw" class="raw-line" />
        <polyline :points="curve.avg" class="avg-line" />
      </svg>
      <p class="legend">浅色 = 单回合原始值；深色 = 20 回合移动平均。共 {{ curve.count }} 回合。</p>
    </div>
    <p v-else class="legend">曲线将在第一个回合结束后出现。</p>

    <p class="lab-note">
      观察结论：ε=0 时若初始路线碰巧很差，Q 表可能长期停在差路线（第 1 课误区 2）；
      ε=0.2 训练 200~300 回合后，贪心路线通常稳定绕开两个陷阱。γ 调到 0.5，
      路线常常变长——机器人变得短视，不再在乎远处的终点加分。
    </p>
  </section>
</template>

<style scoped>
.q-lab {
  margin: 1.5rem 0 2rem;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--course-panel);
}

.lab-heading p {
  margin: 6px 0 0;
  color: var(--vp-c-text-2);
  font-size: 0.92rem;
}

.lab-body {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(240px, 1fr);
  gap: 18px;
  margin-top: 14px;
}

.lab-grid-panel {
  min-width: 0;
}

.grid {
  display: grid;
  gap: 5px;
}

.cell-row {
  display: grid;
  grid-template-columns: subgrid;
  grid-column: 1 / -1;
  gap: 5px;
}

.cell {
  position: relative;
  display: grid;
  min-height: 62px;
  place-items: center;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  font-size: 0.78rem;
}

.cell.wall {
  background: repeating-linear-gradient(45deg, var(--vp-c-divider), var(--vp-c-divider) 4px, transparent 4px, transparent 8px);
  color: var(--vp-c-text-3);
}

.cell.trap {
  border-color: var(--course-red);
  color: var(--course-red);
  background: color-mix(in srgb, var(--course-red) 14%, var(--vp-c-bg));
}

.cell.goal {
  border-color: var(--course-green);
  color: var(--course-green);
  background: color-mix(in srgb, var(--course-green) 16%, var(--vp-c-bg));
  font-weight: 700;
}

.cell.start {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.cell.route {
  outline: 2px solid var(--course-green);
  outline-offset: -2px;
}

.cell.agent::after {
  content: '🤖';
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: 1.25rem;
}

.arrow {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
}

.q-value {
  color: var(--vp-c-text-2);
  font-size: 0.68rem;
}

.all-q {
  position: absolute;
  bottom: 2px;
  color: var(--vp-c-text-3);
  font-size: 0.56rem;
  letter-spacing: -0.02em;
}

.legend {
  margin: 10px 0 0;
  color: var(--vp-c-text-2);
  font-size: 0.84rem;
  line-height: 1.6;
}

.toggle {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  margin-left: 8px;
  cursor: pointer;
}

.lab-controls {
  display: grid;
  gap: 12px;
  align-content: start;
}

.lab-controls label {
  display: grid;
  gap: 4px;
  color: var(--vp-c-text-2);
  font-size: 0.88rem;
}

.lab-controls input[type='range'] {
  width: 100%;
  accent-color: var(--vp-c-brand-1);
}

.buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.buttons button {
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  cursor: pointer;
}

.buttons button.primary {
  border-color: var(--vp-c-brand-1);
  color: white;
  background: var(--vp-c-brand-1);
}

.stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.stats > div {
  padding: 8px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  background: var(--vp-c-bg);
}

.stats dt {
  color: var(--vp-c-text-3);
  font-size: 0.74rem;
}

.stats dd {
  margin: 2px 0 0;
  font-weight: 700;
  font-size: 0.95rem;
}

.trace {
  padding: 10px 12px;
  border-left: 4px solid var(--course-purple);
  border-radius: 0 10px 10px 0;
  background: var(--vp-c-bg);
  font-size: 0.84rem;
  line-height: 1.7;
}

.trace strong {
  color: var(--course-purple);
}

.trace p {
  margin: 6px 0 0;
}

.muted {
  color: var(--vp-c-text-3);
}

.curve-panel {
  margin-top: 16px;
}

.curve-panel svg {
  width: 100%;
  height: auto;
  margin-top: 6px;
}

.axis {
  stroke: var(--vp-c-divider);
}

.axis-label {
  fill: var(--vp-c-text-3);
  font-size: 10px;
}

.raw-line {
  fill: none;
  stroke: var(--vp-c-brand-1);
  stroke-width: 1;
  opacity: 0.4;
}

.avg-line {
  fill: none;
  stroke: var(--vp-c-brand-1);
  stroke-width: 2.4;
}

.lab-note {
  margin: 14px 0 0;
  padding: 11px 14px;
  border-left: 4px solid var(--vp-c-brand-1);
  border-radius: 0 10px 10px 0;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
  font-size: 0.9rem;
  line-height: 1.7;
}

@media (max-width: 720px) {
  .lab-body {
    grid-template-columns: 1fr;
  }

  .cell {
    min-height: 48px;
  }

  .all-q {
    display: none;
  }
}
</style>
