<script setup lang="ts">
import { computed, ref } from 'vue'

// DQN 高估偏差的真实蒙特卡洛实验：所有数字来自浏览器内的真实随机采样。
// 场景与第 9 课一致：k 个动作的真实价值全部相等（无差别），估计值带高斯噪声。
// max(估计) 的期望必然 ≥ 真实值——高估不是玄学，是 max 运算的统计性质。

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

// Box-Muller 变换：均匀随机数 → 标准正态
function gaussian(rng: () => number) {
  const u1 = Math.max(rng(), 1e-12)
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

const actionCount = ref(4)
const noise = ref(0.5)
const rng = ref(makeRng(9))

const draws = ref(0)
const sumMaxSingle = ref(0)
const sumMaxDouble = ref(0)

const TRUE_VALUE = 1.0

function sampleOnce() {
  const estimates = Array.from({ length: actionCount.value }, () => TRUE_VALUE + gaussian(rng.value) * noise.value)
  // 普通 DQN：同一组估计既用来选动作、又用来打分
  const best = Math.max(...estimates)
  // Double DQN：用第一组估计选动作，用独立第二组估计给该动作打分
  const evaluations = Array.from({ length: actionCount.value }, () => TRUE_VALUE + gaussian(rng.value) * noise.value)
  const chosen = estimates.indexOf(best)
  draws.value += 1
  sumMaxSingle.value += best
  sumMaxDouble.value += evaluations[chosen]
}

function sampleMany(count: number) {
  for (let i = 0; i < count; i += 1) sampleOnce()
}

function resetRun() {
  draws.value = 0
  sumMaxSingle.value = 0
  sumMaxDouble.value = 0
  rng.value = makeRng(9)
}

const avgSingle = computed(() => (draws.value === 0 ? null : sumMaxSingle.value / draws.value))
const avgDouble = computed(() => (draws.value === 0 ? null : sumMaxDouble.value / draws.value))
const overestimate = computed(() =>
  avgSingle.value === null ? null : avgSingle.value - TRUE_VALUE,
)
const doubleBias = computed(() => (avgDouble.value === null ? null : avgDouble.value - TRUE_VALUE))

const fmt = (v: number | null, digits = 3) => (v === null ? '—' : v.toFixed(digits))
</script>

<template>
  <section class="course-lab over-lab" aria-labelledby="over-title">
    <div class="lab-heading">
      <h3 id="over-title">实验台：用真实随机采样复现“max 造成的高估”</h3>
      <p>
        任务：保持“k 个动作真实价值全部 = 1.0”，采样上千次，看「同组估计取 max」的平均值
        与真实值 1.0 差多少，再对比 Double 式“选与评分开”的结果。
      </p>
    </div>

    <div class="controls">
      <label>
        动作数 k = {{ actionCount }}
        <input v-model.number="actionCount" type="range" min="2" max="10" step="1">
      </label>
      <label>
        估计噪声 σ = {{ noise.toFixed(2) }}
        <input v-model.number="noise" type="range" min="0.1" max="1.5" step="0.05">
      </label>
      <div class="buttons">
        <button type="button" class="primary" @click="sampleOnce">采样 1 次</button>
        <button type="button" @click="sampleMany(1000)">采样 1000 次</button>
        <button type="button" @click="resetRun">重置</button>
      </div>
    </div>

    <div class="bars" aria-live="polite">
      <div class="bar-row">
        <span class="bar-name">真实最优值</span>
        <div class="bar-track">
          <div class="bar" :style="{ width: `${Math.min(100, TRUE_VALUE / 2 * 100)}%` }" />
        </div>
        <span class="bar-value">1.000</span>
      </div>
      <div class="bar-row">
        <span class="bar-name">DQN 式 max 估计均值</span>
        <div class="bar-track">
          <div class="bar bad" :style="{ width: `${Math.min(100, (avgSingle ?? 0) / 2 * 100)}%` }" />
        </div>
        <span class="bar-value">{{ fmt(avgSingle) }}</span>
      </div>
      <div class="bar-row">
        <span class="bar-name">Double 式选评分开均值</span>
        <div class="bar-track">
          <div class="bar good" :style="{ width: `${Math.min(100, (avgDouble ?? 0) / 2 * 100)}%` }" />
        </div>
        <span class="bar-value">{{ fmt(avgDouble) }}</span>
      </div>
    </div>

    <p class="reading" aria-live="polite">
      已采样 {{ draws.toLocaleString() }} 次。
      <template v-if="draws > 0">
        普通 DQN 式估计平均高估 <strong>{{ fmt(overestimate) }}</strong>；
        Double 式平均偏差 {{ fmt(doubleBias) }}（正为偏高、负为偏低）。
        <template v-if="draws < 300">采样还少，数字会继续漂移，建议点几次「采样 1000 次」。</template>
      </template>
      <template v-else>点击上方按钮开始采样。</template>
    </p>

    <p class="lab-note">
      读图结论：动作数越多、噪声越大，max 的平均高估越严重（max 挑的总是“被噪声偶然抬高”的那个）。
      Double 的“选动作”与“打分”用两组独立噪声，正负误差互相抵消，偏差显著变小——但不是零，
      也不是永远低估。
    </p>
  </section>
</template>

<style scoped>
.over-lab {
  margin: 1.5rem 0 2rem;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--course-panel);
}

.lab-heading h3 {
  margin: 0;
  font-size: 1.02rem;
}

.lab-heading p {
  margin: 6px 0 0;
  color: var(--vp-c-text-2);
  font-size: 0.92rem;
  line-height: 1.7;
}

.controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.3fr);
  gap: 14px;
  align-items: end;
  margin: 14px 0;
}

.controls label {
  display: grid;
  gap: 4px;
  color: var(--vp-c-text-2);
  font-size: 0.88rem;
}

.controls input {
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

.bars {
  display: grid;
  gap: 10px;
  margin: 6px 0 12px;
}

.bar-row {
  display: grid;
  grid-template-columns: 170px 1fr 70px;
  gap: 10px;
  align-items: center;
}

.bar-name {
  color: var(--vp-c-text-2);
  font-size: 0.84rem;
}

.bar-track {
  height: 22px;
  overflow: hidden;
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.bar {
  height: 100%;
  border-radius: 7px;
  background: var(--vp-c-text-3);
}

.bar.bad {
  background: var(--course-red);
}

.bar.good {
  background: var(--course-green);
}

.bar-value {
  color: var(--vp-c-text-1);
  font-variant-numeric: tabular-nums;
  font-size: 0.9rem;
  font-weight: 700;
  text-align: right;
}

.reading {
  margin: 10px 0 0;
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  line-height: 1.7;
}

.reading strong {
  color: var(--course-red);
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
  .controls {
    grid-template-columns: 1fr;
  }

  .bar-row {
    grid-template-columns: 1fr 90px;
  }

  .bar-track {
    grid-column: 1 / -1;
    order: 3;
  }
}
</style>
