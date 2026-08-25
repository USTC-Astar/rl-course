<script setup lang="ts">
import { computed, ref } from 'vue'

// PPO 裁剪目标的真实函数曲线：所有点都由 min(rA, clip(r,1-ε,1+ε)A) 直接计算。
const epsilon = ref(0.2)
const advantage = ref(1)

const W = 640
const H = 320
const R_MAX = 2
const Y_MAX = 2.2
const Y_MIN = -2.2

const px = (r: number) => 40 + (r / R_MAX) * (W - 60)
const py = (v: number) => H / 2 - (v / Y_MAX) * (H / 2 - 24)

const objective = (r: number, a: number, eps: number) =>
  Math.min(r * a, Math.max(1 - eps, Math.min(1 + eps, r)) * a)

const series = computed(() => {
  const points: string[] = []
  const unclipped: string[] = []
  for (let i = 0; i <= 160; i += 1) {
    const r = (i / 160) * R_MAX
    points.push(`${px(r).toFixed(1)},${py(objective(r, advantage.value, epsilon.value)).toFixed(1)}`)
    unclipped.push(`${px(r).toFixed(1)},${py(r * advantage.value).toFixed(1)}`)
  }
  return { clipped: points.join(' '), unclipped: unclipped.join(' ') }
})

const probe = ref(1.5)
const probeValue = computed(() => objective(probe.value, advantage.value, epsilon.value))
const probeUnclipped = computed(() => probe.value * advantage.value)
const clipping = computed(() => Math.abs(probeValue.value - probeUnclipped.value) > 1e-6)
const noGradient = computed(() =>
  (advantage.value > 0 && probe.value > 1 + epsilon.value)
  || (advantage.value < 0 && probe.value < 1 - epsilon.value),
)
</script>

<template>
  <section class="course-lab clip-lab" aria-labelledby="clip-title">
    <div class="lab-heading">
      <h3 id="clip-title">实验台：拖动参数，观察裁剪目标的真实曲线</h3>
      <p>任务：把优势切成正负两种情况，观察“没有额外收益的水平段”出现在哪一侧。</p>
    </div>

    <div class="controls">
      <label>
        裁剪系数 ε = {{ epsilon.toFixed(2) }}
        <input v-model.number="epsilon" type="range" min="0.05" max="0.5" step="0.05">
      </label>
      <label>
        优势值 A = {{ advantage.toFixed(1) }}
        <input v-model.number="advantage" type="range" min="-2" max="2" step="0.5">
      </label>
      <label>
        探针位置 r = {{ probe.toFixed(2) }}
        <input v-model.number="probe" type="range" min="0.2" max="2" step="0.05">
      </label>
    </div>

    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="PPO 裁剪目标函数曲线图">
      <!-- 纵轴 -->
      <line :x1="px(0)" :y1="py(Y_MIN)" :x2="px(0)" :y2="py(Y_MAX)" class="axis" />
      <!-- 横轴 -->
      <line :x1="px(0)" :y1="py(0)" :x2="px(R_MAX)" :y2="py(0)" class="axis" />
      <!-- 裁剪区间底色 -->
      <rect
        :x="px(1 - epsilon)"
        :y="24"
        :width="px(1 + epsilon) - px(1 - epsilon)"
        :height="H - 48"
        class="clip-zone"
      />
      <!-- 参考线 r=1 -->
      <line :x1="px(1)" :y1="24" :x2="px(1)" :y2="H - 24" class="reference" />
      <text :x="px(1) + 4" :y="38" class="axis-label">r = 1（新旧策略相同）</text>
      <text :x="px(1 - epsilon)" :y="H - 12" text-anchor="middle" class="axis-label">1−ε</text>
      <text :x="px(1 + epsilon)" :y="H - 12" text-anchor="middle" class="axis-label">1+ε</text>
      <text :x="px(R_MAX)" :y="py(0) + 16" text-anchor="end" class="axis-label">概率比 r →</text>
      <text :x="px(0) + 6" :y="py(Y_MAX) + 4" class="axis-label">L 值 ↑</text>

      <!-- 未裁剪的 r·A 虚线 -->
      <polyline :points="series.unclipped" class="unclipped-line" />
      <!-- 裁剪后的目标实线 -->
      <polyline :points="series.clipped" class="clipped-line" />

      <!-- 探针 -->
      <line :x1="px(probe)" :y1="py(0)" :x2="px(probe)" :y2="py(probeValue)" class="probe-line" />
      <circle :cx="px(probe)" :cy="py(probeValue)" r="6" class="probe-dot" />
      <text :x="px(probe) + 10" :y="py(probeValue) - 8" class="probe-text">L={{ probeValue.toFixed(2) }}</text>
    </svg>

    <p class="reading" aria-live="polite">
      当探针 r={{ probe.toFixed(2) }}、A={{ advantage.toFixed(1) }} 时：
      未裁剪目标是 {{ probeUnclipped.toFixed(2) }}，裁剪后是 {{ probeValue.toFixed(2) }}。
      <template v-if="clipping">
        <strong>裁剪已介入</strong>：继续朝这个方向增大（或减小）概率不再带来任何额外收益<template v-if="noGradient">，此段梯度为 0，优化器失去继续推的动力</template>。
      </template>
      <template v-else>当前还在裁剪区间内，目标与未裁剪版本一致。</template>
    </p>

    <p class="lab-note">
      读图结论：A&gt;0 时水平段出现在右侧（概率已被抬够了），A&lt;0 时水平段出现在左侧（概率已被压够了）。
      PPO 不禁止策略变化，只是让“过度纠正”不再有奖励——这正是“限速器”的几何含义。
    </p>
  </section>
</template>

<style scoped>
.clip-lab {
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
}

.controls {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
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

svg {
  width: 100%;
  height: auto;
}

.axis {
  stroke: var(--vp-c-divider);
  stroke-width: 1.4;
}

.axis-label {
  fill: var(--vp-c-text-3);
  font-size: 11px;
}

.clip-zone {
  fill: color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent);
}

.reference {
  stroke: var(--vp-c-text-3);
  stroke-dasharray: 4 4;
}

.unclipped-line {
  fill: none;
  stroke: var(--vp-c-text-3);
  stroke-width: 1.6;
  stroke-dasharray: 7 5;
}

.clipped-line {
  fill: none;
  stroke: var(--vp-c-brand-1);
  stroke-width: 3;
}

.probe-line {
  stroke: var(--course-amber);
  stroke-width: 1.6;
}

.probe-dot {
  fill: var(--course-amber);
  stroke: var(--vp-c-bg);
  stroke-width: 2;
}

.probe-text {
  fill: var(--vp-c-text-1);
  font-size: 12px;
  font-weight: 700;
}

.reading {
  margin: 12px 0 0;
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  line-height: 1.7;
}

.reading strong {
  color: var(--course-amber);
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
}
</style>
