<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    file: string
    path: string
    title: string
    subtitle?: string
    color?: string
    target?: number
    badge?: string
  }>(),
  {
    subtitle: '曲线来自本地 Python 训练脚本生成的 JSON 数据。',
    color: '#2563eb',
    target: undefined,
    badge: '真实训练数据',
  },
)

const canvas = ref<HTMLCanvasElement | null>(null)
const values = ref<number[]>([])
const loading = ref(true)
const error = ref('')
let resizeObserver: ResizeObserver | null = null

const latest = computed(() => values.value.at(-1) ?? 0)
const best = computed(() => values.value.length ? Math.max(...values.value) : 0)
const recentAverage = computed(() => {
  const recent = values.value.slice(-Math.min(50, values.value.length))
  return recent.length ? recent.reduce((sum, value) => sum + value, 0) / recent.length : 0
})

function pickPath(payload: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((current, key) => {
    if (current && typeof current === 'object' && key in current) {
      return (current as Record<string, unknown>)[key]
    }
    return undefined
  }, payload)
}

function movingAverage(source: number[], windowSize: number): number[] {
  return source.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1)
    const window = source.slice(start, index + 1)
    return window.reduce((sum, value) => sum + value, 0) / window.length
  })
}

function drawSeries(
  context: CanvasRenderingContext2D,
  source: number[],
  color: string,
  width: number,
  minValue: number,
  maxValue: number,
  chartWidth: number,
  chartHeight: number,
  padding: number,
) {
  if (source.length < 2) return
  context.beginPath()
  source.forEach((value, index) => {
    const x = padding + (index / (source.length - 1)) * chartWidth
    const ratio = (value - minValue) / Math.max(maxValue - minValue, 1e-6)
    const y = padding + chartHeight - ratio * chartHeight
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.strokeStyle = color
  context.lineWidth = width
  context.stroke()
}

function draw() {
  const element = canvas.value
  if (!element || !values.value.length) return
  const width = Math.max(element.clientWidth, 320)
  const height = 260
  const ratio = window.devicePixelRatio || 1
  element.width = width * ratio
  element.height = height * ratio
  const context = element.getContext('2d')
  if (!context) return
  context.scale(ratio, ratio)

  const padding = 34
  const chartWidth = width - padding * 2
  const chartHeight = height - padding * 2
  const smoothed = movingAverage(values.value, Math.max(5, Math.round(values.value.length / 40)))
  const candidates = props.target === undefined ? values.value : [...values.value, props.target]
  const minValue = Math.min(...candidates)
  const maxValue = Math.max(...candidates)

  context.clearRect(0, 0, width, height)
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-divider')
  context.lineWidth = 1
  for (let line = 0; line <= 4; line += 1) {
    const y = padding + (line / 4) * chartHeight
    context.beginPath()
    context.moveTo(padding, y)
    context.lineTo(width - padding, y)
    context.stroke()
  }

  if (props.target !== undefined) {
    const targetRatio = (props.target - minValue) / Math.max(maxValue - minValue, 1e-6)
    const targetY = padding + chartHeight - targetRatio * chartHeight
    context.setLineDash([6, 5])
    context.strokeStyle = '#059669'
    context.beginPath()
    context.moveTo(padding, targetY)
    context.lineTo(width - padding, targetY)
    context.stroke()
    context.setLineDash([])
  }

  drawSeries(context, values.value, `${props.color}55`, 1.2, minValue, maxValue, chartWidth, chartHeight, padding)
  drawSeries(context, smoothed, props.color, 3, minValue, maxValue, chartWidth, chartHeight, padding)

  context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--vp-c-text-2')
  context.font = '12px system-ui'
  context.fillText(maxValue.toFixed(0), 4, padding + 4)
  context.fillText(minValue.toFixed(0), 4, height - padding + 4)
}

function observeCanvas() {
  if (!canvas.value || resizeObserver) return
  resizeObserver = new ResizeObserver(draw)
  resizeObserver.observe(canvas.value)
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(`/data/${props.file}`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    const selected = pickPath(payload, props.path)
    if (!Array.isArray(selected)) throw new Error(`找不到数组字段 ${props.path}`)
    values.value = selected.map(Number).filter(Number.isFinite)
    await nextTick()
    draw()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    loading.value = false
    // canvas 位于 v-else 中，必须等 loading=false 后的下一次渲染才能取得真实节点。
    await nextTick()
    observeCanvas()
    draw()
  }
}

onMounted(() => {
  load()
})

onBeforeUnmount(() => resizeObserver?.disconnect())
watch(() => [props.file, props.path], load)
</script>

<template>
  <section class="training-curve">
    <header>
      <div>
        <h3>{{ title }}</h3>
        <p>{{ subtitle }}</p>
      </div>
      <span class="data-badge">{{ badge }}</span>
    </header>
    <p v-if="loading" class="status">正在读取训练数据……</p>
    <p v-else-if="error" class="status error">读取失败：{{ error }}</p>
    <template v-else>
      <canvas ref="canvas" :aria-label="`${title}折线图`" />
      <div class="metrics">
        <article><small>最后值</small><strong>{{ latest.toFixed(1) }}</strong></article>
        <article><small>最近平均</small><strong>{{ recentAverage.toFixed(1) }}</strong></article>
        <article><small>历史最好</small><strong>{{ best.toFixed(1) }}</strong></article>
        <article><small>数据点</small><strong>{{ values.length }}</strong></article>
      </div>
    </template>
  </section>
</template>

<style scoped>
.training-curve {
  margin: 22px 0;
  padding: 19px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
}

header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

h3,
p {
  margin: 0;
}

header p {
  margin-top: 5px;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.data-badge {
  flex: none;
  padding: 5px 9px;
  border-radius: 999px;
  color: #047857;
  background: rgba(16, 185, 129, 0.13);
  font-size: 0.76rem;
  font-weight: 700;
}

canvas {
  display: block;
  width: 100%;
  height: 260px;
  margin-top: 14px;
  border-radius: 12px;
  background: var(--vp-c-bg);
}

.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 9px;
  margin-top: 12px;
}

.metrics article {
  padding: 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg);
}

small,
strong {
  display: block;
}

small {
  color: var(--vp-c-text-2);
}

strong {
  margin-top: 3px;
  font-size: 1.08rem;
}

.status {
  margin-top: 16px;
  color: var(--vp-c-text-2);
}

.error {
  color: #dc2626;
}

@media (max-width: 620px) {
  header {
    display: block;
  }

  .data-badge {
    display: inline-block;
    margin-top: 9px;
  }

  .metrics {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
