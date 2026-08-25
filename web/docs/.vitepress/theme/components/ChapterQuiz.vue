<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { questionBank, type Question } from '../quiz-bank'

const props = defineProps<{ lesson: string }>()

const questions = computed(() => questionBank[props.lesson] ?? questionBank['01'])
const selections = ref<Array<number | null>>([])
const submitted = ref(false)

function resetQuiz() {
  selections.value = questions.value.map(() => null)
  submitted.value = false
}

watch(questions, resetQuiz, { immediate: true })

const score = computed(() => selections.value.reduce(
  (total, selection, index) => total + Number(selection === questions.value[index].answer),
  0,
))
</script>

<template>
  <section class="chapter-quiz" aria-labelledby="quiz-title">
    <div class="quiz-heading">
      <div>
        <span>快速自测</span>
        <h3 id="quiz-title">第 {{ Number(lesson) }} 课理解检查</h3>
        <p>不是考背诵，而是检查你能否判断“为什么要用这个方法”。</p>
      </div>
      <strong v-if="submitted">{{ score }} / {{ questions.length }}</strong>
    </div>

    <fieldset v-for="(question, questionIndex) in questions" :key="question.text">
      <legend>{{ questionIndex + 1 }}. {{ question.text }}</legend>
      <label v-for="(option, optionIndex) in question.options" :key="option">
        <input
          v-model="selections[questionIndex]"
          type="radio"
          :name="`lesson-${lesson}-question-${questionIndex}`"
          :value="optionIndex"
        >
        <span>{{ option }}</span>
      </label>
      <p
        v-if="submitted"
        :class="selections[questionIndex] === question.answer ? 'answer-correct' : 'answer-wrong'"
      >
        {{ selections[questionIndex] === question.answer ? '回答正确。' : '需要再想一想。' }}
        {{ question.explanation }}
      </p>
    </fieldset>

    <div class="quiz-actions">
      <button
        type="button"
        class="primary"
        :disabled="selections.some((value) => value === null)"
        @click="submitted = true"
      >
        提交答案
      </button>
      <button type="button" @click="resetQuiz">重新作答</button>
    </div>
  </section>
</template>

<style scoped>
.chapter-quiz {
  margin: 24px 0 34px;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 17px;
  background: var(--vp-c-bg-soft);
}

.quiz-heading {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.quiz-heading span {
  color: var(--vp-c-brand-1);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.quiz-heading h3,
.quiz-heading p {
  margin: 0;
}

.quiz-heading h3 {
  margin-top: 2px;
}

.quiz-heading p {
  margin-top: 5px;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.quiz-heading > strong {
  color: var(--vp-c-brand-1);
  font-size: 1.35rem;
}

fieldset {
  display: grid;
  gap: 9px;
  margin: 18px 0;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
}

legend {
  padding: 0 6px;
  font-weight: 700;
}

fieldset label {
  display: flex;
  gap: 9px;
  align-items: flex-start;
  color: var(--vp-c-text-2);
  cursor: pointer;
}

input {
  margin-top: 6px;
  accent-color: var(--vp-c-brand-1);
}

fieldset p {
  margin: 6px 0 0;
  padding: 10px 11px;
  border-radius: 9px;
  font-size: 0.88rem;
}

.answer-correct {
  color: #047857;
  background: rgba(16, 185, 129, 0.11);
}

.answer-wrong {
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.1);
}

.quiz-actions {
  display: flex;
  gap: 10px;
}

button {
  min-height: 38px;
  padding: 0 15px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  cursor: pointer;
}

button.primary {
  border-color: var(--vp-c-brand-1);
  color: white;
  background: var(--vp-c-brand-1);
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}
</style>
