import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import './custom.css'

import AlgorithmLab from './components/AlgorithmLab.vue'
import ChapterQuiz from './components/ChapterQuiz.vue'
import LearningPath from './components/LearningPath.vue'
import TrainingCurve from './components/TrainingCurve.vue'
import ConceptDiagram from './components/ConceptDiagram.vue'
import NetworkDiagram from './components/NetworkDiagram.vue'
import MechanismDiagram from './components/MechanismDiagram.vue'
import QLearningLab from './components/QLearningLab.vue'
import ClipObjectiveLab from './components/ClipObjectiveLab.vue'
import OverestimateLab from './components/OverestimateLab.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('AlgorithmLab', AlgorithmLab)
    app.component('ChapterQuiz', ChapterQuiz)
    app.component('LearningPath', LearningPath)
    app.component('TrainingCurve', TrainingCurve)
    app.component('ConceptDiagram', ConceptDiagram)
    app.component('NetworkDiagram', NetworkDiagram)
    app.component('MechanismDiagram', MechanismDiagram)
    app.component('QLearningLab', QLearningLab)
    app.component('ClipObjectiveLab', ClipObjectiveLab)
    app.component('OverestimateLab', OverestimateLab)
  },
} satisfies Theme
