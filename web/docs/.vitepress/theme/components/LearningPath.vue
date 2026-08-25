<script setup lang="ts">
const stages = [
  {
    title: '第零阶段：把地基打牢',
    lessons: '第 0 课',
    color: '#0891b2',
    detail: 'MDP、回报、贝尔曼方程与动态规划：后面每个算法都在解同一组方程，先把方程本身搞懂。',
  },
  {
    title: '第一阶段：看懂价值',
    lessons: '第 1—4 课',
    color: '#2563eb',
    detail: '从 Q 表、DQN 到行动者—评论家，先弄清“这个状态和动作到底好不好”。可随时插读第 18 课对照 MC 与 SARSA。',
  },
  {
    title: '第二阶段：稳定更新',
    lessons: '第 5—7 课',
    color: '#7c3aed',
    detail: '进入 PPO、连续动作和 SAC，理解概率策略为什么既要进步，也不能一次改太猛。',
  },
  {
    title: '第三阶段：升级工具箱',
    lessons: '第 8—12 课',
    color: '#d97706',
    detail: '逐个解决回报传播慢、Q 值高估、样本利用低效等 DQN 常见问题。',
  },
  {
    title: '第四阶段：面向真实世界',
    lessons: '第 13—17 课',
    color: '#059669',
    detail: '学习 TD3、规划、多智能体、离线数据和仿真到现实，建立完整工程视角。',
  },
  {
    title: '第五阶段：补全拼图',
    lessons: '第 18—21 课',
    color: '#db2777',
    detail: '探索方法、模仿学习与 RLHF、调试指南：把地基、探索、奖励来源和排查习惯补成完整闭环。',
  },
]
</script>

<template>
  <section class="learning-path" aria-label="课程学习路线">
    <article v-for="(stage, index) in stages" :key="stage.title" :style="{ '--stage-color': stage.color }">
      <span>{{ index + 1 }}</span>
      <div>
        <small>{{ stage.lessons }}</small>
        <h3>{{ stage.title }}</h3>
        <p>{{ stage.detail }}</p>
      </div>
    </article>
  </section>
</template>

<style scoped>
.learning-path {
  display: grid;
  gap: 14px;
  margin: 22px 0 34px;
}

article {
  display: grid;
  grid-template-columns: 46px 1fr;
  gap: 15px;
  align-items: start;
  padding: 18px;
  border: 1px solid var(--vp-c-divider);
  border-left: 5px solid var(--stage-color);
  border-radius: 15px;
  background: var(--vp-c-bg-soft);
}

article > span {
  display: grid;
  width: 40px;
  height: 40px;
  place-items: center;
  border-radius: 12px;
  color: white;
  background: var(--stage-color);
  font-weight: 800;
}

small {
  color: var(--stage-color);
  font-weight: 700;
}

h3,
p {
  margin: 0;
}

h3 {
  margin-top: 2px;
  font-size: 1.06rem;
}

p {
  margin-top: 5px;
  color: var(--vp-c-text-2);
  line-height: 1.7;
}
</style>
