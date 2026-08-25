import { defineConfig } from 'vitepress'

const lesson = (number: number, title: string) => ({
  text: `第 ${number} 课：${title}`,
  link: `/lesson${String(number).padStart(2, '0')}/`,
})

// 部署到 GitHub Pages 项目页（https://<user>.github.io/<repo>/）时，
// 由 CI 设置 BASE_PATH=/<repo>；本地与自定义域名部署保持默认 '/'。
// GitHub Pages 不支持无后缀 URL 重写，CI 部署时设置 CLEAN_URLS=0 关闭。
const base = process.env.BASE_PATH || '/'
const cleanUrls = process.env.CLEAN_URLS !== '0'

export default defineConfig({
  lang: 'zh-CN',
  title: '强化学习项目式课程',
  description: '从贝尔曼方程到仿真到现实的中文交互式强化学习课程',
  base,
  cleanUrls,
  outDir: '../dist',
  markdown: {
    math: true,
  },
  head: [
    ['meta', { name: 'theme-color', content: '#2563eb' }],
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1' }],
    ['link', { rel: 'icon', href: '/course-mark.svg', type: 'image/svg+xml' }],
  ],
  themeConfig: {
    logo: '/course-mark.svg',
    nav: [
      { text: '课程首页', link: '/' },
      {
        text: '课程章节',
        items: [
          { text: '理论基础（0、18）', link: '/lesson00/' },
          { text: '基础与价值学习（1—4）', link: '/lesson01/' },
          { text: '策略与连续控制（5—7）', link: '/lesson05/' },
          { text: 'DQN 进阶工具箱（8—12）', link: '/lesson08/' },
          { text: '高级专题与工程（13—17、19—21）', link: '/lesson13/' },
        ],
      },
      { text: '术语与符号表', link: '/reference/glossary' },
      { text: '本地运行', link: '/practice/local-setup' },
    ],
    sidebar: [
      {
        text: '开始学习',
        items: [
          { text: '课程首页', link: '/' },
          { text: '术语与符号速查', link: '/reference/glossary' },
          { text: '本地环境与硬件', link: '/practice/local-setup' },
        ],
      },
      {
        text: '第零阶段：把地基打牢',
        collapsed: false,
        items: [
          { text: '第 0 课：MDP、贝尔曼方程与动态规划', link: '/lesson00/' },
        ],
      },
      {
        text: '第一阶段：先学会看价值',
        collapsed: false,
        items: [
          lesson(1, '网格世界与 Q 学习'),
          lesson(2, 'CartPole 与 DQN'),
          lesson(3, 'REINFORCE 策略梯度'),
          lesson(4, '行动者—评论家'),
        ],
      },
      {
        text: '第二阶段：让策略更稳',
        collapsed: false,
        items: [
          lesson(5, 'PPO 策略限速器'),
          lesson(6, '连续动作与高斯策略'),
          lesson(7, 'SAC 与最大熵学习'),
        ],
      },
      {
        text: '第三阶段：升级价值网络',
        collapsed: false,
        items: [
          lesson(8, '多步回报与 TD(λ)'),
          lesson(9, 'Double DQN'),
          lesson(10, 'Dueling DQN'),
          lesson(11, '优先经验回放'),
          lesson(12, 'Rainbow DQN'),
        ],
      },
      {
        text: '第四阶段：走向真实问题',
        collapsed: false,
        items: [
          lesson(13, 'TD3 连续控制'),
          lesson(14, 'Dyna-Q 学习与规划'),
          lesson(15, '多智能体强化学习'),
          lesson(16, '离线强化学习'),
          lesson(17, '仿真到现实'),
        ],
      },
      {
        text: '第五阶段：补全拼图',
        collapsed: false,
        items: [
          lesson(18, '蒙特卡洛与 SARSA'),
          lesson(19, '探索方法进阶'),
          lesson(20, '模仿学习与人类反馈对齐'),
          lesson(21, '调试强化学习'),
        ],
      },
    ],
    outline: {
      level: [2, 3],
      label: '本页目录',
    },
    search: {
      provider: 'local',
    },
    docFooter: {
      prev: '上一课',
      next: '下一课',
    },
    darkModeSwitchLabel: '切换主题',
    sidebarMenuLabel: '课程目录',
    returnToTopLabel: '返回顶部',
    footer: {
      message: '先建立直觉，再拆公式，最后运行代码。',
      copyright: '强化学习项目式课程',
    },
  },
})
