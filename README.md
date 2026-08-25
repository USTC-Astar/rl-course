<p align="center">
  <img src="web/docs/public/course-mark.svg" width="92" alt="强化学习项目式课程标志">
</p>

<h1 align="center">强化学习项目式学习实验室</h1>

<p align="center">
  <em>Reinforcement Learning Learning Lab — a hands-on Chinese RL course: real local training, interactive web lessons.</em>
</p>

<p align="center">
<!-- 把 YOUR_USERNAME 替换为你的 GitHub 用户名后，徽章才会正常显示 -->
<a href="https://github.com/YOUR_USERNAME/rl-learning-lab/actions/workflows/ci.yml"><img src="https://github.com/YOUR_USERNAME/rl-learning-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
<img src="https://img.shields.io/badge/lessons-22-blue" alt="lessons: 22">
<img src="https://img.shields.io/badge/python-3.11+-blue" alt="python: 3.11+">
<img src="https://img.shields.io/badge/node-18+-blue" alt="node: 18+">
</p>

这是一个面向初学者的“**Python 本地训练 + VitePress 交互网页讲解**”中文强化学习课程。课程从贝尔曼方程的地基讲起，先说明旧方法卡在哪里，再用生活类比、结构图解、数字算例、可交互实验和可运行代码逐步拆解；公式使用 MathJax 渲染，并配有[术语与符号速查表](web/docs/reference/glossary.md)。

## ✨ 主要功能

- **22 课独立页面**（第 0 课 + 第 1—21 课），带侧边栏、本页目录、全文搜索和前后课导航。
- **每课完整学习闭环**：学习目标、前置知识、直觉类比、结构图解、公式拆解、可运行代码、常见误区、动手练习、自测和小结。
- **浏览器内实时运行的算法实验台**：网格世界 Q 学习逐步可视化（第 1 课）、PPO 裁剪目标曲线（第 5 课）、高估蒙特卡洛模拟（第 9 课）——全部真实计算，非动画演示。
- **19 张 SVG 结构图解**：DQN 双网络、Dueling 双支路、SAC 部件、资格迹、RLHF 流水线、域随机化桥梁等，跟随明暗主题。
- **真实数据**：前 7 课使用 PyTorch 与 Gymnasium 真实训练；第 0、18—21 课的表格级实验全部真实运行、固定随机种子、写入课程前先经脚本实际输出校验。
- **分步推导框与伪代码框**：策略梯度定理、SAC 最大熵目标、GAE 推导；Q 学习、PPO 伪代码。
- **110 道自测题**（每课 5 题，正确答案位置打散），Python 核心计算有 45 个 pytest 测试覆盖。
- **真实失败案例库**：第 21 课注入 4 种经典 bug 训练出的失败曲线，用于“看曲线下诊断”。

## 📚 课程目录

### 第零阶段：把地基打牢

0. MDP、贝尔曼方程与动态规划

### 第一阶段：先学会看价值

1. 网格世界与 Q 学习
2. CartPole 与深度 Q 网络（Deep Q-Network, DQN）
3. REINFORCE 策略梯度
4. 行动者—评论家（Actor-Critic）

### 第二阶段：让策略更稳

5. 近端策略优化（Proximal Policy Optimization, PPO）
6. 连续动作与高斯策略
7. 软行动者—评论家（Soft Actor-Critic, SAC）

### 第三阶段：升级价值网络

8. 多步回报与 TD(λ)
9. Double DQN
10. Dueling DQN
11. 优先经验回放（Prioritized Experience Replay, PER）
12. Rainbow DQN

### 第四阶段：走向真实问题

13. 双延迟深度确定性策略梯度（Twin Delayed DDPG, TD3）
14. Dyna-Q 学习与规划
15. 多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）
16. 离线强化学习（Offline Reinforcement Learning, Offline RL）
17. 仿真到现实（Simulation-to-Reality, Sim-to-Real）

### 第五阶段：补全拼图

18. 蒙特卡洛与 SARSA
19. 探索方法进阶
20. 模仿学习与人类反馈对齐
21. 调试强化学习

## 🚀 快速开始

```bash
git clone https://github.com/YOUR_USERNAME/rl-learning-lab.git
cd rl-learning-lab

# 1. 安装隔离的 Python 环境与依赖（不污染系统 Python）
./scripts/bootstrap.sh

# 2. 安装网页依赖
cd web && npm install

# 3. 启动课程网站
npm run dev          # 打开 http://127.0.0.1:5173
```

不想训练也能学：仓库自带全部课程数据（`web/data/*.json`），clone 后直接 `npm run dev` 即可看到带曲线和实验台的完整课程。

## 🧪 训练与生成数据

生成全部课程数据（前 7 课真实训练 + 其余各课轻量实验）：

```bash
# 在项目根目录执行
./scripts/train_all.sh
```

只重新生成轻量实验数据（第 0、8—21 课，CPU 秒级）：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/generate_foundations.py
./scripts/run_python.sh scripts/generate_advanced_lessons.py
./scripts/run_python.sh scripts/generate_td_variants.py
./scripts/run_python.sh scripts/generate_exploration.py
./scripts/run_python.sh scripts/generate_imitation.py
./scripts/run_python.sh scripts/generate_debug_lessons.py
```

单独训练某一课（详见 [本地环境说明](web/docs/practice/local-setup.md)）：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_gridworld.py --episodes 3000
./scripts/run_python.sh scripts/train_cartpole.py --steps 150000
./scripts/run_python.sh scripts/train_policy_gradient.py --episodes 600
./scripts/run_python.sh scripts/train_actor_critic.py --steps 300000
./scripts/run_python.sh scripts/train_ppo.py --steps 120000
./scripts/run_python.sh scripts/train_continuous_ppo.py --steps 200000
./scripts/run_python.sh scripts/train_sac.py --steps 100000
```

训练脚本写入 `web/data/*.json`；VitePress 在开发或构建前自动同步数据，不需要手工复制。

## ✅ 测试与构建

```bash
# 在项目根目录执行：45 个 Python 测试
./scripts/run_python.sh -m pytest

# 构建静态网站到 web/dist/
cd web && npm run build
```

正常情况下 pytest 全部通过，VitePress 输出 `build complete`。CI 也会在每次推送时自动执行这两步（见 `.github/workflows/ci.yml`）。

## 📦 部署到 GitHub Pages（可选）

仓库自带部署工作流 `.github/workflows/deploy-pages.yml`：推送到 `main` 分支即自动构建并发布课程站。

1. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**；
2. 推送到 `main`，等待 Actions 完成后访问 `https://<用户名>.github.io/<仓库名>/`。

工作流会自动设置子路径 `base` 并关闭 `cleanUrls`（GitHub Pages 不支持无后缀 URL 重写），本地开发不受影响。

## 📁 目录结构

```text
rl-learning-lab/
├── .github/                     # CI、Pages 部署、Issue/PR 模板
├── legacy/                      # 已归档的旧版单页应用（仅存档）
├── src/rl_learning_lab/         # 强化学习算法与核心计算
├── scripts/                     # 环境安装、训练、数据生成和服务脚本
├── tests/                       # 45 个自动化测试
├── artifacts/                   # 训练产物（.pt 不入库，可本地重新训练）
├── web/
│   ├── data/                    # 训练和实验生成的课程数据（JSON）
│   ├── docs/                    # 22 课 Markdown 与 Vue 交互组件
│   │   └── .vitepress/          # 站点配置、主题、图解与实验台组件
│   ├── scripts/                 # 网页数据同步脚本
│   └── package.json             # VitePress 依赖与构建命令
├── pyproject.toml               # Python 依赖（uv 管理）
├── CONTRIBUTING.md              # 贡献指南
└── CHANGELOG.md                 # 版本历史
```

## 🛠 技术栈与硬件

| 层 | 技术 |
| --- | --- |
| 算法与训练 | Python 3.11+、PyTorch 2.x、Gymnasium 1.x、MuJoCo、NumPy |
| 课程网站 | VitePress 1.6 + Vue 3、markdown-it-mathjax3（公式渲染）、原生 SVG（图解） |
| 测试 | pytest |

- CPU：最低 4 核，推荐 6～12 核；不需要独立显卡（默认安装 CPU 版 PyTorch）。
- 内存：最低 8 GB，推荐 16 GB；磁盘预留 3～6 GB。
- Node.js 18+（推荐 20）；系统 Linux / macOS / WSL。

网页框架选择 MIT 许可的 VitePress：原生支持 Markdown、Vue 交互组件、本地搜索和静态构建，与本课程“正文 + 交互实验台”的形态最匹配。

## ⚠️ 已知限制

- 第 1—7 课包含完整训练器；第 0、8—21 课以核心算法函数和轻量机制实验为主，**不冒充大型基准训练结果**。
- 第 13 课提供 TD3 的关键目标计算和动作平滑，完整连续训练可复用现有 SAC 骨架继续扩展。
- 第 15 课使用小型双智能体矩阵博弈讲解非平稳性，不等同于复杂机器人编队基准。
- 第 17 课的鲁棒性曲线是机制模拟，不构成真实设备安全证明；真实部署必须重新做参数测量、扰动评估和安全审查。

## 🤝 贡献

欢迎纠错、补充课程和改进实验！请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（含新课写作规范、数据生成流程和 PR 检查清单）。

## 📄 许可

[MIT](LICENSE) © rl-learning-lab 贡献者。课程内容可自由用于学习和教学，转载请注明出处。
