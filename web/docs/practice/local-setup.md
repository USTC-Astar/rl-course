---
title: 本地环境与硬件要求
---

# 本地环境与硬件要求

<div class="lesson-lead">
这套入门课程默认使用中央处理器（Central Processing Unit, CPU）训练。没有独立显卡也能完整学习；显卡只会缩短部分神经网络实验的等待时间。
</div>

## 1. 你的电脑需要什么配置？

| 项目 | 最低建议 | 更舒服的配置 | 说明 |
| --- | --- | --- | --- |
| CPU | 4 核 | 6～12 核 | 大多数环境交互本身由 CPU 完成 |
| 内存 | 8 GB | 16 GB | 经验回放池和 Python 环境会占用内存 |
| 显卡 | 不需要 | NVIDIA 6 GB 以上显存 | 本课程默认安装 CPU 版 PyTorch |
| 磁盘 | 3 GB 可用空间 | 6 GB | 包含 Python、PyTorch、MuJoCo 和训练产物 |
| Python | 3.11 | 3.11 | 项目已锁定依赖版本 |
| Node.js | 18 以上 | 20 | 用于构建 VitePress 教学网页 |

::: warning 关于显卡
CartPole、网格世界和 Pendulum 的网络很小。把它们搬到显卡后，数据来回传输的开销可能抵消加速，因此“有显卡”不等于这些小实验一定更快。
:::

## 2. 第一次安装

在项目根目录执行：

```bash
# 在项目根目录执行
./scripts/bootstrap.sh
cd web
npm install
```

这里发生了两件事：

1. `bootstrap.sh` 创建 Python 虚拟环境并安装 PyTorch、Gymnasium 和 MuJoCo。
2. `npm install` 安装静态站点生成器 VitePress，不会参与强化学习训练。

## 3. 启动网页

开发模式最适合边学边改：

```bash
cd web
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。修改 Markdown 或 Vue 组件后，页面会自动刷新。

构建静态网页并使用 Python 服务：

```bash
# 在项目根目录执行
./scripts/serve.sh
```

浏览器打开 `http://127.0.0.1:8000`。

## 4. 训练一个最小项目

先训练几秒钟的网格世界：

```bash
# 在项目根目录执行
./scripts/run_python.sh scripts/train_gridworld.py --episodes 500
```

正常情况下会看到“网格世界训练完成”，并更新 `web/data/gridworld.json`。重新启动或构建网页后，新数据会出现在第 1 课曲线中。

## 5. 运行全部已有训练

```bash
# 在项目根目录执行
./scripts/train_all.sh
```

这会依次训练前 7 课的真实项目。CPU 速度不同，完整训练可能需要数分钟到更久。第一次学习不必每次全部重训，可以直接使用仓库中已有 JSON 数据。

## 6. 常见问题

### 为什么网页显示旧数据？

VitePress 会在启动或构建前，把 `web/data/` 同步到 `web/docs/public/data/`。训练完成后重新执行 `npm run dev` 或 `npm run build` 即可。

### 为什么不用 Isaac Sim？

前 17 课重点是算法机制，不是高保真机器人建模。Isaac Sim 对显卡、驱动和显存要求更高，适合后续机械臂、移动机器人和视觉导航项目；本课程先用轻量环境降低学习成本。

### Windows 可以运行吗？

推荐使用适用于 Linux 的 Windows 子系统（Windows Subsystem for Linux, WSL）。项目脚本是 Bash 脚本，在 WSL 中与 Linux 使用方式一致。
