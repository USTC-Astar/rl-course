name: 新课提案
description: 建议新增课程、实验或交互组件
labels: [enhancement, 内容]
body:
  - type: textarea
    id: gap
    attributes:
      label: 它填补哪个缺口
      description: 现有 22 课哪里讲不到、读者会卡在哪里。
    validations:
      required: true
  - type: textarea
    id: plan
    attributes:
      label: 内容大纲设想
      description: 一条主线 + 3~6 个核心概念；能否配真实实验（环境、种子、预期输出）。
  - type: input
    id: prereq
    attributes:
      label: 前置课程
      placeholder: "例：第 5 课 PPO、第 8 课 GAE"
  - type: textarea
    id: refs
    attributes:
      label: 参考资料
      description: 教材章节、论文、优秀实现链接。
