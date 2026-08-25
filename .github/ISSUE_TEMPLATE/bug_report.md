name: 内容勘误
description: 课程正文、公式、图解、代码或自测题的内容错误
labels: [bug, 内容]
body:
  - type: input
    id: lesson
    attributes:
      label: 课程与小节
      placeholder: "例：第 8 课，第 3 节「n 越大越好吗？」"
    validations:
      required: true
  - type: textarea
    id: what
    attributes:
      label: 错误内容
      description: 现在写的是什么、你认为应该是什么、依据是什么（教材/论文/计算过程）。
    validations:
      required: true
  - type: textarea
    id: context
    attributes:
      label: 补充信息（可选）
      description: 截图、复现命令与输出等。
