# Chain of Thought (CoT) 模式实现

本文档介绍了 OpenManus 中的 Chain of Thought (CoT) 模式实现及使用方法。

## 什么是 Chain of Thought

Chain of Thought（思维链）是一种提示大语言模型展示其推理过程的方法，通过显示模型如何一步步思考问题，从而得出更准确的结论。与 ReAct（Reasoning and Acting）模式不同，CoT 专注于思考过程的展示，而不涉及工具的使用和执行。

## CoT 与 ReAct 的区别

|特性|CoT（思维链）|ReAct（推理行动）|
|---|---|---|
|主要目的|展示详细的思考过程|执行操作并与环境交互|
|使用工具|不使用工具|使用各种工具执行操作|
|步骤数|通常只需一步完成|多步骤循环（思考-行动-观察）|
|适用场景|复杂推理、数学问题、逻辑题|任务执行、信息检索、编程等|
|输出重点|思考过程和最终答案|工具调用结果和执行状态|

## 实现概述

CoT 模式的实现包括以下文件：

1. `app/agent/cot.py` - CoT 代理实现
2. `app/prompt/cot.py` - CoT 特定的提示模板

CoT Agent 继承自 BaseAgent，实现了一个简单的思考步骤，不包含工具调用。

## 使用方法

### 方法一：使用选择器脚本

使用项目根目录的 `select_agent.py` 脚本，可以选择不同的代理模式：

```bash
python select_agent.py
```

然后选择 "2" 使用 CoT 模式。

### 方法二：直接运行示例

项目包含两个示例脚本：

1. 基本 CoT 示例：

```bash
python examples/cot_agent_example.py
```

2. 多步骤 CoT 对话示例：

```bash
python examples/multi_step_cot_example.py
```

### 方法三：在代码中使用

在你的代码中使用 CoT 代理：

```python
from app.agent import CoTAgent

async def main():
    agent = CoTAgent()
    result = await agent.run("你的问题或提示")
    print(result)
```

## 定制 CoT

要定制 CoT 的行为，可以修改以下部分：

1. 修改系统提示：编辑 `app/prompt/cot.py` 中的 `SYSTEM_PROMPT`
2. 调整输出格式：通过修改系统提示中的指示来改变输出格式
3. 增加特定领域知识：根据需要在系统提示中添加特定领域的专业知识或指导

## 高级用法

### 多步骤推理

对于需要多步推理的复杂问题，可以设置更多的步骤：

```python
agent = CoTAgent(max_steps=3)
```

然后使用 `step()` 方法进行逐步推理，在每步之间可以添加新的用户输入。

### 与其他模式结合

可以考虑将 CoT 与其他模式结合使用，例如：

- 先使用 CoT 分析问题和制定计划
- 然后使用 ReAct 执行具体操作

## 故障排除

如果遇到以下问题：

1. **回答过于简短**：调整系统提示，强调需要详细的思考过程
2. **没有遵循格式**：检查系统提示中的格式指导，确保清晰明确
3. **性能问题**：CoT 适合较复杂的推理任务，对于简单问题可能显得冗长
