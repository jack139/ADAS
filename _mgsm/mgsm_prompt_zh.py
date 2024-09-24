# -*- coding: utf-8 -*-

import json

QWEN2_5_CODER_7B = ("Qwen2.5-Coder-7B-Instruct", 'http://localhost:8001/v1')
QWEN2_5_7B = ("Qwen2.5-7B-Instruct", 'http://localhost:8000/v1')
QWEN2_5_14B = ("Qwen2.5-14B-Instruct", 'http://localhost:8000/v1')
QWEN2_5_32B = ("Qwen2.5-32B-Instruct", 'http://localhost:8000/v1')

META_AGENT = QWEN2_5_CODER_7B
BASE_AGENT = QWEN2_5_7B


EXAMPLE = {
    "thought": "**见解：**\n您对下一个有趣的智能体的见解。\n**总体思路：**\n您的推理和智能体设计背后的总体概念。\n**实施：**\n逐步描述实施过程。",
    "name": "您建议的智能体名称",
    "code": """def forward(self, taskInfo):
    # 您的代码在这里
    return answer
"""
}

COT = {
    "thought": "通过鼓励 LLM 逐步思考而不是直接输出答案，CoT推理能够通过中间步骤解决复杂问题。这种做法提高了模型处理需要更深入推理的任务的能力，并提供了对其决策过程的洞察。",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # 思维链 (CoT) 的指令
    # 这是让LLM在解决任务之前能够一步步思考的重要实践。
    cot_instruction = "请逐步思考然后解决任务。"

    # 实例化一个专门用于 CoT 的新 LLM 智能体
    # 为了让LLM在回答之前思考，我们需要设置一个额外的输出字段'thinking'。
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # 准备 CoT 智能体的输入
    # 输入应该是 Info 列表，第一个通常是 taskInfo
    cot_agent_inputs = [taskInfo]

    # 获取 CoT 智能体的响应
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)

    # 仅返回最终答案
    return answer
"""
}

COT_SC = {"thought": "虽然 LLM 可以得出正确答案，但其推理方式可能有所不同。通过在temperature设置下反复询问同一个问题，我们可以生成不同的推理路径。然后，我们将这些思维链 (CoT) 智能体的多个答案组合起来，通过集成产生更准确的最终答案。",
          "name": "Self-Consistency with Chain-of-Thought",
          "code": """def forward(self, taskInfo):
    # 逐步推理的指令
    cot_instruction = "请逐步思考然后解决任务。"
    N = 5 # CoT 智能体的数量

    # 初始化多个 CoT 智能体，使其具有更高的temperature，以实现不同的推理
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    # 多数投票函数用于选择最常见的答案
    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]
    
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    # 整合来自多个 CoT 智能体的答案
    answer = majority_voting(possible_answers)
    return answer  
"""
          }

Reflexion = {
    "thought": "为了提高其性能，LLM 可以根据反馈反复改进其答案。通过反思之前的尝试并结合反馈，该模型可以改进其推理并提供更准确的解决方案。",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    # 初步推理的指令
    cot_initial_instruction = "请逐步思考然后解决任务。"

    # 反思以前的尝试并提出改进意见的指令
    cot_reflect_instruction = "根据之前的尝试和反馈，仔细考虑在最近的尝试中可能出错的地方。利用之前尝试中的经验，尝试更好地解决任务。"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # 提供反馈和纠正答案的指令
    critic_instruction = "请检查以上答案，并批评其中可能错误的地方。如果你绝对确定它是正确的，请在'correct'中输出'True'。"
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')
    
    N_max = 5 # 最大尝试次数

    # 初次尝试
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # 从评论者那里获得反馈和正确的状态
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # 为下一次迭代的输入添加反馈
        cot_inputs.extend([thinking, answer, feedback])

        # 反思之前的尝试并完善答案
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
"""
}

LLM_debate = {
    "thought": "通过让不同的LLM互相辩论，我们可以利用他们不同的观点来找到更好的任务解决方案。",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    # 初步推理的指令
    debate_initial_instruction = "请逐步思考然后解决任务。"

    # 根据其他智能体的解决方案进行讨论和更新解决方案的指令
    debate_instruction = "鉴于其他智能体对问题的解决方案，请将他们的意见视为补充建议。请仔细思考并提供更新的答案。"
    
    # 初始化具有不同角色和适度temperature的辩论智能体，以进行不同的推理
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['数学教授', '小学教师', '数学爱好者']]

    # 根据所有辩论和解决方案做出最终决策的指令
    final_decision_instruction = "综合以上思考和回答，仔细推理，给出最终答案。"
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)

    max_round = 2 # 辩论最大轮数
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # 进行辩论
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # 根据所有辩论结果和解决方案做出最终决定
    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer
"""
}

Take_a_step_back = {"thought": "让 LLM 首先思考解决此任务所涉及的原理，这可能会有所帮助。通过理解底层原理，模型可以更好地推理问题并提供更准确的解决方案。",
                    "name": "Step-back Abstraction",
                    "code": """def forward(self, taskInfo):
        # 理解任务所涉及原理的指令
        principle_instruction = "解决这个任务涉及哪些物理、化学或生物原理和概念？首先一步一步思考。然后列出所有涉及的原理并解释它们。"
        
        # 根据原则解决任务的指令
        cot_instruction = "给出问题以及问题背后涉及的原理，一步步思考，然后解决任务。"
        
        # 实例化 LLM 智能体
        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
        
        # 获取任务中涉及的原则
        thinking, principle = principle_agent([taskInfo], principle_instruction)

        # 运用原则解决任务
        thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)
        return answer
"""
                    }

QD = {"thought": "与质量多样性方法类似，让 LLM 生成多个不同的有趣解决方案可能会有所帮助。通过鼓励模型探索不同的推理路径，我们可以增加找到最佳解决方案的机会。",
      "name": "Quality-Diversity",
      "code": """def forward(self, taskInfo):
    # 初步推理的指令
    cot_initial_instruction = "请逐步思考然后解决任务。"

    # 给出不同答案的指令
    qd_instruction = "鉴于以前的尝试，尝试想出另一种有趣的方法来解决任务。"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # 根据收集到的推理和答案进行最终决策的指令
    final_decision_instruction = "给出上述所有解决方案，仔细推理并给出最终答案。"
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3 # 最大尝试次数

    # 初次尝试
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    # 将答案添加到可能的答案列表中
    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        # 反思之前的尝试并产生另一个有趣的答案
        cot_inputs.extend([thinking, answer])

        # 生成另一个有趣的答案
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    # 根据所有生成的答案做出最终决定
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
"""
      }

Role_Assignment = {"thought": "与 Auto-GPT 和专家提示类似，我们可以在设计中使用动态控制流让智能体决定我们应该使用哪个专家。",
                   "name": "Dynamic Assignment of Roles",
                   "code": """def forward(self, taskInfo):
        # 逐步推理的指令
        cot_instruction = "请逐步思考然后解决任务。"
        expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['数学教授', '小学教师', '数学爱好者', 'Helpful Assistant']]

        # 将任务分配给适当专家的说令
        routing_instruction = "给出任务后，请选择一位专家来回答问题。选择范围：数学教授、小学教师、数学爱好者。"
        routing_agent = LLMAgentBase(['choice'], 'Routing agent')

        # 选择专家来安排任务
        choice = routing_agent([taskInfo], routing_instruction)[0]

        if '教授' in choice.content.lower():
            expert_id = 0
        elif '教师' in choice.content.lower() or '老师' in choice.content.lower():
            expert_id = 1
        elif '爱好者' in choice.content.lower():
            expert_id = 2
        else:
            expert_id = 3 # Default to helpful assistant

        thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
        return answer
"""
                   }

system_prompt = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """# 概述
您是一位专业的机器学习研究员，正在测试各种智能体系统。您的目标是设计这些系统中的提示和控制流等构建块来解决复杂的任务。您的目标是设计一个在多语言小学数学基准 (MGSM) 上表现良好的最佳智能体，该基准评估各种语言的数学问题解决能力，以确保广泛有效的多语言性能。

## 来自 MGSM 的一个示例问题：

**问题**： 请解答这道数学题。 \n\n在我家附近，宠物兔子的数量比宠物狗和猫的总和还要少 12 只。如果每只狗有 2 只猫，狗的数量为 60 只，那么附近一共有多少只宠物？

**答案（未给出）**： 348

# 实用程序代码：

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI(
                    base_url='http://localhost:8000/v1',
                    timeout=60,
                    max_retries=3,
                    api_key="fake_key",
)

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    \"""
    Function to get JSON response from GPT model.
    
    Args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.
    
    Returns:
    - dict: The JSON response.
    \"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    #print(f\"gpt\\n--> {system_message} | {msg}\\n<-- {json_dict}\\n\")
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.
    
    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - model (str): Model to be used. (option. Keep it default.)
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='[BASE_AGENT]', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        Generates a prompt for the LLM.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        
        Returns:
        - tuple: System prompt and user prompt.

        An example of a generated prompt:
        ""
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.
        ""
        \"""
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.
        
        Returns:
        - output_infos (list[Info]): Output information.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - taskInfo (Info): Task information.
        
        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string of answers.
        \"""
        pass
```
# 已发现的智能体架构存档
以下是已发现的智能体架构存档：

[存档]

适应度值是验证问题集正确率的中位数和 95% Bootstrap 置信区间。您的目标是最大化“适应度”。

# 输出说明及示例:
第一个关键点应该是("thought")，它应该记录你设计下一个功能的思考过程。在"thought"部分，首先推理下一个值得尝试的有趣智能体是什么，然后描述你的推理和智能体设计背后的总体概念，最后详细说明实施步骤。
第二个键("name")对应于您的下一个智能体架构的名称。
最后，最后一个键("code")对应于您想要尝试的 Python 代码中的 forward() 函数。您必须在“code”中编写完整的代码：您的代码将成为整个项目的一部分，因此请实现完整、可靠、可重复使用的代码片段。

以下是下一个智能体架构的输出格式的示例：

[示例]

您必须使用上面使用一致的函数接口。您需要为各种 LLM 智能体指定指令、输入信息和所需的输出字段，以执行其架构的特定部分。
此外，设置 LLM 的 role 和 temperature 以进一步控制 LLM 的响应可能会有所帮助。请注意，LLMAgentBase() 将自动解析输出并返回“Infos”列表。您可以通过 Infos.content 获取内容。
如果您认为需要，请不要忘记将 taskInfo 输入到 LLM，否则 LLM 将不知道该任务。

## 错误的实施示例：
以下是您可能会犯的一些错误：

1. 这是错误的： ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
使用“Info('feedback', 'Critic Agent', thinking, 0)”是错误的。LLMAgentBase 返回的“feedback”已经是 Info。

2. 这是错误的： ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
首先，len(feedback_info) 将不起作用。
其次，您永远不应返回错误消息。您应该始终返回您能得到的最佳答案。
第三，您永远不应在代码中打印任何内容。
最后，再次强调，不要自己创建 Info 对象。

3. 这是错误的： ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '\n'.join(all_thinking)
aggregated_answers = '\n'.join(all_answers)
```
您不应该自己从 Info 对象中提取内容。您应该直接使用 Info 对象。如果您想聚合内容，您应该将这些 Info 对象放入列表中，然后使用该列表作为下一个 LLM 智能体的输入。

4. 这是错误的： ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
您不应该自己提取最终答案。您应该直接返回答案信息。此外，您应该始终返回您能得到的最佳答案。
正确示例： ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer
```

# 你的任务
您非常熟悉 LLM 提示技术和文献中的 LLM 智能体工作。您的目标是通过提出有趣的新智能体来最大化“适应度”。
仔细观察发现的智能体架构并思考从中可以学到什么见解、教训或垫脚石。
发挥创造力，思考下一个值得尝试的有趣智能体架构。我们鼓励您从相关的 LLM 智能体论文或其他研究领域的学术论文中汲取灵感。
利用从存档中学到的知识和从学术文献中得到的灵感来设计下一个有趣的智能体架构。
跳出框框思考。
"""

Reflexion_prompt_1 = f""""[示例]仔细审查提议的新架构并反思以下几点："

1. **有趣性**：评估您提出的架构与存档中现有的方法相比是否有趣或具有创新性。如果您确定所提出的架构不有趣，请建议一种解决这些缺点的新架构。
- 确保检查所提出的架构与以前的尝试之间的差异。
- 仔细比较提案和档案中的架构，包括它们在实施中的实际差异。
- 确定当前架构是否具有创新性。
- 运用批判性思维！

2. **实施错误**：找出您在实施过程中可能犯下的任何错误。仔细检查代码，调试您发现的任何问题，并提供更正后的版本。记得在提示中勾选“## 错误的实施示例”。

3. **改进**：根据所提议的架构，建议改进详细实现，以提高其性能或效率。在此步骤中，重点是在不改变整体设计框架的情况下改进和优化现有实现，除非您想在当前架构不有趣的情况下提出不同的架构。
- 仔细观察实现是否真正做了它应该做的事情。
- 检查实现中是否有冗余代码或不必要的步骤。用有效的实现替换它们。
- 尽量避免实现与之前的智能体太相似。

然后，您需要改进或修改实现，或者基于反射实现新提出的架构。

你的回复应该按照以下方式组织：

"reflection"：提供您对架构有趣之处的想法，识别实施中的任何错误，并提出改进建议。

"thought"：修改您之前的提案或提出新的架构（如有必要），使用与示例响应相同的格式。

"name"：为修订或新的架构提供名称。（不要在名称中添加“新”或“改进”等字词。）

"code"：提供更正后的代码或改进的实施。确保您确实在此代码中实施了修复和改进。
"""

Reflexion_prompt_2 = """使用“##错误的实现示例”部分中的提示，进一步修改代码。
您的回复应按以下方式组织：
将您新的反思思维放在"reflection"中。重复之前的"thought"和"name"，并在"code"中更新更正后的代码版本。
"""


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]


def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol, ensure_ascii=False) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[BASE_AGENT]", BASE_AGENT[0]).replace("[存档]", archive_str)
    prompt = prompt.replace("[示例]", json.dumps(EXAMPLE, ensure_ascii=False))

    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "这是您之前尝试过的智能体：\n" + json.dumps(prev_example, ensure_ascii=False) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[示例]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[示例]", "")
    return r1, Reflexion_prompt_2
