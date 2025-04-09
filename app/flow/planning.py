# 导入所需的标准库和第三方库
import json
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

# 导入项目内部模块
from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanStepStatus(str, Enum):
    """计划步骤状态的枚举类，继承自str和Enum，用于定义计划中每个步骤的可能状态"""

    # 定义四种基本状态
    NOT_STARTED = "not_started"  # 未开始
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"  # 已完成
    BLOCKED = "blocked"  # 被阻塞

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """获取所有可能的步骤状态值列表"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """获取表示活动状态（未开始或进行中）的状态值列表"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """获取状态到其标记符号的映射字典，用于可视化显示状态"""
        return {
            cls.COMPLETED.value: "[✓]",  # 已完成使用对勾标记
            cls.IN_PROGRESS.value: "[→]",  # 进行中使用箭头标记
            cls.BLOCKED.value: "[!]",  # 被阻塞使用感叹号标记
            cls.NOT_STARTED.value: "[ ]",  # 未开始使用空方框标记
        }


class PlanningFlow(BaseFlow):
    """一个用于管理任务规划和执行的流程类，继承自BaseFlow。
    该类负责创建计划、跟踪执行进度、分配执行器并管理整个任务的生命周期。
    """

    # 类属性定义，使用pydantic的Field进行类型和默认值管理
    llm: LLM = Field(default_factory=lambda: LLM())  # 语言模型实例，用于生成和理解计划
    planning_tool: PlanningTool = Field(
        default_factory=PlanningTool
    )  # 计划工具实例，用于管理计划的具体操作
    executor_keys: List[str] = Field(
        default_factory=list
    )  # 执行器的键列表，用于存储可用的执行器标识
    active_plan_id: str = Field(
        default_factory=lambda: f"plan_{int(time.time())}"
    )  # 当前活动计划的ID
    current_step_index: Optional[int] = None  # 当前执行步骤的索引

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        """初始化PlanningFlow实例

        参数:
            agents: 可以是单个代理、代理列表或代理字典，用于执行计划中的任务
            **data: 额外的配置参数
                - executors: 指定执行器列表（会被转换为executor_keys）
                - plan_id: 指定计划ID（会被转换为active_plan_id）
                - planning_tool: 可选的PlanningTool实例
        """
        # 处理executors参数，转换为executor_keys
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 设置计划ID（如果提供）
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 如果未提供planning_tool，则初始化一个新的实例
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 调用父类的初始化方法
        super().__init__(agents, **data)

        # 如果未指定executor_keys，则使用所有代理的键
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """获取适合当前步骤的执行器代理

        根据步骤类型选择合适的执行器。如果找不到匹配的执行器，则按照优先级顺序选择：
        1. 指定类型的执行器（如果step_type匹配某个代理）
        2. executor_keys列表中的第一个可用执行器
        3. 主要代理（作为后备选项）

        参数:
            step_type: 步骤类型，用于匹配特定的执行器

        返回:
            BaseAgent: 选中的执行器代理实例
        """
        # 如果提供了步骤类型且匹配某个代理，则使用该代理
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则使用第一个可用的执行器或回退到主要代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 如果没有找到合适的执行器，使用主要代理
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """执行计划流程

        这是整个计划流程的主要执行方法，负责：
        1. 创建初始计划（如果提供了输入文本）
        2. 循环执行计划中的每个步骤
        3. 处理执行过程中的异常情况
        4. 生成执行结果报告

        参数:
            input_text: 用于创建初始计划的输入文本

        返回:
            str: 执行结果的描述文本

        异常:
            ValueError: 当没有可用的主要代理时抛出
        """
        try:
            # 检查是否有可用的主要代理
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # 如果提供了输入文本，则创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划是否创建成功
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            result = ""
            while True:
                # 获取当前需要执行的步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多步骤或计划已完成，则退出循环
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 使用适当的执行器执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 检查执行器是否请求终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """创建初始计划

        使用语言模型和计划工具根据用户请求创建一个初始计划。如果计划创建失败，
        会创建一个包含基本步骤的默认计划。

        参数:
            request: 用户的请求文本，用于生成计划

        工作流程:
        1. 创建系统消息和用户消息
        2. 调用语言模型生成计划
        3. 处理语言模型的响应
        4. 如果计划创建失败，则创建默认计划
        """
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # 创建用于计划生成的系统消息
        system_message = Message.system_message(
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )

        # 创建包含用户请求的消息
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # 调用语言模型，使用PlanningTool生成计划
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理语言模型返回的工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析工具调用的参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    # 确保使用正确的计划ID
                    args["plan_id"] = self.active_plan_id

                    # 通过工具集合执行计划创建
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"Plan creation result: {str(result)}")
                    return

        # 如果上述过程失败，创建默认计划
        logger.warning("Creating default plan")

        # 使用工具集合创建包含基本步骤的默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """获取当前步骤信息

        解析当前计划以识别第一个未完成步骤的索引和信息。
        如果找不到活动步骤，则返回(None, None)。

        返回:
            tuple: (步骤索引, 步骤信息字典)
                - 如果没有活动步骤，两个值都为None
                - 步骤信息字典包含步骤文本和类型（如果指定）
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # 直接从计划工具存储中获取计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤信息
                    step_info = {"text": step}

                    # 尝试从文本中提取步骤类型（例如 [SEARCH] 或 [CODE]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 将当前步骤标记为进行中
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # 如果工具调用失败，直接更新状态
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 没有找到活动步骤

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """执行当前步骤

        使用指定的执行器代理执行当前步骤。包括：
        1. 准备执行环境和上下文
        2. 创建执行提示
        3. 执行步骤并处理结果
        4. 更新步骤状态

        参数:
            executor: 执行步骤的代理实例
            step_info: 步骤信息字典，包含步骤文本等信息

        返回:
            str: 步骤执行的结果描述
        """
        # 准备带有当前计划状态的上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 创建执行步骤的提示
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Please execute this step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # 使用执行器运行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 执行成功后将步骤标记为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """将当前步骤标记为已完成

        首先尝试使用计划工具更新步骤状态，如果失败则直接在存储中更新状态。
        这个方法确保即使在工具调用失败的情况下也能维护正确的步骤状态。
        """
        if self.current_step_index is None:
            return

        try:
            # 使用计划工具标记步骤为已完成
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            # 如果工具调用失败，直接在计划工具存储中更新状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保状态列表长度足够
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本表示

        首先尝试使用计划工具获取计划文本，如果失败则从存储中直接生成文本。

        返回:
            str: 格式化的计划文本，包含标题、进度和步骤信息
        """
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """直接从存储中生成计划文本

        当计划工具无法正常工作时，这个方法作为后备方案，直接从存储中读取数据并生成格式化文本。

        返回:
            str: 格式化的计划文本，包含以下内容：
                - 计划标题和ID
                - 总体进度统计
                - 各状态步骤数量
                - 带状态标记的步骤列表
        """
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            # 获取计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保状态和注释列表长度与步骤数量匹配
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 统计各状态的步骤数量
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            # 计算总体进度
            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            # 生成计划文本
            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            # 添加进度信息
            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            # 获取状态标记
            status_marks = PlanStepStatus.get_status_marks()

            # 添加步骤列表
            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 使用状态标记显示步骤状态
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """完成计划并生成总结

        使用语言模型或代理生成计划执行的总结报告。如果语言模型调用失败，
        会尝试使用代理作为后备方案。

        返回:
            str: 计划完成的总结报告
        """
        plan_text = await self._get_plan_text()

        # 使用语言模型生成总结
        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # 如果语言模型失败，使用代理生成总结
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
