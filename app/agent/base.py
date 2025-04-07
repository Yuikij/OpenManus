from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message

# 导入必要的模块和类：
# - ABC和abstractmethod用于定义抽象基类和抽象方法
# - asynccontextmanager用于创建异步上下文管理器
# - List和Optional是类型提示
# - BaseModel, Field和model_validator来自pydantic，用于数据验证
# - 其他是项目内部模块，包括LLM（语言模型）、日志记录器、沙盒客户端和各种数据模型

# 导入必要的模块和类：
# - ABC和abstractmethod用于定义抽象基类和抽象方法
# - asynccontextmanager用于创建异步上下文管理器
# - List和Optional是类型提示
# - BaseModel, Field和model_validator来自pydantic，用于数据验证
# - 其他是项目内部模块，包括LLM（语言模型）、日志记录器、沙盒客户端和各种数据模型


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # 代理的抽象基类，用于管理代理状态和执行流程。
    #
    # 提供了状态转换、内存管理和基于步骤的执行循环的基础功能。
    # 子类必须实现`step`方法。

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    # 核心属性：代理的唯一名称
    description: Optional[str] = Field(None, description="Optional agent description")
    # 可选的代理描述

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    # 系统级指令提示，用于初始化代理的行为和能力
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )
    # 用于确定下一步行动的提示

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    # 依赖项：语言模型实例，用于代理的思考和决策
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    # 代理的内存存储，用于保存对话历史和状态
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )
    # 当前代理状态（空闲、运行中、已完成、错误）

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    # 执行控制：终止前的最大步骤数
    current_step: int = Field(default=0, description="Current step in execution")
    # 执行中的当前步骤

    duplicate_threshold: int = 2
    # 重复阈值：检测代理是否陷入循环的阈值

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses
        # 配置类：
        # - arbitrary_types_allowed=True 允许使用任意类型
        # - extra="allow" 允许子类添加额外字段，提高灵活性

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        # 初始化代理，如果未提供默认设置则使用默认值
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        # 如果LLM未提供或类型不正确，创建一个新的LLM实例，使用代理名称的小写形式作为配置名
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        # 如果内存不是Memory类型，创建一个新的Memory实例
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        # 异步上下文管理器，用于安全的代理状态转换
        #
        # 参数:
        #     new_state: 上下文期间要转换到的状态
        #
        # 产出:
        #     None: 允许在新状态下执行
        #
        # 异常:
        #     ValueError: 如果new_state无效
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")
        # 验证状态是否为有效的AgentState枚举值

        previous_state = self.state
        self.state = new_state
        # 保存先前状态并设置新状态
        try:
            yield
            # 允许在新状态下执行代码
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            # 如果执行失败，转换到ERROR状态
            raise e
        finally:
            self.state = previous_state  # Revert to previous state
            # 最终恢复到先前的状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        # 向代理的内存添加消息
        #
        # 参数:
        #     role: 消息发送者的角色（用户、系统、助手、工具）
        #     content: 消息内容
        #     base64_image: 可选的base64编码图像
        #     **kwargs: 额外参数（例如，工具消息的tool_call_id）
        #
        # 异常:
        #     ValueError: 如果角色不受支持
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }
        # 消息映射字典，将角色映射到相应的消息创建方法

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")
        # 检查角色是否受支持

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        # 根据角色准备适当的参数，对于工具消息，包含额外的kwargs
        self.memory.add_message(message_map[role](content, **kwargs))
        # 使用相应的消息创建方法创建消息并添加到内存中

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        # 异步执行代理的主循环
        #
        # 参数:
        #     request: 可选的初始用户请求
        #
        # 返回:
        #     总结执行结果的字符串
        #
        # 异常:
        #     RuntimeError: 如果代理在开始时不处于IDLE状态
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")
        # 验证代理是否处于IDLE状态，否则抛出异常

        if request:
            self.update_memory("user", request)
        # 如果提供了请求，将其作为用户消息添加到内存中

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            # 使用状态上下文管理器，将状态设置为RUNNING
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                # 当未达到最大步骤数且状态不是FINISHED时继续执行
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()
                # 执行单个步骤并获取结果

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()
                # 检查代理是否陷入循环，如果是则处理卡住状态

                results.append(f"Step {self.current_step}: {step_result}")
                # 将步骤结果添加到结果列表

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
                # 如果达到最大步骤数，重置步骤计数器，将状态设置为IDLE，并添加终止消息
        await SANDBOX_CLIENT.cleanup()
        # 清理沙盒客户端资源
        return "\n".join(results) if results else "No steps executed"
        # 返回结果字符串，如果没有执行步骤则返回默认消息

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """
        # 执行代理工作流中的单个步骤
        #
        # 这是一个抽象方法，必须由子类实现以定义特定行为
        # 每个子类应该根据其特定目的和功能实现此方法

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        # 通过添加提示来处理卡住状态，鼓励代理改变策略
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        # 卡住状态提示：观察到重复响应，考虑新策略并避免重复已尝试的无效路径
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        # 将卡住提示添加到下一步提示前面
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")
        # 记录警告日志，表明代理检测到卡住状态

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        # 通过检测重复内容来检查代理是否陷入循环
        if len(self.memory.messages) < 2:
            return False
        # 如果消息少于2条，不可能陷入循环

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False
        # 如果最后一条消息没有内容，不考虑卡住

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )
        # 计算与最后一条消息内容相同的助手消息数量

        return duplicate_count >= self.duplicate_threshold
        # 如果重复计数达到或超过阈值，则认为代理陷入循环

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        # 属性getter：获取代理内存中的消息列表
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        # 属性setter：设置代理内存中的消息列表
        self.memory.messages = value
