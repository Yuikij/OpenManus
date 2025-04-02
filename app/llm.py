# 导入必要的Python标准库
import math
from typing import Dict, List, Optional, Union

# 导入第三方库
import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,  # Azure OpenAI API客户端
    AsyncOpenAI,       # OpenAI API异步客户端
    AuthenticationError,  # 认证错误
    OpenAIError,       # OpenAI通用错误
    RateLimitError,    # 速率限制错误
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (  # 重试机制相关装饰器
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# 导入应用内部模块
from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# 定义推理模型列表
REASONING_MODELS = ["o1", "o3-mini"]

# 定义支持多模态（图像处理）的模型列表
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    """Token计数器类

    用于计算文本和图像消息的token数量。包含了对不同类型内容（文本、图像）的token计算方法，
    以及对OpenAI API消息格式的token计算支持。
    """
    # Token相关常量
    BASE_MESSAGE_TOKENS = 4    # 每条消息的基础token数
    FORMAT_TOKENS = 2          # 格式化token数
    LOW_DETAIL_IMAGE_TOKENS = 85    # 低细节图片的固定token数
    HIGH_DETAIL_TILE_TOKENS = 170   # 高细节图片每个tile的token数

    # 图像处理相关常量
    MAX_SIZE = 2048                        # 图片最大尺寸
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768    # 高细节模式下短边目标尺寸
    TILE_SIZE = 512                        # 图片分块大小

    def __init__(self, tokenizer):
        """初始化Token计数器

        Args:
            tokenizer: 用于文本token计算的分词器实例
        """
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """计算文本字符串的token数量

        Args:
            text: 需要计算token数的文本字符串

        Returns:
            int: 文本包含的token数量
        """
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """计算图像的token数量

        根据图像的细节级别和尺寸计算token数量：
        - 低细节模式：固定85 tokens
        - 高细节模式：
          1. 将图片缩放至2048x2048范围内
          2. 将短边缩放至768px
          3. 计算512px大小的分块数量（每块170 tokens）
          4. 额外添加85 tokens基础消耗

        Args:
            image_item: 包含图像信息的字典，可包含detail（细节级别）和dimensions（尺寸）字段

        Returns:
            int: 图像消耗的token数量
        """
        detail = image_item.get("detail", "medium")

        # 低细节模式返回固定token数
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # 对于中等细节模式（OpenAI默认），使用高细节计算方式
        # OpenAI未指定中等细节的单独计算方法

        # 高细节模式，基于图像尺寸计算（如果提供）
        if detail == "high" or detail == "medium":
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # 当未提供尺寸或细节级别未知时的默认值
        if detail == "high":
            # 高细节模式默认使用1024x1024尺寸计算
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # 中等细节模式使用默认中等大小图片
            return 1024  # 与原始默认值匹配
        else:
            # 未知细节级别默认使用中等模式
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """计算高细节图像的token数量

        Args:
            width: 图像宽度
            height: 图像高度

        Returns:
            int: 计算得到的token数量
        """
        # 步骤1：缩放至MAX_SIZE x MAX_SIZE范围内
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # 步骤2：将短边缩放至HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # 步骤3：计算TILE_SIZE大小的分块数量
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # 步骤4：计算最终token数量
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """计算消息内容的token数量

        支持计算文本字符串或包含文本和图像的混合内容列表的token数量

        Args:
            content: 消息内容，可以是字符串或包含文本和图像的字典列表

        Returns:
            int: 内容的总token数量
        """
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """计算工具调用的token数量

        Args:
            tool_calls: 工具调用列表，每个调用包含函数名和参数

        Returns:
            int: 工具调用消耗的token数量
        """
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的总token数量

        包括基础格式token、每条消息的基础token、角色token、内容token和工具调用token

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            int: 消息列表的总token数量
        """
        total_tokens = self.FORMAT_TOKENS  # 基础格式token数

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # 每条消息的基础token数

            # 添加角色token数
            tokens += self.count_text(message.get("role", ""))

            # 添加内容token数
            if "content" in message:
                tokens += self.count_content(message["content"])

            # 添加工具调用token数
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # 添加名称和工具调用ID的token数
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    """大语言模型接口类

    实现了与OpenAI、Azure OpenAI和AWS Bedrock等大语言模型服务的交互。
    使用单例模式确保每个配置只创建一个实例。包含token计数、消息格式化、
    错误重试等功能。
    """
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        """实现单例模式

        Args:
            config_name: 配置名称，用于区分不同的LLM实例
            llm_config: LLM配置对象，可选

        Returns:
            LLM: 对应配置的LLM实例
        """
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        """初始化LLM实例

        仅在实例首次创建时执行初始化。设置模型参数、API配置，
        初始化token计数器和客户端连接。

        Args:
            config_name: 配置名称
            llm_config: LLM配置对象，如果为None则使用默认配置
        """
        if not hasattr(self, "client"):  # 仅在未初始化时执行
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # 添加token计数相关属性
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # 初始化分词器
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # 如果模型不在tiktoken预设中，使用cl100k_base作为默认
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            # 根据API类型初始化对应的客户端
            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量

        Args:
            text: 需要计算的文本字符串

        Returns:
            int: 文本的token数量
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的token数量

        Args:
            messages: 消息列表

        Returns:
            int: 消息列表的总token数量
        """
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """更新token使用计数

        记录输入和完成的token数量，并输出日志信息

        Args:
            input_tokens: 输入token数量
            completion_tokens: 完成token数量，默认为0
        """
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """检查是否超出token限制

        Args:
            input_tokens: 要检查的输入token数量

        Returns:
            bool: 如果未超出限制返回True，否则返回False
        """
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """生成token限制错误消息

        Args:
            input_tokens: 输入token数量

        Returns:
            str: 格式化的错误消息
        """
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """格式化消息列表为OpenAI API所需的格式

        将消息对象或字典转换为标准的OpenAI消息格式，支持处理文本和图像内容。

        Args:
            messages: 消息列表，可以是字典或Message对象
            supports_images: 是否支持图像输入的标志

        Returns:
            List[dict]: OpenAI格式的消息列表

        Raises:
            ValueError: 消息格式无效或缺少必要字段时
            TypeError: 消息类型不支持时
        """
        formatted_messages = []

        for message in messages:
            # 将Message对象转换为字典
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # 确保消息字典包含必要字段
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # 处理base64编码的图像（如果存在且模型支持）
                if supports_images and message.get("base64_image"):
                    # 初始化或转换content为适当格式
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # 将字符串项转换为正确的文本对象
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # 添加图像到content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # 移除base64_image字段
                    del message["base64_image"]
                # 如果模型不支持图像但消息包含base64_image，优雅处理
                elif not supports_images and message.get("base64_image"):
                    # 仅移除base64_image字段并保留文本内容
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # 验证所有消息都具有必要字段
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)  # 不重试TokenLimitExceeded
        ),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """向大语言模型发送请求并获取响应

        支持流式和非流式响应，可以处理系统消息和用户消息，包含完整的错误处理和重试机制。

        Args:
            messages: 对话消息列表
            system_msgs: 系统消息列表（可选）
            stream: 是否使用流式响应
            temperature: 采样温度，控制响应的随机性

        Returns:
            str: 模型生成的响应文本

        Raises:
            TokenLimitExceeded: 超出token限制时
            ValueError: 消息格式无效或响应为空时
            OpenAIError: API调用失败且重试耗尽时
            Exception: 其他意外错误
        """
        try:
            # 检查模型是否支持图像
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化系统消息和用户消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入token数量
            input_tokens = self.count_message_tokens(messages)

            # 检查是否超出token限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出特殊异常，不会重试
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            # 根据模型类型设置不同的参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # 非流式请求：一次性获取完整响应
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                # 验证响应是否有效
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # 更新输入和完成的token计数
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                # 返回模型生成的文本内容
                return response.choices[0].message.content

            # 流式请求：实时获取并输出响应
            # 在请求前更新预估的输入token计数
            self.update_token_count(input_tokens)

            # 创建流式请求
            response = await self.client.chat.completions.create(**params, stream=True)

            # 收集和处理流式响应
            collected_messages = []  # 存储所有响应片段
            completion_text = ""    # 完整的响应文本
            async for chunk in response:
                # 获取当前响应片段的文本内容
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                # 实时打印响应内容
                print(chunk_message, end="", flush=True)

            print()  # 流式输出后的换行
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # 估算流式响应的完成token数量
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # 不记录日志直接重新抛出token限制错误
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # Always use non-streaming for tool requests
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
