"""大模型客户端

- 从 .env 文件读取 API_KEY / BASE_URL / MODEL
- 未配置有效密钥时进入 Mock 模式，返回模拟回答
- 配置有效密钥时调用 OpenAI 兼容接口
"""

import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 未安装 python-dotenv 时，仍可读取系统环境变量
    pass


class LLMClient:
    """大模型客户端，支持真实调用与 Mock 模式自动切换"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = (api_key or os.getenv("API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("BASE_URL", "")).strip()
        self.model = (model or os.getenv("MODEL", "deepseek-v4-flash")).strip()

        # 既没有显式传入，也没有从环境读到有效值，则进入 Mock 模式
        self.mock_mode = not (self.api_key and self.base_url)

        if self.mock_mode:
            print("[LLMClient] 未检测到有效的 API_KEY / BASE_URL，已进入 Mock 模式")
        else:
            print(f"[LLMClient] 已配置真实接口：{self.base_url}，模型：{self.model}")

    def chat(self, messages: List[Dict]) -> str:
        """接收 messages 列表，返回 AI 回答文本

        Args:
            messages: OpenAI 格式的消息列表，
                      例如 [{"role": "user", "content": "你好"}]

        Returns:
            AI 回答字符串
        """
        if self.mock_mode:
            return self._mock_chat(messages)
        return self._real_chat(messages)

    def _mock_chat(self, messages: List[Dict]) -> str:
        """Mock 模式：根据最后一条用户消息返回模拟回答"""
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        if not user_msg:
            return "[Mock 回答] 你好，我是模拟助手，请输入你的问题。"

        return (
            f"[Mock 回答] 你刚才说：「{user_msg}」。\n"
            f"这是一条来自 Mock 模式的模拟回答。"
            f"配置有效的 API_KEY 与 BASE_URL 后即可获取真实回答。"
        )

    def _real_chat(self, messages: List[Dict]) -> str:
        """真实模式：调用 OpenAI 兼容接口"""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "调用真实接口需要安装 openai 库，请执行：pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # 简单测试代码
    client = LLMClient()

    test_messages = [
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": "请用一句话介绍一下 Python。"},
    ]

    print("\n--- 测试 chat 方法 ---")
    answer = client.chat(test_messages)
    print("AI 回答：")
    print(answer)
