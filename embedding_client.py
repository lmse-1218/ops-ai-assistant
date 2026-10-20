"""向量化客户端

- 从 .env 文件读取 EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL
- 未配置有效密钥时进入 Mock 模式，返回基于字符哈希的伪向量
  （同一进程内稳定，语义相近的文本会得到相近的向量，可用于本地演示）
- 配置有效密钥时调用 OpenAI 兼容的 Embedding 接口
"""

import os
import math
import zlib
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 未安装 python-dotenv 时，仍可读取系统环境变量
    pass


class EmbeddingClient:
    """向量化客户端，支持真实调用与 Mock 模式自动切换"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dim: int = 256,
    ):
        self.api_key = (api_key or os.getenv("EMBED_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("EMBED_BASE_URL", "")).strip()
        self.model = (model or os.getenv("EMBED_MODEL", "text-embedding-3-small")).strip()
        self.dim = dim  # Mock 模式下的向量维度

        # 没有有效密钥则进入 Mock 模式
        self.mock_mode = not (self.api_key and self.base_url)

        if self.mock_mode:
            print("[EmbeddingClient] 未检测到有效的 EMBED_API_KEY / EMBED_BASE_URL，已进入 Mock 模式")
        else:
            print(f"[EmbeddingClient] 已配置真实接口：{self.base_url}，模型：{self.model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，返回与输入等长的向量列表"""
        if not texts:
            return []
        if self.mock_mode:
            return self._mock_embed(texts)
        return self._real_embed(texts)

    def _mock_embed(self, texts: List[str]) -> List[List[float]]:
        """Mock 模式：基于字符哈希的词袋伪向量（归一化）

        - 相同字符出现越多，对应维度值越大
        - 文本共享的字符越多，余弦相似度越高，可满足本地演示检索需求
        """
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            # 逐字符累加到对应哈希维度，形成词袋向量
            for ch in text:
                idx = zlib.crc32(ch.encode("utf-8")) % self.dim
                vec[idx] += 1.0
            # L2 归一化，方便后续直接点积计算余弦相似度
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def _real_embed(self, texts: List[str]) -> List[List[float]]:
        """真实模式：调用 OpenAI 兼容的 Embedding 接口"""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "调用真实 Embedding 接口需要安装 openai 库，请执行：pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # 按 index 排序，保证向量顺序与输入文本一一对应
        data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in data]


if __name__ == "__main__":
    # 简单测试代码
    ec = EmbeddingClient()
    vecs = ec.embed(["你好世界", "你好世界", "完全不同的文本"])
    print(f"向量数量：{len(vecs)}，维度：{len(vecs[0])}")

    # 计算两组余弦相似度验证效果
    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    print(f"相同文本相似度：{cosine(vecs[0], vecs[1]):.4f}")   # 应接近 1
    print(f"不同文本相似度：{cosine(vecs[0], vecs[2]):.4f}")   # 应明显更低
