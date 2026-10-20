"""RAGCore：基于 ChromaDB 向量检索 + BM25 关键词检索的混合检索核心

- 初始化 ChromaDB 持久化客户端（数据落盘到 ./chroma_db）
- rebuild_knowledge_base：从 chunks_result.json 读取分块
  - 向量化后写入 ChromaDB collection
  - 同时构建 BM25 索引（基于 rank_bm25）
- search：混合检索（向量 + BM25），用 RRF（Reciprocal Rank Fusion）融合两路结果
- 向量化复用项目已有的 EmbeddingClient（Mock/真实接口自动切换）
  且显式传入向量，不触发 chromadb 默认的 ONNX embedding（Windows 下易缺 onnxruntime）
"""

import os
import re
import json
from typing import List, Dict, Optional

from embedding_client import EmbeddingClient

# 中文分词：优先用 jieba（若已安装），否则降级为正则切词（字符级 + 英文词）
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def _tokenize(text: str) -> List[str]:
    """BM25 用的分词器：中文按词/字切分，英文按词切分，去掉标点和空白"""
    if not text:
        return []
    if _HAS_JIEBA:
        # jieba 对中文分词效果更好
        tokens = list(jieba.cut(text))
    else:
        # 降级方案：英文按空白/标点切词，中文按单字切
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fa5]", text)
    # 去掉纯标点和空白，统一小写
    return [t.lower() for t in tokens if t and not t.isspace()]


class RAGCore:
    """RAG 混合检索核心：向量检索（ChromaDB）+ 关键词检索（BM25），RRF 融合"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        embedder: Optional[EmbeddingClient] = None,
    ):
        # 向量化客户端：未传入则新建一个（未配置密钥时自动 Mock 模式）
        self.embedder = embedder or EmbeddingClient()
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # BM25 索引相关状态：rebuild_knowledge_base 时构建
        self.bm25 = None                 # BM25Okapi 实例
        self.bm25_chunks: List[str] = []  # 与 BM25 索引一一对应的原文
        self.bm25_tokenized: List[List[str]] = []  # 分词后的列表（调试用）

        # 延迟导入 chromadb，避免模块导入期就触发其默认组件初始化
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "未安装 chromadb，请先执行：pip install chromadb"
            ) from e

        # 持久化客户端：数据保存在本地目录，重启不丢失
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 获取/创建 collection。
        # 注意：不在这里传 embedding_function，后续 add/query 都显式传入向量，
        # 这样 chromadb 的默认 ONNX embedding 永远不会被调用（规避 Windows 依赖问题）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

        print(f"[RAGCore] 初始化完成，持久化目录：{persist_dir}，"
              f"collection：{collection_name}，当前文档数：{self.collection.count()}")
        if not _HAS_JIEBA:
            print("[RAGCore] 未安装 jieba，BM25 分词降级为正则切词；"
                   "如需更精准的中文检索可执行：pip install jieba")

    def rebuild_knowledge_base(self, chunks_file: str = "chunks_result.json") -> int:
        """
        从分块结果文件重建知识库

        - 读取 JSON 中的所有文本块
        - 用 EmbeddingClient 批量向量化
        - 清空旧 collection 后重新写入

        Args:
            chunks_file: 分块结果 JSON 文件路径

        Returns:
            写入的文档块数量
        """
        # 1. 读取分块文件
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(
                f"分块文件不存在：{chunks_file}，请先生成分块结果"
            )

        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容多种 JSON 结构：纯列表 / {"chunks": [...]} / 列表里是字符串或字典
        if isinstance(data, dict):
            raw_chunks = data.get("chunks", [])
        else:
            raw_chunks = data

        chunks: List[str] = []
        for item in raw_chunks:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                # 兼容 content / text / chunk 几种常见字段名
                text = item.get("content") or item.get("text") or item.get("chunk") or ""
            else:
                text = str(item)
            text = text.strip()
            if text:
                chunks.append(text)

        if not chunks:
            print("[RAGCore] 文件中没有有效分块，跳过重建")
            return 0

        print(f"[RAGCore] 读取到 {len(chunks)} 个文本块，开始向量化...")

        # 2. 批量向量化所有文本块
        vectors = self.embedder.embed(chunks)

        # 3. 重建 collection：先删除旧的，再创建新的（rebuild 语义）
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # 旧 collection 不存在时忽略
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 4. 写入向量库：id / 原文 / 向量 / 元数据
        #    分批写入，避免一次性数据量过大
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch_chunks = chunks[start:end]
            batch_vectors = vectors[start:end]
            # id 和元数据的序号按实际批次长度生成
            batch_ids = [f"chunk_{i}" for i in range(start, end)]
            batch_metas = [{"index": i, "source": chunks_file} for i in range(start, end)]

            self.collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=batch_vectors,   # 显式传入向量，不走 chromadb 默认 embedding
                metadatas=batch_metas,
            )

        # 5. 构建 BM25 索引（与向量库共用同一份 chunks，按相同顺序）
        #    这样 BM25 命中的第 i 个片段，就是向量库中 chunk_i
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError(
                "未安装 rank_bm25，请先执行：pip install rank-bm25"
            ) from e

        self.bm25_chunks = chunks[:]                          # 原文
        self.bm25_tokenized = [_tokenize(c) for c in chunks]  # 分词后的列表
        self.bm25 = BM25Okapi(self.bm25_tokenized)

        print(f"[RAGCore] 知识库重建完成，共写入 {self.collection.count()} 个文档块"
              f"（向量库 + BM25 索引已同步）")
        return len(chunks)

    def _search_vector(self, query: str, n: int) -> List[Dict]:
        """向量检索：返回 [{id, content, metadata, distance, rank}]，rank 从 1 开始"""
        query_vector = self.embedder.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_vector],  # 显式传入问题向量
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits = []
        for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            hits.append({
                "id": (meta or {}).get("index", rank - 1),
                "content": doc,
                "metadata": meta or {},
                "distance": round(float(dist), 4),
                "rank": rank,
            })
        return hits

    def _search_bm25(self, query: str, n: int) -> List[Dict]:
        """BM25 关键词检索：返回 [{id, content, score, rank}]，rank 从 1 开始"""
        if not self.bm25 or not self.bm25_chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # 按得分降序取前 n
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        hits = []
        for rank, (idx, score) in enumerate(ranked, 1):
            hits.append({
                "id": idx,
                "content": self.bm25_chunks[idx],
                "score": round(float(score), 4),
                "rank": rank,
            })
        return hits

    @staticmethod
    def _rrf_fuse(vec_hits: List[Dict], bm25_hits: List[Dict],
                  k: int = 60) -> List[Dict]:
        """
        RRF（Reciprocal Rank Fusion）融合两路结果
        - score(d) = Σ 1 / (k + rank(d))，对每一路的排名求和
        - k=60 是 RRF 原论文经验值，平滑靠前结果的得分差
        """
        rrf_score: Dict[int, float] = {}
        meta_map: Dict[int, Dict] = {}

        for hit in vec_hits:
            cid = hit["id"]
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + hit["rank"])
            meta_map[cid] = hit
        for hit in bm25_hits:
            cid = hit["id"]
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + hit["rank"])
            if cid not in meta_map:
                meta_map[cid] = hit

        # 按 RRF 得分降序排列
        ordered = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for rank, (cid, score) in enumerate(ordered, 1):
            base = meta_map[cid]
            fused.append({
                "id": cid,
                "content": base["content"],
                "metadata": base.get("metadata", {}),
                "distance": base.get("distance"),    # 来自向量检索（可能为 None）
                "bm25_score": base.get("bm25_score", base.get("score")),
                "rrf_score": round(score, 4),
                "rank": rank,
            })
        return fused

    def search(self, query: str, n_results: int = 3, mode: str = "hybrid") -> List[Dict]:
        """
        检索：默认 hybrid（向量 + BM25 + RRF 融合），也支持单路对比测试

        Args:
            query: 用户问题
            n_results: 最终返回数量
            mode: "hybrid" 混合 / "vector" 仅向量 / "bm25" 仅 BM25

        Returns:
            结果列表，每项含 content / metadata / rank，
            以及相应模式的得分（distance / bm25_score / rrf_score）
        """
        if not query.strip():
            return []

        # 单路检索，为保证两路候选足够，各自取较多候选再做融合
        candidate_n = max(n_results * 3, 10)
        vec_hits = self._search_vector(query, candidate_n)
        bm25_hits = self._search_bm25(query, candidate_n)

        if mode == "vector":
            fused = vec_hits[:n_results]
        elif mode == "bm25":
            fused = bm25_hits[:n_results] if bm25_hits else vec_hits[:n_results]
        else:
            # 混合检索：RRF 融合后取前 n
            fused = self._rrf_fuse(vec_hits, bm25_hits)[:n_results]

        # 统一返回结构：把 rrf_score / distance / bm25_score 都塞进 distance 字段，
        # 方便上层接口（group_project 的 /rag-query）保持原有 {"content", "distance"} 用法
        out = []
        for hit in fused:
            if mode == "vector":
                score_str = hit.get("distance")
            elif mode == "bm25":
                score_str = hit.get("score") or hit.get("bm25_score")
            else:
                score_str = hit.get("rrf_score")
            out.append({
                "content": hit["content"],
                "metadata": hit.get("metadata", {}),
                "distance": score_str,
                # 额外字段：方便调试与上层展示来源
                "rrf_score": hit.get("rrf_score"),
                "vector_distance": hit.get("distance") if mode == "hybrid" else None,
                "bm25_score": hit.get("bm25_score") or hit.get("score") if mode != "hybrid" else None,
            })
        return out


if __name__ == "__main__":
    # ---------- 测试代码 ----------
    CHUNKS_FILE = "chunks_result.json"

    # 如果分块文件不存在，先用 group_project 里的分块函数生成一份
    if not os.path.exists(CHUNKS_FILE):
        print(f"[测试] 未找到 {CHUNKS_FILE}，自动从 test_content 生成...")
        from group_project import test_content, chunk_semantic
        chunks = chunk_semantic(test_content, max_size=200)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
        print(f"[测试] 已生成 {CHUNKS_FILE}，共 {len(chunks)} 块\n")

    # 1. 初始化 RAGCore
    rag = RAGCore(persist_dir="./chroma_db")

    # 2. 重建知识库（同时构建向量库和 BM25 索引）
    print("\n--- 重建知识库（向量 + BM25 索引）---")
    rag.rebuild_knowledge_base(CHUNKS_FILE)

    # 3. 检索对比测试：纯向量 vs 纯 BM25 vs 混合（RRF 融合）
    print("\n--- 检索对比测试 ---")
    question = "SQL注入如何修复"
    print(f"问题：{question}\n")

    for mode in ["vector", "bm25", "hybrid"]:
        print(f"\n>>> 模式：{mode}")
        results = rag.search(question, n_results=3, mode=mode)

        if not results:
            print("  （无结果）")
            continue

        for i, hit in enumerate(results):
            # 展示该模式下的主得分
            score = hit.get("rrf_score") or hit.get("bm25_score") or hit.get("vector_distance")
            print(f"  【结果 {i + 1}】得分={hit['distance']}  片段={hit['content'][:40]}...")

    # 4. 默认混合检索的最终输出（与 /rag-query 接口返回结构一致）
    print("\n\n--- 混合检索最终结果（/rag-query 接口使用）---")
    final = rag.search(question, n_results=3)
    print(f"问题：{question}")
    print(f"Top-{len(final)} 结果：")
    for i, hit in enumerate(final):
        print(f"\n【结果 {i + 1}】RRF 得分={hit['rrf_score']}（向量距离={hit.get('vector_distance')}, "
              f"BM25 得分={hit.get('bm25_score')}）")
        print(f"内容：{hit['content']}")
