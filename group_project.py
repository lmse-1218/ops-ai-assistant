import json
import asyncio
import inspect

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import math

from llm_client import LLMClient
from embedding_client import EmbeddingClient
from rag_core import RAGCore
from tools import (vulnerability_analysis, windows_ops_troubleshoot,
                   security_audit, execute_tool, TOOL_CALL_LOG, TOOLS_REGISTRY)
from agent import ReActAgent

# 创建大模型客户端（未配置有效密钥时自动进入 Mock 模式）
llm = LLMClient()

# 创建向量化客户端（未配置有效密钥时自动进入 Mock 模式）
embed_client = EmbeddingClient()

# 创建 RAG 检索核心（ChromaDB 持久化在 ./chroma_db，需先重建知识库）
rag = RAGCore(persist_dir="./chroma_db")

# 创建 ReAct Agent（全局实例，跨请求保留对话记忆）
agent = ReActAgent(max_steps=4)

app = FastAPI(title="Browser Assistant Service")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 全局异常处理：兜底所有未捕获的异常，保证服务不会因单个接口崩溃
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常兜底处理器
    - 任何接口内部未被 try/except 捕获的异常都会进入这里
    - 统一返回 500 + {"error": "错误信息"} 格式
    - HTTPException（如 404/400）不受影响，仍走 FastAPI 默认处理
    """
    # 打印到服务端日志，方便排查问题
    print(f"[全局异常] {request.method} {request.url.path} -> "
          f"{type(exc).__name__}: {exc}")

    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {exc}"},
    )


def split_text_chunk(text: str, chunk_size: int = 400, overlap: int = 100) -> list:
    """
    把长文本切成小块（滑动窗口分块）

    - 块与块之间保留 overlap 个字符的重叠部分，保证上下文语义不被切断
    - 返回分片列表

    Args:
        text: 原始长文本
        chunk_size: 每个分块的最大字符数，默认 400
        overlap: 相邻分块之间的重叠字符数，默认 100

    Returns:
        分片字符串列表
    """
    # 参数校验：overlap 必须小于 chunk_size，否则会死循环
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    # 空文本直接返回空列表
    if not text:
        return []

    chunks = []
    start = 0          # 当前分块在原文中的起始位置
    step = chunk_size - overlap  # 每次窗口向后滑动的步长

    while start < len(text):
        # 取出当前窗口对应的文本块
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

        # 如果窗口已经覆盖到文本末尾，则切分完成
        if start + chunk_size >= len(text):
            break

        # 窗口向后滑动 step 个字符，保留 overlap 的重叠区域
        start += step

    return chunks


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """计算两个向量的余弦相似度，取值范围 [-1, 1]，越接近 1 越相似"""
    # 点积：对应元素相乘再求和
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    # 两个向量各自的 L2 范数（模长）
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    # 任一 向量为零向量时无法计算，返回 0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _serve_html(filename: str) -> HTMLResponse:
    """读取并返回本地 HTML 文件（找不到时返回 404）"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content=f"<h1>{filename} not found</h1>", status_code=404)


@app.get("/", response_class=HTMLResponse)
async def index():
    """产品级统一首页（index.html，整合四大功能面板）"""
    return _serve_html("index.html")


@app.get("/group_index.html", response_class=HTMLResponse)
async def group_index_page():
    """旧版浏览器助手页面（group_index.html），保留原有入口"""
    return _serve_html("group_index.html")


@app.post("/browser-assistant")
async def browser_assistant(request: Request):
    """
    浏览器助手接口
    - 接收 content 和 action 参数
    - action 支持 summarize / explain / translate
    - 调用大模型 API 返回真实回答
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    content = data.get("content", "")
    action = data.get("action", "").strip().lower()

    # 支持的 action 列表
    supported_actions = {"summarize", "explain", "translate"}

    if action not in supported_actions:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": f"Unsupported action: {action}. "
                           f"Supported: {', '.join(sorted(supported_actions))}",
                "data": None,
            },
        )

    if not content:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": "content is required",
                "data": None,
            },
        )

    # 根据 action 构建不同的 system prompt
    system_prompts = {
        "summarize": "你是一个内容摘要助手。请用简洁、准确的语言总结用户提供的文本核心内容，控制在 200 字以内。",
        "explain": "你是一个概念解释助手。请用通俗易懂的语言向用户解释给定文本或概念的含义，必要时给出例子。",
        "translate": "你是一个翻译助手。请将用户提供的文本翻译为英文，只输出译文，不要附加解释。",
    }

    messages = [
        {"role": "system", "content": system_prompts[action]},
        {"role": "user", "content": content},
    ]

    # 调用大模型获取真实回答
    try:
        answer = llm.chat(messages)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"LLM 调用失败：{e}",
                "data": None,
            },
        )

    return {
        "code": 200,
        "message": "success",
        "action": action,
        "data": answer,
    }


@app.post("/rag-query")
async def rag_query(request: Request):
    """
    RAG 知识库问答接口（基于 ChromaDB 持久化知识库）
    - 接收 {"question": "用户问题"}
    - 用 RAGCore.search() 从知识库检索最相关的 Top-3 文档块
    - 把检索到的资料交给大模型生成回答
    - 返回 {"answer": 回答, "sources": [来源片段...]}
    """
    # 解析请求体
    try:
        data = await request.json()
    except Exception:
        data = {}

    question = data.get("question", "").strip()

    # 参数校验
    if not question:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "question is required", "data": None},
        )

    # 1 & 2. 调用 RAGCore 检索知识库，获取最相关的 3 个文档块
    try:
        hits = rag.search(question, n_results=3)
    except Exception as e:
        # 检索失败时返回明确错误，不静默降级
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"知识库检索失败：{e}", "data": None},
        )

    # 知识库为空时提示先重建
    if not hits:
        return JSONResponse(
            status_code=404,
            content={
                "code": 404,
                "message": "知识库为空，请先调用 RAGCore.rebuild_knowledge_base() 重建知识库",
                "data": None,
            },
        )

    # 3. 构建 System Prompt：要求 AI 仅基于检索到的资料回答
    system_prompt = (
        "你是一个知识库问答助手。请严格基于下面提供的参考资料回答用户问题，"
        "不要编造资料以外的内容；如果资料中没有足够信息，请如实说明。"
    )

    # 4. 构建 User Prompt：检索到的资料 + 用户问题
    context_text = "\n\n".join(
        f"【资料{i + 1}】（相关度距离 {hit['distance']}）\n{hit['content']}"
        for i, hit in enumerate(hits)
    )
    user_prompt = f"参考资料：\n{context_text}\n\n用户问题：{question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 5. 调用大模型生成回答
    try:
        answer = llm.chat(messages)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"LLM 调用失败：{e}", "data": None},
        )

    # 6. 返回回答与引用的来源片段
    return {
        "answer": answer,
        "sources": [hit["content"] for hit in hits],
    }


# ============================================================
# 工具调用接口（基于 tools.py 的三个知识库工具 + 统一调度器）
# ============================================================

async def _parse_body(request: Request) -> dict:
    """解析 JSON 请求体，解析失败返回空 dict"""
    try:
        return await request.json()
    except Exception:
        return {}


@app.post("/tool/analyze")
async def tool_analyze(request: Request):
    """
    漏洞分析工具接口
    - 请求体：{"input_text": "待分析文本"}
    - 调用 vulnerability_analysis 返回格式化漏洞报告
    """
    data = await _parse_body(request)
    input_text = (data.get("input_text") or "").strip()

    if not input_text:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "input_text is required", "data": None},
        )

    # 调用知识库工具（内部自动记录 TOOL_CALL_LOG）
    report = vulnerability_analysis(input_text)

    return {
        "tool": "vulnerability",
        "input_text": input_text,
        "report": report,
    }


@app.post("/tool/troubleshoot")
async def tool_troubleshoot(request: Request):
    """
    运维故障排查工具接口
    - 请求体：{"input_text": "故障描述"}
    - 调用 windows_ops_troubleshoot 返回格式化排查方案
    """
    data = await _parse_body(request)
    input_text = (data.get("input_text") or "").strip()

    if not input_text:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "input_text is required", "data": None},
        )

    report = windows_ops_troubleshoot(input_text)

    return {
        "tool": "troubleshoot",
        "input_text": input_text,
        "report": report,
    }


@app.post("/tool/audit")
async def tool_audit(request: Request):
    """
    安全配置审计工具接口
    - 请求体：{"input_text": "配置描述"}
    - 调用 security_audit 返回格式化审计报告
    """
    data = await _parse_body(request)
    input_text = (data.get("input_text") or "").strip()

    if not input_text:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "input_text is required", "data": None},
        )

    report = security_audit(input_text)

    return {
        "tool": "audit",
        "input_text": input_text,
        "report": report,
    }


@app.post("/tool/execute")
async def tool_execute(request: Request):
    """
    统一工具调用入口
    - 请求体：{"tool_name": "vulnerability|troubleshoot|audit", "input_text": "..."}
    - 通过 execute_tool 按名称调度工具，找不到工具时在 result 中返回错误提示
    """
    data = await _parse_body(request)
    tool_name = (data.get("tool_name") or "").strip()
    input_text = data.get("input_text") or ""

    if not tool_name:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "tool_name is required", "data": None},
        )

    # execute_tool 内部：查找注册表 -> 执行 -> 记录日志，永不抛异常
    result = execute_tool(tool_name, input_text=input_text)

    return {
        "tool_name": tool_name,
        "input_text": input_text,
        "result": result,
    }


# ============================================================
# 审计日志接口
# ============================================================
@app.get("/logs")
async def get_logs():
    """
    审计日志：返回 TOOL_CALL_LOG 中的所有工具调用记录
    每条记录包含：时间戳 / 工具名 / 输入参数 / 执行结果
    """
    data = [
        {
            "time": log["time"],                                   # 时间戳
            "tool": log["tool"],                                   # 工具名
            "input": log["input_preview"],                         # 输入参数（截断预览）
            "result": "、".join(log["matched"]),                   # 执行结果（命中项/调度状态/错误信息）
        }
        for log in TOOL_CALL_LOG
    ]
    return {"code": 200, "total": len(data), "data": data}


# ============================================================
# MCP 风格接口：对标 MCP 协议的工具注册标准（tools/list）
# ============================================================

# Python 类型 -> JSON Schema 类型的映射
_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _extract_param_docs(fn) -> dict:
    """
    从函数 docstring 的 Args 段提取参数说明
    例如 docstring 中 "input_text: 待分析的文本..." -> {"input_text": "待分析的文本..."}
    """
    docs = {}
    in_args = False
    for line in (inspect.getdoc(fn) or "").splitlines():
        s = line.strip()
        if not in_args:
            if s.lower().rstrip(":") == "args":
                in_args = True
            continue
        # 遇到空行或下一个段落（Returns: / Raises:）时结束
        if not s or s.lower().rstrip(":") in ("returns", "raises"):
            break
        if ":" in s:
            name, desc = s.split(":", 1)
            docs[name.strip()] = desc.strip()
    return docs


def _build_tool_schema(name: str, fn) -> dict:
    """
    把注册表中的一个工具函数转成 MCP 风格的 Schema
    {
      "name": 工具名,
      "description": 描述（docstring 首行）,
      "parameters": {参数名: {"type": 类型, "description": 说明, "default": 默认值(可选)}}
    }
    """
    description = (inspect.getdoc(fn) or "无描述").split("\n")[0].strip()
    param_docs = _extract_param_docs(fn)

    parameters = {}
    for pname, p in inspect.signature(fn).parameters.items():
        # 跳过 *args / **kwargs
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        # 推断参数类型：有注解用注解，没有默认按 string
        ptype = _TYPE_MAP.get(p.annotation, "string") \
            if p.annotation is not inspect.Parameter.empty else "string"
        entry = {
            "type": ptype,
            "description": param_docs.get(pname, f"参数 {pname}"),
        }
        # 有默认值时附带默认值
        if p.default is not inspect.Parameter.empty:
            entry["default"] = p.default
        parameters[pname] = entry

    return {"name": name, "description": description, "parameters": parameters}


@app.get("/mcp/tools")
async def mcp_tools():
    """
    MCP 风格工具注册接口（对标 MCP 协议的 tools/list）
    返回 TOOLS_REGISTRY 中所有工具的 JSON Schema 描述，
    外部 Agent 平台可据此发现并调用本服务的工具。
    """
    try:
        tools = [_build_tool_schema(name, fn) for name, fn in TOOLS_REGISTRY.items()]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"生成工具 Schema 失败：{e}"})

    return {"tools": tools}


# ============================================================
# Agent SSE 流式接口
# ============================================================
@app.post("/agent/stream")
async def agent_stream(request: Request):
    """
    ReAct Agent 流式问答接口（SSE）
    - 请求体：{"query": "用户问题"}
    - 调用 ReActAgent.run_stream，把推理每一步用 Server-Sent Events 推给前端
    - 事件序列（每步三个事件 + 结束标记）：
        data: {"type": "step_start",  "content": "thought"}
        data: {"type": "step_content", "content": "用户询问..."}
        data: {"type": "step_end"}
        ...
        data: {"type": "done"}
        data: [DONE]
    """
    data = await _parse_body(request)
    query = (data.get("query") or "").strip()

    if not query:
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "query is required", "data": None},
        )

    async def event_generator():
        # 队列：Agent 推理线程/协程 与 SSE 输出之间的桥梁
        queue: asyncio.Queue = asyncio.Queue()

        def callback(step_type: str, content: str) -> None:
            """把 Agent 的每一步推进队列（同步回调，配合无界队列不阻塞）"""
            queue.put_nowait((step_type, content))

        async def run_task():
            """后台执行 Agent 循环，结束后放入结束哨兵"""
            try:
                await agent.run_stream(query, callback)
            except Exception as e:
                queue.put_nowait(("error", f"Agent 执行出错：{e}"))
            finally:
                queue.put_nowait(None)  # None 表示事件流结束

        task = asyncio.create_task(run_task())

        # 逐条取出事件，按约定的三段式格式输出
        while True:
            item = await queue.get()
            if item is None:
                break
            step_type, content = item
            yield 'data: ' + json.dumps(
                {"type": "step_start", "content": step_type}, ensure_ascii=False) + "\n\n"
            yield 'data: ' + json.dumps(
                {"type": "step_content", "content": content}, ensure_ascii=False) + "\n\n"
            yield 'data: {"type": "step_end"}\n\n'

        # 结束标记
        yield 'data: ' + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止 Nginx 等代理缓冲导致前端收不到流
        },
    )


@app.get("/agent", response_class=HTMLResponse)
@app.get("/agent_index.html", response_class=HTMLResponse)  # 别名：供 Trae 预览按文件名访问
async def agent_page():
    """ReAct Agent 推理演示页面（agent_index.html），复用 _serve_html 的异常处理"""
    return _serve_html("agent_index.html")


# ============================================================
# 以下为新增内容：测试文本 + 三种分块函数 + 测试代码
# ============================================================

# 测试文本
test_content = """Web 安全漏洞知识库

攻击原理
攻击者通过利用用户输入中拼写错误 SQL 注入漏洞，成功获取用户的敏感信息。

识别特征（日志回显）
请求参数中出现以下关键词：OR 1=1、UNION SELECT、' OR '1'='1

SQL 注入
SQL 注入是攻击者常用的手法之一，通过向数据库插入恶意代码，实现远程命令执行。

XSS
XSS 攻击是指攻击者在网站上注入恶意代码，当用户浏览该网站时，恶意代码会被执行，导致浏览器执行恶意脚本，从而窃取用户的敏感信息。

文件上传
文件上传漏洞是指攻击者在网站上传包含恶意代码的文件，当用户访问该网站时，恶意代码会被执行，导致系统权限提升、数据泄露等安全问题。

CSRF
CSRF 攻击是指攻击者在网站上发送恶意请求，当用户点击该请求时，恶意请求会被执行，导致用户执行某些操作，如修改个人信息、转账等。"""


def chunk_fixed(text: str, size: int = 300) -> list:
    """固定大小分块，不重叠。直接按字符数切片。"""
    if not text or size <= 0:
        return []
    chunks = []
    # 每次从 start 位置取 size 个字符，步长就是 size（无重叠）
    for start in range(0, len(text), size):
        chunks.append(text[start:start + size])
    return chunks


def chunk_overlap(text: str, size: int = 300, overlap: int = 100) -> list:
    """固定大小 + 重叠分块。相邻块保留 overlap 个字符的重叠区域，保证语义不被切断。"""
    if not text or size <= 0:
        return []
    # overlap 不能大于等于 size，否则会死循环
    if overlap < 0 or overlap >= size:
        overlap = 0

    chunks = []
    start = 0            # 当前块的起始位置
    step = size - overlap  # 每次窗口向后滑动的距离
    while start < len(text):
        chunks.append(text[start:start + size])
        # 已经覆盖到末尾则退出
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunk_semantic(text: str, max_size: int = 400) -> list:
    """按句子边界分块，尽量保留完整句子。

    思路：先按句子结束符（。！？.!?\n）切出短句，再把短句逐个累加到当前块，
    超过 max_size 时另起一块。保证不把一个句子拆到两块里。
    """
    if not text:
        return []

    import re
    # 按常见句末标点切分，保留分隔符
    sentences = re.split(r'([。！？.!?\n])', text)

    # 重组：split 会把分隔符单独作为一项，需要把句子和紧随的分隔符合并
    merged = []
    i = 0
    while i < len(sentences):
        seg = sentences[i]
        # 如果下一项是分隔符，就合并进去
        if i + 1 < len(sentences) and re.fullmatch(r'[。！？.!?\n]', sentences[i + 1]):
            seg += sentences[i + 1]
            i += 2
        else:
            i += 1
        if seg.strip():
            merged.append(seg)

    chunks = []
    current = ""
    for sentence in merged:
        # 加进去后还没超 max_size，就继续累加
        if len(current) + len(sentence) <= max_size:
            current += sentence
        else:
            # 超了就先存下当前块，再把当前句子放到新块
            if current.strip():
                chunks.append(current.strip())
            # 单个句子本身就超过 max_size，直接作为一块，避免死循环
            if len(sentence) > max_size:
                chunks.append(sentence.strip())
                current = ""
            else:
                current = sentence

    # 循环结束后把最后一个块也加上
    if current.strip():
        chunks.append(current.strip())

    return chunks


if __name__ == "__main__":
    # 先跑三种分块的测试
    print("=" * 60)
    print("【分块函数测试】")
    print("=" * 60)
    print(f"原始文本长度：{len(test_content)} 字符\n")

    # 测试 chunk_fixed
    print("─" * 60)
    print("① chunk_fixed(size=300) — 固定大小，不重叠")
    fixed_chunks = chunk_fixed(test_content, 300)
    print(f"共 {len(fixed_chunks)} 块")
    for i, c in enumerate(fixed_chunks):
        print(f"\n[块 {i + 1}] 长度={len(c)}")
        print(c)

    # 测试 chunk_overlap
    print("\n" + "─" * 60)
    print("② chunk_overlap(size=300, overlap=100) — 固定大小 + 重叠")
    overlap_chunks = chunk_overlap(test_content, 300, 100)
    print(f"共 {len(overlap_chunks)} 块")
    for i, c in enumerate(overlap_chunks):
        print(f"\n[块 {i + 1}] 长度={len(c)}")
        print(c)

    # 测试 chunk_semantic
    print("\n" + "─" * 60)
    print("③ chunk_semantic(max_size=400) — 按句子边界分块")
    semantic_chunks = chunk_semantic(test_content, 400)
    print(f"共 {len(semantic_chunks)} 块")
    for i, c in enumerate(semantic_chunks):
        print(f"\n[块 {i + 1}] 长度={len(c)}")
        print(c)

    print("\n" + "=" * 60)
    print("【FastAPI 服务启动】访问 http://0.0.0.0:8002")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002)
