# 安全智能运维助手（RAG + ReAct Agent）

面向网络运维场景的智能助手。用户用自然语言描述问题，Agent 按 ReAct 范式循环推理：
**思考 → 调用工具 → 观察结果 → 继续推理**，直到给出最终回答。

---

## 三个界面

| 页面 | 路径 | 用途 |
|---|---|---|
| 安全智能运维平台 | `/` | 漏洞分析、故障排查、配置审计三个工具的统一入口 |
| 浏览器智能助手 | `/group_index.html` | 对话式问答，带检索和工具调用 |
| ReAct Agent 推理演示 | `/agent` | 把每一步思考、工具调用、观察结果铺开显示 |

---

## 能做什么

| 能力 | 说明 |
|---|---|
| 漏洞分析 | 内置 8 类常见漏洞知识：等级 / CVSS / CWE / OWASP / 描述 / 修复建议 |
| 运维故障排查 | 内置 6 类 Windows 运维故障：现象 / 成因 / 修复方案 |
| 配置审计 | 内置 5 类配置项审计规则 |
| 混合检索问答 | 向量检索 + 关键词检索，覆盖自然语言提问和专有名词查询 |
| 推理过程可视化 | 每一步思考与工具调用都可查看，便于定位是检索错了还是推理错了 |

---

## 技术结构

```
用户提问
   ↓
[检索层]  ChromaDB 向量检索  +  BM25 关键词检索
   ↓              RRF（倒数排名融合）合并两路结果
[推理层]  ReAct Agent —— 解析 <<THINK>> / <<ACTION>> 标签
   ↓
[工具层]  漏洞分析 / 故障排查 / 配置审计
   ↓
[输出层]  最终回答 + 工具调用日志
```

### 几个设计取舍

**1. 检索用混合，而不是纯向量**
纯向量检索对「CVSS」「CWE-79」这类专有名词召回不稳定——语义相近但字面不同时容易漏。
加上 BM25 关键词检索后，用 RRF 把两路结果的排名融合，自然语言提问和专有名词查询都能覆盖。

**2. 用 ReAct 循环，而不是一次性生成**
运维问题经常要串起来查：先查漏洞库判断类型，再查故障库对照现象。
分步推理的好处不是答得更快，而是**看得出它错在哪一步**——是一开始检索就偏了，还是工具调对了但结论推错了。

**3. 每一步工具调用都记日志**
记录时间 / 工具名 / 输入摘要 / 命中结果。
出问题时能直接回溯，不用靠猜。

**4. 没有密钥也能跑**
大模型客户端和向量化客户端都做了 Mock 模式：未配置密钥时自动切换到模拟返回，
保证项目在任何机器上 clone 下来就能启动，不会卡在环境配置上。

---

## 项目结构

```
├── group_project.py        FastAPI 服务入口（路由 + 页面渲染）
├── agent.py                ReAct Agent 主循环
├── agent_core.py           <<THINK>> / <<ACTION>> 标签解析
├── rag_core.py             混合检索核心（ChromaDB + BM25 + RRF）
├── tools.py                三套知识库与工具注册表
├── llm_client.py           大模型客户端（OpenAI 兼容，含 Mock 模式）
├── embedding_client.py     向量化客户端（含 Mock 模式）
├── chunks_result.json      知识库分块数据
├── index.html              安全智能运维平台
├── group_index.html        浏览器智能助手
├── agent_index.html        ReAct Agent 推理演示
└── requirements.txt
```

---

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 安全智能运维平台页面 |
| GET | `/group_index.html` | 浏览器智能助手页面 |
| GET | `/agent` | Agent 推理演示页面 |
| POST | `/browser-assistant` | 对话式问答 |
| POST | `/rag-query` | 单独调用混合检索 |
| POST | `/tool/analyze` | 漏洞分析 |
| POST | `/tool/troubleshoot` | 故障排查 |
| POST | `/tool/audit` | 配置审计 |
| POST | `/tool/execute` | 按工具名统一调度 |
| POST | `/agent/stream` | Agent 流式推理 |
| GET | `/logs` | 查看工具调用日志 |

---

## 本地运行

```bash
pip install fastapi uvicorn pydantic openai python-dotenv chromadb rank-bm25
```

在项目根目录创建 `.env`：

```
API_KEY=你的密钥
BASE_URL=接口的 base_url
```

启动：

```bash
uvicorn group_project:app --reload --port 8002
```

浏览器打开 <http://127.0.0.1:8002>

> 不配置 `.env` 也能启动，此时自动进入 Mock 模式，返回模拟数据，用于验证流程是否跑通。

---

## 技术栈

Python 3 / FastAPI / ChromaDB / rank-bm25 / OpenAI 兼容接口 / 原生 HTML + JS

---

## 说明

本项目为个人独立完成。代码部分借助 AI 编程工具辅助生成；
检索方案、工具知识库结构与验收标准由本人确定。
