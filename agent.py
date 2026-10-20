"""agent.py - ReAct Agent（思考 → 行动 → 观察 → 重复）

串联项目各模块：
- llm_client.LLMClient          大模型客户端（Day 2）
- tools.TOOLS_REGISTRY/execute_tool  工具注册表与统一调度器（Day 5）
- agent_core.parse_agent_output 解析 LLM 输出中的 <<THINK>>/<<ACTION>> 标签（Day 6-2）

工作流程：
1. 用户问题加入记忆
2. ReAct 循环：LLM 思考并输出 <<ACTION>>: 工具名|参数 → 执行工具 → 观察结果回填记忆
3. Agent 认为信息足够时输出 <<ACTION>>: final|最终回答，循环结束
4. Agent 回答加入记忆，实现多轮对话上下文
"""

import asyncio
import inspect
from typing import Dict, List, Optional

from llm_client import LLMClient
from tools import TOOLS_REGISTRY, execute_tool
from agent_core import parse_agent_output


class ReActAgent:
    """基于 ReAct 范式的智能体：思考(THINK) → 行动(ACTION) → 观察(Observation) → 循环"""

    def __init__(self, llm: Optional[LLMClient] = None, max_steps: int = 5):
        """
        Args:
            llm: 大模型客户端，不传则新建（未配置密钥时自动 Mock 模式）
            max_steps: 单次任务的最大思考-行动循环次数（防止死循环）
        """
        # 1. 初始化 LLM 客户端
        self.llm = llm or LLMClient()
        self.max_steps = max_steps

        # 2. 发现可用工具：从 TOOLS_REGISTRY 提取工具名和 docstring 首行描述
        self.tools_desc: Dict[str, str] = self._discover_tools()

        # 3. 构建系统提示词（告诉 Agent 如何思考、有哪些工具）
        self.system_prompt = self._build_system_prompt()

        # 4. 对话记忆：[{"role": "user"/"assistant", "content": ...}, ...]
        self.memory: List[Dict[str, str]] = []

        print(f"[ReActAgent] 初始化完成，可用工具：{list(self.tools_desc.keys())}，"
              f"最大循环步数：{max_steps}")

    # --------------------------------------------------------
    # 工具发现与系统提示词
    # --------------------------------------------------------
    @staticmethod
    def _discover_tools() -> Dict[str, str]:
        """发现可用工具：工具名 -> 描述（取函数 docstring 的第一行）"""
        desc = {}
        for name, fn in TOOLS_REGISTRY.items():
            doc = inspect.getdoc(fn) or "无描述"
            desc[name] = doc.split("\n")[0].strip()
        return desc

    def _build_system_prompt(self) -> str:
        """构建系统提示词：角色设定 + 工具清单 + ReAct 输出格式约定"""
        # 把工具清单拼成 "- 工具名: 描述" 的列表
        tool_lines = "\n".join(f"- {name}: {desc}" for name, desc in self.tools_desc.items())

        return f"""你是一个安全运维智能助手（ReAct Agent），可以通过调用工具来回答用户问题。

可用工具：
{tool_lines}

你必须严格按照以下格式输出（每次一组）：
<<THINK>>: 你的思考过程（分析问题、判断是否已有足够信息）
<<ACTION>>: 工具名|参数

规则：
1. 需要工具帮忙时，ACTION 填 "工具名|参数"，参数是要传给工具的文本
2. 已有足够信息时，ACTION 填 "final|给用户的最终回答"
3. 每次只输出一组 THINK 和 ACTION，不要编造工具的输出结果
4. 工具的执行结果会以 [工具观察] 开头的消息提供给你
5. 闲聊或不需要工具的问题，直接输出 <<ACTION>>: final|你的回答"""

    # --------------------------------------------------------
    # 工具执行
    # --------------------------------------------------------
    def _execute_action(self, tool_name: str, args: str) -> str:
        """
        执行工具调用，返回观察结果

        Args:
            tool_name: 工具名（来自 ACTION 标签）
            args: 工具参数（第一个 | 之后的内容）

        Returns:
            工具执行结果字符串（超长时截断，避免撑爆上下文）
        """
        # execute_tool 内部：查注册表 → 执行 → 记日志，找不到工具也不会抛异常
        observation = execute_tool(tool_name, input_text=(args or "").strip())

        # 观察结果过长时截断，保留关键开头部分
        if len(observation) > 500:
            observation = observation[:500] + "\n...(观察结果过长已截断)"
        return observation

    # --------------------------------------------------------
    # 记忆管理
    # --------------------------------------------------------
    def clear_memory(self) -> None:
        """清空对话记忆，开始全新会话"""
        self.memory = []
        print("[ReActAgent] 记忆已清空")

    def get_memory_summary(self) -> Dict:
        """
        获取记忆摘要（不返回全文，只统计条数和最近一轮问答）
        注意：以 [工具观察] 开头的 user 消息是中间观察，不计入用户提问
        """
        user_msgs = [m["content"] for m in self.memory
                     if m["role"] == "user" and not m["content"].startswith("[工具观察]")]
        assistant_msgs = [m["content"] for m in self.memory if m["role"] == "assistant"]

        return {
            "总消息数": len(self.memory),
            "用户提问数": len(user_msgs),
            "助手消息数": len(assistant_msgs),
            "最近提问": user_msgs[-1] if user_msgs else None,
            "最近回答": (assistant_msgs[-1][:60] + "...") if assistant_msgs and len(assistant_msgs[-1]) > 60
                        else (assistant_msgs[-1] if assistant_msgs else None),
        }

    # --------------------------------------------------------
    # ReAct 主循环
    # --------------------------------------------------------
    def run(self, user_input: str) -> str:
        """
        处理一次用户输入：ReAct 循环（思考 → 行动 → 观察 → 重复）

        Args:
            user_input: 用户问题

        Returns:
            Agent 的最终回答字符串
        """
        # 用户问题加入记忆
        self.memory.append({"role": "user", "content": user_input})

        answer: Optional[str] = None

        for step in range(1, self.max_steps + 1):
            # 构建 messages = [system] + memory
            messages = [{"role": "system", "content": self.system_prompt}] + self.memory

            # 调用 LLM（失败则直接返回错误信息，不中断会话）
            try:
                resp = self.llm.chat(messages)
            except Exception as e:
                answer = f"LLM 调用失败：{e}"
                self.memory.append({"role": "assistant", "content": answer})
                return answer

            # 解析 LLM 输出中的 THINK / ACTION
            parsed = parse_agent_output(resp)
            tool, args = parsed["tool"], parsed["args"]

            # ---- 情况 1：Agent 决定直接回答（final 动作）----
            if tool == "final":
                answer = args or parsed["think"] or resp
                self.memory.append({"role": "assistant", "content": answer})
                print(f"  [第{step}轮] 思考: {parsed['think'][:40]}")
                print(f"  [第{step}轮] 决定直接回答")
                break

            # ---- 情况 2：没有 ACTION 标签（THINK 即答案，或用原始输出兜底）----
            if tool is None:
                answer = parsed["think"] or parsed["action_raw"] or resp
                self.memory.append({"role": "assistant", "content": answer})
                print(f"  [第{step}轮] 无动作，直接回答")
                break

            # ---- 情况 3：调用工具 ----
            print(f"  [第{step}轮] 思考: {parsed['think'][:40]}")
            print(f"  [第{step}轮] 调用工具: {tool} | 参数: {(args or '')[:30]}")

            # 思考与动作加入记忆（保留完整决策轨迹）
            self.memory.append({"role": "assistant", "content": resp})

            # 执行工具并把观察结果加入记忆（用 user 角色回填，供下一轮 LLM 看到）
            observation = self._execute_action(tool, args)
            print(f"  [第{step}轮] 观察: {observation[:40].replace(chr(10), ' ')}...")
            self.memory.append({"role": "user", "content": f"[工具观察] {observation}"})

        # 达到最大循环次数仍未得出答案
        if answer is None:
            answer = f"抱歉，经过 {self.max_steps} 轮思考仍未能完成任务，请换一种问法或缩小问题范围。"
            self.memory.append({"role": "assistant", "content": answer})
            print(f"  [循环终止] 达到最大步数 {self.max_steps}")

        return answer

    # --------------------------------------------------------
    # ReAct 流式循环（异步，供 SSE 接口使用）
    # --------------------------------------------------------
    async def run_stream(self, user_input: str, callback) -> str:
        """
        run 的流式异步版本：每完成一个步骤，调用 callback(step_type, content)

        Args:
            user_input: 用户问题
            callback:   回调函数，参数为 (step_type, content)
                        step_type 取值：thought / action / observation / answer

        Returns:
            Agent 的最终回答字符串（与 run 一致，记忆逻辑完全相同）
        """
        # 用户问题加入记忆
        self.memory.append({"role": "user", "content": user_input})

        answer: Optional[str] = None

        for step in range(1, self.max_steps + 1):
            # 构建 messages = [system] + memory
            messages = [{"role": "system", "content": self.system_prompt}] + self.memory

            # LLM 调用是同步阻塞的，放到线程池执行，避免卡住事件循环
            try:
                resp = await asyncio.to_thread(self.llm.chat, messages)
            except Exception as e:
                answer = f"LLM 调用失败：{e}"
                self.memory.append({"role": "assistant", "content": answer})
                callback("answer", answer)
                return answer

            # 解析 THINK / ACTION
            parsed = parse_agent_output(resp)
            tool, args = parsed["tool"], parsed["args"]

            # 步骤 1：思考过程（每轮必有）
            callback("thought", parsed["think"] or "（本轮无思考内容）")

            # ---- 情况 1：final 直接回答 ----
            if tool == "final":
                answer = args or parsed["think"] or resp
                self.memory.append({"role": "assistant", "content": answer})
                callback("action", "final（信息已足够，直接回答）")
                callback("answer", answer)
                return answer

            # ---- 情况 2：没有 ACTION 标签 ----
            if tool is None:
                answer = parsed["think"] or parsed["action_raw"] or resp
                self.memory.append({"role": "assistant", "content": answer})
                callback("answer", answer)
                return answer

            # ---- 情况 3：调用工具 ----
            callback("action", f"调用工具 {tool}，参数：{(args or '').strip()[:80]}")
            # 思考与动作加入记忆（保留完整决策轨迹）
            self.memory.append({"role": "assistant", "content": resp})

            # 执行工具（同步快速调用，放线程池保持事件循环畅通）
            observation = await asyncio.to_thread(self._execute_action, tool, args)
            callback("observation", observation)

            # 观察结果回填记忆，供下一轮 LLM 参考
            self.memory.append({"role": "user", "content": f"[工具观察] {observation}"})

        # 达到最大循环次数仍未得出答案
        if answer is None:
            answer = f"抱歉，经过 {self.max_steps} 轮思考仍未能完成任务，请换一种问法或缩小问题范围。"
            self.memory.append({"role": "assistant", "content": answer})
            callback("answer", answer)

        return answer


# ============================================================
# 测试代码：多轮对话（工具调用 + 记忆 + 无需工具的闲聊）
# ============================================================
if __name__ == "__main__":
    agent = ReActAgent(max_steps=4)

    # ---------- 第一轮：需要调用工具 ----------
    print("\n" + "=" * 60)
    print("【第一轮】漏洞分析问题（应触发 vulnerability 工具）")
    print("=" * 60)
    q1 = "日志里出现了 ' OR '1'='1，这是什么安全问题？该怎么处理？"
    print(f"用户：{q1}")
    a1 = agent.run(q1)
    print(f"\nAgent：{a1[:200]}")

    # ---------- 第二轮：依赖记忆的追问（不含关键词） ----------
    print("\n" + "=" * 60)
    print("【第二轮】追问（不包含任何工具关键词，考验记忆）")
    print("=" * 60)
    q2 = "这个漏洞的严重程度到底如何？"
    print(f"用户：{q2}")
    a2 = agent.run(q2)
    print(f"\nAgent：{a2[:200]}")

    # ---------- 第三轮：无需工具的一般性问题 ----------
    print("\n" + "=" * 60)
    print("【第三轮】一般性问候（Agent 应直接回答，不调用工具）")
    print("=" * 60)
    q3 = "你好，请用一句话介绍你自己"
    print(f"用户：{q3}")
    a3 = agent.run(q3)
    print(f"\nAgent：{a3[:200]}")

    # ---------- 记忆摘要 ----------
    print("\n" + "=" * 60)
    print("【记忆摘要】")
    print("=" * 60)
    for k, v in agent.get_memory_summary().items():
        print(f"{k}：{v}")

    # ---------- 清空记忆 ----------
    print("\n" + "=" * 60)
    print("【清空记忆】")
    print("=" * 60)
    agent.clear_memory()
    for k, v in agent.get_memory_summary().items():
        print(f"{k}：{v}")
