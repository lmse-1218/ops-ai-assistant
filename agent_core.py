"""agent_core.py - Agent 输出解析器

从 LLM 输出文本中提取 <<THINK>>（思考过程）和 <<ACTION>>（动作）标签的内容：
- 兼容中英文冒号（: 和 ：），且允许冒号省略
- ACTION 内容格式：工具名|参数，只按第一个 | 分割（参数内部可含 |）
- 健壮处理边界情况：标签不存在、内容为空、多行内容

典型的 LLM（ReAct Agent）输出格式：
    <<THINK>>: 用户想分析漏洞，我需要调用漏洞分析工具
    <<ACTION>>: vulnerability|日志中出现 UNION SELECT
"""

import re
from typing import Dict, Optional

# 匹配 <<THINK>> / <<ACTION>> 标签及其内容（内容到下一个标签或文本结束为止）
# - [:：]?      兼容中英文冒号，冒号也可省略
# - .*?         非贪婪匹配标签内容
# - (?=...|$)   前瞻：遇到下一个标签或字符串结尾即停止
# - re.DOTALL    让 . 也能匹配换行，支持多行内容
# - re.IGNORECASE 兼容 <<think>> 等大小写变体
_TAG_RE = re.compile(
    r"<<\s*(THINK|ACTION)\s*>>\s*[:：]?\s*(.*?)\s*(?=<<\s*(?:THINK|ACTION)\s*>>|$)",
    re.DOTALL | re.IGNORECASE,
)


def parse_agent_output(text: str) -> Dict[str, Optional[str]]:
    """
    解析 LLM 输出，提取 <<THINK>> 与 <<ACTION>> 标签内容

    Args:
        text: LLM 的原始输出文本

    Returns:
        dict，包含四个字段：
        - think:      思考过程内容（标签不存在或为空时为 ""）
        - action_raw: ACTION 标签的原始内容（不存在时为 ""）
        - tool:       工具名（ACTION 缺失或内容为空时为 None）
        - args:       第一个 | 之后的参数串（无参数时为 ""；ACTION 缺失时为 None）

        若同一标签出现多次，以最后出现的为准（Agent 重新决策时最新决策生效）
    """
    # 默认返回结构：所有字段按"不存在"处理
    result: Dict[str, Optional[str]] = {
        "think": "",
        "action_raw": "",
        "tool": None,
        "args": None,
    }

    # 边界情况：输入为 None / 空串 / 纯空白
    if not text or not text.strip():
        return result

    # 1. 提取所有标签内容；同一标签多次出现时后者覆盖前者
    for tag, content in _TAG_RE.findall(text):
        content = content.strip()
        if tag.upper() == "THINK":
            result["think"] = content
        else:
            result["action_raw"] = content

    # 2. 解析 ACTION：按第一个 | 分割为 工具名 和 参数
    if result["action_raw"]:
        parts = result["action_raw"].split("|", 1)
        result["tool"] = parts[0].strip()
        # 分割后还有第二段则取参数（内部的 | 原样保留），没有参数给空串
        result["args"] = parts[1].strip() if len(parts) > 1 else ""

    return result


# ============================================================
# 测试代码：覆盖正常情况与各类边界情况
# ============================================================
if __name__ == "__main__":
    test_cases = [
        # (用例名, 输入文本, 期望字段)
        ("英文冒号",
         "<<THINK>>: 用户想分析漏洞\n<<ACTION>>: vulnerability|日志出现 UNION SELECT",
         {"think": "用户想分析漏洞", "tool": "vulnerability", "args": "日志出现 UNION SELECT"}),

        ("中文冒号",
         "<<THINK>>：需要排查CPU问题\n<<ACTION>>：troubleshoot|CPU过高",
         {"think": "需要排查CPU问题", "tool": "troubleshoot", "args": "CPU过高"}),

        ("无标签（普通文本）",
         "这是一段普通文本，没有任何标签",
         {"think": "", "tool": None, "args": None}),

        ("空字符串输入",
         "",
         {"think": "", "tool": None, "args": None}),

        ("THINK 内容为空",
         "<<THINK>>:\n<<ACTION>>: audit|检查密码策略",
         {"think": "", "tool": "audit", "args": "检查密码策略"}),

        ("ACTION 内容为空",
         "<<THINK>>: 让我想一想\n<<ACTION>>:",
         {"think": "让我想一想", "tool": None, "args": None}),

        ("ACTION 无参数（没有 |）",
         "<<ACTION>>: security_audit",
         {"think": "", "tool": "security_audit", "args": ""}),

        ("参数含多个 | 只切第一个",
         "<<THINK>>: 组合参数\n<<ACTION>>: execute|a|b|c",
         {"think": "组合参数", "tool": "execute", "args": "a|b|c"}),

        ("多行 THINK 内容",
         "<<THINK>>: 第一步先分析日志\n第二步决定调用漏洞工具\n<<ACTION>>: vulnerability|SQL注入",
         {"think": "第一步先分析日志\n第二步决定调用漏洞工具",
          "tool": "vulnerability", "args": "SQL注入"}),
    ]

    passed = 0
    for name, text, expect in test_cases:
        r = parse_agent_output(text)
        # 逐字段比对期望值
        ok = all(r[k] == v for k, v in expect.items())
        passed += ok
        print(f"{'✓ 通过' if ok else '✗ 失败'}  {name}")
        if not ok:
            print(f"    输入     : {text!r}")
            print(f"    实际结果 : {r}")
            print(f"    期望匹配 : {expect}")
        else:
            print(f"    THINK -> {r['think']!r}")
            print(f"    ACTION -> tool={r['tool']!r}, args={r['args']!r}")

    print("-" * 50)
    print(f"测试结果：{passed}/{len(test_cases)} 通过")
