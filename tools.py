"""tools.py - 安全运维工具集

包含三套知识库和对应的分析工具（供后续 Function Calling / Agent 使用）：
1. VULNERABILITY_KB           漏洞知识库（8 种漏洞）
2. vulnerability_analysis()   漏洞分析工具：识别文本中的漏洞特征并输出报告
3. OPS_KB                     运维故障知识库（6 类故障）
4. windows_ops_troubleshoot() 故障排查工具：匹配故障特征输出排查方案
5. CONFIG_AUDIT_KB            配置审计知识库（5 类配置项）
6. security_audit()           配置审计工具：识别不安全配置输出加固建议
7. TOOL_CALL_LOG              全局工具调用日志（每次调用都会记录）
"""

import re
from datetime import datetime

# ============================================================
# 7. 工具调用日志：记录每次工具调用的 时间 / 工具名 / 输入摘要 / 命中结果
# ============================================================
TOOL_CALL_LOG = []


def _log_call(tool_name: str, input_text: str, matched_types: list) -> None:
    """向全局日志追加一条工具调用记录"""
    TOOL_CALL_LOG.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": tool_name,
        "input_preview": input_text[:50] + ("..." if len(input_text) > 50 else ""),
        "matched": matched_types,
    })


def _match_kb(input_text: str, kb: list) -> list:
    """
    通用知识库匹配：遍历知识库条目，检查其关键词是否出现在输入文本中
    - 不区分大小写（英文关键词统一转小写后匹配）
    - 返回命中的条目列表（可能命中多个）
    """
    text = input_text.lower()
    hits = []
    for entry in kb:
        # 任一关键词命中即认为该条目相关
        if any(kw.lower() in text for kw in entry["keywords"]):
            hits.append(entry)
    return hits


# ============================================================
# 1. 漏洞知识库（8 种漏洞）
#    每个漏洞包含：type / level / cvss / cwe / owasp / desc / fix / keywords
#    keywords 为匹配用的特征关键词
# ============================================================
VULNERABILITY_KB = [
    {
        "type": "SQL注入",
        "level": "高危",
        "cvss": 9.8,
        "cwe": "CWE-89",
        "owasp": "A03:2021 – Injection（注入）",
        "desc": "攻击者在用户输入中拼接恶意 SQL 语句，欺骗数据库执行非授权查询，导致数据泄露、篡改甚至远程命令执行。",
        "fix": "1) 使用参数化查询/预编译语句，禁止字符串拼接 SQL；2) 对输入做白名单校验；3) 数据库账号最小权限；4) 部署 WAF，对 OR 1=1、UNION SELECT 等特征告警。",
        "keywords": ["sql注入", "sql 注入", "sql inject", "or 1=1", "union select", "' or '1'='1", "sqlmap", "注入漏洞"],
    },
    {
        "type": "XSS（跨站脚本）",
        "level": "中危",
        "cvss": 6.1,
        "cwe": "CWE-79",
        "owasp": "A03:2021 – Injection（注入）",
        "desc": "攻击者向页面注入恶意脚本，用户浏览时脚本在浏览器执行，窃取 Cookie/会话或以用户身份伪造操作。",
        "fix": "1) 输出按 HTML/JS/URL 上下文编码；2) 输入过滤危险标签与事件属性；3) 配置 CSP 内容安全策略；4) Cookie 设置 HttpOnly；5) 优先使用框架自动转义。",
        "keywords": ["xss", "跨站脚本", "<script", "script>alert", "onerror", "onload=", "javascript:"],
    },
    {
        "type": "命令注入",
        "level": "严重",
        "cvss": 9.8,
        "cwe": "CWE-78",
        "owasp": "A03:2021 – Injection（注入）",
        "desc": "用户输入被拼接到操作系统命令中执行，攻击者可借此获取服务器 shell 权限。",
        "fix": "1) 避免拼接用户输入调用系统命令；2) 用语言内置库替代 shell 调用；3) 必要时对参数做白名单校验与转义；4) 以最小权限运行进程。",
        "keywords": ["命令注入", "command inject", "os.system", "exec(", "system(", "; rm ", "&& ping", "| cat", "cmd.exe"],
    },
    {
        "type": "权限绕过（越权访问）",
        "level": "高危",
        "cvss": 8.8,
        "cwe": "CWE-285",
        "owasp": "A01:2021 – Broken Access Control（失效的访问控制）",
        "desc": "鉴权逻辑缺陷使攻击者可访问未授权的功能或数据，包括水平越权（访问同角色他人数据）和垂直越权（获得更高权限）。",
        "fix": "1) 服务端对每个请求做权限校验，默认拒绝；2) 使用成熟鉴权框架，禁止仅前端隐藏入口；3) 敏感操作二次确认；4) 记录越权尝试审计日志。",
        "keywords": ["权限绕过", "越权", "未授权", "unauthorized", "broken access", "水平越权", "垂直越权", "access control"],
    },
    {
        "type": "目录遍历（路径穿越）",
        "level": "高危",
        "cvss": 7.5,
        "cwe": "CWE-22",
        "owasp": "A01:2021 – Broken Access Control（失效的访问控制）",
        "desc": "通过 ../ 等序列跳出受限目录，读取服务器任意文件（配置文件、密码文件等）。",
        "fix": "1) 对路径规范化后校验是否仍在允许目录内；2) 文件名白名单；3) 禁止用用户输入拼接文件路径；4) 运行账号最小文件系统权限。",
        "keywords": ["目录遍历", "路径穿越", "directory traversal", "path traversal", "../", "..\\", "/etc/passwd", "c:\\windows"],
    },
    {
        "type": "反序列化漏洞",
        "level": "严重",
        "cvss": 9.8,
        "cwe": "CWE-502",
        "owasp": "A08:2021 – Software and Data Integrity Failures（软件与数据完整性失效）",
        "desc": "不可信数据被反序列化为对象时触发已知 gadget 链，导致远程代码执行。",
        "fix": "1) 禁止对不可信数据做原生态反序列化；2) 改用 JSON 等纯数据格式；3) 必要时启用类型白名单校验；4) 及时更新反序列化组件。",
        "keywords": ["反序列化", "deserialization", "unserialize", "pickle.load", "yaml.load", "readobject", "objectinputstream"],
    },
    {
        "type": "任意文件上传",
        "level": "高危",
        "cvss": 8.8,
        "cwe": "CWE-434",
        "owasp": "A04:2021 – Insecure Design（不安全设计）",
        "desc": "未限制上传文件的类型与内容，攻击者可上传 webshell 获取服务器控制权。",
        "fix": "1) 扩展名与 MIME 白名单校验；2) 校验文件头（magic number）；3) 重命名并隔离存储；4) 上传目录禁止执行权限；5) 限制文件大小。",
        "keywords": ["文件上传", "上传漏洞", "webshell", "一句话木马", "shell.php", "shell.jsp", "upload"],
    },
    {
        "type": "CSRF（跨站请求伪造）",
        "level": "中危",
        "cvss": 6.5,
        "cwe": "CWE-352",
        "owasp": "A01:2021 – Broken Access Control（失效的访问控制）",
        "desc": "诱导已登录用户在第三方页面发起伪造请求，以用户身份执行转账、改密等敏感操作。",
        "fix": "1) 使用 CSRF Token（同步器模式）；2) Cookie 设置 SameSite 属性；3) 校验 Referer/Origin；4) 敏感操作要求二次认证。",
        "keywords": ["csrf", "xsrf", "跨站请求伪造", "跨站伪造"],
    },
]


# ============================================================
# 2. 漏洞分析工具
# ============================================================
def vulnerability_analysis(input_text: str) -> str:
    """
    漏洞分析工具：扫描输入文本，匹配漏洞知识库关键词，返回格式化漏洞报告

    Args:
        input_text: 待分析的文本（如日志片段、漏洞描述、代码片段）

    Returns:
        格式化的漏洞分析报告字符串；未命中时返回友好提示
    """
    input_text = (input_text or "").strip()
    if not input_text:
        return "⚠️ 输入为空：请提供待分析的文本（日志、漏洞描述或代码片段）。"

    hits = _match_kb(input_text, VULNERABILITY_KB)
    _log_call("vulnerability_analysis", input_text, [h["type"] for h in hits])

    # 未命中：返回友好提示，并给出使用建议
    if not hits:
        return (
            "✅ 未在输入中检测到已知漏洞特征。\n"
            "提示：可尝试包含以下关键词的文本：SQL注入 / XSS / 命令注入 / 越权 /\n"
            "目录遍历 / 反序列化 / 文件上传 / CSRF 等。"
        )

    # 命中：逐个输出漏洞详情
    lines = []
    lines.append("=" * 50)
    lines.append("🔍 漏洞分析报告")
    lines.append("=" * 50)
    lines.append(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"输入长度：{len(input_text)} 字符")
    lines.append(f"检出漏洞：{len(hits)} 个（按危险等级排序）")

    # 按危险等级从高到低排序：严重 > 高危 > 中危 > 低危
    level_order = {"严重": 0, "高危": 1, "中危": 2, "低危": 3}
    hits.sort(key=lambda h: level_order.get(h["level"], 9))

    for i, v in enumerate(hits, 1):
        lines.append("-" * 50)
        lines.append(f"【{i}】{v['type']}    等级：{v['level']}（CVSS {v['cvss']}）")
        lines.append(f"  CWE 编号　：{v['cwe']}")
        lines.append(f"  OWASP 分类：{v['owasp']}")
        lines.append(f"  漏洞描述　：{v['desc']}")
        lines.append(f"  修复建议　：{v['fix']}")

    lines.append("=" * 50)
    return "\n".join(lines)


# ============================================================
# 3. 运维故障知识库（6 类故障）
#    每个故障包含：type / level / symptom / cause / fix / keywords
# ============================================================
OPS_KB = [
    {
        "type": "CPU过高",
        "level": "P2-紧急",
        "symptom": "CPU 使用率持续高于 90%，系统卡顿、接口响应明显变慢。",
        "cause": "常见原因：死循环或正则回溯、慢 SQL 全表扫描、流量突增、异常进程（如挖矿程序）。",
        "fix": "1) top / 任务管理器定位高 CPU 进程；2) 分析线程栈或慢查询日志找到热点代码；3) 优化代码/加索引，必要时限流扩容；4) Windows 下用性能监视器与资源监视器持续跟踪。",
        "keywords": ["cpu过高", "cpu 过高", "cpu使用率", "cpu 使用率", "cpu占用", "cpu 100", "load过高", "高负载"],
    },
    {
        "type": "内存不足",
        "level": "P2-紧急",
        "symptom": "可用内存持续走低，出现 OutOfMemory 报错或进程被 OOM Killer 杀死。",
        "cause": "常见原因：内存泄漏、缓存无上限增长、一次性加载大文件、并发突增。",
        "fix": "1) 定位内存占用最高的进程；2) 抓取堆转储分析泄漏点；3) 给缓存设置上限并修复泄漏；4) 调整 JVM/.NET 内存参数或扩容。",
        "keywords": ["内存不足", "内存溢出", "oom", "out of memory", "内存占用高", "内存泄漏"],
    },
    {
        "type": "磁盘空间满",
        "level": "P2-紧急",
        "symptom": "写文件报 No space left on device，日志无法写入，数据库或服务异常。",
        "cause": "常见原因：日志未轮转清理、大文件/临时文件堆积、数据库或备份文件膨胀。",
        "fix": "1) df -h / 资源管理器定位大目录；2) 清理或轮转日志与临时文件；3) 配置 logrotate 与磁盘使用率告警（阈值 80%）；4) 必要时扩容磁盘。",
        "keywords": ["磁盘满", "磁盘空间", "disk full", "空间不足", "硬盘满", "no space left"],
    },
    {
        "type": "服务无响应",
        "level": "P1-致命",
        "symptom": "端口不通或请求持续超时，健康检查失败，用户无法访问。",
        "cause": "常见原因：进程崩溃、线程池/连接池耗尽、死锁、依赖服务阻塞。",
        "fix": "1) 检查进程与端口（netstat / Get-NetTCPConnection）；2) 查看崩溃日志与线程转储；3) 重启服务并做限流降级；4) 排查依赖服务健康状态。",
        "keywords": ["服务无响应", "服务挂了", "无法访问", "连接不上", "connection refused", "请求超时", "服务宕机", "timeout"],
    },
    {
        "type": "网络延迟高",
        "level": "P3-一般",
        "symptom": "ping/RTT 显著上升，页面加载慢，接口超时率升高、偶发丢包。",
        "cause": "常见原因：带宽打满、DNS 解析慢、跨机房链路抖动、路由异常。",
        "fix": "1) 分段 ping / tracert 定位延迟点；2) 检查带宽利用率与连接数；3) 优化 DNS、接入 CDN；4) 必要时联系运营商或切换线路。",
        "keywords": ["网络延迟", "延迟高", "网络慢", "丢包", "ping 高", "latency", "网络抖动"],
    },
    {
        "type": "日志报错",
        "level": "P3-一般",
        "symptom": "日志中密集出现 ERROR/Exception，可能伴随部分功能异常。",
        "cause": "常见原因：代码缺陷、配置错误、依赖服务异常、系统资源不足。",
        "fix": "1) 按时间与关键字聚类定位高频报错；2) 还原请求链路并复现问题；3) 修复后补充监控告警；4) 避免静默吞异常导致问题被掩盖。",
        "keywords": ["日志报错", "error", "exception", "异常日志", "traceback", "堆栈", "报错"],
    },
]


# ============================================================
# 4. 故障排查工具
# ============================================================
def windows_ops_troubleshoot(input_text: str) -> str:
    """
    故障排查工具：扫描输入文本，匹配运维知识库关键词，返回格式化排查方案

    Args:
        input_text: 故障描述或监控告警文本

    Returns:
        格式化的排查方案字符串；未命中时返回友好提示
    """
    input_text = (input_text or "").strip()
    if not input_text:
        return "⚠️ 输入为空：请描述故障现象（如“CPU 使用率 95% 持续 10 分钟”）。"

    hits = _match_kb(input_text, OPS_KB)
    _log_call("windows_ops_troubleshoot", input_text, [h["type"] for h in hits])

    if not hits:
        return (
            "✅ 未匹配到已知故障类型。\n"
            "提示：可尝试包含以下关键词的描述：CPU过高 / 内存不足 / 磁盘满 /\n"
            "服务无响应 / 网络延迟 / 日志报错 等。"
        )

    lines = []
    lines.append("=" * 50)
    lines.append("🖥️ 运维故障排查方案")
    lines.append("=" * 50)
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"匹配故障：{len(hits)} 类")

    # P1 最优先，依次排列
    level_order = {"P1-致命": 0, "P2-紧急": 1, "P3-一般": 2}
    hits.sort(key=lambda h: level_order.get(h["level"], 9))

    for i, f in enumerate(hits, 1):
        lines.append("-" * 50)
        lines.append(f"【{i}】{f['type']}    优先级：{f['level']}")
        lines.append(f"  故障现象　：{f['symptom']}")
        lines.append(f"  可能原因　：{f['cause']}")
        lines.append(f"  处理步骤　：{f['fix']}")

    lines.append("=" * 50)
    return "\n".join(lines)


# ============================================================
# 5. 配置审计知识库（5 类配置项）
#    每个配置项包含：type / level / desc / fix / keywords
# ============================================================
CONFIG_AUDIT_KB = [
    {
        "type": "密码策略",
        "level": "高",
        "desc": "弱口令策略（长度不足、无复杂度要求、不过期、无失败锁定）极易被暴力破解。",
        "fix": "1) 密码最小长度 ≥ 8（建议 12）；2) 必须包含大小写+数字+特殊字符；3) 定期更换并限制历史密码复用；4) 连续失败次数锁定账号；5) 禁用默认口令。",
        "keywords": ["密码策略", "密码强度", "弱口令", "password policy", "密码复杂度", "口令", "密码", "登录失败"],
    },
    {
        "type": "日志审计",
        "level": "中",
        "desc": "未开启或未集中收集审计日志，安全事件发生后无法追溯定位。",
        "fix": "1) 记录登录、鉴权、敏感操作日志；2) 日志集中收集到 SIEM；3) 日志保留 ≥ 180 天；4) 全网 NTP 时间同步；5) 日志防篡改（只追加存储）。",
        "keywords": ["日志审计", "审计日志", "操作日志", "audit", "日志留存", "审计策略"],
    },
    {
        "type": "防火墙规则",
        "level": "高",
        "desc": "规则过宽（如 0.0.0.0/0 放行管理端口）会显著扩大攻击面。",
        "fix": "1) 遵循最小化放行原则；2) 管理端口限制源 IP 或走 VPN；3) 定期梳理并回收僵尸规则；4) 出入站默认拒绝、按需放行。",
        "keywords": ["防火墙", "firewall", "端口开放", "安全组", "iptables", "0.0.0.0/0"],
    },
    {
        "type": "权限配置",
        "level": "高",
        "desc": "服务以管理员/root 运行或目录权限过宽，被攻破后影响面扩大。",
        "fix": "1) 使用最小权限专用账号运行服务；2) 文件/目录按需授权；3) 关键目录移除 Everyone/匿名写权限；4) 定期复核账号与权限清单。",
        "keywords": ["权限配置", "文件权限", "最小权限", "administrator", "root", "everyone", "授权"],
    },
    {
        "type": "SSL/TLS 配置",
        "level": "高",
        "desc": "使用过时协议（SSLv3/TLS 1.0）或弱加密套件、证书过期，可被降级或中间人攻击。",
        "fix": "1) 仅启用 TLS 1.2/1.3；2) 禁用弱加密套件；3) 证书到期监控与自动续期；4) 启用 HSTS；5) 定期用 SSL Labs 等工具扫描评分。",
        "keywords": ["ssl", "tls", "https", "证书", "加密套件", "cipher", "弱加密"],
    },
]


# ============================================================
# 6. 配置审计工具
# ============================================================
def security_audit(input_text: str) -> str:
    """
    配置审计工具：扫描输入文本，匹配配置审计知识库关键词，返回格式化审计报告

    Args:
        input_text: 待审计的配置描述（如扫描报告、配置清单、检查记录）

    Returns:
        格式化的配置审计报告字符串；未命中时返回友好提示
    """
    input_text = (input_text or "").strip()
    if not input_text:
        return "⚠️ 输入为空：请提供待审计的配置信息（如“服务器密码最小长度为 6 位”）。"

    hits = _match_kb(input_text, CONFIG_AUDIT_KB)
    _log_call("security_audit", input_text, [h["type"] for h in hits])

    if not hits:
        return (
            "✅ 未发现已知的不安全配置项。\n"
            "提示：可尝试包含以下关键词的配置描述：密码策略 / 日志审计 / 防火墙 /\n"
            "权限配置 / SSL / TLS 等。"
        )

    lines = []
    lines.append("=" * 50)
    lines.append("🛡️ 安全配置审计报告")
    lines.append("=" * 50)
    lines.append(f"审计时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"发现问题：{len(hits)} 项")

    for i, c in enumerate(hits, 1):
        lines.append("-" * 50)
        lines.append(f"【{i}】{c['type']}    风险等级：{c['level']}")
        lines.append(f"  问题描述　：{c['desc']}")
        lines.append(f"  加固建议　：{c['fix']}")

    lines.append("=" * 50)
    return "\n".join(lines)


# ============================================================
# 9. 工具注册表 + 统一调度入口（供 Function Calling / Agent 使用）
# ============================================================

# 工具注册表：工具名 -> 工具函数，Agent 只需知道名字即可调用
TOOLS_REGISTRY = {
    "vulnerability": vulnerability_analysis,
    "troubleshoot": windows_ops_troubleshoot,
    "audit": security_audit,
}


def execute_tool(tool_name: str, **kwargs) -> str:
    """
    统一工具调度入口：按名称从 TOOLS_REGISTRY 查找并执行工具

    Args:
        tool_name: 工具名称（vulnerability / troubleshoot / audit）
        **kwargs: 透传给工具的参数（如 input_text="..."）

    Returns:
        工具执行结果字符串；找不到工具或执行出错时返回错误提示
    """
    # 1. 查找工具
    tool = TOOLS_REGISTRY.get(tool_name)

    # 2. 找不到：记录失败日志并返回可用工具提示
    if tool is None:
        available = "、".join(TOOLS_REGISTRY.keys())
        _log_call(f"execute_tool({tool_name})", str(kwargs), ["未找到工具"])
        return f"❌ 未找到工具 '{tool_name}'，当前可用工具：{available}"

    # 3. 找到：执行工具（工具内部还会记录一次自己的调用日志）
    try:
        result = tool(**kwargs)
        _log_call(f"execute_tool({tool_name})",
                  str(kwargs.get("input_text", "")), ["调度成功"])
        return result
    except Exception as e:
        # 4. 执行出错：记录并返回错误信息，不抛出异常中断调用方
        _log_call(f"execute_tool({tool_name})",
                  str(kwargs.get("input_text", "")), [f"执行出错: {e}"])
        return f"❌ 工具 '{tool_name}' 执行出错：{e}"


# ============================================================
# 8. 测试代码：测试所有工具
# ============================================================
if __name__ == "__main__":
    print("========== 测试 1：漏洞分析（命中 SQL注入 + XSS）==========")
    text1 = "日志显示请求参数中出现 ' OR '1'='1，同时页面被注入 <script>alert(1)</script>"
    print(vulnerability_analysis(text1))

    print("\n========== 测试 2：漏洞分析（未命中）==========")
    print(vulnerability_analysis("今天天气不错，适合写代码"))

    print("\n========== 测试 3：故障排查（命中 磁盘满 + CPU过高）==========")
    text3 = "监控告警：服务器 CPU 使用率 98%，同时磁盘空间不足，日志无法写入"
    print(windows_ops_troubleshoot(text3))

    print("\n========== 测试 4：故障排查（未命中）==========")
    print(windows_ops_troubleshoot("食堂今天的菜不错"))

    print("\n========== 测试 5：配置审计（命中 SSL + 密码策略 + 防火墙）==========")
    text5 = "检查发现：服务器仍支持 TLS 1.0，密码最小长度为 6 位，防火墙对 3389 端口开放 0.0.0.0/0"
    print(security_audit(text5))

    print("\n========== 测试 6：配置审计（未命中）==========")
    print(security_audit("会议室投影仪需要更换"))

    print("\n========== 工具调用日志 TOOL_CALL_LOG ==========")
    for i, log in enumerate(TOOL_CALL_LOG, 1):
        print(f"{i}. [{log['time']}] {log['tool']} | 输入: {log['input_preview']} | 命中: {log['matched']}")

    print("\n========== 测试 7：execute_tool 统一调度 ==========")
    # 用同一段文本依次调用注册表中的全部三个工具
    demo_text = "发现请求参数出现 UNION SELECT 攻击特征，同时服务器 CPU 过高"
    for name in TOOLS_REGISTRY:
        print(f"\n--- 调用工具: {name} ---")
        print(execute_tool(name, input_text=demo_text))

    # 调用不存在的工具，验证错误提示
    print("\n--- 调用不存在的工具: not_exist ---")
    print(execute_tool("not_exist", input_text="任意输入"))

    print("\n========== 最终工具调用日志（含 execute_tool 调度记录）==========")
    for i, log in enumerate(TOOL_CALL_LOG, 1):
        print(f"{i}. [{log['time']}] {log['tool']} | 输入: {log['input_preview']} | 命中: {log['matched']}")
