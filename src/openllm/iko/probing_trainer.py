"""
Probing Trainer — IKO 追问训练器
=================================

根据用户会话成熟度（M1/M2/M3）和输出内容，决定是否提示追问训练。
成熟度模型：
- M1 (<30 sessions): 新用户，每 5 次会话提示 1 次追问
- M2 (30-90 sessions): 中级用户，每 10 次会话提示 1 次追问
- M3 (>90 sessions): 老用户，不再提示追问

追问建议基于 last_output 的特征分析，生成有针对性的追问提示。

用法：
    trainer = ProbingTrainer()
    if trainer.should_prompt_probing(user_id="u-001", session_count=15):
        suggestion = trainer.get_probing_suggestion(last_output="...")
"""

from __future__ import annotations


class ProbingTrainer:
    """追问训练器——根据用户成熟度和输出特征生成追问建议。

    管理每个用户的会话计数和提示状态，按照成熟度阶段（M1/M2/M3）
    以不同的频率提示用户进行追问训练，帮助提升输出质量。

    Attributes:
        _user_sessions: 每个用户的会话计数 {user_id: session_count}。
        _prompt_counts: 每个用户的已提示次数 {user_id: prompt_count}。
    """

    # 成熟度阈值
    M1_MAX: int = 30       # M1 最大会话数
    M2_MAX: int = 90       # M2 最大会话数

    # M1 每 N 次提示 1 次
    M1_INTERVAL: int = 5
    # M2 每 N 次提示 1 次
    M2_INTERVAL: int = 10

    def __init__(self) -> None:
        """初始化追问训练器。"""
        self._user_sessions: dict[str, int] = {}
        self._prompt_counts: dict[str, int] = {}

    def should_prompt_probing(self, user_id: str, session_count: int) -> bool:
        """判断是否应该向用户提示追问训练。

        成熟度模型：
        - M1 (session_count < 30): 每 5 次会话提示 1 次
        - M2 (30 <= session_count <= 90): 每 10 次会话提示 1 次
        - M3 (session_count > 90): 不再提示

        Args:
            user_id: 用户唯一标识。
            session_count: 用户当前的总会话数。

        Returns:
            True 表示应提示追问，False 表示不提示。
        """
        if session_count < self.M1_MAX:
            # M1: 新用户，每 5 次提示 1 次
            return session_count % self.M1_INTERVAL == 0 and session_count > 0
        elif session_count <= self.M2_MAX:
            # M2: 中级用户，每 10 次提示 1 次
            return session_count % self.M2_INTERVAL == 0
        else:
            # M3: 老用户，不提示
            return False

    def get_probing_suggestion(self, last_output: str) -> str:
        """根据最近一次输出生成追问建议。

        分析 last_output 的特征（长度、问号、代码块等），
        生成针对性的追问提示字符串。

        Args:
            last_output: 最近一次 IKO 输出的内容。

        Returns:
            追问建议字符串，如果无需追问则返回空字符串。
        """
        if not last_output or not last_output.strip():
            return ""

        output = last_output.strip()
        suggestions: list[str] = []

        # 检测是否包含代码
        has_code = "```" in output or "def " in output or "class " in output
        if has_code:
            suggestions.append("这段代码的边界条件和错误处理是否完善？")

        # 检测是否包含问号（可能是回答而非建议）
        has_question = "?" in output or "？" in output

        # 检测长度
        if len(output) > 500:
            suggestions.append("这段较长的输出是否有更简洁的表达方式？")
        elif len(output) < 50:
            suggestions.append("这个回答是否足够详尽？是否有遗漏的关键信息？")

        # 检测是否包含列表/结构化内容
        has_list = any(line.strip().startswith(("-", "*", "1.", "2.")) for line in output.split("\n"))
        if has_list:
            suggestions.append("列表中各项的优先级和关联性是否清晰？")

        # 默认建议（无特殊特征时）
        if not suggestions:
            suggestions.append("能否对以上回答进行进一步的追问和澄清？")

        return "; ".join(suggestions)
