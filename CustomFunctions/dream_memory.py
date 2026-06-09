#!/usr/bin/env python3
# coding=utf8
"""
梦的记忆 — 离线经验回放模块

每个 demo 结束时调用一次 log_session()，
将当天发生的所有交互追加记录到 DreamLogs/{YYYY-MM-DD}.json。

「梦」的概念：
  白天机器人记录每一次动作执行、交互、失败。
  待机时（充电/空闲），DreamReflector 读取今天的日志，
  用 LLM 分析失败模式，生成改进策略。
  下次开机时 TTS 报告：「我昨晚梦到昨天摔倒的事了…」
"""

import os
import sys
import json
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DREAM_DIR = os.path.join(_PROJECT_ROOT, "DreamLogs")


def _ensure_dir():
    """确保 DreamLogs 目录存在"""
    os.makedirs(DREAM_DIR, exist_ok=True)


def _today_path():
    """返回今天的日志路径: DreamLogs/2024-01-15.json"""
    return os.path.join(DREAM_DIR, datetime.now().strftime("%Y-%m-%d") + ".json")


def _read_today():
    """读取今日的全部日志记录（追加前读取）"""
    path = _today_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, Exception):
        return []


def _write_today(entries):
    """覆写今日日志（完整数组）"""
    _ensure_dir()
    path = _today_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


class DreamLogger:
    """
    「梦」日志记录器

    用法（在每个 demo 结尾调用）:
        from CustomFunctions.dream_memory import DreamLogger
        DreamLogger.log_session(demo_name="demo_01_offline_voice")

    带详细参数:
        DreamLogger.log_session(
            demo_name="demo_07_memory_dialog",
            user_input="做一个俯卧撑",
            llm_intent="做俯卧撑",
            actions=[{"action": "push_ups", "params": {"times": 3}, "result": "success"}],
            tts_output="好的，我来做三个俯卧撑",
            fell_during=False,
            success=True,
        )
    """

    @staticmethod
    def log_session(
        demo_name: str = "",
        user_input: str = "",
        llm_intent: str = "",
        actions: list = None,
        tts_output: str = "",
        fell_during: bool = False,
        success: bool = True,
        terminal_summary: str = "",
    ):
        """
        记录一次 demo 运行 session，追加到今天的日志文件

        参数:
            demo_name:      demo 标识，如 "demo_01_offline_voice"
            user_input:     用户的输入（语音识别的文本 / 键盘输入）
            llm_intent:     LLM 解析出的意图
            actions:        执行的步骤列表 [{"action":..., "params":..., "result":...}, ...]
            tts_output:     机器人 TTS 播报的内容
            fell_during:    运行期间是否跌倒
            success:        是否成功完成
            terminal_summary: 终端输出的总结文字

        所有参数都有默认值，demo 只需要传 demo_name 即可。
        """
        entry = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "demo_name": demo_name,
            "user_input": user_input,
            "llm_intent": llm_intent,
            "actions": actions or [],
            "tts_output": tts_output,
            "fell_during": fell_during,
            "success": success,
            "terminal_summary": terminal_summary,
        }

        entries = _read_today()
        entries.append(entry)
        _write_today(entries)

        print(f"  [梦] 已记录 → {os.path.basename(_today_path())} (#{len(entries)})")

    @staticmethod
    def get_today_logs() -> list:
        """读取今天的全部日志"""
        return _read_today()

    @staticmethod
    def get_recent_days(days: int = 7) -> dict:
        """读取最近 N 天的日志"""
        result = {}
        _ensure_dir()
        now = datetime.now()
        for i in range(days):
            date_str = (now.timestamp() - i * 86400)
            date_label = datetime.fromtimestamp(date_str).strftime("%Y-%m-%d")
            path = os.path.join(DREAM_DIR, date_label + ".json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            result[date_label] = data
                except Exception:
                    pass
        return result

    @staticmethod
    def count_sessions_today() -> int:
        """今日已记录多少条 session"""
        return len(_read_today())

    @staticmethod
    def count_successes_today() -> int:
        """今日成功次数"""
        entries = _read_today()
        return sum(1 for e in entries if e.get("success", False))

    @staticmethod
    def count_falls_today() -> int:
        """今日跌倒次数"""
        entries = _read_today()
        return sum(1 for e in entries if e.get("fell_during", False))


# ── 快速测试 ──
if __name__ == "__main__":
    print("=" * 50)
    print("梦的记忆 — 测试")
    print("=" * 50)

    # 模拟记录几次
    DreamLogger.log_session(
        demo_name="demo_test",
        user_input="你好",
        actions=[{"action": "wave", "params": {}, "result": "success"}],
        tts_output="你好，我是TonyPi",
        success=True,
    )

    DreamLogger.log_session(
        demo_name="demo_test",
        user_input="做俯卧撑",
        actions=[{"action": "push_ups", "params": {"times": 3}, "result": "success"}],
        tts_output="好的，我来做三个俯卧撑",
        success=True,
    )

    print(f"\n今日共有 {DreamLogger.count_sessions_today()} 条记录")
    print(f"成功: {DreamLogger.count_successes_today()}, 跌倒: {DreamLogger.count_falls_today()}")

    logs = DreamLogger.get_today_logs()
    print(f"\n全部记录:")
    for entry in logs:
        print(f"  [{entry['timestamp']}] {entry['demo_name']} — {entry['user_input']}")
