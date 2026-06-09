#!/usr/bin/env python3
# coding=utf8
"""
Demo 09 — 梦的反思（离线经验回放）
══════════════════════════════════════

核心概念（海马体回放理论）：
  动物在睡眠时会重放白天的经历，用于巩固记忆和优化行为。
  本 demo 模拟这一过程——读取今天的 DreamLogs，
  用 LLM 分析失败模式，生成一段 ≤50 字的「梦」的反思，
  最后通过 TTS 讲出来。

流程:
    1. 读取 DreamLogs/ 中最近一天的日志
    2. 将日志条目发给 DeepSeek LLM
    3. LLM 生成一段 ≤50 字的第一人称梦境反思
    4. 保存到 DreamLogs/dream_reflections/
    5. TTS 播报反思结果

运行:
    python3 Demo/demo_09_dream_reflector.py
    python3 Demo/demo_09_dream_reflector.py --sim   模拟模式

前置条件:
    config.yaml 中已配置 DeepSeek API（deepseek 段）
"""

import os
import sys
import json
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer,
    HAS_AGC,
)

DREAM_DIR = os.path.join(_PROJECT_ROOT, "DreamLogs")
REFLECTION_DIR = os.path.join(DREAM_DIR, "dream_reflections")


# ═══════════════════════════════════════════════
#  日志读取
# ═══════════════════════════════════════════════

def find_latest_log() -> str:
    """
    从 DreamLogs/ 中找到最新的日志文件（按文件名排序）
    返回完整路径，没有日志则返回 None
    """
    if not os.path.isdir(DREAM_DIR):
        return None

    files = [f for f in os.listdir(DREAM_DIR)
             if f.endswith(".json") and not f.startswith(".")]
    if not files:
        return None

    # 按文件名（日期）降序排序，最新的排最前
    files.sort(reverse=True)
    return os.path.join(DREAM_DIR, files[0])


def load_logs(log_path: str) -> list:
    """加载日志文件中的条目"""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except (json.JSONDecodeError, Exception):
        return []


def summarize_logs(entries: list) -> str:
    """
    将日志条目压缩为 LLM 友好的文字摘要
    避免日志太长超出 LLM 上下文
    """
    if not entries:
        return "今天没有任何活动记录。"

    total = len(entries)
    successes = sum(1 for e in entries if e.get("success", False))
    failures = total - successes
    falls = sum(1 for e in entries if e.get("fell_during", False))

    lines = [f"今天共运行 {total} 次（成功 {successes} 次，失败 {failures} 次，跌倒 {falls} 次）"]

    for e in entries:
        demo = e.get("demo_name", "?")
        user = e.get("user_input", "")
        intent = e.get("llm_intent", "")
        actions = e.get("actions", [])
        fell = e.get("fell_during", False)
        ok = e.get("success", False)

        action_str = "、".join(
            f"{a.get('action','?')}({','.join(f'{k}={v}' for k,v in a.get('params',{}).items())})"
            for a in actions
        ) if actions else "无动作"

        flags = []
        if fell:
            flags.append("❗跌倒")
        if not ok:
            flags.append("❗失败")

        line = f"  · {demo}"
        if user:
            line += f" 用户说「{user}」"
        if intent:
            line += f" 意图: {intent}"
        line += f" → {action_str}"
        if flags:
            line += f" {' '.join(flags)}"
        lines.append(line)

    return "\n".join(lines)


# ═══════════════════════════════════════════════
#  LLM 反思
# ═══════════════════════════════════════════════

REFLECT_PROMPT = """你是TonyPi机器人的潜意识，正在「做梦」——重放白天的经历。

以下是今天的活动日志，请以第一人称「我」的口吻，
生成一段简短梦境反思，要求：

1. 必须 ≤ 50 个中文字符（一个汉字算一个字符）
2. 如果当天有失败/跌倒 → 反思原因 + 改进想法
3. 如果全部成功 → 表达成长或满足感
4. 用第一人称，像在说梦话
5. 只输出反思内容，不要解释，不要加引号

示例输出（≤50字）：
  我梦见今天摔倒了好多次，下次在光滑地面要走慢一点。
  我梦到学会跳舞了，好开心。
  我梦见今天左转时总偏右，下次步幅要减半。

今天的日志：
{log_summary}
"""


def call_deepseek(log_summary: str) -> str:
    """调用 DeepSeek 生成反思"""
    try:
        import yaml
        cfg_path = os.path.join(_PROJECT_ROOT, "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg.get("deepseek", {})
        api_key = ds_cfg.get("api_key", "")
        base_url = ds_cfg.get("base_url", "https://api.deepseek.com")
        model = ds_cfg.get("model", "deepseek-chat")
    except ImportError:
        print("  [反思] ⚠ 缺少 PyYAML 库 (pip3 install pyyaml)")
        return fallback_reflection(log_summary)
    except Exception as e:
        print(f"  [反思] ⚠ 加载配置失败: {e}")
        return fallback_reflection(log_summary)

    if not api_key or api_key == "你的DeepSeekAPIKey":
        print("  [反思] ⚠ DeepSeek API Key 未配置")
        return fallback_reflection(log_summary)

    try:
        import requests
    except ImportError:
        print("  [反思] ⚠ 缺少 requests 库")
        return fallback_reflection(log_summary)

    prompt = REFLECT_PROMPT.format(log_summary=log_summary)

    print(f"  [反思] 正在分析今天的 {log_summary.count('条')}条日志...")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个机器人梦境生成器。输出 ≤50 个中文字符的梦境反思，只输出反思本身。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"  [反思] API 返回 {resp.status_code}")
            return fallback_reflection(log_summary)

        reply = resp.json()["choices"][0]["message"]["content"].strip()
        # 去掉引号和多余的空白
        reply = reply.strip("「」\"'“”").strip()

        # 截断到 50 个字
        if len(reply) > 50:
            reply = reply[:50]

        print(f"  [反思] LLM 生成: {reply}")
        return reply

    except Exception as e:
        print(f"  [反思] API 调用失败: {e}")
        return fallback_reflection(log_summary)


def fallback_reflection(log_summary: str) -> str:
    """
    当 API 不可用时的兜底方案
    根据日志关键词生成简单的反思
    """
    if "跌倒" in log_summary:
        return "我梦见今天摔倒了，下次走慢一点。"
    if "失败" in log_summary or "❗" in log_summary:
        return "我梦见今天遇到困难，但我会继续加油。"
    if "跳舞" in log_summary:
        return "我梦到学会了跳舞，好开心呀。"
    if "俯卧撑" in log_summary or "push_ups" in log_summary:
        return "我梦见自己在做运动，越来越强壮了。"
    return "我梦到和主人一起度过快乐的一天。"


# ═══════════════════════════════════════════════
#  保存反思
# ═══════════════════════════════════════════════

def save_reflection(reflection: str, source_log: str, entries: list):
    """
    保存反思结果到 DreamLogs/dream_reflections/
    多次运行会覆写当天同一日期的反思（保留最新）
    """
    os.makedirs(REFLECTION_DIR, exist_ok=True)

    # 从源文件名提取日期
    date_str = os.path.splitext(os.path.basename(source_log))[0]

    reflection_data = {
        "date": date_str,
        "timestamp": datetime.now().isoformat(),
        "reflection": reflection,
        "char_count": len(reflection),
        "source_log": source_log,
        "summary": {
            "total_sessions": len(entries),
            "successes": sum(1 for e in entries if e.get("success", False)),
            "failures": sum(1 for e in entries if not e.get("success", True)),
            "falls": sum(1 for e in entries if e.get("fell_during", False)),
        },
    }

    path = os.path.join(REFLECTION_DIR, f"{date_str}_reflection.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reflection_data, f, ensure_ascii=False, indent=2)

    print(f"  [反思] 已保存 → {os.path.basename(path)}")
    return path


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def run_reflection():
    """
    完整反思流程：
    1. 找最新日志
    2. 加载并摘要
    3. LLM 反思
    4. 保存 + TTS
    """
    # 1. 找日志
    log_path = find_latest_log()
    if log_path is None:
        print("  [反思] DreamLogs/ 中没有找到日志文件")
        print("  [反思] 请先运行其他 demo 产生日志")
        tts_speak("今天没有什么可梦到的")
        return

    date_str = os.path.splitext(os.path.basename(log_path))[0]
    print(f"  [反思] 加载日志: {os.path.basename(log_path)}")

    # 2. 加载
    entries = load_logs(log_path)
    if not entries:
        print("  [反思] 日志为空")
        tts_speak("今天没有什么可梦到的")
        return

    print(f"  [反思] 共 {len(entries)} 条记录")

    # 3. 摘要
    summary = summarize_logs(entries)
    print(f"\n{'─' * 50}")
    print(summary)
    print(f"{'─' * 50}\n")

    # 4. LLM 反思
    reflection = call_deepseek(summary)

    # 确保不超过 50 字
    if len(reflection) > 50:
        reflection = reflection[:50]

    print(f"\n  💭「梦」: {reflection}")
    print(f"  📏 字数: {len(reflection)}/50")

    # 5. 保存
    save_reflection(reflection, log_path, entries)

    # 6. TTS 播报（≤50 字，一次读完）
    print(f"\n  🔊 播报反思...")
    tts_speak(reflection)

    print("  [反思] 完成")


# ═══════════════════════════════════════════════
#  模拟模式
# ═══════════════════════════════════════════════

def run_simulated():
    """模拟模式 — 展示「梦」的概念（不需要 API）"""
    print_header(
        "Demo 09 — 梦的反思（模拟版）",
        "离线经验回放 — 海马体回放理论模拟"
    )

    # 模拟几条日志
    print("  [模拟] 假设今天产生了几条日志：")
    print()
    print("    [demo_07_memory_dialog] 用户说「做三个俯卧撑」")
    print("      → push_ups(times=3) ✅ 成功")
    print()
    print("    [demo_01_offline_voice] 用户说「跳舞」")
    print("      → dance ✅ 成功")
    print()
    print("    [demo_04_object_tracking] 自动检测")
    print("      → 发现人 🧑 → wave ✅ 成功")
    print()

    # 模拟反思
    print("  [模拟] 正在做梦...")
    time.sleep(1.5)

    reflection = "我梦见今天学会了好多新动作，好充实呀。"
    print(f"\n  💭「梦」: {reflection}")
    print(f"  📏 字数: {len(reflection)}/50")
    print()
    print("  [模拟] 反思完成")

    tts_speak(reflection)
    robot_stand()
    print_footer(success=True)


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    robot_init()

    print_header(
        "Demo 09 — 梦的反思（离线经验回放）",
        "读取 DreamLogs → DeepSeek 分析 → 生成梦境反思 → TTS 播报"
    )

    print("  这个过程模拟海马体回放理论：")
    print("  机器人「睡着」后重放白天的经历，")
    print("  从失败中学习，巩固成功的经验。")
    print()
    print("  LLM 将分析今天的日志，生成一段 ≤50 字的梦境")
    print("  最后通过 TTS 讲出来。")
    print("-" * 56)

    import argparse
    parser = argparse.ArgumentParser(description="梦的反思 Demo")
    parser.add_argument("--sim", action="store_true", help="模拟模式")
    args, extra = parser.parse_known_args()

    if args.sim:
        run_simulated()
        return

    tts_speak("开始回想今天发生的事情")
    run_reflection()

    robot_stand()

    from CustomFunctions.dream_memory import DreamLogger
    DreamLogger.log_session(demo_name="demo_09_dream_reflector")

    print_footer(success=True)


if __name__ == "__main__":
    main()
