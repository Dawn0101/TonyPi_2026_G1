#!/usr/bin/env python3
# coding=utf8
"""
Demo 01 — 离线语音识别 + Dance 特色动作
═══════════════════════════════════════════

演示内容:
  - 硬件离线 ASR 芯片（I2C）识别语音关键词
  - 匹配后执行对应的动作组
  - 重点展示 Dance 特色动作
  - 支持: 跳舞 / 鞠躬 / 挥手 / 俯卧撑 / 前进 / 后退 / 左移 / 右移 / 左右转

运行:
    python3 Demo/demo_01_offline_voice.py
"""

import os
import sys
import time

# 项目根路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer, wait_key,
    HAS_AGC, AGC,
)

# ── 关键词 → 动作映射 ────────────────────────

# 硬件 ASR 注册的关键词（拼音 → 中文 → 动作）
HARDWARE_KEYWORDS = {
    "往前走": "go_forward",
    "前进": "go_forward",
    "直走": "go_forward",
    "往后退": "back_fast",
    "后退": "back_fast",
    "向左移": "left_move_fast",
    "向右移": "right_move_fast",
    "鞠躬": "bow",
    "挥手": "wave",
    "俯卧撑": "push_ups",
    "左右转": "twist",
    "跳舞": "dance",              # ⭐ 特色动作
}

# 动作的中文描述
ACTION_DESCRIPTIONS = {
    "go_forward": "前进",
    "back_fast": "后退",
    "left_move_fast": "向左移",
    "right_move_fast": "向右移",
    "bow": "鞠躬",
    "wave": "挥手",
    "push_ups": "俯卧撑",
    "twist": "扭腰",
    "dance": "跳舞 💃",
    "stand": "立正",
    "squat": "下蹲",
    "left_kick": "左侧踢",
    "right_kick": "右侧踢",
    "stepping": "原地踏步",
    "chest": "庆祝",
    "turn_left": "左转",
    "turn_right": "右转",
}

# 需要 with_stand 收步的动作
ACTIONS_NEED_STAND = (
    "go_forward", "back_fast", "left_move_fast", "right_move_fast",
    "turn_left", "turn_right",
)


def execute_action(action_name):
    """执行单个动作"""
    if not HAS_AGC:
        desc = ACTION_DESCRIPTIONS.get(action_name, action_name)
        print(f"  [模拟] 执行动作: {desc}")
        tts_speak(desc)
        return

    desc = ACTION_DESCRIPTIONS.get(action_name, action_name)
    print(f"  [执行] {desc}")
    tts_speak(desc)

    if action_name in ACTIONS_NEED_STAND:
        AGC.runActionGroup(action_name, times=2, with_stand=True)
    else:
        AGC.runActionGroup(action_name, times=1, with_stand=True)


def run_hardware_mode():
    """硬件 ASR 芯片模式"""
    try:
        from CustomFunctions.STT_Control import STT_Control
        stt = STT_Control(mode="hardware")
    except Exception as e:
        print(f"[Demo01] 硬件 ASR 初始化失败: {e}")
        return False

    if stt._asr is None:
        print("[Demo01] 硬件 ASR 芯片不可用")
        return False

    print_header(
        "Demo 01 — 离线语音识别 + Dance",
        "说关键词触发动作 | 说'退出'结束"
    )
    print("  支持关键词: 跳舞 / 鞠躬 / 挥手 / 俯卧撑 / 前进 / 后退 / ...")
    print("  ⭐ 试试说: '跳舞'")
    print("-" * 56)

    while True:
        text, action = stt.listen_with_action(timeout=10)

        if text is None:
            print("  [ASR] 等待语音...")
            continue

        if "退出" in text or "结束" in text:
            print("  [Demo01] 退出")
            break

        if action:
            execute_action(action)
        else:
            # 尝试直接匹配关键词
            matched = None
            for keyword, act in HARDWARE_KEYWORDS.items():
                if keyword in text:
                    matched = act
                    break
            if matched:
                execute_action(matched)
            else:
                print(f"  [未匹配] 识别到: {text}")

    return True


def run_keyboard_mode():
    """键盘模拟模式（非树莓派环境）"""
    print_header(
        "Demo 01 — 离线语音识别 + Dance（键盘模拟）",
        "输入关键词模拟语音指令 | 输入'q'退出"
    )
    print("  ⭐ 试试输入: 跳舞")
    print("  其他: 鞠躬 / 挥手 / 俯卧撑 / 前进 / 后退 / 左移 / 右移 / 左右转")
    print("-" * 56)

    while True:
        try:
            cmd = input("\n  请输入指令 > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue

        if cmd.lower() in ("q", "quit", "退出", "结束"):
            break

        # 匹配关键词
        matched = None
        for keyword, act in HARDWARE_KEYWORDS.items():
            if keyword in cmd:
                matched = act
                break

        if matched:
            execute_action(matched)
        else:
            print(f"  [未匹配] 未识别的指令: {cmd}")
            print(f"  可用: {', '.join(sorted(set(HARDWARE_KEYWORDS.values())))}")

    return True


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    robot_init()

    # 尝试硬件 ASR
    success = run_hardware_mode()

    if not success:
        # 降级到键盘模拟
        print("\n[Demo01] 硬件 ASR 不可用，切换到键盘模拟模式")
        print("[Demo01] 输入关键词即可模拟语音指令")
        wait_key(2)
        run_keyboard_mode()

    robot_stand()

    from CustomFunctions.dream_memory import DreamLogger
    DreamLogger.log_session(demo_name="demo_01_offline_voice")

    print_footer(success=True)


if __name__ == "__main__":
    main()
