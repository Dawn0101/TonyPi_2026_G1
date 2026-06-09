#!/usr/bin/env python3
# coding=utf8
"""
Demo 05 — 讯飞在线语音识别 + 动作
════════════════════════════════════════

演示内容:
  使用科大讯飞云端 API 进行在线语音识别（完整句子）
  → 关键词匹配 → 执行对应动作组
  → TTS 语音反馈

  与 Demo 01（离线关键词）的区别:
    - 离线: 只能识别预注册的拼音关键词
    - 在线: 可识别任意自然语句，如"请往前走两步"

运行:
    python3 Demo/demo_05_online_voice.py

前置条件:
    config.yaml 中已配置讯飞 API（xunfei 段）
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer, wait_key,
    HAS_AGC, AGC,
)


# ── 关键词 → 动作映射（比离线模式更丰富） ──

XUNFEI_KEYWORDS = {
    # 移动
    "往前走": "go_forward",  "前进": "go_forward",    "直走": "go_forward",
    "向前": "go_forward",
    "往后退": "back_fast",   "后退": "back_fast",     "向后": "back_fast",
    "向左移": "left_move_fast", "左移": "left_move_fast",
    "向右移": "right_move_fast", "右移": "right_move_fast",
    "左转": "turn_left",     "右转": "turn_right",
    # 表演
    "鞠躬": "bow",
    "挥手": "wave",          "挥挥手": "wave",         "打招呼": "wave",
    "跳舞": "dance",        "跳一支舞": "dance",      "跳支舞": "dance",
    "跳个舞": "dance",       "来段舞": "dance",
    "俯卧撑": "push_ups",    "做俯卧撑": "push_ups",
    "扭腰": "twist",         "左右转": "twist",
    "下蹲": "squat",         "蹲下": "squat",
    "踏步": "stepping",      "原地踏步": "stepping",
    "踢腿": "left_kick",     "踢": "left_kick",
    "庆祝": "chest",
    "立正": "stand",         "站好": "stand",
    "仰卧起坐": "sit_ups",
    # 综合
    "红色": None,            # 只是识别，不做动作（留给 LLM 处理）
    "绿色": None,
    "蓝色": None,
}

ACTION_DESCRIPTIONS = {
    "go_forward": "前进",      "back_fast": "后退",
    "left_move_fast": "左移",  "right_move_fast": "右移",
    "turn_left": "左转",       "turn_right": "右转",
    "bow": "鞠躬",             "wave": "挥手",
    "dance": "跳舞",         "push_ups": "俯卧撑",
    "twist": "扭腰",           "squat": "下蹲",
    "stepping": "原地踏步",    "left_kick": "踢腿",
    "chest": "庆祝",           "stand": "立正",
    "sit_ups": "仰卧起坐",
}

ACTIONS_NEED_STAND = (
    "go_forward", "back_fast", "left_move_fast", "right_move_fast",
    "turn_left", "turn_right",
)


def execute_action(action_name):
    """执行动作"""
    desc = ACTION_DESCRIPTIONS.get(action_name, action_name)
    print(f"  [执行] {desc}")
    tts_speak(desc)

    if not HAS_AGC:
        return

    if action_name in ACTIONS_NEED_STAND:
        AGC.runActionGroup(action_name, times=2, with_stand=True)
    elif action_name == "sit_ups":
        AGC.runActionGroup("sit_ups", times=3, with_stand=True)
    else:
        AGC.runActionGroup(action_name, times=1, with_stand=True)


def match_actions(text):
    """从识别文本中提取所有匹配的动作"""
    matched = []
    for keyword, action in XUNFEI_KEYWORDS.items():
        if keyword in text and action is not None and action not in matched:
            matched.append(action)
    return matched


# ═══════════════════════════════════════════════
#  主演示
# ═══════════════════════════════════════════════

def main():
    robot_init()

    print_header(
        "Demo 05 — 讯飞在线语音识别 + 动作",
        "云端识别完整句子 → 关键词提取 → 动作执行"
    )

    # 尝试加载讯飞 STT
    try:
        from CustomFunctions.STT_Control import STT_Control
        stt = STT_Control(mode="xunfei")
    except Exception as e:
        print(f"  [Demo05] STT 初始化失败: {e}")
        print("  切换到键盘模拟模式...")
        stt = None

    if stt is None:
        print("  请在 config.yaml 中配置讯飞 API")
        wait_key(2)
        run_keyboard_mode()
        return

    print("  请对着麦克风说出完整指令")
    print("  例如: '请往前走两步' / '跳个舞吧' / '鞠躬加挥手'")
    print("  说 '退出' 或 '结束' 退出")
    print("-" * 56)

    tts_speak("在线语音识别已就绪，请说话")

    round_num = 0
    while True:
        round_num += 1
        print(f"\n  [第 {round_num} 轮] 录音中（{stt._record_seconds} 秒）...")

        text = stt.transcribe()

        if text is None or not text.strip():
            print("  [未识别] 请再说一遍")
            continue

        print(f"  [识别] \"{text}\"")

        if "退出" in text or "结束" in text:
            print("  [Demo05] 退出")
            break

        # 匹配动作
        actions = match_actions(text)

        if actions:
            print(f"  [匹配] {len(actions)} 个动作: {', '.join(actions)}")
            for action in actions:
                execute_action(action)
                time.sleep(0.3)
        else:
            print(f"  [无匹配] 未识别到可执行的动作")
            print(f"  识别文本: \"{text}\"")
            tts_speak("抱歉，我没听懂您的指令")
            print(f"  可用关键词: 前进/后退/左移/右移/鞠躬/挥手/跳舞/俯卧撑/...")

        # 记录本轮动作
        from CustomFunctions.dream_memory import DreamLogger
        DreamLogger.log_session(
            demo_name="demo_05_online_voice",
            user_input=text,
            actions=[{"action": a} for a in actions],
            tts_output="、".join(ACTION_DESCRIPTIONS.get(a, a) for a in actions) if actions else "",
            success=bool(actions),
        )

    robot_stand()
    print_footer(success=True)


def run_keyboard_mode():
    """键盘模拟模式"""
    print_header("Demo 05 — 讯飞在线语音 模拟版", "输入完整句子模拟语音识别")

    print("  输入完整句子（如 '请往前走两步'）")
    print("  输入 q 退出")
    print("-" * 56)

    while True:
        try:
            text = input("\n  请输入指令 > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not text:
            continue
        if text.lower() in ("q", "quit", "退出"):
            break

        print(f"  [模拟识别] \"{text}\"")
        actions = match_actions(text)

        if actions:
            print(f"  [匹配] {', '.join(actions)}")
            for action in actions:
                execute_action(action)
                time.sleep(0.3)
        else:
            print(f"  [无匹配] 试试: 前进/后退/鞠躬/挥手/跳舞/俯卧撑")

    robot_stand()
    print_footer(success=True)


if __name__ == "__main__":
    main()
