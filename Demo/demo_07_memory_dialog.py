#!/usr/bin/env python3
# coding=utf8
"""
Demo 07 — DeepSeek 多轮记忆对话
══════════════════════════════════════

演示内容:
  - 语音输入 → DeepSeek LLM 解析 → 步骤执行 → TTS 反馈
  - 多轮对话上下文记忆（记住之前的对话内容）
  - 复合指令理解（一句话包含多个动作）
  - 指代消解（"再做三个" 知道"再"指的是什么）

  这是最高阶的演示，展示完整的 AI 语音助手能力

运行:
    python3 Demo/demo_07_memory_dialog.py

前置条件:
    config.yaml 中已配置 DeepSeek API（deepseek 段）
"""

import os
import sys
import time
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer, wait_key,
)


# ═══════════════════════════════════════════════
#  主演示
# ═══════════════════════════════════════════════

def main():
    robot_init()

    print_header(
        "Demo 07 — DeepSeek 多轮记忆对话",
        "语音→LLM理解→步骤执行→TTS反馈，带上下文记忆"
    )

    # ── 加载模块 ──
    try:
        from CustomFunctions.STT_Control import STT_Control
        stt = STT_Control(mode="xunfei")
    except Exception as e:
        print(f"  [Demo07] STT 加载失败: {e}")
        stt = None

    try:
        from CustomFunctions.LLM_Control import LLM_Control
        llm = LLM_Control()
    except Exception as e:
        print(f"  [Demo07] LLM 加载失败: {e}")
        print("  请在 config.yaml 中配置 deepseek.api_key")
        wait_key(2)
        run_keyboard_mode()
        return

    try:
        from CustomFunctions.step_executor import StepExecutor
        executor = StepExecutor()
    except Exception as e:
        print(f"  [Demo07] StepExecutor 加载失败: {e}")
        executor = None

    if stt is None:
        print("  讯飞 STT 不可用，使用键盘输入模式")
        run_keyboard_mode_with_llm(llm, executor)
        return

    print("  请自然地说出你的指令，机器人会记住对话上下文")
    print()
    print("  试试这样对话:")
    print("    你说: '你好，你叫什么名字'")
    print("    你说: '我刚才问你叫什么来着'        ← 考验记忆")
    print("    你说: '做一个俯卧撑'")
    print("    你说: '再做两个'                    ← 考验指代理解")
    print("    你说: '看到有人就打个招呼然后跳舞'   ← 考验复合指令")
    print()
    print("  说 '清除记忆' 重置对话 | 说 '退出' 结束")
    print("-" * 56)

    tts_speak("语音助手已就绪，请说话")

    round_num = 0
    try:
        while True:
            round_num += 1
            print(f"\n  [第 {round_num} 轮] 录音中（{stt._record_seconds} 秒）...")

            text = stt.transcribe()

            if text is None or not text.strip():
                print("  [未识别] 请再说一遍")
                continue

            print(f"  [识别] \"{text}\"")

            # ── 特殊命令 ──
            if "退出" in text or "结束" in text:
                print("  [Demo07] 退出")
                break

            if "清除记忆" in text or "重置对话" in text:
                llm.reset_history()
                print("  [记忆] 对话历史已清除")
                tts_speak("记忆已清除，让我们重新开始")
                round_num = 0
                # 也记录这条操作
                from CustomFunctions.dream_memory import DreamLogger
                DreamLogger.log_session(
                    demo_name="demo_07_memory_dialog",
                    user_input=text,
                    llm_intent="清除记忆",
                    tts_output="记忆已清除，让我们重新开始",
                    success=True,
                )
                continue

            # ── LLM 解析 ──
            print("  [LLM] 正在理解...")
            try:
                plan = llm.parse(text, keep_history=True)
            except Exception as e:
                print(f"  [LLM] 解析失败: {e}")
                plan = llm._fallback_parse(text)

            # 打印解析结果
            intent = plan.get("intent", "")
            tts_text = plan.get("tts_response", "")
            steps = plan.get("steps", [])

            print(f"  [意图] {intent}")
            if tts_text:
                print(f"  [TTS]  {tts_text}")
            if steps:
                print(f"  [步骤] {len(steps)} 步:")
                for i, s in enumerate(steps, 1):
                    params = ", ".join(f"{k}={v}" for k, v in s.get("params", {}).items())
                    print(f"         {i}. {s.get('action','?')}({params}) — {s.get('description','')}")
            else:
                print(f"  [步骤] 无（纯对话）")

            # ── 执行 ──
            if executor and steps:
                print("  [执行] 开始执行步骤链...")
                executor.execute(plan)
            elif not steps:
                # 纯对话，只播报 TTS
                if tts_text:
                    tts_speak(tts_text)

            # ── 记录本轮梦日志（带上详细内容）──
            from CustomFunctions.dream_memory import DreamLogger
            DreamLogger.log_session(
                demo_name="demo_07_memory_dialog",
                user_input=text,
                llm_intent=intent,
                actions=steps,
                tts_output=tts_text,
                success=True,
            )

            print(f"  [完成] 等待下一轮指令...")

    except KeyboardInterrupt:
        print("\n  [Demo07] 用户退出")

    robot_stand()
    print_footer(success=True)


# ═══════════════════════════════════════════════
#  键盘模拟模式
# ═══════════════════════════════════════════════

def run_keyboard_mode():
    """简化键盘模式（无 LLM）"""
    print_header("Demo 07 — 多轮记忆对话 模拟版", "键盘输入，展示 LLM 对话概念")

    print("  模拟对话:")
    print("  ═══════════════════════════════════════")
    print("  你: 你好，你叫什么名字")
    print("  机器人: 你好，我是TonyPi智能机器人")
    print("  你: 我刚才问你叫什么来着")
    print("  机器人: TonyPi智能机器人  ← 记住了上一轮")
    print("  你: 做一个俯卧撑")
    print("  机器人: 好的 [执行 push_ups]")
    print("  你: 再做两个")
    print("  机器人: 好的，再做两个俯卧撑 [执行 push_ups×2]")
    print("  ═══════════════════════════════════════")

    tts_speak("这是多轮记忆对话的概念演示")

    robot_stand()
    print_footer(success=True)


def run_keyboard_mode_with_llm(llm, executor):
    """键盘 + LLM 模式"""
    print_header("Demo 07 — 多轮记忆对话（键盘输入）", "输入自然语言，LLM 解析执行")

    print("  输入指令（如 '做一个俯卧撑然后鞠躬'）")
    print("  输入 '清除记忆' 重置 | 输入 'q' 退出")
    print("-" * 56)

    round_num = 0
    while True:
        round_num += 1
        try:
            text = input(f"\n  [{round_num}] 请输入 > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not text:
            continue
        if text.lower() in ("q", "quit", "退出"):
            break
        if "清除记忆" in text:
            llm.reset_history()
            print("  [记忆] 已清除")
            round_num = 0
            continue

        # LLM 解析
        try:
            plan = llm.parse(text, keep_history=True)
        except Exception as e:
            print(f"  [LLM] 失败: {e}")
            plan = llm._fallback_parse(text)

        intent = plan.get("intent", "")
        tts_text = plan.get("tts_response", "")
        steps = plan.get("steps", [])

        print(f"  [意图] {intent}")
        if tts_text:
            print(f"  [TTS]  {tts_text}")
        if steps:
            print(f"  [步骤] {len(steps)} 步")
            for i, s in enumerate(steps, 1):
                print(f"         {i}. {s.get('action','?')} — {s.get('description','')}")

        # 执行
        if executor and steps:
            executor.execute(plan)

        # 记录本轮
        from CustomFunctions.dream_memory import DreamLogger
        DreamLogger.log_session(
            demo_name="demo_07_memory_dialog",
            user_input=text,
            llm_intent=intent,
            actions=steps,
            tts_output=tts_text,
            success=True,
        )

    robot_stand()
    print_footer(success=True)


if __name__ == "__main__":
    main()
