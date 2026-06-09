#!/usr/bin/env python3
# coding=utf8
"""
Demo 06 — 语音驱动视觉查询（VLM JSON 输出）
══════════════════════════════════════════════════

演示内容:
  语音提问 → 机器人拍照 → 语音识别结果作为提示词 → 智谱 GLM-4V
  → VLM 返回 JSON 格式结果 → 保存到文件 → TTS 朗读回答

  相比旧的 scene_describer（只能做固定场景描述），这个版本:
  · ❌ 不再绑定固定触发词 — 你可以问任何问题
  · ✅ 语音结果直接作为 VLM 的提示词
  · ✅ VLM 输出结构化 JSON，而不是自由文本
  · ✅ 自动保存 JSON 到 vlm_results/ 目录

  VLM 输出 JSON 结构:
  {
    "question": "用户的问题",
    "answer": "详细回答（60字以内，直接用于语音播报）",
    "details": { ... }
  }

运行:
    python3 Demo/demo_06_visual_query.py

前置条件:
    config.yaml 中已配置:
      · xunfei (语音识别)
      · zhipu.api_key (视觉大模型)
      · tts (语音播报)

旧版:
    Demo/demo_06_scene_describe.py（已废弃，迁移到本文件）
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer,
)


# ═══════════════════════════════════════════════
#  主演示
# ═══════════════════════════════════════════════

def main():
    robot_init()

    print_header(
        "Demo 06 — 语音驱动视觉查询 (VLM JSON 输出)",
        "语音提问 → 拍照 → STT+图像 → GLM-4V → JSON 文件"
    )

    # 尝试加载 VisualQuery
    try:
        from CustomFunctions.visual_query import VisualQuery
        vq = VisualQuery()
    except Exception as e:
        print(f"  [Demo06] VisualQuery 加载失败: {e}")
        print("  请确认:")
        print("    · config.yaml 中有 zhipu.api_key")
        print("    · 已安装依赖: pip3 install requests pyyaml")
        print("  ---")
        print("  使用 --sim 参数查看模拟演示")

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--sim", action="store_true")
        args, _ = parser.parse_known_args()

        if args.sim:
            run_simulated()
        else:
            print("  提示: python3 Demo/demo_06_visual_query.py --sim")
        return

    print()
    print("  你可以对麦克风说出关于当前场景的任何问题，例如:")
    print("    · 「桌子上有什么物体」")
    print("    · 「杯子是什么颜色的」")
    print("    · 「房间里光线怎么样」")
    print("    · 「这里安全吗」")
    print()
    print("  VLM 将以 JSON 格式回答并保存到 vlm_results/ 目录")
    print("  按 Ctrl+C 退出")
    print("-" * 56)

    tts_speak("视觉查询已就绪，请说出你的问题")

    try:
        round_num = 0
        while True:
            round_num += 1
            print(f"\n{'─' * 40}")
            print(f"  第 {round_num} 轮")
            print(f"{'─' * 40}")

            ok, json_path, question = vq.run_once()

            if ok:
                print(f"\n  ✅ 第 {round_num} 轮完成")
                print(f"     问题: {question}")
                print(f"     结果: {json_path}")
                print(f"   (按 Ctrl+C 退出，等待 2 秒后继续下一轮)")
                time.sleep(2)

            else:
                print(f"\n  ⏭️  第 {round_num} 轮跳过（未识别到有效语音）")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n  [Demo06] 用户退出")
    finally:
        try:
            vq._close_camera()
        except Exception:
            pass

    robot_stand()

    from CustomFunctions.dream_memory import DreamLogger
    DreamLogger.log_session(demo_name="demo_06_visual_query")

    print_footer(success=True)


def run_simulated():
    """模拟模式：展示概念（不需要硬件和 API 密钥）"""
    print_header(
        "Demo 06 — 视觉查询 模拟版",
        "展示语音驱动 VLM JSON 输出的概念流程"
    )

    print("  模拟流程:")
    print("    1. 说出你的问题 → 如「桌子上有什么物体」")
    print("    2. 拍照 → 摄像头抓取一帧")
    print("    3. 语音文本 + 图片 → 智谱 GLM-4V")
    print("    4. VLM 返回 JSON → 保存到 vlm_results/xxx.json")
    print("    5. TTS 朗读回答")
    print()
    print("  [模拟] 假设场景: 桌上有红杯子和绿球")
    print("  [模拟] 假设提问: 桌子上有什么物体")
    print("  [模拟] VLM JSON 返回:")

    import json
    mock_result = {
        "question": "桌子上有什么物体",
        "answer": "桌面上有一个红色马克杯和一个绿色网球",
        "details": {
            "objects": [
                {"name": "马克杯", "color": "红色", "position": "桌子左侧"},
                {"name": "网球", "color": "绿色", "position": "桌子右侧"},
            ]
        },
    }
    print(f"  {json.dumps(mock_result, ensure_ascii=False, indent=4)}")
    print()
    print("  [模拟] 结果将保存到 vlm_results/ 目录")

    tts_speak("模拟结果，桌上有红色杯子和绿色网球")

    robot_stand()
    print_footer(success=True)


if __name__ == "__main__":
    main()
