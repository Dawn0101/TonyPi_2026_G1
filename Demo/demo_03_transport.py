#!/usr/bin/env python3
# coding=utf8
"""
Demo 03 — 智能搬运
══════════════════════════════════

演示内容:
  机器人通过颜色识别定位目标物体 → 自主靠近 → 抓取
  → AprilTag 标签导航 → 放置到目标区域

  完整展示: 视觉定位 + 自主导航 + 机械臂抓取 + 标签导航放置

硬件准备:
  ⚠ 需要以下道具:
    - 红色/绿色/蓝色方块（至少1个）
    - 3 个 AprilTag 标签（tag36h11，ID: 1=红色区, 2=绿色区, 3=蓝色区）
    - 平整地面，标签间距约 30-50cm

运行:
    python3 Demo/demo_03_transport.py
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer, wait_key,
    CameraManager, HAS_CAMERA, HAS_AGC,
)

# ═══════════════════════════════════════════════
#  智能搬运（使用原厂 Transport 模块）
# ═══════════════════════════════════════════════

def run_transport(cam, target_color="red"):
    """
    运行智能搬运
    参数:
        cam: CameraManager 实例
        target_color: 目标颜色 "red" / "green" / "blue"
    """
    try:
        import Functions.Transport as Transport
    except Exception as e:
        print(f"[Demo03] Transport 模块加载失败: {e}")
        print("[Demo03] 此演示需要 hiwonder 硬件环境")
        return False

    # ── 配置搬运参数 ──
    Transport.object_color = target_color
    Transport.color_list = [target_color]  # 只搬运指定颜色
    Transport.find_box = True              # 从寻找物体阶段开始
    Transport.step = 1
    Transport.stop_detect = False
    Transport.lock_servos = ""

    color_cn = {"red": "红色", "green": "绿色", "blue": "蓝色"}.get(target_color, target_color)

    print_header(
        f"Demo 03 — 智能搬运 ({color_cn}物体)",
        "机器人将自主完成: 寻找→靠近→抓取→导航→放置"
    )

    print(f"  目标颜色: {color_cn}")
    print(f"  阶段 1: 寻找{color_cn}物体并抓取")
    print(f"  阶段 2: 通过 AprilTag 导航到{color_cn}放置区")
    print(f"  阶段 3: 放下物体")
    print("  按 Ctrl+C 中断")
    print("-" * 56)

    tts_speak(f"开始搬运{color_cn}物体")

    # ── 初始化 Transport ──
    Transport.init()
    Transport.start()

    success = False
    try:
        start_time = time.time()
        # 最长运行 60 秒
        while time.time() - start_time < 60:
            frame = cam.grab(timeout=0.5)
            if frame is None:
                time.sleep(0.03)
                continue

            # 喂帧给 Transport.run()，move() 线程会自动处理导航
            try:
                Transport.run(frame)
            except Exception as e:
                print(f"  [Transport] run() 异常: {e}")

            # 检查是否完成搬运
            if not Transport.color_list:
                print(f"\n  ✓ 所有{color_cn}物体搬运完成！")
                success = True
                break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n  [Demo03] 用户中断")
    finally:
        Transport.stop()
        Transport.exit()

    if success:
        tts_speak(f"{color_cn}物体搬运完成")
    else:
        tts_speak("搬运中断")

    return success


# ═══════════════════════════════════════════════
#  简化版搬运（纯动作组演示 — 非树莓派环境）
# ═══════════════════════════════════════════════

def run_transport_simulated(target_color="red"):
    """
    简化版搬运：用动作组模拟搬运全流程
    适合无完整硬件环境时展示逻辑
    """
    from Demo.common import HAS_AGC, AGC
    import hiwonder.Board as Board

    color_cn = {"red": "红色", "green": "绿色", "blue": "蓝色"}.get(target_color, target_color)

    print_header(
        f"Demo 03 — 智能搬运 模拟版 ({color_cn})",
        "用预设动作组模拟搬运全流程"
    )

    steps = [
        ("寻找物体", "go_forward_one_step", 1),
        ("靠近物体", "go_forward_one_step", 2),
        ("抓取物体", "move_up", 1),
        ("后退", "back_fast", 2),
        ("放置物体", "put_down", 1),
        ("归位", "back", 3),
    ]

    for i, (desc, action, times) in enumerate(steps, 1):
        print(f"  步骤 {i}/{len(steps)}: {desc}")
        tts_speak(desc)
        if HAS_AGC:
            if action == "move_up" or action == "put_down":
                AGC.runActionGroup(action)
            else:
                AGC.runActionGroup(action, times=times, with_stand=True)
        time.sleep(0.5)

    print("  ✓ 搬运流程演示完成")
    tts_speak("搬运完成")
    return True


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    robot_init()

    # 选择目标颜色
    import argparse
    parser = argparse.ArgumentParser(description="智能搬运 Demo")
    parser.add_argument("--color", choices=["red", "green", "blue"], default="red",
                        help="目标颜色 (默认: red)")
    parser.add_argument("--sim", action="store_true",
                        help="使用模拟模式（无硬件时）")
    args = parser.parse_args()

    if args.sim or not HAS_CAMERA:
        # 模拟模式
        run_transport_simulated(args.color)
    else:
        # 完整硬件模式
        cam = CameraManager()
        if not cam.open():
            print("[Demo03] 摄像头不可用，切换到模拟模式")
            wait_key(2)
            run_transport_simulated(args.color)
            robot_stand()
            return

        try:
            run_transport(cam, args.color)
        finally:
            cam.close()

    robot_stand()

    from CustomFunctions.dream_memory import DreamLogger
    DreamLogger.log_session(demo_name="demo_03_transport")

    print_footer(success=True)


if __name__ == "__main__":
    main()
