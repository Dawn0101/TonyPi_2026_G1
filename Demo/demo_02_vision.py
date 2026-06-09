#!/usr/bin/env python3
# coding=utf8
"""
Demo 02 — 颜色识别 + 颜色追踪 + 巡线追踪
══════════════════════════════════════════════

演示内容:
  1. 颜色识别: 摄像头检测红/绿/蓝物体，实时标注 + TTS 播报
  2. 颜色追踪: 云台（头部舵机）自动跟踪指定颜色物体（PID控制）
  3. 巡线追踪: 沿地面黑色线条自主行走

运行:
    python3 Demo/demo_02_vision.py

按 Ctrl+C 跳过当前子演示
"""

import os
import sys
import time
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer, wait_key,
    CameraManager, HAS_CAMERA, HAS_AGC,
)


# ═══════════════════════════════════════════════
#  子演示 1: 颜色识别
# ═══════════════════════════════════════════════

def subdemo_color_detect(cam):
    """颜色识别：检测红/绿/蓝物体"""
    print_header("子演示 2-1: 颜色识别", "请在摄像头前展示红/绿/蓝物体")

    from CustomFunctions.step_executor import detect_color_sync

    COLOR_NAMES = {"red": "红色", "green": "绿色", "blue": "蓝色"}
    detected_history = set()
    start_time = time.time()

    print("  将依次扫描红、绿、蓝三色物体...")
    print("  按 Ctrl+C 跳过")
    tts_speak("开始颜色识别，请展示彩色物体")

    try:
        while time.time() - start_time < 15:
            frame = cam.grab(timeout=0.5)
            if frame is None:
                time.sleep(0.05)
                continue

            display = frame.copy()
            any_found = False

            for color_en, color_cn in COLOR_NAMES.items():
                found, cx, cy = detect_color_sync(frame, color_en)
                if found:
                    any_found = True
                    # 画圆标注
                    cv2.circle(display, (cx, cy), 40, (0, 255, 0), 2)
                    cv2.putText(display, color_cn, (cx + 30, cy - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if color_cn not in detected_history:
                        detected_history.add(color_cn)
                        print(f"  [检测] ✓ 发现{color_cn}物体")
                        tts_speak(f"发现了{color_cn}物体")

            # 显示状态
            status = f"已发现: {', '.join(detected_history)}" if detected_history else "扫描中..."
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if any_found:
                time.sleep(0.5)  # 检测到后稍作停留

    except KeyboardInterrupt:
        pass

    print(f"\n  颜色识别结果: 共发现 {len(detected_history)} 种颜色 — "
          f"{', '.join(detected_history) if detected_history else '无'}")

    if not detected_history:
        tts_speak("未检测到彩色物体，请调整物体位置")
    else:
        tts_speak(f"检测完毕，发现了{'、'.join(detected_history)}")


# ═══════════════════════════════════════════════
#  子演示 2: 颜色追踪（云台跟踪）
# ═══════════════════════════════════════════════

def subdemo_color_track(cam):
    """颜色追踪：云台 PID 跟踪红色物体"""
    print_header("子演示 2-2: 颜色追踪", "云台将自动跟踪红色物体")

    try:
        import Functions.ColorTrack as ColorTrack
    except Exception as e:
        print(f"  [跳过] ColorTrack 模块加载失败: {e}")
        print("  (ColorTrack 需要 hiwonder 硬件环境)")
        return

    ColorTrack.debug = False

    # 设置追踪颜色为红色
    try:
        ColorTrack.setTargetColor(("red",))
    except Exception:
        print("  [警告] setTargetColor 可能不支持，使用默认红色")
    ColorTrack.init()
    ColorTrack.start()

    print("  云台正在跟踪红色物体，移动红色物体试试...")
    print("  按 Ctrl+C 停止追踪")
    tts_speak("开始追踪红色物体")

    start_time = time.time()
    try:
        while time.time() - start_time < 15:
            frame = cam.grab(timeout=0.5)
            if frame is None:
                time.sleep(0.03)
                continue

            display = ColorTrack.run(frame)
            if display is not None:
                cv2.putText(display, "Color Track - 追踪红色物体", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except KeyboardInterrupt:
        pass
    finally:
        ColorTrack.stop()
        ColorTrack.exit()
        print("  颜色追踪已停止")
        tts_speak("追踪结束")


# ═══════════════════════════════════════════════
#  子演示 3: 巡线追踪
# ═══════════════════════════════════════════════

def subdemo_line_patrol(cam):
    """巡线追踪：沿黑色线条自主行走"""
    print_header("子演示 2-3: 巡线追踪", "机器人将沿地面黑色线条行走")

    try:
        import Functions.VisualPatrol as VisualPatrol
    except Exception as e:
        print(f"  [跳过] VisualPatrol 模块加载失败: {e}")
        print("  (VisualPatrol 需要 hiwonder 硬件环境)")
        return

    # 设置目标颜色为黑色
    try:
        VisualPatrol.setLineTargetColor(("white",))
    except Exception:
        print("  [警告] setLineTargetColor 可能不支持，使用默认黑色")
    VisualPatrol.init()
    VisualPatrol.start()

    print("  ⚠ 请确保地面有黑色线条！")
    print("  机器人将沿着黑线自主行走...")
    print("  按 Ctrl+C 停止巡线")

    if HAS_AGC:
        tts_speak("开始巡线，请确保地面有黑色引导线")

    start_time = time.time()
    try:
        while time.time() - start_time < 15:
            frame = cam.grab(timeout=0.5)
            if frame is None:
                time.sleep(0.03)
                continue

            display = VisualPatrol.run(frame)
            if display is not None:
                cv2.putText(display, "Line Patrol - 巡线追踪", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except KeyboardInterrupt:
        pass
    finally:
        VisualPatrol.stop()
        VisualPatrol.exit()
        print("  巡线追踪已停止")

    if HAS_AGC:
        tts_speak("巡线结束")


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    robot_init()

    cam = CameraManager()
    if not cam.open():
        print("[Demo02] 摄像头不可用，退出")
        return

    try:
        # 子演示 1: 颜色识别
        subdemo_color_detect(cam)
        wait_key(2)

        # 子演示 2: 颜色追踪
        subdemo_color_track(cam)
        wait_key(2)

        # 子演示 3: 巡线追踪
        subdemo_line_patrol(cam)

    finally:
        cam.close()
        robot_stand()

    from CustomFunctions.dream_memory import DreamLogger
    DreamLogger.log_session(demo_name="demo_02_vision")

    print_footer(success=True)


if __name__ == "__main__":
    main()
