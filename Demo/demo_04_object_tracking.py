#!/usr/bin/env python3
# coding=utf8
"""
Demo 04 — 实时目标检测与追踪
══════════════════════════════════════

演示内容:
  使用 MobileNet SSD (Caffe) 在摄像头画面中实时检测
  21 种常见物体，重点追踪"人"并做出交互响应。

  相比原厂 FaceDetect.py（只识别人脸）：
    · 检测 21 种物体（人/瓶子/猫/狗/椅子...）
    · 结构化 JSON 输出（label, conf, bbox, center）
    · 头部主动扫描 + 目标跟踪
    · 发现重要物体时 TTS 播报

运行:
    python3 Demo/demo_04_object_tracking.py
    python3 Demo/demo_04_object_tracking.py --sim    模拟模式

前置条件:
    models/ 目录下有 MobileNet SSD Caffe 模型文件:
      - MobileNetSSD_deploy.prototxt
      - MobileNetSSD_deploy.caffemodel
"""

import os
import sys
import time
import math
import cv2
import threading
import signal

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer,
    HAS_AGC, AGC,
)

try:
    import hiwonder.Board as Board
    import hiwonder.Camera as Camera
    import hiwonder.yaml_handle as yaml_handle
    HAS_HIWONDER = True
except ImportError:
    HAS_HIWONDER = False

from CustomFunctions.object_detector import ObjectDetector


# ═══════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════

SCAN_RANGE_MIN = 1000       # 头部扫描左极限
SCAN_RANGE_MAX = 2000       # 头部扫描右极限
SCAN_STEP = 15              # 每帧扫描步进（像素）
TRACK_CONFIDENCE = 0.3      # 追踪目标的最低置信度
GREET_COOLDOWN = 10         # 打招呼冷却（秒）
TTS_COOLDOWN = 5            # 物体播报冷却（秒）
SHOW_CAMERA = True          # 是否显示 cv2.imshow 画面窗口
SHOW_FPS_INTERVAL = 10      # 每 N 帧在画面上刷新一次标签（防闪烁）

# ── 检测是否有显示器（SSH 无界面时自动关闭 cv2.imshow）──
try:
    import os as _os
    _has_display = bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))
except Exception:
    _has_display = False
if not _has_display:
    SHOW_CAMERA = False
    print("[Demo04] ℹ 无显示器，关闭画面窗口（仅 SSH 下运行）")

# 值得 TTS 播报的物体（中文名 → 动作）
INTERESTING_OBJECTS = {
    "人":       "wave",
    "杯子":     "speak",
    "手机":     "speak",
    "球":       "speak",
    "书":       "speak",
    "瓶子":     "speak",
    "玩具":     "speak",
    "猫":       "speak",
    "狗":       "speak",
}

# 重要物体中英文映射（用于 console 打印）
OBJECT_EMOJI = {
    "人": "🧑", "杯子": "🥤", "手机": "📱", "球": "⚽",
    "书": "📖", "瓶子": "🍾", "玩具": "🧸", "猫": "🐱", "狗": "🐶",
}


# ═══════════════════════════════════════════════
#  舵机控制
# ═══════════════════════════════════════════════

def load_servo_center():
    """读取舵机中位

    servo1 = 头部俯仰（1500 = 水平）
    servo2 = 头部左右（1500 = 中）
    
    注意: servo_config.yaml 中的 servo1 是校准偏差值，
    不是水平位置。水平位置固定为 1500 PWM。
    """
    try:
        servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        s2 = servo_data.get("servo2", 1465)
        return 1500, s2  # servo1 固定 1500（水平），不从 config 读
    except Exception:
        return 1500, 1500


SERVO1_CENTER, SERVO2_CENTER = load_servo_center()


def set_head(pulse1, pulse2, duration=100):
    """设置头部舵机位置"""
    if HAS_HIWONDER:
        try:
            Board.setPWMServoPulse(1, int(pulse1), duration)
            Board.setPWMServoPulse(2, int(pulse2), duration)
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  主演示
# ═══════════════════════════════════════════════

def main():
    robot_init()

    # ── 解析命令行参数（提前解析，--sim 走模拟模式）──
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args, _ = parser.parse_known_args()

    print_header(
        "Demo 04 — 实时目标检测与追踪",
        "MobileNet SSD · 21 种物体实时检测"
    )

    # ── 模拟模式（无需模型/摄像头）──
    if args.sim:
        run_simulated()
        return

    # ── 加载检测器 ──
    detector = ObjectDetector(confidence_threshold=TRACK_CONFIDENCE)
    if not detector.available:
        print("  ⚠ 模型未加载")
        print("  请确认 models/ 目录下有:")
        print("    - MobileNetSSD_deploy.prototxt")
        print("    - MobileNetSSD_deploy.caffemodel")
        print("  ---")
        print("  提示: python3 Demo/demo_04_object_tracking.py --sim")
        return

    # ── 打开摄像头 ──
    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(0.5)
    print("[相机] 摄像头已打开")

    # ── 扫描状态 ──
    scan_pulse = SERVO2_CENTER
    scan_direction = 1   # 1=右, -1=左
    current_target = None   # 当前追踪的人 {"cx": int, "last_seen": float}
    last_greet_time = 0
    last_tts_times = {}     # {label_cn: last_time}
    tts_consecutive = {}    # {label_cn: 连续帧计数}
    frame_count = 0

    # TTS 防误报参数
    TTS_MIN_CONFIDENCE = 0.55         # TTS 最低置信度
    TTS_CONSECUTIVE_FRAMES = 5        # 连续 N 帧检测到才播报

    print("\n  正在检测... 机器人会主动扫描周围环境")
    print('  发现"人"时 → 头部跟踪 + 靠近打招呼')
    print("  发现重要物体 → TTS 播报")
    print("  按 Ctrl+C 退出")
    print("-" * 56)

    # ── 注册退出信号（覆盖 common.py 的 os._exit(0)，确保头部归位）──
    _exit_flag = [False]
    def _demo_exit(signum, frame):
        _exit_flag[0] = True
    signal.signal(signal.SIGINT, _demo_exit)
    signal.signal(signal.SIGTERM, _demo_exit)

    try:
        while not _exit_flag[0]:
            frame_count += 1

            # ── 更快地取帧（减少等待）──
            frame = None
            try:
                if cam.frame is not None:
                    frame = cam.frame.copy()
            except Exception:
                pass
            if frame is None:
                try:
                    cam.frame  # 尝试触发一次刷新
                except Exception:
                    pass
                time.sleep(0.01)
                continue

            img_h, img_w = frame.shape[:2]

            # ── 运行检测 ──
            t0 = time.time()
            try:
                results = detector.detect(frame)
            except Exception as e:
                print(f"  [检测器] ⚠ detect() 异常: {e}")
                results = []
            elapsed_ms = int((time.time() - t0) * 1000)

            # 提取"人"
            persons = [r for r in results if r["label"] == "person"]
            # 提取重要物体（除人之外，且非 background）
            interesting = [r for r in results
                           if r["label_cn"] in INTERESTING_OBJECTS
                           and r["label"] != "person"
                           and r["label"] != "background"]
            # 提取所有非背景物体（用于可视化展示）
            all_objects = [r for r in results
                           if r["label"] != "background"
                           and r["confidence"] >= 0.3]

            # ── 控制台输出（每秒约 2 次，不刷屏）──
            if frame_count % 5 == 0:
                status = f"  [{elapsed_ms}ms] "
                if persons:
                    p = persons[0]
                    third = img_w / 3
                    if p["center"][0] < third:
                        dir_str = "左侧"
                    elif p["center"][0] > img_w - third:
                        dir_str = "右侧"
                    else:
                        dir_str = "中央"
                    status += f"🧑 人在{dir_str} (conf={p['confidence']:.2f})"
                else:
                    status += "🔍 扫描中..."

                if interesting:
                    objs = [f"{OBJECT_EMOJI.get(o['label'], '')}{o['label']}"
                            for o in interesting[:3]]
                    status += f" | 发现: {' '.join(objs)}"
                elif all_objects:
                    # 显示其他非重点物体
                    objs = [f"{o['label']}({o['confidence']:.2f})"
                            for o in all_objects[:3]]
                    status += f" | {', '.join(objs)}"
                print(status)

            # ── 画面可视化（cv2.imshow）——只画概率最高的物体──
            if SHOW_CAMERA and frame_count % 2 == 0:  # 隔帧绘制，减轻负担
                frame_display = frame.copy()
                img_h2, img_w2 = frame_display.shape[:2]

                # 只取 top-1 物体画框（画面简洁）
                top_obj = all_objects[0] if all_objects else None
                if top_obj is not None:
                    x1, y1, x2, y2 = top_obj["bbox"]
                    label = top_obj["label"]
                    conf = top_obj["confidence"]

                    # 人的框用绿色，其他用青色
                    color = (0, 255, 0) if top_obj["label"] == "person" else (255, 255, 0)
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 3)
                    label_text = f"{label} ({conf:.2f})"
                    cv2.putText(frame_display, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cx, cy = top_obj["center"]
                    cv2.circle(frame_display, (cx, cy), 5, color, -1)

                # 左上角信息（显示置信度，方便调试）
                if top_obj:
                    conf_pct = int(top_obj["confidence"] * 100)
                    info = f"{elapsed_ms}ms | {top_obj['label']} ({conf_pct}%)"
                else:
                    info = f"{elapsed_ms}ms | 无检测"
                cv2.putText(frame_display, info, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("TonyPi - Object Detection", frame_display)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    _exit_flag[0] = True
                    break

            # ── 人追踪逻辑 ──
            if persons:
                target = persons[0]  # 取置信度最高的人
                cx, cy = target["center"]
                area = target["area"]
                img_area = img_h * img_w
                area_ratio = area / img_area

                # 更新追踪目标
                current_target = {"cx": cx, "cy": cy, "last_seen": time.time()}

                # 头部跟随 — PID 风格的简单跟随
                target_pulse2 = SERVO2_CENTER + int((cx / img_w - 0.5) * 400)
                target_pulse2 = max(SCAN_RANGE_MIN, min(SCAN_RANGE_MAX, target_pulse2))
                target_pulse1 = SERVO1_CENTER + int((0.5 - cy / img_h) * 200)
                target_pulse1 = max(SERVO1_CENTER - 200, min(SERVO1_CENTER + 200, target_pulse1))
                set_head(target_pulse1, target_pulse2, 50)
                scan_pulse = target_pulse2  # 同步扫描位置

                # 人靠近且在画面中央 → 打招呼
                if area_ratio > 0.05:  # 面积占比 > 5% = 距离近
                    now = time.time()
                    if now - last_greet_time > GREET_COOLDOWN:
                        third = img_w / 3
                        if third < cx < img_w - third:  # 人在中央
                            print(f"\n  👋 人靠近了，打招呼！")
                            tts_speak("你好")
                            if HAS_AGC:
                                AGC.runActionGroup("wave")
                                # wave 会压低头部，执行完后复位到水平
                                set_head(SERVO1_CENTER, SERVO2_CENTER, 500)
                            last_greet_time = now

            else:
                # 没有检测到人 → 头部主动扫描
                current_target = None
                scan_pulse += SCAN_STEP * scan_direction
                if scan_pulse > SCAN_RANGE_MAX:
                    scan_pulse = SCAN_RANGE_MAX
                    scan_direction = -1
                elif scan_pulse < SCAN_RANGE_MIN:
                    scan_pulse = SCAN_RANGE_MIN
                    scan_direction = 1
                set_head(SERVO1_CENTER, scan_pulse, 30)

            # ── TTS 播报（连续帧确认 + 高置信度门槛，防误报）──
            for obj in all_objects:
                now = time.time()
                label_cn = obj["label_cn"]
                label = obj["label"]
                conf = obj["confidence"]

                if label == "person":
                    continue

                # 连续帧计数：物体必须连续出现 N 帧才播报
                count = tts_consecutive.get(label_cn, 0)
                if conf >= TTS_MIN_CONFIDENCE:
                    tts_consecutive[label_cn] = count + 1
                else:
                    tts_consecutive[label_cn] = 0
                    continue

                if tts_consecutive[label_cn] < TTS_CONSECUTIVE_FRAMES:
                    continue

                last = last_tts_times.get(label_cn, 0)
                if now - last > TTS_COOLDOWN:
                    # 选择合体的量词/语气
                    if label_cn in ("猫", "狗"):
                        tts_msg = f"我看到一只{label_cn}"
                    elif label_cn in ("书", "手机", "杯子", "瓶子", "玩具"):
                        tts_msg = f"发现一个{label_cn}"
                    elif label_cn == "球":
                        tts_msg = "发现一个球"
                    else:
                        tts_msg = f"发现{label_cn}"
                    print(f"\n  📢 {tts_msg} (conf={conf:.2f}, {tts_consecutive[label_cn]}帧连续)")
                    tts_speak(tts_msg)
                    last_tts_times[label_cn] = now
                    tts_consecutive[label_cn] = 0  # 播报后重置

            # 小幅休眠防树莓派过热，0.02s ≈ 50fps 上限
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        # 头部归位（必须在关闭舵机前执行）
        print("[Demo04] 头部归位...")
        set_head(SERVO1_CENTER, SERVO2_CENTER, 500)
        time.sleep(0.3)
        try:
            cam.camera_close()
        except Exception:
            pass
        print("[相机] 摄像头已关闭")

    # 恢复 SIGINT 让 common.py 的 robot_stand() 能正常跑
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    robot_stand()

    from CustomFunctions.dream_memory import DreamLogger
    DreamLogger.log_session(demo_name="demo_04_object_tracking")

    print_footer(success=True)


# ═══════════════════════════════════════════════
#  模拟模式
# ═══════════════════════════════════════════════

def run_simulated():
    """模拟模式 — 展示 ObjectDetector 的概念"""
    print_header(
        "Demo 04 — 实时目标检测（模拟版）",
        "MobileNet SSD 21 种物体实时检测"
    )

    print("  [模拟] 摄像头画面流:")
    print("    ┌──────────────────────────┐")
    print("    │    🧑    🥤              │  ← 检测到: 人(中央), 杯子(右侧)")
    print("    │                          │")
    print("    │         📖               │  ← 检测到: 书(左下)")
    print("    └──────────────────────────┘")
    print()
    print("  [模拟] 头部正在左右扫描...")
    print()

    import random

    scenarios = [
        ("🧑 人在左侧", "turn_left"),
        ("🧑 人在中央 距离近", "wave"),
        ("🥤 发现杯子", "speak('发现杯子')"),
        ("🧑 人在右侧", "turn_right"),
        ("📱 发现手机", "speak('发现手机')"),
        ("⚽ 发现球", "speak('发现球')"),
    ]

    tts_speak("模拟模式，实时目标检测演示")

    try:
        for r in range(1, 9):
            desc, action = random.choice(scenarios)
            print(f"  [{r}] 帧分析完成: {desc}")
            print(f"      → 执行: {action}")

            if "wave" in action:
                tts_speak("你好")
                if HAS_AGC:
                    AGC.runActionGroup("wave")
            elif "speak" in action:
                tts_speak(action.split("'")[1])
            elif "turn" in action:
                pass  # 模拟版不做实际转向

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n  [模拟] 退出")

    print()
    print("  [模拟] 检测结束")
    robot_stand()
    print_footer(success=True)


if __name__ == "__main__":
    main()
