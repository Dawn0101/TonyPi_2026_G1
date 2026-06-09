#!/usr/bin/env python3
# coding=utf8
"""
步骤执行引擎 (Step Executor)
接收 LLM_Control 输出的结构化 JSON，逐条执行动作步骤。

职责：
  1. 解析 JSON steps
  2. 动作 → AGC.runActionGroup()
  3. 视觉检测 → OpenCV 实时检测
  4. TTS 播报 → hiwonder.TTS
  5. 状态反馈 → 终端打印
"""

import os
import sys
import time
import math
import cv2

# ── 项目路径 ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ── TonyPi 原生模块 ────────────────────────────
try:
    import hiwonder.ActionGroupControl as AGC
    import hiwonder.Board as Board
    import hiwonder.Camera as Camera
    import hiwonder.yaml_handle as yaml_handle
    import hiwonder.Misc as Misc
    HAS_HIWONDER = True
except ImportError:
    HAS_HIWONDER = False
    print("[Executor] 警告: 未检测到 hiwonder 模块，部分功能不可用")

# ── TTS（统一走 common.tts_speak，由 config.yaml 控制模式）──
from Demo.common import tts_speak



# ═══════════════════════════════════════════════
#  颜色检测（内嵌，同步调用，不走原有 Functions 线程）
# ═══════════════════════════════════════════════

# LAB 阈值（从 lab_config.yaml 加载）
_lab_data = None
_range_rgb = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def _load_lab_data():
    global _lab_data
    if _lab_data is not None:
        return
    try:
        lab_path = yaml_handle.lab_file_path
        _lab_data = yaml_handle.get_yaml_data(lab_path)
    except:
        # 默认阈值（兜底）
        _lab_data = {
            "red": {"min": [0, 167, 135], "max": [255, 255, 255]},
            "green": {"min": [47, 0, 135], "max": [255, 110, 255]},
            "blue": {"min": [0, 0, 0], "max": [255, 146, 120]},
        }


def _get_area_max_contour(contours, min_area=300):
    """找出面积最大的轮廓"""
    contour_area_max = 0
    area_max_contour = None
    for c in contours:
        area = math.fabs(cv2.contourArea(c))
        if area > contour_area_max:
            contour_area_max = area
            if area >= min_area:
                area_max_contour = c
    return area_max_contour, contour_area_max


def detect_color_sync(frame, target_color):
    """
    同步颜色检测
    参数:
        frame: BGR 图像
        target_color: "red" / "green" / "blue"
    返回: (found, center_x, center_y)
    """
    _load_lab_data()
    if _lab_data is None or target_color not in _lab_data:
        return False, -1, -1

    size = (320, 240)
    img_h, img_w = frame.shape[:2]

    frame_resize = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

    color_info = _lab_data[target_color]
    frame_mask = cv2.inRange(
        frame_lab,
        (color_info["min"][0], color_info["min"][1], color_info["min"][2]),
        (color_info["max"][0], color_info["max"][1], color_info["max"][2]),
    )
    eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    area_max_contour, area_max = _get_area_max_contour(contours)

    if area_max_contour is not None and area_max > 500:
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(area_max_contour)
        center_x = int(Misc.map(center_x, 0, size[0], 0, img_w))
        center_y = int(Misc.map(center_y, 0, size[1], 0, img_h))
        print(f"  [检测] 发现{target_color}物体 @ ({center_x}, {center_y})")
        return True, center_x, center_y

    return False, -1, -1


# ═══════════════════════════════════════════════
#  人脸检测（内嵌，同步调用）
# ═══════════════════════════════════════════════

_face_net = None
FACE_CONFIDENCE = 0.6


def _load_face_model():
    global _face_net
    if _face_net is not None:
        return
    model_file = os.path.join(PROJECT_ROOT, "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    config_file = os.path.join(PROJECT_ROOT, "models/deploy.prototxt")
    if os.path.exists(model_file) and os.path.exists(config_file):
        _face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)


def detect_face_sync(frame):
    """
    同步人脸检测
    返回: (found, face_count)
    """
    _load_face_model()
    if _face_net is None:
        return False, 0

    img_h, img_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1, (150, 150), [104, 117, 123], False, False)
    _face_net.setInput(blob)
    detections = _face_net.forward()

    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > FACE_CONFIDENCE:
            count += 1

    return count > 0, count


# ═══════════════════════════════════════════════
#  StepExecutor
# ═══════════════════════════════════════════════

class StepExecutor:
    """步骤执行引擎"""

    def __init__(self):
        self._camera = None
        self._camera_opened = False
        self._servo_data = None

        if HAS_HIWONDER:
            try:
                self._servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
            except:
                self._servo_data = {"servo1": 1035, "servo2": 1465}

    # ── 摄像头管理 ──────────────────────────────

    def _open_camera(self):
        """打开摄像头"""
        if self._camera_opened:
            return True
        try:
            self._camera = Camera.Camera()
            self._camera.camera_open()
            self._camera_opened = True
            print("[相机] 摄像头已打开")
            return True
        except Exception as e:
            print(f"[相机] 打开失败: {e}")
            return False

    def _close_camera(self):
        """关闭摄像头"""
        if self._camera_opened and self._camera:
            try:
                self._camera.camera_close()
            except:
                pass
        self._camera_opened = False
        print("[相机] 摄像头已关闭")

    def _grab_frame(self):
        """采集一帧图像"""
        if not self._camera_opened:
            return None
        try:
            if self._camera.frame is not None:
                return self._camera.frame.copy()
            # 等待帧
            for _ in range(30):
                time.sleep(0.03)
                if self._camera.frame is not None:
                    return self._camera.frame.copy()
            return None
        except:
            return None

    # ── TTS（统一走 common.tts_speak，由 config.yaml 控制模式）──

    def speak(self, text):
        """TTS 语音播报 — 统一走 common.tts_speak()"""
        if not text:
            return
        print(f"[TTS] {text}")
        tts_speak(text)

    # ── 单步执行 ────────────────────────────────

    def _execute_step(self, step):
        """执行单步动作"""
        action = step.get("action", "")
        params = step.get("params", {})
        desc = step.get("description", "")

        print(f"\n  ▶ [{action}] {desc}")

        # ── 动作类 ──
        if action == "go_forward":
            steps = int(params.get("steps", 1))
            AGC.runActionGroup("go_forward", times=steps)
            return True

        elif action == "go_back":
            steps = int(params.get("steps", 1))
            AGC.runActionGroup("back_fast", times=steps)
            return True

        elif action == "turn_left":
            steps = int(params.get("steps", 1))
            AGC.runActionGroup("turn_left", times=steps)
            return True

        elif action == "turn_right":
            steps = int(params.get("steps", 1))
            AGC.runActionGroup("turn_right", times=steps)
            return True

        elif action == "left_move":
            steps = int(params.get("steps", 1))
            AGC.runActionGroup("left_move_fast", times=steps)
            return True

        elif action == "right_move":
            steps = int(params.get("steps", 1))
            AGC.runActionGroup("right_move_fast", times=steps)
            return True

        elif action in ("bow", "wave", "push_ups", "stand", "squat",
                        "chest", "twist", "stepping", "left_kick",
                        "right_kick", "left_shot", "right_shot"):
            times = int(params.get("times", 1)) or int(params.get("steps", 1))
            AGC.runActionGroup(action, times=times, with_stand=True)
            return True

        # ── 视觉检测类 ──
        elif action == "detect_color":
            target = params.get("target", "red")
            self.speak(f"我找找有没有{target}色的物体")

            found = False
            for attempt in range(5):  # 最多尝试 5 帧
                frame = self._grab_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                found, cx, cy = detect_color_sync(frame, target)
                if found:
                    color_name = {"red": "红色", "green": "绿色", "blue": "蓝色"}.get(target, target)
                    self.speak(f"找到了{color_name}物体")
                    return True
                time.sleep(0.1)

            self.speak(f"没有找到{target}色的物体")
            return False

        elif action == "detect_face":
            self.speak("我看看周围有没有人")

            found = False
            for attempt in range(5):
                frame = self._grab_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                found, count = detect_face_sync(frame)
                if found:
                    self.speak(f"发现了{count}个人，你好")
                    return True
                time.sleep(0.1)

            self.speak("没有看到人")
            return False

        # ── 交互类 ──
        elif action == "speak":
            self.speak(params.get("text", ""))
            time.sleep(0.5)
            return True

        elif action == "wait":
            seconds = float(params.get("seconds", 1))
            time.sleep(seconds)
            return True

        else:
            print(f"  [未知动作] {action}")
            return False

    # ── 执行完整计划 ────────────────────────────

    def execute(self, plan):
        """
        执行 LLM 输出的完整计划

        参数:
            plan: {
                "intent": str,
                "steps": [{...}],
                "tts_response": str
            }
        返回: bool 是否全部成功
        """
        steps = plan.get("steps", [])
        if not steps:
            tts = plan.get("tts_response", "")
            if tts:
                self.speak(tts)
            return True

        # 先播报意图
        # 但如果第一步已经是 speak 同一段话，跳过避免重复播报
        tts = plan.get("tts_response", "")
        first_step_is_same_speak = (
            steps
            and steps[0].get("action") == "speak"
            and steps[0].get("params", {}).get("text", "") == tts
        )
        if tts and not first_step_is_same_speak:
            self.speak(tts)

        # 如果有视觉步骤，提前打开摄像头
        needs_camera = any(
            s["action"] in ("detect_color", "detect_face")
            for s in steps
            if "action" in s
        )
        if needs_camera:
            if not self._open_camera():
                print("[Executor] 摄像头不可用，跳过视觉步骤")

        time.sleep(0.3)

        # 逐条执行
        all_success = True
        for i, step in enumerate(steps, 1):
            if "action" not in step:
                print(f"  [跳过] 步骤 {i} 缺少 action 字段: {step}")
                continue

            print(f"\n[步骤 {i}/{len(steps)}] ", end="")
            success = self._execute_step(step)
            if not success:
                all_success = False
            time.sleep(0.2)

        # 收尾
        if needs_camera:
            self._close_camera()

        if all_success:
            print("\n[Executor] 全部步骤执行完成")
        else:
            print("\n[Executor] 部分步骤执行失败")

        return all_success


# ── 独立测试 ──────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("StepExecutor 测试")
    print("=" * 50)

    executor = StepExecutor()

    # 测试 1: 动作测试
    print("\n--- 测试 1: 简单动作 ---")
    plan = {
        "intent": "鞠躬打招呼",
        "steps": [
            {"action": "speak", "params": {"text": "你好，我是TonyPi"}, "description": "打招呼"},
            {"action": "bow", "params": {}, "description": "鞠躬"},
            {"action": "speak", "params": {"text": "很高兴见到你"}, "description": "说高兴"},
        ],
        "tts_response": "好的",
    }
    executor.execute(plan)

    # 测试 2: 视觉测试（需要摄像头）
    print("\n--- 测试 2: 颜色检测 ---")
    plan = {
        "intent": "找红色物体",
        "steps": [
            {"action": "detect_color", "params": {"target": "red"}, "description": "找红色"},
            {"action": "speak", "params": {"text": "检测完成"}, "description": "报告结果"},
        ],
        "tts_response": "开始检测",
    }
    executor.execute(plan)
