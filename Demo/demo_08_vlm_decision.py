#!/usr/bin/env python3
# coding=utf8
"""
Demo 08 — VLM 多帧视觉动作决策
══════════════════════════════════════════════════

核心逻辑:
  5 秒观察窗口 → 每秒拍 1 帧 (共 5 帧)
  → 5 帧 + 可用动作列表 → VLM
  → VLM 输出 {"action_id": "...", "reason": "..."}
  → 执行动作 → 重新观察

相比传统 CV 追踪（写死颜色阈值/轮廓参数），这个方案:
  · VLM 直接理解画面语义 — "用户在向前走" vs "色块偏移"
  · 动作列表由 VLM 自主选择，不用手动调 PID/阈值
  · 可通过修改动作列表改变机器人行为，不改代码

给 VLM 的动作列表:
  go_forward    向前走一步        turn_right    右转
  go_back        向后退一步        bow           鞠躬
  turn_left      左转              wave          挥手
  stand          立正              speak         说话 (TTS)
  wait           等待 5 秒再观察

VLM 输出格式:
  {
    "action_id": "go_forward",
    "reason": "用户正在向前走，机器人应跟随",
    "speak_text": "我跟着你"
  }

运行:
    python3 Demo/demo_08_vlm_decision.py
    python3 Demo/demo_08_vlm_decision.py --sim    模拟模式

前置条件:
    config.yaml 中配置了 zhipu.api_key（GLM-4V 视觉大模型）
"""

import os
import sys
import time
import json
import base64

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Demo.common import (
    robot_init, robot_stand, tts_speak,
    print_header, print_footer,
    HAS_AGC, AGC,
)

import hiwonder.Camera as Camera
import cv2


# ═══════════════════════════════════════════════
#  可用动作列表（给 VLM 的决策选项）
# ═══════════════════════════════════════════════

AVAILABLE_ACTIONS = [
    {"id": "go_forward", "name": "前进", "desc": "向前走一步（约 10cm）"},
    {"id": "go_back",    "name": "后退", "desc": "向后退一步（约 10cm）"},
    {"id": "turn_left",  "name": "左转", "desc": "向左转 30 度"},
    {"id": "turn_right", "name": "右转", "desc": "向右转 30 度"},
    {"id": "bow",        "name": "鞠躬", "desc": "鞠躬打招呼"},
    {"id": "wave",       "name": "挥手", "desc": "挥手打招呼"},
    {"id": "dance",      "name": "跳舞", "desc": "跳舞"},
    {"id": "stand",      "name": "立正", "desc": "回到初始立正姿势"},
    {"id": "speak",      "name": "说话", "desc": "通过扬声器说一句话"},
    {"id": "wait",       "name": "等待", "desc": "不做动作，等待 5 秒后重新观察"},
]


# ═══════════════════════════════════════════════
#  动作模板加载
# ═══════════════════════════════════════════════

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def load_templates():
    """
    扫描 templates/ 目录，加载所有 .png 模板图片
    文件名（不含扩展名）作为动作ID，例如:
      templates/bow.png   → {"action_id": "bow",   "image": ...}
      templates/dance.png → {"action_id": "dance", "image": ...}
      templates/wave.png  → {"action_id": "wave",  "image": ...}
    返回: list[{"action_id": str, "name": str, "b64": str}]
    """
    if not os.path.isdir(TEMPLATES_DIR):
        print(f"  [模板] 目录不存在: {TEMPLATES_DIR}")
        return []

    templates = []
    for fname in sorted(os.listdir(TEMPLATES_DIR)):
        if not fname.lower().endswith(".png"):
            continue
        path = os.path.join(TEMPLATES_DIR, fname)
        action_id = os.path.splitext(fname)[0].lower()  # bow.png → "bow"
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"  [模板] ⚠ 无法读取: {fname}")
                continue
            # 缩放到统一宽度
            h, w = img.shape[:2]
            scale = RESIZE_WIDTH / w
            new_size = (RESIZE_WIDTH, int(h * scale))
            small = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
            _, buf = cv2.imencode('.jpg', small)
            b64 = base64.b64encode(buf.tobytes()).decode()
            templates.append({"action_id": action_id, "name": fname, "b64": b64})
            print(f"  [模板] 加载: {fname} → action_id='{action_id}' ({len(b64)//1024} KB)")
        except Exception as e:
            print(f"  [模板] ⚠ 加载失败 {fname}: {e}")

    return templates

VLM_SYSTEM_PROMPT = f"""你是一个机器人视觉决策助手。
我给你:
  1) 参考模板照片 — 每种动作的样子（如 bow=鞠躬、dance=跳舞）
  2) 5 张连续拍摄的实时照片（每秒 1 张）
  3) 机器人可执行的动作列表

请做两件事：
  A) 看实时照片的变化趋势 → 判断人在往哪移动（空间判断）
  B) 拿实时照片和参考模板对比 → 判断人在做什么动作（动作识别）

可用动作列表：
{json.dumps(AVAILABLE_ACTIONS, ensure_ascii=False, indent=2)}

输出严格的 JSON 格式，不要包含任何其他文字：
{{
    "action_id": "从动作列表中选一个 id",
    "reason": "为什么选这个动作的中文说明",
    "speak_text": "执行动作时机器人说的话（不需要说话时填空字符串"
}}

判断规则：
1. 先看参考模板 — 如果实时照片中人的姿势和某个模板高度相似 → 执行对应的动作（如模板是 bow → 选 bow）
2. 如果人在画面中移动（左右偏移或大小变化）→ 选 go_forward / go_back / turn_left / turn_right
3. 如果画面没有变化或没有目标 → 选 wait
4. 不要重复选同一个动作超过 3 次
5. speak_text 填写执行动作时机器人该说的话（如"你好""我来了"等），不需要说话时填空字符串""
"""


# ═══════════════════════════════════════════════
#  配置加载
# ═══════════════════════════════════════════════

def _load_config():
    import yaml
    cfg_path = os.path.join(_PROJECT_ROOT, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

OBSERVE_SECONDS = 5       # 观察窗口（秒）
OBSERVE_FRAMES = 5        # 窗口内采集的帧数
RESIZE_WIDTH = 320        # 送给 VLM 的图片宽度（等比例缩放）


# ═══════════════════════════════════════════════
#  VLM 调用
# ═══════════════════════════════════════════════

def vlm_decide(frames, api_key, templates=None):
    """
    将 5 帧图片 + 模板 + 动作列表发给 VLM，返回决策结果
    参数:
        frames: list of BGR images (numpy arrays)
        api_key: 智谱 API 密钥
        templates: list[{"action_id": str, "name": str, "b64": str}] — 参考模板
    返回:
        dict: {"action_id": str, "reason": str, "speak_text": str}
    """
    try:
        import requests
    except ImportError:
        return {"action_id": "wait", "reason": "依赖缺失", "speak_text": ""}

    # ── 编码图片（等比例缩放到 320px 宽以节省 token）──
    encoded = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_size = (RESIZE_WIDTH, int(h * scale))
        small = cv2.resize(frame, new_size, interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.jpg', small)
        b64 = base64.b64encode(buf.tobytes()).decode()
        encoded.append(b64)

    print(f"  [VLM] {len(frames)} 帧已编码 ({sum(len(e) for e in encoded)//1024} KB)")

    # ── 构建消息 ──
    content = []

    # 先放参考模板（如果有）
    if templates:
        tmpl_text = "【参考模板】以下是不同动作的参考照片，用于对比识别："
        for t in templates:
            tmpl_text += f"\n  - {t['name']}（动作ID: {t['action_id']}）"
        content.append({"type": "text", "text": tmpl_text})
        for t in templates:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{t['b64']}"}
            })

    # 再放实时帧
    live_text = f"\n【实时画面】这是连续{OBSERVE_FRAMES}张实时照片（每秒1张），请与上面的参考模板对比，结合画面变化趋势，选择最合适的动作。"
    content.append({"type": "text", "text": live_text})
    for i, b64 in enumerate(encoded):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "glm-4.1v-thinking-flashx",
        "messages": [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.3,
    }

    try:
        resp = requests.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  [VLM] HTTP {resp.status_code}")
            return {"action_id": "wait", "reason": f"API 错误 {resp.status_code}", "speak_text": ""}

        reply = resp.json()["choices"][0]["message"]["content"]
        reply = reply.strip()
        if "```json" in reply:
            reply = reply.split("```json", 1)[1]
        if "```" in reply:
            reply = reply.rsplit("```", 1)[0]
        reply = reply.strip()

        # ── 健壮的 JSON 解析 ──
        result = None
        parse_attempts = []

        # 尝试 1: 标准解析
        try:
            result = json.loads(reply)
        except json.JSONDecodeError as e1:
            parse_attempts.append(f"标准解析失败: {e1}")

        # 尝试 2: 找到第一个 { 和最后一个 } 截取
        if result is None:
            try:
                brace_start = reply.find('{')
                brace_end = reply.rfind('}')
                if brace_start != -1 and brace_end > brace_start:
                    candidate = reply[brace_start:brace_end + 1]
                    result = json.loads(candidate)
                    print(f"  [VLM] ⚠ 用 {{…}} 截取修复了 JSON")
            except (json.JSONDecodeError, ValueError) as e2:
                parse_attempts.append(f"{{…}} 截取失败: {e2}")

        # 尝试 3: 替换转义问题（常见：引号内换行、未转义引号）
        if result is None:
            try:
                # 用更宽松的解析 — 替换单引号为双引号、移除控制字符
                fixed = reply.replace("'", '"')
                # 修复常见未转义内嵌引号（如 reason 字段中的 " 或「」）
                import re
                fixed = re.sub(r'(?<!\\)"', '"', fixed)
                brace_start = fixed.find('{')
                brace_end = fixed.rfind('}')
                if brace_start != -1 and brace_end > brace_start:
                    candidate = fixed[brace_start:brace_end + 1]
                    result = json.loads(candidate)
                    print(f"  [VLM] ⚠ 转义修复后 JSON 解析成功")
            except (json.JSONDecodeError, ValueError) as e3:
                parse_attempts.append(f"转义修复失败: {e3}")

        # ── 所有尝试都失败 → 输出调试信息 ──
        if result is None:
            print(f"\n  [VLM] ═══════════════════════════════")
            print(f"  [VLM] ❌ JSON 解析全部失败")
            print(f"  [VLM] 原始回复 ({len(reply)} 字符):")
            # 截取前 300 字符展示，防止刷屏
            preview = reply[:300].replace('\n', '\\n').replace('\r', '\\r')
            print(f"  [VLM] >>> {preview} {'...' if len(reply) > 300 else ''}")
            print(f"  [VLM] 解析尝试:")
            for a in parse_attempts:
                print(f"  [VLM]   · {a}")
            print(f"  [VLM] ═══════════════════════════════\n")
            return {"action_id": "wait", "reason": "VLM 返回了无效 JSON", "speak_text": ""}

        print(f"  [VLM] 决策: {result.get('action_id')} — {result.get('reason', '')[:60]}")
        return result

    except Exception as e:
        print(f"  [VLM] 调用失败: {e}")
        return {"action_id": "wait", "reason": str(e), "speak_text": ""}


# ═══════════════════════════════════════════════
#  动作执行
# ═══════════════════════════════════════════════

def execute_action(action_id, reason, speak_text):
    """执行 VLM 选择的动作"""
    print(f"\n  [执行] {action_id} — {reason}")

    # 先说话
    if speak_text:
        tts_speak(speak_text)
        time.sleep(0.5)

    # 映射到动作组
    action_map = {
        "go_forward":  ("go_forward", {"times": 2, "with_stand": True}),
        "go_back":     ("back_fast",  {"times": 2, "with_stand": True}),
        "turn_left":   ("turn_left",  {"times": 1}),
        "turn_right":  ("turn_right", {"times": 1}),
        "bow":         ("bow",        {"times": 1}),
        "wave":        ("wave",       {"times": 1}),
        "dance":       ("dance",      {"times": 1}),  # dance.d6b
        "stand":       ("stand",      {"times": 1}),
    }

    if action_id == "speak":
        # speak 动作的文本已经在 speak_text 里了，不需要再执行动作组
        pass
    elif action_id == "wait":
        print("  [等待] 5 秒后重新观察")
        time.sleep(OBSERVE_SECONDS)
    elif action_id in action_map:
        group, kwargs = action_map[action_id]
        if HAS_AGC:
            try:
                AGC.runActionGroup(group, **kwargs)
            except Exception as e:
                print(f"  [执行] 动作失败: {e}")
    else:
        print(f"  [执行] 未知动作: {action_id}，跳过")


# ═══════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════

def decision_loop(api_key):
    """
    VLM 动作决策主循环
    流程: 观察 5 秒(拍 5 帧) → VLM 决策 → 执行 → 重复
    """
    # ── 加载动作模板 ──
    templates = load_templates()
    if templates:
        print(f"  [模板] 已加载 {len(templates)} 个动作模板: {[t['action_id'] for t in templates]}")
    else:
        print("  [模板] 无模板，VLM 将仅靠画面变化趋势做空间判断")
    # ── 打开摄像头 ──
    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(0.5)
    print("[相机] 摄像头已打开")

    round_num = 0
    try:
        while True:
            round_num += 1
            print(f"\n{'=' * 50}")
            print(f"  第 {round_num} 轮决策 — 观察中 ({OBSERVE_SECONDS}s / {OBSERVE_FRAMES} 帧)")
            print(f"{'=' * 50}")

            # ── 采集 5 帧（每秒 1 帧）──
            frames = []
            for i in range(OBSERVE_FRAMES):
                frame = None
                for _ in range(10):
                    try:
                        if cam.frame is not None:
                            frame = cam.frame.copy()
                            break
                    except Exception:
                        pass
                    time.sleep(0.05)
                if frame is not None:
                    frames.append(frame)
                    print(f"    [{i+1}/{OBSERVE_FRAMES}] 帧已采集 ({frame.shape[1]}×{frame.shape[0]})")
                else:
                    print(f"    [{i+1}/{OBSERVE_FRAMES}] ⚠ 采集失败")
                time.sleep(1.0)  # 每秒 1 帧

            if len(frames) < 2:
                print("  [主循环] 有效帧不足，重新观察")
                continue

            # ── VLM 决策 ──
            print(f"\n  [VLM] 正在分析 {len(frames)} 帧画面...")
            decision = vlm_decide(frames, api_key, templates=templates)

            action_id = decision.get("action_id", "wait")
            reason = decision.get("reason", "")
            speak_text = decision.get("speak_text", "")

            print(f"  [决策] → {action_id}")
            print(f"  [理由] {reason}")

            # ── 执行 ──
            execute_action(action_id, reason, speak_text)

            print(f"\n  [完成] 第 {round_num} 轮结束，准备下一轮观察...")

    except KeyboardInterrupt:
        print("\n  [主循环] 用户中断")
    finally:
        try:
            cam.camera_close()
        except Exception:
            pass
        print("[相机] 摄像头已关闭")


# ═══════════════════════════════════════════════
#  模拟模式
# ═══════════════════════════════════════════════

def run_simulated():
    """模拟模式 — 用预置场景展示概念"""
    print_header(
        "Demo 08 — VLM 多帧视觉动作决策（模拟版）",
        "5 帧 → VLM → 动作决策 → 执行 → 循环"
    )

    import random

    scenarios = [
        ("go_forward", "用户正在远离摄像头，需要前进跟随", "我来了"),
        ("turn_left", "用户向左走，应左转跟随", "我往左边走"),
        ("bow", "用户走近了，应鞠躬打招呼", "你好"),
        ("wave", "用户挥手示意，应挥手回应", "你好"),
        ("wait", "画面没有变化，等待", ""),
    ]

    print("  模拟流程:")
    print("  ═══════════════════════════════════════════")
    print("  第 1 秒: 拍照 → 用户站在画面中央")
    print("  第 2 秒: 拍照 → 用户向左移动")
    print("  第 3 秒: 拍照 → 用户继续向左")
    print("  第 4 秒: 拍照 → 用户已经到画面左侧边缘")
    print("  第 5 秒: 拍照 → 用户消失")
    print()
    print("  5 帧 → GLM-4V → 分析变化趋势 → 选择动作")
    print()

    tts_speak("模拟模式，VLM 多帧决策演示")

    try:
        for r in range(1, 6):
            action, reason, speak = random.choice(scenarios)
            print(f"\n  ── 第 {r} 轮决策 ──")
            print(f"  VLM 分析 5 帧完成")
            print(f"  决策: {action}  —  {reason}")
            print(f"  执行: {'说话「' + speak + '」' if speak else '无语音'}")

            if action == "go_forward":
                print("         → 前进半步")
            elif action == "bow":
                print("         → 鞠躬")
            elif action == "wave":
                print("         → 挥手")
            elif action == "turn_left":
                print("         → 左转")
            else:
                print("         → 等待 5 秒")

            if speak:
                tts_speak(speak)

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n  [模拟] 退出")

    print()
    print("  ═══════════════════════════════════════════")
    print("  模拟结束")
    robot_stand()
    print_footer(success=True)


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    robot_init()

    print_header(
        "Demo 08 — VLM 多帧视觉动作决策",
        "观察 5 秒 → VLM 决策 → 执行动作 → 循环"
    )

    import argparse
    parser = argparse.ArgumentParser(description="VLM 多帧动作决策 Demo")
    parser.add_argument("--sim", action="store_true", help="模拟模式")
    args, _ = parser.parse_known_args()

    if args.sim:
        run_simulated()
        return

    # ── 检查 API 密钥 ──
    cfg = _load_config()
    api_key = cfg.get("zhipu", {}).get("api_key", "")
    if not api_key or api_key == "your_zhipu_api_key":
        print("  ⚠ config.yaml 中未配置 zhipu.api_key")
        print("  使用 --sim 参数进入模拟模式")
        robot_stand()
        print_footer(success=False)
        return

    print(f"\n  观察窗口: {OBSERVE_SECONDS}s / {OBSERVE_FRAMES} 帧")
    print(f"  可用动作: {len(AVAILABLE_ACTIONS)} 个")
    print()
    print("  VLM 将分析连续帧的画面变化趋势")
    print("  自主选择最合适的动作执行")
    print("  按 Ctrl+C 退出")
    print("-" * 56)

    tts_speak("VLM 视觉决策已启动")

    try:
        decision_loop(api_key)
    except KeyboardInterrupt:
        print("\n  [Demo08] 用户退出")
    finally:
        robot_stand()

        from CustomFunctions.dream_memory import DreamLogger
        DreamLogger.log_session(demo_name="demo_08_vlm_decision")

        print_footer(success=True)


if __name__ == "__main__":
    main()
