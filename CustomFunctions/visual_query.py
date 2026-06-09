#!/usr/bin/env python3
# coding=utf8
"""
视觉查询模块 (Visual Query)
══════════════════════════════════════

语音提问 → 拍照 → STT结果作为 VLM 提示词 → JSON 输出 → 保存文件

相比旧的 SceneDescriber（只做固定场景描述），这个模块:
  · 不限触发词 — 你可以问任何关于场景的问题
  · VLM 输出结构化的 JSON 结果，而不是自由文本
  · 自动保存 JSON 到文件，方便下游程序读取

流程:
    1. 用讯飞 STT 监听语音（用户自由提问）
    2. 用摄像头拍照
    3. STT 结果 + 图片 发送到智谱 GLM-4V
    4. VLM 返回 JSON 格式结果（含 question/answer/details）
    5. 保存 JSON 到 vlm_results/ 目录
    6. TTS 朗读 answer 内容
"""

import os
import sys
import time
import base64
import json
from datetime import datetime

# 添加项目根目录到 Python 路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import yaml
import hiwonder.Camera as Camera
import cv2

from Demo.common import tts_speak
from CustomFunctions.STT_Control import STT_Control


_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VisualQuery:
    """
    视觉查询器

    用法:
        vq = VisualQuery()
        ok, json_path, question = vq.run_once()
        if ok:
            print(f"结果已保存: {json_path}")
    """

    def __init__(self):
        cfg = _load_config()

        # 智谱 API 配置
        zhipu_cfg = cfg.get("zhipu", {})
        self._zhipu_api_key = zhipu_cfg.get("api_key", "")

        # STT（讯飞模式，能识别自由语音，不限关键词）
        self._stt = STT_Control(mode="xunfei")

        # TTS — 统一走 common.tts_speak()，由 config.yaml 控制模式

        # 摄像头
        self._camera = None

        # JSON 结果保存目录
        self._result_dir = os.path.join(_PROJECT_ROOT, "vlm_results")
        os.makedirs(self._result_dir, exist_ok=True)

        print("[VQ] VisualQuery 就绪")
        print("[VQ] 流程: 语音提问 → 拍照 → VLM → JSON 文件")

    # ── 摄像头 ──────────────────────────────

    def _open_camera(self):
        if self._camera is None:
            self._camera = Camera.Camera()
            self._camera.camera_open()
            time.sleep(0.5)
            print("[VQ] 摄像头已打开")

    def _close_camera(self):
        if self._camera is not None:
            try:
                self._camera.camera_close()
            except Exception:
                pass
            self._camera = None
            print("[VQ] 摄像头已关闭")

    # ── 拍照 ────────────────────────────────

    def _take_photo(self, save_path=None):
        """拍照并保存 JPEG，返回路径"""
        self._open_camera()

        if save_path is None:
            photo_dir = os.path.join(self._result_dir, "photos")
            os.makedirs(photo_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(photo_dir, f"vlm_{timestamp}.jpg")

        print(f"[VQ] 正在拍照... -> {save_path}")
        # 抓取几帧让摄像头稳定
        for _ in range(5):
            ret, frame = self._camera.read()
            if ret and frame is not None:
                break
            time.sleep(0.1)

        # 取一帧并保存
        for _ in range(3):
            ret, frame = self._camera.read()
            if ret and frame is not None:
                cv2.imwrite(save_path, frame)
                file_size = os.path.getsize(save_path)
                print(f"[VQ] 拍照完成 ({file_size} bytes)")
                return save_path
            time.sleep(0.1)

        print("[VQ] 拍照失败")
        return None

    # ── VLM 调用 ────────────────────────────

    def _query_vlm(self, question, image_path):
        """
        用用户的语音文本作为提示词，调用智谱 GLM-4V
        要求返回 JSON 格式（含 question / answer / details）

        参数:
            question: STT 识别出的用户提问
            image_path: 照片路径

        返回: dict（解析后的 JSON）
        """
        if not self._zhipu_api_key or self._zhipu_api_key == "your_zhipu_api_key":
            return {
                "question": question,
                "answer": "请先配置智谱 API 密钥",
                "details": {},
                "error": "api_key_not_configured",
            }

        try:
            import requests
        except ImportError:
            print("[VQ] 请先安装 requests: pip3 install requests")
            return {
                "question": question,
                "answer": "请先安装 requests",
                "details": {},
                "error": "missing_requests",
            }

        # 读取图片并 base64 编码
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()
        data_url = f"data:image/jpeg;base64,{b64_data}"

        print(f"[VQ] 正在调用 GLM-4V...")
        print(f"[VQ] 用户提问: {question}")

        headers = {
            "Authorization": f"Bearer {self._zhipu_api_key}",
            "Content-Type": "application/json",
        }

        # ── system prompt 要求 JSON 输出 ──
        system_prompt = (
            "你是一个视觉理解助手。请观察图片，回答用户的问题。\n"
            "你必须以严格的 JSON 格式输出。只输出纯 JSON，不要包含任何 Markdown 代码块标记（如 ```json）。\n\n"
            "输出格式：\n"
            "{\n"
            '  "question": "用户的问题原文",\n'
            '  "answer": "详细的中文回答（50字以内，完整自然的一句话，直接用于语音播报）",\n'
            '  "details": {}  // 根据问题的额外结构化数据，如物体列表、颜色、位置等\n'
            "}\n\n"
            "示例：\n"
            "用户问「桌子上有什么物体」\n"
            "输出：\n"
            "{\n"
            '  "question": "桌子上有什么物体",\n'
            '  "answer": "桌面上有一个红色马克杯和一个绿色网球",\n'
            '  "details": {\n'
            '    "objects": [\n'
            '      {"name": "马克杯", "color": "红色", "position": "桌子左侧"},\n'
            '      {"name": "网球", "color": "绿色", "position": "桌子右侧"}\n'
            '    ]\n'
            '  }\n'
            "}"
        )

        payload = {
            "model": "glm-4.1v-thinking-flashx",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.3,  # 低温度让输出更稳定、可复现
        }

        resp = requests.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"[VQ] API 调用失败: {resp.status_code} {resp.text}")
            return {
                "question": question,
                "answer": f"API 调用失败: {resp.status_code}",
                "details": {},
                "error": f"http_{resp.status_code}",
            }

        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"[VQ] VLM 原始返回: {content[:200]}...")

        # ── 解析 JSON（容错处理 Markdown 包裹）──
        content_clean = content.strip()
        # 去掉 ```json ... ``` 包裹
        if "```json" in content_clean:
            content_clean = content_clean.split("```json", 1)[1]
        if "```" in content_clean:
            content_clean = content_clean.rsplit("```", 1)[0]
        content_clean = content_clean.strip()

        try:
            json_result = json.loads(content_clean)
            print(f"[VQ] JSON 解析成功")

            # 补全缺失字段
            if "question" not in json_result:
                json_result["question"] = question
            if "answer" not in json_result:
                json_result["answer"] = str(json_result)

            # 截断 answer 到 60 字以内，确保 TTS 播报不超长
            if "answer" in json_result and len(json_result["answer"]) > 60:
                json_result["answer"] = json_result["answer"][:60]

            if "details" not in json_result:
                json_result["details"] = {}

            return json_result

        except json.JSONDecodeError as e:
            print(f"[VQ] JSON 解析失败: {e}，将原始内容包装为 JSON")
            return {
                "question": question,
                "answer": content_clean,
                "details": {"raw_content": content_clean},
                "error": "json_parse_failed",
            }

    # ── 保存 JSON ──────────────────────────

    def _save_json_result(self, data, photo_path=None):
        """
        保存 JSON 结果到 vlm_results/ 目录
        文件名: {问题关键词}_{时间戳}.json

        返回: 文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 从问题中提取关键词做文件名前缀（最多 10 个字母数字）
        question = data.get("question", "query")
        prefix = "".join(c for c in question[:10] if c.isalnum() or c in "_")
        if not prefix:
            prefix = "visual_query"

        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self._result_dir, filename)

        # 附加元信息
        save_data = dict(data)
        save_data["_meta"] = {
            "timestamp": timestamp,
            "model": "glm-4.1v-thinking-flashx",
        }
        if photo_path:
            save_data["_meta"]["photo"] = photo_path

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"[VQ] JSON 已保存: {filepath}")
        return filepath

    # ── TTS（统一走 common.tts_speak，由 config.yaml 控制模式）──

    def _speak(self, text):
        """统一走 common.tts_speak() — hardware / xunfei / i2c 由 config 切换"""
        if not text:
            return
        tts_speak(text)

    # ── 主循环 ──────────────────────────────

    def run_once(self):
        """
        执行一轮完整流程:
          录音（STT）→ 拍照 → VLM 查询 → 保存 JSON → TTS 朗读回答

        返回:
            (ok: bool, json_path: str, question: str)
              ok=True 表示成功完成一轮
              json_path 是保存的 JSON 文件路径
              question 是用户的问题原文
        """
        print("\n[VQ] ═══════════ 等待语音提问 ═══════════")
        print("[VQ] 请对着麦克风说出你想问的问题（如「桌子上有什么」「这个杯子是什么颜色」）")
        print(f"[VQ] 录音时长: {self._stt._record_seconds} 秒")

        # 1. 语音识别 — 结果直接作为 VLM 提示词
        question = self._stt.transcribe()

        if not question:
            print("[VQ] 未识别到语音，跳过本轮")
            return False, "", ""

        print(f"[VQ] ✅ 识别到问题: {question}")

        # 2. 拍照
        photo_path = self._take_photo()
        if photo_path is None:
            self._close_camera()
            print("[VQ] 拍照失败")
            return False, "", question

        # 3. VLM 查询（STT 结果作为提示词）
        result = self._query_vlm(question, photo_path)

        # 4. 保存 JSON
        json_path = self._save_json_result(result, photo_path)

        # 5. 清理照片（保留 JSON 即可，照片太大占地）
        try:
            os.unlink(photo_path)
            print(f"[VQ] 临时照片已清理")
        except Exception:
            pass

        self._close_camera()

        # 6. TTS 朗读 answer（60字以内完整自然的一句话）
        tts_text = result.get("answer", "")
        if tts_text:
            print(f"\n[VQ] 📋 完整结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
            print(f"\n[VQ] 🔊 TTS 播报: {tts_text}")
            self._speak(tts_text)
        else:
            print(f"\n[VQ] VLM 返回为空")

        return True, json_path, question


# ── 独立测试 ──────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("VisualQuery 测试")
    print("=" * 50)
    print("提示: 对着麦克风说出你想问的问题")
    print("  · 这个桌子上有什么？")
    print("  · 杯子是什么颜色的？")
    print("  · 我面前有几个人？")

    vq = VisualQuery()

    try:
        round_num = 0
        while True:
            round_num += 1
            print(f"\n{'=' * 50}")
            print(f"第 {round_num} 轮")
            print(f"{'=' * 50}")

            ok, json_path, question = vq.run_once()
            if ok:
                print(f"\n✅ 第 {round_num} 轮完成")
                print(f"   问题: {question}")
                print(f"   结果: {json_path}")
            else:
                print(f"\n⏭️  第 {round_num} 轮跳过")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[VQ] 用户退出")
    finally:
        try:
            vq._close_camera()
        except Exception:
            pass

    print("\n[VQ] 退出")
