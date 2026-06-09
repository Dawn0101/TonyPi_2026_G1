#!/usr/bin/env python3
# coding=utf8
"""
语音转文字模块 (STT_Control)
支持两种模式：
  - hardware: 使用 TonyPi 自带的离线 ASR 硬件芯片（关键词→ID→映射为文字）
  - xunfei:   从 WAV 文件读取音频，调用科大讯飞 API 转文字（参照 TonyPi_2/ASR_xf.py）

用法:
    stt = STT_Control(mode="hardware")
    text = stt.listen()  # 返回字符串
"""

import os
import sys
import time
import json
import base64
import hashlib
import hmac
import struct
import subprocess
import threading
import tempfile
import ssl
import wave
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode
from queue import Queue

import yaml
import websocket

# ── 添加项目根目录到 Python 路径 ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# 硬件 ASR 依赖
HAS_HARDWARE_ASR = False
try:
    import hiwonder.ASR as ASR
    import hiwonder.ActionGroupControl as AGC
    import hiwonder.Board as Board
    from ActionGroupDict import action_group_dict
    HAS_HARDWARE_ASR = True
except ImportError as e:
    print(f"[STT] 硬件 ASR 模块导入失败: {e}")
    ASR = None


# ──────────────────────────────────────────────
# 科大讯飞 WebSocket API 常量
# ──────────────────────────────────────────────
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


# ── 配置文件 ──────────────────────────────────
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config():
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class STT_Control:
    """语音转文字控制类"""

    MODE_HARDWARE = "hardware"
    MODE_XUNFEI = "xunfei"

    def __init__(self, mode=None):
        cfg = _load_config()
        if mode is None:
            mode = cfg.get("stt", {}).get("default_mode", "hardware")

        self.mode = mode
        self._record_seconds = cfg.get("stt", {}).get("record_seconds", 5)
        self._alsa_device = cfg.get("stt", {}).get("alsa_device", "plughw:2,0")

        if mode == self.MODE_HARDWARE:
            self._init_hardware()
        if mode == self.MODE_XUNFEI:
            xunfei_cfg = cfg.get("xunfei", {})
            if not xunfei_cfg.get("appid") or xunfei_cfg["appid"] == "你的APPID":
                print("[STT] 警告: 科大讯飞 API 未配置")
            self._xunfei_cfg = xunfei_cfg
            # 讯飞模式的中文关键词 → 动作名映射
            self._xunfei_keywords = {
                "往前走": "go_forward",
                "前进": "go_forward",
                "直走": "go_forward",
                "往后退": "back_fast",
                "后退": "back_fast",
                "向左移": "left_move_fast",
                "左移": "left_move_fast",
                "向右移": "right_move_fast",
                "右移": "right_move_fast",
                "鞠躬": "bow",
                "挥手": "wave",
                "挥挥手": "wave",
                "俯卧撑": "push_ups",
                "左右转": "twist",
                "扭腰": "twist",
                "跳舞": "dance",
                "跳一支舞": "dance",
                "跳支舞": "dance",
                "跳个舞": "dance",
            }

    # ══════════════════════════════════════════
    # 硬件 ASR 模式（参照 ASRControl.py）
    # ══════════════════════════════════════════

    def _init_hardware(self):
        if not HAS_HARDWARE_ASR:
            print("[STT] hiwonder.ASR 模块不可用，硬件模式无法使用")
            self._asr = None
            return

        try:
            self._asr = ASR.ASR()
            data = self._asr.getResult()
            if data is None:
                raise IOError("ASR 芯片 I2C 通信失败")

            self._asr.eraseWords()
            self._asr.setMode(1)

            self._asr.addWords(2, 'wang qian zou')
            self._asr.addWords(2, 'qian jin')
            self._asr.addWords(2, 'zhi zou')
            self._asr.addWords(3, 'wang hou tui')
            self._asr.addWords(4, 'xiang zuo yi')
            self._asr.addWords(5, 'xiang you yi')
            self._asr.addWords(6, 'ju gong')
            self._asr.addWords(7, 'hui shou')
            self._asr.addWords(8, 'fu wo cheng')
            self._asr.addWords(9, 'zuo you zhuan')
            self._asr.addWords(10, 'tiao wu')

            self._id_to_text = {
                2: "往前走", 3: "往后退", 4: "向左移", 5: "向右移",
                6: "鞠躬", 7: "挥手", 8: "俯卧撑", 9: "左右转", 10: "跳舞",
            }
            self._id_to_action = {
                2: "go_forward", 3: "back_fast", 4: "left_move_fast",
                5: "right_move_fast", 6: "bow", 7: "wave",
                8: "push_ups", 9: "twist", 10: "dance",
            }

            print("[STT] 硬件 ASR 初始化完成（循环识别模式）")
        except Exception as e:
            print(f"[STT] 硬件 ASR 初始化失败: {e}")
            self._asr = None

    def _hardware_listen(self, timeout=10):
        if self._asr is None:
            return None, None
        print(f"[STT] 正在监听语音指令...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            data = self._asr.getResult()
            if data and data in self._id_to_text:
                text = self._id_to_text[data]
                action = self._id_to_action.get(data, "")
                print(f"[STT] 识别到: {text}")
                return text, action
            time.sleep(0.05)
        return None, None

    # ══════════════════════════════════════════
    # 科大讯飞 API 调用（完全参照 TonyPi_2/ASR_xf.py）
    # ══════════════════════════════════════════

    def _create_xunfei_url(self):
        """生成讯飞 WebSocket 鉴权 URL（跟 ASR_xf.py 完全一致）"""
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: ws-api.xfyun.cn\ndate: " + date + "\nGET /v2/iat HTTP/1.1"
        signature_sha = hmac.new(
            self._xunfei_cfg["api_secret"].encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
            self._xunfei_cfg["api_key"], "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        return url + '?' + urlencode(v)

    def xunfei_transcribe(self, wav_path):
        """
        读取 WAV 文件，调用讯飞 API 转文字
        跟 ASR_xf.py 的 Speech2text() 完全一致

        参数:
            wav_path: WAV 音频文件路径（16kHz, 16-bit, 单声道）
        返回:
            识别出的文字（字符串）
        """
        if not os.path.exists(wav_path):
            print(f"[STT] 文件不存在: {wav_path}")
            return ""

        # 检查文件大小
        file_size = os.path.getsize(wav_path)
        print(f"[STT] 音频文件: {file_size} bytes, 16kHz/16bit/单声道 PCM")

        ws_url = self._create_xunfei_url()
        result_queue = Queue()

        # ── on_message ──
        def on_message(ws, message):
            try:
                resp = json.loads(message)
                code = resp["code"]
                if code != 0:
                    print(f"[STT] 讯飞错误: {resp.get('message','')} (code={code})")
                    return
                data = resp.get("data", {})
                result_data = data.get("result", {})
                ws_data = result_data.get("ws", [])
                result = ""
                for i in ws_data:
                    for w in i.get("cw", []):
                        result += w.get("w", "")
                # 调试：打印返回状态
                status = data.get("status", -1)
                sn = result_data.get("sn", -1)
                if result:
                    print(f"[STT] 讯飞返回 sn={sn} status={status}: '{result}'")
                # 有结果就放进去（跟 ASR_xf.py 一样，取第一个非空结果）
                if result.strip():
                    result_queue.put(result)
            except Exception as e:
                print(f"[STT] 解析响应异常: {e}")
                import traceback
                traceback.print_exc()

        def on_error(ws, error):
            print(f"[STT] WebSocket 错误: {error}")

        def on_close(ws, a, b):
            pass

        # ── on_open ──
        def on_open(ws):
            def run():
                frameSize = 16000
                intervel = 0.04
                status = STATUS_FIRST_FRAME

                with open(wav_path, "rb") as fp:
                    while True:
                        buf = fp.read(frameSize)
                        if not buf:
                            status = STATUS_LAST_FRAME

                        if status == STATUS_FIRST_FRAME:
                            d = {
                                "common": {"app_id": self._xunfei_cfg["appid"]},
                                "business": {
                                    "domain": "iat",
                                    "language": "zh_cn",
                                    "accent": "mandarin",
                                    "vinfo": 1,
                                    "vad_eos": 10000,
                                },
                                "data": {
                                    "status": 0,
                                    "format": "audio/L16;rate=16000",
                                    "audio": str(base64.b64encode(buf), 'utf-8'),
                                    "encoding": "raw"
                                }
                            }
                            ws.send(json.dumps(d))
                            status = STATUS_CONTINUE_FRAME

                        elif status == STATUS_CONTINUE_FRAME:
                            d = {
                                "data": {
                                    "status": 1,
                                    "format": "audio/L16;rate=16000",
                                    "audio": str(base64.b64encode(buf), 'utf-8'),
                                    "encoding": "raw"
                                }
                            }
                            ws.send(json.dumps(d))

                        elif status == STATUS_LAST_FRAME:
                            d = {
                                "data": {
                                    "status": 2,
                                    "format": "audio/L16;rate=16000",
                                    "audio": str(base64.b64encode(buf), 'utf-8'),
                                    "encoding": "raw"
                                }
                            }
                            ws.send(json.dumps(d))
                            time.sleep(1)
                            break

                        time.sleep(intervel)

                ws.close()

            thread = threading.Thread(target=run)
            thread.start()

        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.on_open = on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        try:
            result = result_queue.get(timeout=30)
        except:
            result = ""

        result = result.strip()
        if result:
            print(f"[STT] 识别结果: {result}")
        else:
            print("[STT] 未识别到语音")
        return result

    # ══════════════════════════════════════════
    # 录音模块（参照 TonyPi_2/Record.py）
    # ══════════════════════════════════════════

    def record_to_wav(self, save_path=None, duration=5):
        """
        用 arecord 录音到原始 PCM 文件（无 WAV 头部，给讯飞直接用）
        arecord -l 显示 USB 音频设备在 card 2, device 0
        参数:
            save_path: 保存路径
            duration: 录音时长（秒）
        返回: PCM 文件路径，失败返回 None
        """
        if save_path is None:
            save_path = tempfile.mktemp(suffix=".pcm")

        print("[STT] 录音中（{}秒，请对着麦克风说话）...".format(duration))

        cmd = [
            "arecord",
            "-D", self._alsa_device,    # USB 麦克风（从 config.yaml 读取）
            "-d", str(duration),
            "-f", "S16_LE",        # 16-bit 有符号小端
            "-r", "16000",         # 16kHz 采样率
            "-c", "1",             # 单声道
            "-t", "raw",           # 原始 PCM，无 WAV 头部
            save_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("[STT] arecord 错误:", result.stderr.strip())
            try:
                os.unlink(save_path)
            except:
                pass
            return None

        file_size = os.path.getsize(save_path)
        if file_size < 100:
            print("[STT] 录音文件过小 ({} bytes)，可能没收到声音".format(file_size))
            try:
                os.unlink(save_path)
            except:
                pass
            return None

        duration_actual = file_size / (16000 * 2)
        print("[STT] 录音完成 ({} bytes, {:.1f}秒)".format(file_size, duration_actual))
        return save_path

    def xunfei_listen(self, save_path=None, duration=None):
        """
        完整流程：录音 → 调讯飞 API → 返回文字

        参数:
            save_path: 录音保存路径，None=保存到 Audio_file
            duration: 录音秒数，None=从 config 读取
        """
        print("[STT] 开始录音...")
        if duration is None:
            duration = self._record_seconds
        if save_path is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audio_file")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "xunfei_record.pcm")

        wav_path = self.record_to_wav(save_path, duration=duration)
        if wav_path is None:
            return ""

        # 检查录音音量，确认是否有声音
        try:
            with open(wav_path, 'rb') as f:
                raw = f.read()
            import struct
            samples = struct.unpack_from('<' + 'h' * (len(raw) // 2), raw)
            max_vol = max(abs(s) for s in samples) if samples else 0
            avg_vol = sum(abs(s) for s in samples) // len(samples) if samples else 0
            print(f"[STT] 音量检测: 最大={max_vol}, 平均={avg_vol}")
            if max_vol < 500:
                print("[STT] 警告: 音量过低，麦克风可能没收到声音")
        except Exception as e:
            print(f"[STT] 音量检测失败: {e}")

        print(f"[STT] 录音已保存: {wav_path}")
        text = self.xunfei_transcribe(wav_path)

        # 匹配并执行动作
        if text:
            actions = self._match_xunfei_action(text)
            if actions:
                print(f"[STT] 匹配到动作: {', '.join(actions)}")
                for action in actions:
                    self.execute_action(action)
            else:
                print("[STT] 未匹配到可执行的动作")

        # 保留录音文件，不清理
        return text

    # ══════════════════════════════════════════
    # 统一接口
    # ══════════════════════════════════════════

    def listen(self, timeout=10):
        """
        录音 → 识别 → 返回文字
        注意: xunfei 模式下会同时用内置 dict 匹配关键词并执行动作
              如果不需要 dict 匹配（如给 LLM 用），请用 transcribe()
        """
        if self.mode == self.MODE_HARDWARE:
            text, action = self._hardware_listen(timeout)
            return text
        elif self.mode == self.MODE_XUNFEI:
            return self.xunfei_listen()
        return None

    def transcribe(self, duration=None):
        """
        纯转录：录音 → 讯飞 API → 返回文字
        不匹配关键词、不执行动作，适合 demo_07 等 LLM 驱动场景
        """
        if self.mode != self.MODE_XUNFEI:
            return self.listen()
        print("[STT] 开始录音...")
        if duration is None:
            duration = self._record_seconds
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audio_file")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "xunfei_record.pcm")
        wav_path = self.record_to_wav(save_path, duration=duration)
        if wav_path is None:
            return ""
        text = self.xunfei_transcribe(wav_path)
        return text

    def listen_with_action(self, timeout=10):
        if self.mode == self.MODE_HARDWARE:
            return self._hardware_listen(timeout)
        return None, None

    def _match_xunfei_action(self, text):
        """
        从讯飞识别的文字中匹配关键词，返回要执行的动作列表
        例如: "你好请你往前走" → ["go_forward"]
        """
        matched = []
        for keyword, action in self._xunfei_keywords.items():
            if keyword in text:
                if action not in matched:
                    matched.append(action)
                    print(f"[STT] 关键词 '{keyword}' → 动作 '{action}'")
        return matched

    def execute_action(self, action_name):
        try:
            print(f"[STT] 执行动作: {action_name}")
            AGC.runActionGroup(action_name, 2, True)
        except Exception as e:
            print(f"[STT] 执行动作失败: {e}")

    def listen_and_execute(self, timeout=10):
        text, action = self._hardware_listen(timeout)
        if action:
            self.execute_action(action)
        return text


# ── 独立测试 ──────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("STT_Control 测试")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "xunfei":
        # python3 STT_Control.py xunfei        → 录音 + 讯飞 API
        # python3 STT_Control.py xunfei <文件>  → 读取 WAV 文件 + 讯飞 API
        stt = STT_Control(mode="xunfei")

        if len(sys.argv) > 2:
            # 从文件读取
            wav_file = sys.argv[2]
            print(f"\n--- 科大讯飞 API 模式（文件模式）---")
            print(f"读取文件: {wav_file}")
            text = stt.xunfei_transcribe(wav_file)
        else:
            # 录音 + 讯飞
            print(f"\n--- 科大讯飞 API 模式（录音模式）---")
            text = stt.xunfei_listen()

        print(f"结果: {text}")

    else:
        # 默认硬件模式
        print("\n--- 硬件 ASR 模式 ---")
        print("提示: python3 STT_Control.py xunfei <wav文件> 使用讯飞 API")
        stt = STT_Control(mode="hardware")

        if stt._asr is not None:
            for i in range(3):
                print(f"\n--- 第 {i+1} 次 ---")
                text = stt.listen_and_execute(timeout=30)
                if text:
                    print(f"识别文字: {text}")
                else:
                    print("未识别到指令")
        else:
            print("\n硬件 ASR 不可用")
