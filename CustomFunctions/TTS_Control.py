#!/usr/bin/env python3
# coding=utf8
"""
文字转语音模块 (TTS_Control)
双模式: hardware（I2C TTS 芯片驱动板载扬声器）/ xunfei（云端合成，需外接 USB 音箱）

硬件状况（经诊断确认）:
    - 扬声器接在 I2C 地址 0x40 的 TTS 合成芯片上
    - Pi 的 3.5mm 音频口空接，aplay 无法驱动扬声器
    - 默认走 hardware 模式，云端模式仅在接了 USB/蓝牙音箱时使用

云端流程（供外接音箱时使用）:
    文字 → 讯飞 WebSocket API → 16kHz/16bit PCM 音频
    → 加 WAV 头 → 保存到 CustomFunctions/tts_audio/
    → aplay 播放

发音人参考:
    x4_yezi   叶子（女，推荐）
    xiaoyan   小燕（女，标准）
    aisjping  艾小萍（女，温柔）
    aisjiuxu  艾久旭（男）
    aisbabyxu 徐小童（童声）

依赖:
    pip3 install websocket-client

配置 (config.yaml):
    tts_xunfei:     ← 与 STT 的 xunfei 段是不同账号
        appid: "…"
        api_secret: "…"
        api_key: "…"
    tts:
        default_mode: "hardware" | "xunfei"
        default_voice: "x4_yezi"
        default_speed: 50
        default_volume: 50
        save_audio: false
"""

import os
import sys
import json
import time
import base64
import hashlib
import hmac
import struct
import threading
import subprocess
from queue import Queue, Empty
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode

import websocket
import ssl

# ── 项目路径 ──────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import yaml

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config():
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── WebSocket 帧状态 ──────────────────────────
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


class TTS_Control:
    """文字转语音控制类"""

    # ══════════════════════════════════════════
    #  初始化
    # ══════════════════════════════════════════

    def __init__(self, mode=None):
        """
        参数:
            mode: "hardware"（I2C 硬件 TTS，驱动板载扬声器—默认）
                  "xunfei"（讯飞云端合成，需外接 USB/蓝牙音箱）
                  默认从 config.yaml 的 tts.default_mode 读取
        """
        cfg = _load_config()

        # ── TTS 云端 API 凭证（tts_xunfei 段，与 STT 的 xunfei 段区分）──
        tts_xf = cfg.get("tts_xunfei", {})
        self._appid = tts_xf.get("appid", "")
        self._api_secret = tts_xf.get("api_secret", "")
        self._api_key = tts_xf.get("api_key", "")

        # ── TTS 播报参数 ──
        tts_cfg = cfg.get("tts", {})
        self._mode = mode or tts_cfg.get("default_mode", "hardware")
        self._default_voice = tts_cfg.get("default_voice", "x4_yezi")
        self._default_speed = tts_cfg.get("default_speed", 50)
        self._default_volume = tts_cfg.get("default_volume", 50)
        self._save_audio = tts_cfg.get("save_audio", False)

        # ── 硬件 TTS 降级参数 ──
        self._hw_max_chars = tts_cfg.get("max_chars", 20)
        self._hw_char_speed = tts_cfg.get("char_speed", 0.3)

        # ── 输出目录 ──
        self._audio_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tts_audio"
        )
        os.makedirs(self._audio_dir, exist_ok=True)

        # ── 硬件 TTS 引擎（惰性加载，只用一次）──
        self._hw_tts = None
        self._hw_available = False

        print(f"[TTS] 模式: {self._mode}")
        if self._mode == "xunfei":
            if not self._appid or self._appid == "你的TTS_APPID":
                print("[TTS] ⚠ 讯飞 TTS API 未配置，将使用硬件 TTS 降级")
                self._mode = "hardware"
            else:
                print(f"[TTS] 发音人: {self._default_voice}, "
                      f"语速: {self._default_speed}, 音量: {self._default_volume}")
        print(f"[TTS] 音频目录: {self._audio_dir}")

    # ══════════════════════════════════════════
    #  公开接口
    # ══════════════════════════════════════════

    def speak(self, text, voice=None, speed=None, volume=None, block=True):
        """
        文字 → 语音合成 → 保存 WAV → 播放

        参数:
            text:   要朗读的文字
            voice:  发音人（仅云端模式），默认取 config
            speed:  语速 0-100（仅云端模式），默认取 config
            volume: 音量 0-100（仅云端模式），默认取 config
            block:  是否等待播完（默认 True）
        返回:
            str: 音频文件路径（成功）| "hardware"（硬件降级）| None（失败）
        """
        if not text or not text.strip():
            print("[TTS] 没有文字需要朗读")
            return None

        print(f"[TTS] ▶ 准备朗读 ({len(text)} 字): "
              f"\"{text[:60]}{'...' if len(text) > 60 else ''}\"")

        if self._mode == "xunfei":
            result = self._speak_xunfei(
                text, voice or self._default_voice,
                speed if speed is not None else self._default_speed,
                volume if volume is not None else self._default_volume,
                block=block,
            )
            if result is not None:
                return result
            # 云端失败 → 自动降级硬件
            print("[TTS] 云端合成失败，自动降级到硬件 TTS...")

        return self._speak_hardware(text)

    # ══════════════════════════════════════════
    #  云端 TTS（讯飞 WebSocket API）
    # ══════════════════════════════════════════

    @staticmethod
    def _build_ws_url(appid, api_key, api_secret):
        """生成讯飞 TTS WebSocket 鉴权 URL（与 demo 完全一致）"""
        url = "wss://tts-api.xfyun.cn/v2/tts"
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: ws-api.xfyun.cn\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET /v2/tts HTTP/1.1"

        signature_sha = hmac.new(
            api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode("utf-8")

        authorization_origin = (
            'api_key="%s", algorithm="%s", headers="%s", signature="%s"'
            % (api_key, "hmac-sha256", "host date request-line", signature_sha)
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

        params = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        return url + "?" + urlencode(params)

    def _speak_xunfei(self, text, voice, speed, volume, block=True):
        """
        调用讯飞 WebSocket API 合成语音 → 保存 WAV → 播放
        返回: WAV 路径（成功）| None（失败）
        """
        audio_chunks = []
        result_queue = Queue()
        error_msg = [None]

        # ── WebSocket 回调 ──────────────────────

        def on_message(ws, message):
            try:
                resp = json.loads(message)
                code = resp.get("code", -1)

                if code != 0:
                    err_msg = resp.get("message", "未知错误")
                    sid = resp.get("sid", "")
                    print(f"[TTS] API 错误 (code={code}, sid={sid}): {err_msg}")
                    error_msg[0] = f"code={code}: {err_msg}"
                    ws.close()
                    return

                data = resp.get("data", {})
                audio_b64 = data.get("audio", "")
                status = data.get("status", -1)

                if audio_b64:
                    chunk = base64.b64decode(audio_b64)
                    audio_chunks.append(chunk)

                if status == STATUS_LAST_FRAME:
                    total = sum(len(c) for c in audio_chunks)
                    print(f"[TTS] 音频接收完成（共 {total} 字节）")
                    result_queue.put(True)
                    ws.close()

            except Exception as e:
                print(f"[TTS] 解析响应异常: {e}")
                error_msg[0] = str(e)

        def on_error(ws, error):
            print(f"[TTS] WebSocket 错误: {error}")
            error_msg[0] = str(error)
            try:
                result_queue.put_nowait(False)
            except:
                pass

        def on_close(ws, *args):
            try:
                result_queue.put_nowait(False)
            except:
                pass

        def on_open(ws):
            def run():
                payload = {
                    "common": {"app_id": self._appid},
                    "business": {
                        "aue": "raw",                     # 原始 PCM（无头）
                        "auf": "audio/L16;rate=16000",    # 16kHz 16-bit PCM
                        "vcn": voice,                     # 发音人
                        "speed": speed,                   # 语速
                        "volume": volume,                 # 音量
                        "tte": "utf8",                    # 文本编码
                    },
                    "data": {
                        "status": 2,  # 一次性发送全部文本
                        "text": str(base64.b64encode(text.encode("utf-8")), "UTF8"),
                    },
                }
                ws.send(json.dumps(payload))
                print(f"[TTS] 文本已发送，等待音频响应...")

            threading.Thread(target=run, daemon=True).start()

        # ── 发起连接 ──────────────────────────

        ws_url = self._build_ws_url(self._appid, self._api_key, self._api_secret)
        websocket.enableTrace(False)

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.on_open = on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        # ── 等待结果 ──────────────────────────

        try:
            success = result_queue.get(timeout=30)
        except Empty:
            print("[TTS] WebSocket 超时（30 秒）")
            success = False

        if not success or error_msg[0]:
            return None

        if not audio_chunks:
            print("[TTS] 未收到音频数据")
            return None

        # ── 保存 WAV ──────────────────────────

        pcm_data = b"".join(audio_chunks)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(self._audio_dir, f"tts_{timestamp}.wav")
        self._write_wav(pcm_data, wav_path)

        # ── 播放 ──────────────────────────────

        if block:
            self._play_audio(wav_path)

        # 不保留则播完删除
        if not self._save_audio:
            os.unlink(wav_path)
            print(f"[TTS] 临时音频已删除")
            return wav_path  # 仍返回路径（已删除，但调用方可知成功）

        return wav_path

    # ══════════════════════════════════════════
    #  WAV 文件处理（16kHz / 16-bit / 单声道）
    # ══════════════════════════════════════════

    @staticmethod
    def _write_wav(pcm_data, output_path):
        """给原始 PCM 数据添加 44 字节 WAV 头部并写入文件"""
        sample_rate = 16000
        bits = 16
        channels = 1
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        data_size = len(pcm_data)
        header_size = 44

        with open(output_path, "wb") as f:
            # RIFF
            f.write(b"RIFF")
            f.write(struct.pack("<I", data_size + header_size - 8))
            f.write(b"WAVE")
            # fmt
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))          # chunk size
            f.write(struct.pack("<H", 1))           # PCM = 1
            f.write(struct.pack("<H", channels))    # 单声道
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", byte_rate))
            f.write(struct.pack("<H", block_align))
            f.write(struct.pack("<H", bits))
            # data
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(pcm_data)

    # ══════════════════════════════════════════
    #  音频播放
    # ══════════════════════════════════════════

    @staticmethod
    def _play_audio(file_path):
        """
        调用 aplay 播放 WAV，依次尝试多个 ALSA 设备名
        TonyPi 的扬声器通常接在 I2C TTS 芯片上，不一定被 ALSA 识别。
        如所有设备都失败，请跑 diagnose_audio() 查看可用设备。
        """
        if not os.path.exists(file_path):
            print(f"[TTS] 音频文件不存在: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        duration = file_size / (16000 * 2)
        print(f"[TTS] 🔊 播放中（{file_size} bytes，约 {duration:.1f} 秒）...")

        # 依次尝试多个 ALSA 设备
        devices = [
            None,               # aplay xxx.wav        — 默认设备
            "default",          # aplay -D default
            "sysdefault",       # aplay -D sysdefault
            "plughw:0,0",       # aplay -D plughw:0,0  — 常用
            "plughw:1,0",       # aplay -D plughw:1,0
            "plughw:2,0",       # aplay -D plughw:2,0  — USB 麦克风有时占 card 2
            "hw:0,0",           # aplay -D hw:0,0      — 直通硬件
            "hw:1,0",
            "hw:2,0",
        ]

        for dev in devices:
            cmd = ["aplay"]
            if dev is not None:
                cmd += ["-D", dev]
            cmd.append(file_path)
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=duration + 5,
                )
                if r.returncode == 0:
                    print(f"[TTS] ✓ aplay -D {dev or '（默认）'} 播放成功")
                    return True
                else:
                    # 静默失败，继续试下一个设备
                    pass
            except FileNotFoundError:
                print("[TTS] aplay 未安装")
                break
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

        # aplay 全部失败 → 打印 stderr 帮助诊断
        print("[TTS] ✗ aplay 在所有设备上均失败")
        for dev in devices[:3]:
            cmd = ["aplay", "-D", dev, file_path] if dev else ["aplay", file_path]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if r.returncode != 0:
                    stderr = r.stderr.strip()[:200]
                    print(f"     aplay -D {dev or '默认'}: {stderr}")
            except:
                pass

        # 备选: paplay（PulseAudio）
        try:
            subprocess.run(
                ["paplay", file_path],
                capture_output=True,
                timeout=duration + 5,
            )
            print("[TTS] ✓ paplay 播放成功")
            return True
        except:
            pass

        # 备选: ffplay（ffmpeg）
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path],
                capture_output=True,
                timeout=duration + 5,
            )
            print("[TTS] ✓ ffplay 播放成功")
            return True
        except:
            pass

        print("[TTS] ❌ 所有播放器均失败")
        print("[TTS] 建议: 运行 tts.diagnose_audio() 查看可用音频设备")
        return False

    # ══════════════════════════════════════════
    #  硬件 TTS 降级 (I2C)
    # ══════════════════════════════════════════

    def _init_hardware(self):
        """惰性初始化硬件 TTS"""
        if self._hw_tts is not None:
            return self._hw_available
        try:
            import hiwonder.TTS as TTS
            self._hw_tts = TTS.TTS()
            self._hw_available = True
            print("[TTS] 硬件 TTS 已就绪 → 驱动板载扬声器")
        except ImportError:
            print("[TTS] hiwonder.TTS 不可用（非树莓派环境？）")
            self._hw_available = False
        return self._hw_available

    def _speak_hardware(self, text):
        """
        用硬件 I2C TTS 芯片逐段朗读
        每段不超过 max_chars 字，依标点切分
        """
        if not self._init_hardware():
            print("[TTS] 硬件 TTS 不可用，无法朗读")
            return None

        print(f"[TTS] 硬件 TTS 朗读（{len(text)} 字）...")

        # 按标点 + max_chars 拆分
        separators = ["。", "！", "？", "；", "，", "\n"]
        raw = [text]
        for sep in separators:
            parts = []
            for s in raw:
                parts.extend(s.split(sep))
            raw = [p.strip() for p in parts if p.strip()]

        chunks = []
        for s in raw:
            while len(s) > self._hw_max_chars:
                chunks.append(s[: self._hw_max_chars])
                s = s[self._hw_max_chars :]
            if s:
                chunks.append(s)

        print(f"[TTS] 分 {len(chunks)} 段朗读...")
        for i, chunk in enumerate(chunks, 1):
            try:
                print(f"[TTS]   [{i}/{len(chunks)}] {chunk}")
                self._hw_tts.TTSModuleSpeak("[h0][v10][m3]", chunk)
                time.sleep(len(chunk) * self._hw_char_speed + 0.3)
            except Exception as e:
                print(f"[TTS] 硬件 TTS 段 {i} 失败: {e}")
                time.sleep(0.3)

        return "hardware"

    # ══════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════

    # ══════════════════════════════════════════
    #  音频设备诊断
    # ══════════════════════════════════════════

    @staticmethod
    def diagnose_audio():
        """
        诊断机器人上的可用音频输出设备
        在机器人上跑: python3 -c "from CustomFunctions.TTS_Control import TTS_Control; TTS_Control.diagnose_audio()"
        """
        print("=" * 50)
        print("音频设备诊断")
        print("=" * 50)

        # 1. aplay -l
        print("\n--- aplay -l（ALSA 设备列表）---")
        try:
            r = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
            out = (r.stdout + r.stderr).strip()
            print(out if out else "(无输出)")
            print(f"  返回码: {r.returncode}")
        except FileNotFoundError:
            print("  aplay 未安装")
        except Exception as e:
            print(f"  错误: {e}")

        # 2. aplay -L（设备名称）
        print("\n--- aplay -L（PCM 设备名）---")
        try:
            r = subprocess.run(["aplay", "-L"], capture_output=True, text=True, timeout=5)
            out = (r.stdout + r.stderr).strip()
            # 只显示非注释行
            for line in out.split("\n"):
                if line and not line.startswith(" "):
                    print(f"  {line}")
        except:
            pass

        # 3. 检查 /proc/asound/cards
        print("\n--- /proc/asound/cards ---")
        try:
            r = subprocess.run(["cat", "/proc/asound/cards"], capture_output=True, text=True, timeout=3)
            print(r.stdout.strip() or "(无)")
        except:
            print("  无法读取")

        # 4. 检查扬声器物理路径（GPIO / I2C）
        print("\n--- I2C TTS 芯片检测 ---")
        try:
            r = subprocess.run(
                ["i2cdetect", "-y", "1"],
                capture_output=True, text=True, timeout=5,
            )
            if "40" in r.stdout:
                print("  在 I2C 地址 0x40 检测到 TTS 芯片（合成语音芯片）")
                print("  扬声器接在该芯片上 → aplay 无法直接驱动")
                print("  如需使用云端合成语音，请考虑:")
                print("    a. 将扬声器改接到 Pi 3.5mm 音频口")
                print("    b. 使用 USB 音箱")
                print("    c. 使用蓝牙音箱")
            else:
                print("  I2C 0x40 未检测到 TTS 芯片")
        except:
            print("  i2cdetect 不可用")

        print("\n" + "=" * 50)
        print("诊断完成")
        print("=" * 50)

    @staticmethod
    def list_voices():
        """打印可用的发音人（仅参考，实际以讯飞平台为准）"""
        voices = {
            "x4_yezi": "叶子（女，推荐）",
            "xiaoyan": "小燕（女，标准）",
            "aisjping": "艾小萍（女，温柔）",
            "aisjiuxu": "艾久旭（男）",
            "aisbabyxu": "徐小童（童声）",
            "aisxping": "艾小评（女）",
            "aisxying": "艾小樱（女）",
        }
        print("\n可用发音人（讯飞 TTS）:")
        for key, desc in voices.items():
            print(f"  {key:20s}  {desc}")
        return list(voices.keys())

    def cleanup(self, max_age_hours=24):
        """清理过期的临时音频文件"""
        now = time.time()
        count = 0
        for fname in os.listdir(self._audio_dir):
            fpath = os.path.join(self._audio_dir, fname)
            if os.path.isfile(fpath) and fname.endswith(".wav"):
                age = now - os.path.getmtime(fpath)
                if age > max_age_hours * 3600:
                    os.unlink(fpath)
                    count += 1
        if count:
            print(f"[TTS] 已清理 {count} 个旧音频文件")


# ═══════════════════════════════════════════════
#  独立测试
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("TTS_Control 测试")
    print("=" * 55)

    tts = TTS_Control()

    # 列出可用发音人
    tts.list_voices()

    # 测试朗读
    tests = [
        "你好，我是TonyPi机器人，很高兴见到你！",
        "好的，我这就去找红色杯子。",
        "检测完毕，没有找到目标物体，请换个位置再试。",
    ]

    for i, text in enumerate(tests, 1):
        print(f"\n{'─' * 55}")
        print(f"测试 {i}: \"{text}\"")
        print(f"{'─' * 55}")
        result = tts.speak(text, block=True)
        status = "✓" if result else "✗"
        print(f"结果: {status} → {result}")
