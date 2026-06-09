#!/usr/bin/env python3
# coding=utf8
"""
Demo 公共模块
提供机器人初始化、摄像头管理、TTS 播报、安全退出等公共函数
所有 Demo 脚本零侵入 — 只 import 现有模块
"""

import os
import sys
import time
import signal

# 确保项目根在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 硬件模块（可选导入，非树莓派环境降级） ──

HAS_AGC = False
HAS_BOARD = False
HAS_TTS = False
HAS_CAMERA = False
HAS_MPU = False

try:
    import hiwonder.ActionGroupControl as AGC
    HAS_AGC = True
except ImportError:
    print("[common] hiwonder.ActionGroupControl 不可用")
    AGC = None

try:
    import hiwonder.Board as Board
    HAS_BOARD = True
except ImportError:
    print("[common] hiwonder.Board 不可用")

try:
    import hiwonder.TTS as _TTS
    _tts = _TTS.TTS()
    HAS_TTS = True
except Exception:
    _tts = None
    HAS_TTS = False

try:
    import hiwonder.Camera as Camera
    HAS_CAMERA = True
except ImportError:
    Camera = None
    HAS_CAMERA = False

try:
    import hiwonder.Mpu6050 as Mpu6050
    HAS_MPU = True
except ImportError:
    Mpu6050 = None
    HAS_MPU = False


# ═══════════════════════════════════════════════
#  机器人初始化
# ═══════════════════════════════════════════════

def robot_init():
    """机器人初始化为立正姿势"""
    print("\n[Init] 机器人初始化...")
    if HAS_AGC:
        try:
            AGC.runActionGroup("stand")
        except Exception as e:
            print(f"[Init] stand 动作组异常: {e}")
    if HAS_BOARD:
        try:
            Board.setPWMServoPulse(1, 1500, 500)
            Board.setPWMServoPulse(2, 1500, 500)
        except Exception as e:
            print(f"[Init] 舵机复位异常: {e}")
    print("[Init] 初始化完成")


def robot_stand():
    """机器人归位立正"""
    if HAS_AGC:
        try:
            AGC.runActionGroup("stand")
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  摄像头管理
# ═══════════════════════════════════════════════

class CameraManager:
    """简单的摄像头上下文管理器"""

    def __init__(self):
        self._cam = None
        self._opened = False

    def open(self):
        if self._opened:
            return True
        if not HAS_CAMERA:
            print("[Camera] hiwonder.Camera 不可用")
            return False
        try:
            self._cam = Camera.Camera()
            self._cam.camera_open()
            self._opened = True
            time.sleep(0.5)
            print("[Camera] 摄像头已打开")
            return True
        except Exception as e:
            print(f"[Camera] 打开失败: {e}")
            return False

    def close(self):
        if self._opened and self._cam:
            try:
                self._cam.camera_close()
            except Exception:
                pass
        self._opened = False
        self._cam = None
        print("[Camera] 摄像头已关闭")

    def grab(self, timeout=1.0):
        """抓取一帧，返回 BGR 图像或 None"""
        if not self._opened or self._cam is None:
            return None
        start = time.time()
        while time.time() - start < timeout:
            try:
                if self._cam.frame is not None:
                    return self._cam.frame.copy()
            except Exception:
                pass
            time.sleep(0.03)
        return None

    @property
    def is_open(self):
        return self._opened


# ═══════════════════════════════════════════════
#  TTS 播报 — 三模式
#    hardware: 讯飞云端合成 → aplay ALSA 设备（外接音箱），整句播报
#    xunfei:   云端合成 → aplay 自动检测设备，整句播报
#    i2c:      I2C 芯片 → 板载扬声器，按 max_chars 切段
# ═══════════════════════════════════════════════

_TTS_CONFIG_LOADED = False
_TTS_MODE = "hardware"
_TTS_MAX_CHARS = 6
_TTS_ALSA_DEVICE = "plughw:2,0"
_TTS_APLAY_TIMEOUT = 0
_TTS_XUNFEI_INSTANCE = None


def _load_tts_config():
    """惰性加载 config.yaml 中的 TTS 配置"""
    global _TTS_CONFIG_LOADED, _TTS_MODE, _TTS_MAX_CHARS, _TTS_ALSA_DEVICE, _TTS_APLAY_TIMEOUT
    if _TTS_CONFIG_LOADED:
        return
    try:
        import yaml
        cfg_path = os.path.join(_PROJECT_ROOT, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            tts_cfg = cfg.get("tts", {})
            _TTS_MODE = tts_cfg.get("default_mode", "hardware")
            _TTS_MAX_CHARS = tts_cfg.get("max_chars", 6)
            _TTS_ALSA_DEVICE = tts_cfg.get("alsa_device", "plughw:2,0")
            _TTS_APLAY_TIMEOUT = tts_cfg.get("aplay_timeout", 0)
    except Exception:
        pass
    _TTS_CONFIG_LOADED = True


def _split_into_chunks(text, max_chars):
    """
    将文本按标点 + max_chars 边界拆分成短段
    硬件 TTS 芯片一次只能朗读有限字数，超过会吞字
    """
    if len(text) <= max_chars:
        return [text]

    # 先按中文标点拆分，尽量在标点处断句
    import re
    raw_parts = re.split(r'([。！？；，、\n])', text)
    segments = []
    buf = ""
    for part in raw_parts:
        buf += part
        # 遇到句末标点或长度超限时切分
        if part in '。！？；\n' or len(buf) >= max_chars:
            if buf.strip():
                segments.append(buf.strip())
            buf = ""
    if buf.strip():
        segments.append(buf.strip())

    # 确保没有单段超过 max_chars
    result = []
    for seg in segments:
        while len(seg) > max_chars:
            result.append(seg[:max_chars])
            seg = seg[max_chars:]
        if seg:
            result.append(seg)

    return result


def _tts_i2c_speak(text):
    """I2C TTS 芯片：按 max_chars 切段 → 板载扬声器"""
    chunks = _split_into_chunks(text, _TTS_MAX_CHARS)

    if not HAS_TTS or _tts is None:
        # 硬件不可用，至少打印文字
        for chunk in chunks:
            print(f"[TTS(i2c)] {chunk}")
        return

    for i, chunk in enumerate(chunks, 1):
        try:
            print(f"[TTS(i2c)] ({i}/{len(chunks)}) {chunk}")
            _tts.TTSModuleSpeak("[h0][v10][m3]", chunk)
            # 短停顿防止芯片缓冲区溢出
            time.sleep(len(chunk) * 0.3 + 0.2)
        except Exception as e:
            print(f"[TTS(i2c)] 段 {i} 播报失败: {e}")
            time.sleep(0.3)


def _tts_hardware_speak(text):
    """
    硬件模式：讯飞云端合成 WAV → aplay 指定 ALSA 设备（外接音箱）
    通过 config.yaml 的 tts.alsa_device 指定设备名（默认 plughw:2,0）
    """
    global _TTS_XUNFEI_INSTANCE
    import subprocess
    try:
        if _TTS_XUNFEI_INSTANCE is None:
            from CustomFunctions.TTS_Control import TTS_Control
            _TTS_XUNFEI_INSTANCE = TTS_Control(mode="xunfei")

        # 临时保留音频文件，以便用指定设备播放
        orig_save = _TTS_XUNFEI_INSTANCE._save_audio
        _TTS_XUNFEI_INSTANCE._save_audio = True

        print(f"[TTS(hw)] {text}")
        # block=False 只合成不播放（用我们自己的 aplay）
        wav_path = _TTS_XUNFEI_INSTANCE.speak(text, block=False)

        _TTS_XUNFEI_INSTANCE._save_audio = orig_save

        if wav_path and os.path.exists(wav_path):
            # 用指定 ALSA 设备播放
            device = _TTS_ALSA_DEVICE
            file_size = os.path.getsize(wav_path)
            duration = file_size / (16000 * 2)
            print(f"[TTS(hw)] 🔊 aplay -D {device}（{file_size} bytes，约 {duration:.1f} 秒）...")
            kwargs = {"capture_output": True, "text": True}
            if _TTS_APLAY_TIMEOUT > 0:
                kwargs["timeout"] = _TTS_APLAY_TIMEOUT
            subprocess.run(["aplay", "-D", device, wav_path], **kwargs)
            # 不保留则播完删除
            if not orig_save:
                try:
                    os.unlink(wav_path)
                    print(f"[TTS(hw)] 临时音频已删除")
                except Exception:
                    pass
        else:
            print("[TTS(hw)] 合成失败，降级到 I2C")
            _tts_i2c_speak(text)
    except Exception as e:
        print(f"[TTS(hw)] 播报失败: {e}")
        _tts_i2c_speak(text)


def _tts_xunfei_speak(text):
    """讯飞云端 TTS：整句合成 → aplay 自动检测设备（不分段）"""
    global _TTS_XUNFEI_INSTANCE
    try:
        if _TTS_XUNFEI_INSTANCE is None:
            from CustomFunctions.TTS_Control import TTS_Control
            _TTS_XUNFEI_INSTANCE = TTS_Control(mode="xunfei")
        print(f"[TTS(xf)] {text}")
        _TTS_XUNFEI_INSTANCE.speak(text, block=True)
    except Exception as e:
        print(f"[TTS(xf)] 播报失败，降级到 I2C: {e}")
        _tts_i2c_speak(text)


def tts_speak(text, mode=None):
    """
    TTS 语音播报（三模式）

    参数:
        text: 要朗读的文字
        mode: "hardware" | "xunfei" | "i2c"
              None 时从 config.yaml 的 tts.default_mode 读取

    模式说明:
        hardware — 讯飞云端合成 WAV + aplay 指定 ALSA 设备（需外接音箱）
                   设备名由 config.yaml 的 tts.alsa_device 控制
        xunfei   — 讯飞云端合成 + aplay 自动检测设备
        i2c      — I2C TTS 芯片驱动板载扬声器，自动按 max_chars 切句
    """
    if not text:
        return

    _load_tts_config()
    mode = mode or _TTS_MODE

    if mode == "xunfei":
        _tts_xunfei_speak(text)
    elif mode == "i2c":
        _tts_i2c_speak(text)
    else:
        # "hardware" 走新硬件模式（讯飞合成 + aplay 指定设备）
        _tts_hardware_speak(text)


# ═══════════════════════════════════════════════
#  安全退出
# ═══════════════════════════════════════════════

_exit_handlers = []


def on_exit(handler):
    """注册退出清理函数"""
    _exit_handlers.append(handler)


def _cleanup(signum=None, frame=None):
    """执行所有清理函数"""
    print("\n[Demo] 正在安全退出...")

    # 不在此处写 DreamLogger — 无动作细节的条目会污染反思
    # 各 demo 在自己的主循环内按轮次记录，带有实际动作内容

    for h in _exit_handlers:
        try:
            h()
        except Exception:
            pass
    robot_stand()
    print("[Demo] 退出完成")
    os._exit(0)  # 强制退出，否则自定义 handler 不会中断程序


signal.signal(signal.SIGINT, lambda s, f: _cleanup())
signal.signal(signal.SIGTERM, lambda s, f: _cleanup())


# ═══════════════════════════════════════════════
#  演示辅助
# ═══════════════════════════════════════════════

def print_header(title, description=""):
    """打印演示头部"""
    width = 56
    print("\n" + "=" * width)
    print(f"  {title}")
    if description:
        print(f"  {description}")
    print("=" * width)


def print_footer(success=True):
    """打印演示尾部"""
    status = "✓ 演示完成" if success else "✗ 演示中断"
    print(f"\n  {status}\n")


def wait_key(seconds=0):
    """等待键盘或超时（seconds=0 表示无限等待）"""
    if seconds > 0:
        print(f"  (等待 {seconds}s，按 Ctrl+C 跳过...)")
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            pass
    else:
        try:
            input("  按 Enter 继续...")
        except (KeyboardInterrupt, EOFError):
            pass
