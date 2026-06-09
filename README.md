# TonyPi 智能机器人 — 综合实训项目

<div align="center">

![TonyPi](https://img.shields.io/badge/机器人-TonyPi-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**9 大演示场景 · 逐层递进 · 从关键词控制到多模态 AI 智能体**

</div>

---

## 📋 项目概述

本项目基于 **Hiwonder TonyPi 人形机器人**，通过 **Python 3** 实现了从基础动作控制到多模态 AI 智能体交互的完整技术栈。项目采用**渐进式**的 9 个 Demo 设计，从离线语音关键词控制开始，逐步引入视觉识别、大模型理解、多模态场景感知、记忆对话，最终实现 **VLM 视觉决策** 与 **Dream Reflector 梦境反思** 等前沿概念。

### 硬件平台

| 模块 | 说明 |
|------|------|
| **主控** | Raspberry Pi (树莓派) |
| **舵机** | 总线/ PWM 舵机 × 多个，支持 ActionGroup 动作组 |
| **摄像头** | USB 摄像头，640×480，用于视觉识别 |
| **麦克风** | USB 麦克风（用于讯飞云端 STT） |
| **扬声器** | I2C TTS 芯片驱动板载扬声器（硬件模式） / 3.5mm 接口外接音箱（云端模式） |
| **IMU** | MPU6050 六轴姿态传感器（跌倒检测用） |
| **ASR** | 离线语音识别芯片（I2C 接口，关键词模式） |

### 软件栈

```
应用层 (Demo 01-09)
    ↓
中间层 (CustomFunctions/)
    ├── LLM_Control      — 大语言模型指令解析 (DeepSeek)
    ├── STT_Control      — 语音转文字 (硬件ASR / 讯飞云端)
    ├── TTS_Control      — 文字转语音 (硬件I2C / 讯飞云端)
    ├── step_executor    — 步骤执行引擎 (动作 + 视觉 + 语音)
    ├── visual_query     — VLM 视觉查询 (GLM-4V + JSON输出)
    └── dream_memory     — 梦的记忆日志 (经验回放)
    ↓
硬件层 (HiwonderSDK/)
    ├── ActionGroupControl — 动作组控制
    ├── Board              — 舵机/PWM/蜂鸣器控制
    ├── Camera             — 摄像头接口
    ├── Mpu6050            — 六轴姿态传感器
    ├── ASR / TTS          — 语音合成芯片
    └── ...
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip3 install opencv-python numpy pyyaml requests websocket-client pandas
pip3 install openai            # DeepSeek API

# 安装 Hiwonder SDK
cd HiwonderSDK
python3 setup.py install
```

### 2. 配置 API 密钥

编辑 `config.yaml`，填入必要的 API 密钥：

```yaml
# 至少需要配置以下一个或多个服务：
xunfei:     { appid, api_secret, api_key }   # 语音听写（STT）
deepseek:   { api_key }                       # 大语言模型（LLM 指令解析）
zhipu:      { api_key }                       # 多模态视觉（GLM-4V）
tts_xunfei: { appid, api_secret, api_key }    # 语音合成（云端 TTS）
```

> **提示**: 请替换为自己的密钥以保证稳定性。

### 3. 运行单个 Demo

```bash
# 运行 DeepSeek 多轮记忆对话
python3 Demo/demo_07_memory_dialog.py

# 运行 VLM 多帧视觉动作决策
python3 Demo/demo_08_vlm_decision.py

# 所有 Demo 均支持 --sim 参数（模拟模式，无需硬件）
python3 Demo/demo_04_object_tracking.py --sim
```

### 4. 运行全部演示

```bash
# 按序运行所有 9 个演示
python3 Demo/run_all.py

# 从第 3 个开始
python3 Demo/run_all.py --start 3

# 只运行第 6 个
python3 Demo/run_all.py --only 6

# 查看演示列表
python3 Demo/run_all.py --list
```

---

## 🎯 9 大演示场景

| # | 场景 | 标签 | 核心能力 |
|---|------|------|----------|
| 1 | **离线语音 + Dance** | 基础 | 硬件 ASR 关键词识别 → 动作执行 |
| 2 | **颜色识别 + 追踪 + 巡线** | 视觉 | OpenCV 颜色检测 + PID 舵机跟踪 |
| 3 | **智能搬运** | 综合 | 颜色定位 → AprilTag 导航 → 抓取放置 |
| 4 | **实时目标检测与追踪** | 检测 | MobileNet-SSD 检测 90 种物体 + 追踪人 |
| 5 | **讯飞在线语音 + 动作** | 在线AI | 云端 STT 句子识别 → 关键词匹配动作 |
| 6 | **语音驱动视觉查询** | 多模态 | 语音提问 → 拍照 → GLM-4V → JSON 输出 |
| 7 | **DeepSeek 多轮记忆对话** | LLM | 自然语言 → LLM 步骤链 → 执行 + 记忆 |
| 8 | **VLM 多帧视觉动作决策** | VLM | 5 帧连续画面 → VLM 自主选择动作 |
| 9 | **梦的反思（经验回放）** | 梦 | 读取日志 → LLM 分析 → 生成梦境反思 |

### Demo 详情

#### Demo 01 — 离线语音 + Dance
利用硬件 ASR 芯片识别预设关键词（前进/后退/鞠躬/跳舞等），匹配后执行对应动作组。重点展示 Dance 舞蹈动作。

```bash
python3 Demo/demo_01_offline_voice.py
```

#### Demo 02 — 颜色识别 + 追踪 + 巡线
颜色检测 → 云台 PID 跟踪 → 黑线巡线自主行走。

#### Demo 03 — 智能搬运
颜色识别定位目标 → 自主靠近 → 抓取 → AprilTag 标签导航 → 放置到目标区域。需红色/蓝色方块 + AprilTag 标签道具。

#### Demo 04 — 实时目标检测与追踪
使用 MobileNet SSD (Caffe) 在摄像头画面中实时检测 21 种常见物体，重点追踪"人"并做出交互响应。相比原厂 FaceDetect（只识别人脸），支持结构化 JSON 输出（label, conf, bbox, center），头部主动扫描 + 目标跟踪，发现重要物体时 TTS 播报。

```bash
python3 Demo/demo_04_object_tracking.py
python3 Demo/demo_04_object_tracking.py --sim    # 模拟模式
```

#### Demo 05 — 讯飞在线语音 + 动作
科大讯飞云端 STT，完整句子识别 → 关键词匹配 → 动作执行。与 Demo 01 的区别：离线只能识别预注册拼音关键词，在线可识别任意自然语句如"请往前走两步"。

#### Demo 06 — 语音驱动视觉查询 (VLM JSON 输出)
用户自由语音提问 → 机器人拍照 → 语音文本 + 图片 → 智谱 GLM-4V → **结构化 JSON 输出** → 保存到 `vlm_results/` → TTS 朗读回答。

相比旧版 SceneDescriber（固定提示词"描述场景"），新版 VisualQuery 允许用户问任意问题，VLM 返回 JSON 格式结果。

```bash
python3 Demo/demo_06_visual_query.py
```

**VLM 输出格式：**
```json
{
  "question": "桌子上有什么物体",
  "answer": "桌面上有一个红色马克杯和一个绿色网球",
  "details": { "objects": [...] }
}
```

#### Demo 07 — DeepSeek 多轮记忆对话
最高阶的 AI 语音助手演示。语音输入 → DeepSeek LLM 解析为结构化步骤链 → StepExecutor 执行 → TTS 反馈。支持：
- **多轮上下文记忆**：记住之前对话内容（如"我叫什么名字"）
- **指代消解**：理解"再做两个"中的"再"
- **复合指令**：一句话包含多个动作

```bash
python3 Demo/demo_07_memory_dialog.py
```

#### Demo 08 — VLM 多帧视觉动作决策
**架构创新** — 5 秒观察窗口 → 每秒拍 1 帧（共 5 帧）→ 5 帧 + 可用动作列表 → VLM 自主选择动作 → 执行 → 重新观察。

相比传统 CV 追踪（写死颜色阈值/轮廓参数），这个方案：
- VLM 直接理解画面语义 — "用户在向前走" vs "色块偏移"
- 动作列表由 VLM 自主选择，不用手动调 PID/阈值
- 可通过修改动作列表改变机器人行为，不改代码

```bash
python3 Demo/demo_08_vlm_decision.py
python3 Demo/demo_08_vlm_decision.py --sim      # 模拟模式
```

**VLM 动作列表：**
| 动作 | 说明 | 动作 | 说明 |
|------|------|------|------|
| `go_forward` | 向前走一步 | `turn_right` | 右转 |
| `go_back` | 向后退一步 | `bow` | 鞠躬 |
| `turn_left` | 左转 | `wave` | 挥手 |
| `stand` | 立正 | `speak` | 说话 (TTS) |
| `wait` | 等待 5 秒再观察 | | |

#### Demo 09 — 梦的反思（离线经验回放）
受 **海马体回放理论** 启发——动物在睡眠时会重放白天的经历，用于巩固记忆和优化行为。本 demo 模拟这一过程：读取当天 `DreamLogs/` 日志，用 LLM 分析失败模式，生成一段 ≤50 字的「梦」的反思，最后通过 TTS 讲出来。

```bash
python3 Demo/demo_09_dream_reflector.py
python3 Demo/demo_09_dream_reflector.py --sim   # 模拟模式
```

---

## 🧩 CustomFunctions — 核心中间层详解

`CustomFunctions/` 是项目的中枢层，封装了机器人所有高级能力，Demo 通过调用这些模块实现复杂功能。

### LLM_Control.py — 大语言模型指令解析

将自然语言指令解析为结构化的 JSON 步骤链，供 StepExecutor 逐条执行。

**工作流程：**
```
语音 "帮我拿红色杯子"
  ↓ STT
文字 "帮我拿红色杯子"
  ↓ DeepSeek LLM
JSON 步骤链:
  {
    "intent": "寻找红色杯子",
    "steps": [
      {"action": "detect_color",    "params": {"target": "red"},    "description": "寻找红色"},
      {"action": "go_forward",      "params": {"steps": 3},        "description": "靠近"},
      {"action": "bow",             "params": {},                  "description": "递送"}
    ],
    "tts_response": "好的，我找找看"
  }
  ↓ StepExecutor
逐条执行 → TTS 反馈
```

### STT_Control.py — 语音转文字

支持双模式切换，通过 `config.yaml` 中的 `stt.default_mode` 控制：

| 模式 | 适用场景 | 延迟 | 词汇量 | 原理 |
|------|----------|------|--------|------|
| **hardware** | 离线、关键词控制 | ~200ms | 预设 10-20 个 | I2C ASR 芯片，ID→文字映射 |
| **xunfei** | 自由语音、自然语言 | ~2-3s | 不限 | WAV 音频 → 讯飞 WebSocket API |

### TTS_Control.py — 文字转语音

同样支持双模式：

| 模式 | 适用场景 | 输出设备 |
|------|----------|----------|
| **hardware** | 默认模式，低延迟 | I2C TTS 芯片驱动板载扬声器 |
| **xunfei** | 云端合成，音质更自然 | 外接 USB/蓝牙音箱 (aplay) |

支持多种发音人：叶子（女推荐）、小燕、艾小萍、艾久旭（男）、徐小童（童声）。

### step_executor.py — 步骤执行引擎

接收 LLM_Control 输出的 JSON 步骤链，逐条执行，是连接"理解"与"行动"的桥梁。

**支持的动作类型：**
- `go_forward` / `go_back` / `turn_left` / `turn_right` — 移动
- `detect_color` — 颜色识别
- `say` / `tts` — 语音播报
- `bow` / `wave` / `stand` — 动作组
- 以及通过配置扩展的自定义动作

### visual_query.py — VLM 视觉查询

语音提问 → 拍照 → STT 结果 + 图片 → 智谱 GLM-4V → 结构化 JSON → 保存到 `vlm_results/` → TTS 朗读。

相比传统固定提示词方案，用户可自由提问任意场景相关问题，VLM 返回包含 question/answer/details 的 JSON 结果。

### dream_memory.py — 梦的记忆日志

每次交互结束时自动调用 `log_session()`，将当天所有动作、交互、失败记录到 `DreamLogs/{YYYY-MM-DD}.json`。Demo 09 读取这些日志进行「梦境反思」。

### object_detector.py — 通用目标检测

双方案引擎：

| 方案 | 检测类别 | 模型文件 |
|------|----------|----------|
| **Caffe** (优先) | 21 类 | `models/MobileNetSSD_deploy.{caffemodel,prototxt}` |
| **TFLite** (兜底) | 90 类 (COCO) | `models/detect.tflite` / `efficientdet_lite0.tflite` |

模型下载脚本位于 `scripts/` 目录中。

---

## 📁 上传到 GitHub 的文件结构

```
TonyPi/
├── ActionGroupDict.py          # 动作组编号映射（根目录）
├── config.yaml                 # API 密钥 & 模块配置（根目录）
├── README.md                   # 本文件（根目录）
│
├── ActionGroups/               # 动作组文件 (.d6a) — 30+ 动作
├── CustomFunctions/            # ⭐ 自定义功能模块（核心中间层）
│   ├── LLM_Control.py          #   DeepSeek 指令解析 → JSON 步骤链
│   ├── STT_Control.py          #   语音转文字 (硬件ASR / 讯飞)
│   ├── TTS_Control.py          #   文字转语音 (I2C芯片 / 讯飞)
│   ├── step_executor.py        #   步骤执行引擎 (动作+视觉+TTS)
│   ├── visual_query.py         #   VLM 视觉查询 → JSON 输出
│   ├── dream_memory.py         #   梦的记忆日志
│   └── object_detector.py      #   通用目标检测 (Caffe/TFLite 双方案)
│
├── Demo/                       # ⭐ 9 个渐进式演示场景
│   ├── run_all.py              #   演示总控
│   ├── common.py               #   公共初始化模块
│   ├── demo_01_offline_voice.py   #   离线语音 + Dance
│   ├── demo_02_vision.py          #   颜色识别 + 追踪 + 巡线
│   ├── demo_03_transport.py       #   智能搬运
│   ├── demo_04_object_tracking.py #   实时目标检测与追踪
│   ├── demo_05_online_voice.py    #   讯飞在线语音 + 动作
│   ├── demo_06_visual_query.py    #   语音驱动视觉查询
│   ├── demo_07_memory_dialog.py   #   DeepSeek 多轮记忆对话
│   ├── demo_08_vlm_decision.py    #   VLM 多帧视觉动作决策
│   ├── demo_09_dream_reflector.py #   梦的反思（离线经验回放）
│   └── templates/                 #   演示用图片模板
│
├── Functions/                  # 原厂功能模块
│   ├── Follow.py               #   颜色球体跟随
│   ├── Transport.py            #   智能搬运
│   ├── Fall_and_Stand.py       #   跌倒检测与起立
│   ├── Face_Detect.py          #   人脸检测
│   ├── Color_Recognize.py      #   颜色识别
│   ├── VisualPatrol.py         #   视觉巡逻
│   └── ...
│
├── HiwonderSDK/                # 原厂 Hiwonder Python SDK
│   └── hiwonder/               #   驱动库（舵机/摄像头/传感器）
│
├── ActionGroups/               # 动作组文件 (.d6a)
├── models/                     # 模型文件 (Caffe + TFLite)
├── DreamLogs/                  # 交互日志（供梦境反思使用）
├── scripts/                    # 工具脚本
│   ├── Debug_tflite.py         #   ✅ TFLite 输出检查
│   └── download_tflite_model.sh #  ✅ 模型下载脚本
│
├── TonyPi.py                   # 系统主控（后台服务）
├── RPCServer.py                # RPC 通信服务
├── MjpgServer.py               # MJPEG 视频流服务
├── Joystick.py                 # 手柄控制
├── servo_config.yaml           # 舵机初始位置
└── lab_config.yaml             # LAB 颜色阈值
```

---

## 🔧 配置参考

### config.yaml 主要字段

| 字段 | 用途 | 必需 |
|------|------|------|
| `xunfei.appid` | 科大讯飞语音听写 APPID | 使用 STT 时需要 |
| `deepseek.api_key` | DeepSeek 大模型密钥 | 使用 LLM 指令解析时需要 |
| `zhipu.api_key` | 智谱 GLM-4V 密钥 | 使用视觉查询/混合追踪时需要 |
| `tts_xunfei.appid` | 科大讯飞语音合成 APPID | 使用云端 TTS 时需要 |
| `stt.default_mode` | STT 模式: hardware / xunfei | 默认: xunfei |
| `tts.default_mode` | TTS 模式: hardware / xunfei | 默认: hardware |

---

## 🧠 关键技术点

### LLM 指令解析流程

```
语音输入 → STT 转文字 → DeepSeek LLM → JSON 步骤链 → StepExecutor 逐条执行 → TTS 反馈
```

### VLM 视觉查询流程 (Demo 06)

```
语音"桌子上有什么" → STT → 拍照 → GLM-4V(图片+文本) → JSON → 保存 + TTS 朗读
```

### VLM 多帧视觉决策 (Demo 08)

```
摄像头 5 秒×5 帧 → 5帧 + 可用动作列表 → VLM 自主选择动作 → 执行 → 循环
```

### 梦境反思 (Demo 09)

```
DreamLogs/ 日志 → DeepSeek LLM 分析失败模式 → ≤50 字梦境反思 → TTS 播报
```

---

## 📝 注意事项

1. **API Key 安全**：`config.yaml` 包含敏感密钥，建议添加到 `.gitignore` 或使用环境变量替换
2. **摄像头权限**：树莓派上首次运行需允许摄像头访问
3. **音频设备**：硬件 TTS 模式使用 I2C 芯片驱动板载扬声器；云端 TTS 模式需要外接 USB/蓝牙音箱
4. **模拟模式**：所有 Demo 均支持 `--sim` 参数，可在无硬件环境下运行查看概念演示
5. **动作组**：`ActionGroups/` 中的 `.d6a` 文件是二进制动作数据，如需新增动作需通过原厂上位机工具录制

---

## 📄 License

本项目用于人工智能综合实训教学目的。
