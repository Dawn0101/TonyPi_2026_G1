#!/usr/bin/env python3
# coding=utf8
"""
大语言模型指令解析模块 (LLM_Control)
将自然语言指令解析为结构化的步骤链（JSON 格式）

用法:
    llm = LLM_Control()
    plan = llm.parse("帮我拿红色杯子")
    # 返回:
    # {
    #   "intent": "寻找并拿取红色杯子",
    #   "steps": [
    #     {"action": "detect_color", "params": {"target": "red"}, "description": "寻找红色物体"},
    #     {"action": "go_forward", "params": {"steps": 3}, "description": "靠近红色物体"},
    #     {"action": "bow", "params": {}, "description": "递送给用户"}
    #   ],
    #   "tts_response": "好的，我找找看有没有红色杯子"
    # }
"""

import os
import sys
import json
import time

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _load_config():
    """加载 config.yaml"""
    import yaml
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 可执行动作定义（给 LLM 参考） ──────────────

AVAILABLE_ACTIONS = {
    "go_forward": {
        "description": "向前走",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "go_back": {
        "description": "向后退",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "turn_left": {
        "description": "左转",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "turn_right": {
        "description": "右转",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "left_move": {
        "description": "向左平移",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "right_move": {
        "description": "向右平移",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "bow": {
        "description": "鞠躬",
        "params": {},
    },
    "wave": {
        "description": "挥手打招呼",
        "params": {},
    },
    "push_ups": {
        "description": "做俯卧撑",
        "params": {"times": "次数（整数，默认1）"},
    },
    "stand": {
        "description": "立正站好（回到初始姿势）",
        "params": {},
    },
    "squat": {
        "description": "下蹲",
        "params": {},
    },
    "chest": {
        "description": "挺胸庆祝动作",
        "params": {},
    },
    "twist": {
        "description": "扭腰动作",
        "params": {},
    },
    "stepping": {
        "description": "原地踏步",
        "params": {"steps": "步数（整数，默认1）"},
    },
    "left_kick": {
        "description": "左侧踢",
        "params": {},
    },
    "right_kick": {
        "description": "右侧踢",
        "params": {},
    },
    "left_shot": {
        "description": "左脚踢",
        "params": {},
    },
    "right_shot": {
        "description": "右脚踢",
        "params": {},
    },
    "detect_color": {
        "description": "通过摄像头检测指定颜色的物体",
        "params": {"target": "颜色名称：red/green/blue"},
    },
    "detect_face": {
        "description": "通过摄像头检测人脸",
        "params": {},
    },
    "speak": {
        "description": "让机器人通过扬声器说话（TTS语音合成）",
        "params": {"text": "要说的文字内容"},
    },
    "wait": {
        "description": "等待一段时间",
        "params": {"seconds": "等待秒数（数字，默认1）"},
    },
}

SYSTEM_PROMPT = f"""你是一个机器人控制系统的指令解析器。
用户会输入一段自然语言指令，请将其解析为结构化的步骤列表。

可执行动作列表：
{json.dumps(AVAILABLE_ACTIONS, ensure_ascii=False, indent=2)}

输出格式要求（严格返回 JSON，不要包含 markdown 代码块标记）：
{{
    "intent": "用户意图的简短描述",
    "steps": [
        {{
            "action": "动作名称",
            "params": {{"参数名": "参数值"}},
            "description": "这一步在做什么的中文说明"
        }}
    ],
    "tts_response": "机器人第一步应该说的话"
}}

要求：
1. steps 中的 action 必须从上述可执行动作中选择
2. 如果用户指令无法理解或无法执行，返回：
   {{"intent": "无法理解", "steps": [], "tts_response": "抱歉，我没有理解您的指令"}}
3. 如果用户只是打招呼或闲聊，返回一个 speak 动作即可
4. 所有输出 JSON 的 key 和 string value 使用双引号
5. 不要输出任何除了 JSON 以外的内容
6. 记住用户在对话中提到的个人信息（如名字、喜好、需求等），在后续对话中直接使用这些信息。例如用户说"我叫张三"，你要在 tts_response 中用"张三"称呼他
7. 如果用户询问之前提到过的信息（如"我叫什么名字""我刚才说了什么"），从对话历史中找到正确答案并回复，不要编造"""


class LLM_Control:
    """大语言模型指令解析器"""

    def __init__(self, provider=None):
        cfg = _load_config()
        llm_cfg = cfg.get("llm", {})
        provider = provider or llm_cfg.get("provider", "deepseek")

        if provider == "deepseek":
            ds_cfg = cfg.get("deepseek", {})
            self.api_key = ds_cfg.get("api_key", "")
            self.base_url = ds_cfg.get("base_url", "https://api.deepseek.com")
            self.model = ds_cfg.get("model", "deepseek-chat")
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

        if not self.api_key or self.api_key == "你的DeepSeekAPIKey":
            print("[LLM] 警告: DeepSeek API Key 未配置，请在 config.yaml 中填写")

        self._conversation_history = []
        self.max_history = 10  # 保留最近 10 轮对话

    def parse(self, text, keep_history=True):
        """
        解析自然语言指令
        参数:
            text: 用户输入的文本
            keep_history: 是否保持对话上下文（默认 True）
        返回:
            {
                "intent": str,
                "steps": [{"action": str, "params": dict, "description": str}],
                "tts_response": str
            }
        """
        return self._call_llm(text, keep_history)

    def _call_llm(self, text, keep_history):
        """调用 LLM API（直接 HTTP POST，不依赖 openai 库）"""
        if not self.api_key or self.api_key == "你的DeepSeekAPIKey":
            print("[LLM] API Key 未配置，使用本地规则解析")
            return self._fallback_parse(text)

        try:
            import requests
        except ImportError:
            print("[LLM] 请安装 requests 库: pip install requests")
            return self._fallback_parse(text)

        try:
            # 构建消息
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # 添加上下文历史
            if keep_history:
                messages.extend(self._conversation_history)

            messages.append({"role": "user", "content": text})

            print(f"[LLM] 正在解析: \"{text}\"")

            # DeepSeek 兼容 OpenAI API 格式，直接 HTTP POST
            url = self.base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            if resp.status_code != 200:
                print(f"[LLM] API 返回 {resp.status_code}: {resp.text[:200]}")
                return self._fallback_parse(text)

            reply = resp.json()["choices"][0]["message"]["content"].strip()
            # 去掉可能的 markdown 代码块标记
            reply = reply.replace("```json", "").replace("```", "").strip()

            result = json.loads(reply)

            # 保存对话历史
            if keep_history:
                self._conversation_history.append({"role": "user", "content": text})
                self._conversation_history.append({"role": "assistant", "content": reply})
                # 裁剪历史，避免超出上下文长度
                if len(self._conversation_history) > self.max_history * 2:
                    self._conversation_history = self._conversation_history[-(self.max_history * 2):]

            self._print_plan(result)
            return result

        except Exception as e:
            print(f"[LLM] API 调用失败: {e}")
            # 重试一次
            try:
                print("[LLM] 重试中...")
                time.sleep(1)
                return self._call_llm(text, keep_history)
            except:
                return self._fallback_parse(text)

    def _fallback_parse(self, text):
        """
        本地规则兜底：当 API 不可用时，
        通过关键词匹配返回简单步骤
        """
        text_lower = text.lower()

        # ── 先检查打招呼/自我介绍（不论后面还有没有其他文字）──
        # "我叫XX" → 记住名字并回应
        import re
        name_match = re.search(r'我叫(\S+)', text)
        has_greeting = any(kw in text for kw in ["你好", "您好", "hi", "hello", "打招呼", "认识一下"])

        if name_match:
            user_name = name_match.group(1)
            print(f"[LLM 兜底] 记住名字: {user_name}")
            # 存入历史（模拟记忆，仅在本次 session 生效）
            self._user_name = user_name
            if has_greeting:
                steps = [
                    {"action": "wave", "params": {}, "description": f"向{user_name}挥手打招呼"},
                    {"action": "speak", "params": {"text": f"你好{user_name}，我是TonyPi机器人"}, "description": f"叫出{user_name}的名字"},
                ]
                tts = f"你好{user_name}，我是TonyPi机器人，很高兴认识你"
            else:
                steps = [
                    {"action": "speak", "params": {"text": f"好的{user_name}，我记住了"}, "description": f"记住{user_name}"},
                ]
                tts = f"好的{user_name}，我记住了"
            result = {"intent": f"认识{user_name}", "steps": steps, "tts_response": tts}
            self._print_plan(result)
            return result

        if has_greeting:
            steps = [
                {"action": "wave", "params": {}, "description": "挥手打招呼"},
                {"action": "speak", "params": {"text": "你好，我是TonyPi机器人"}, "description": "自我介绍"},
            ]
            result = {"intent": "打招呼", "steps": steps, "tts_response": "你好，我是TonyPi机器人"}
            self._print_plan(result)
            return result

        # ── 检查名字回忆 ──
        if any(kw in text for kw in ["我叫什么名字", "我是谁", "还记得我吗"]):
            name = getattr(self, '_user_name', None)
            if name:
                steps = [{"action": "speak", "params": {"text": f"你是{name}呀，我当然记得"}, "description": f"回忆出{name}"}]
                result = {"intent": "回忆名字", "steps": steps, "tts_response": f"你是{name}呀，我当然记得"}
            else:
                steps = [{"action": "speak", "params": {"text": "抱歉，我还没有记住你的名字，请告诉我"}, "description": "还没记住"}]
                result = {"intent": "未记住", "steps": steps, "tts_response": "抱歉，我还没有记住你的名字"}
            self._print_plan(result)
            return result

        # ── 关键词 → 动作映射 ──
        keyword_map = [
            (["前进", "往前走", "向前"], [{"action": "go_forward", "params": {"steps": 2}, "description": "向前走"}]),
            (["后退", "往后退", "向后"], [{"action": "go_back", "params": {"steps": 2}, "description": "向后退"}]),
            (["鞠躬"], [{"action": "bow", "params": {}, "description": "鞠躬"}]),
            (["挥手"], [{"action": "wave", "params": {}, "description": "挥手"}]),
            (["左转"], [{"action": "turn_left", "params": {"steps": 2}, "description": "左转"}]),
            (["右转"], [{"action": "turn_right", "params": {"steps": 2}, "description": "右转"}]),
            (["左移"], [{"action": "left_move", "params": {"steps": 2}, "description": "向左平移"}]),
            (["右移"], [{"action": "right_move", "params": {"steps": 2}, "description": "向右平移"}]),
            (["俯卧撑"], [{"action": "push_ups", "params": {"times": 3}, "description": "做俯卧撑"}]),
            (["下蹲"], [{"action": "squat", "params": {}, "description": "下蹲"}]),
            (["踢"], [{"action": "left_kick", "params": {}, "description": "踢腿"}]),
            (["站立", "立正"], [{"action": "stand", "params": {}, "description": "立正站好"}]),
            (["红色", "红"], [{"action": "detect_color", "params": {"target": "red"}, "description": "检测红色物体"}]),
            (["绿色", "绿"], [{"action": "detect_color", "params": {"target": "green"}, "description": "检测绿色物体"}]),
            (["蓝色", "蓝"], [{"action": "detect_color", "params": {"target": "blue"}, "description": "检测蓝色物体"}]),
            (["人脸", "谁", "有人"], [{"action": "detect_face", "params": {}, "description": "检测人脸"}]),
        ]

        for keywords, steps in keyword_map:
            if any(k in text for k in keywords):
                tts = f"好的，{'、'.join(s['description'] for s in steps)}"
                result = {
                    "intent": text,
                    "steps": steps,
                    "tts_response": tts,
                }
                self._print_plan(result)
                return result

        # 默认：无法理解
        result = {
            "intent": "无法理解",
            "steps": [],
            "tts_response": "抱歉，我没有理解您的指令，请说清楚您想让我做什么",
        }
        self._print_plan(result)
        return result

    def reset_history(self):
        """重置对话历史"""
        self._conversation_history = []
        print("[LLM] 对话历史已重置")

    @staticmethod
    def _print_plan(plan):
        """打印解析结果（调试用）"""
        print(f"[LLM] 意图: {plan.get('intent', '未知')}")
        print(f"[LLM] TTS:  {plan.get('tts_response', '')}")
        steps = plan.get("steps", [])
        if steps:
            print(f"[LLM] 步骤链 ({len(steps)} 步):")
            for i, step in enumerate(steps, 1):
                params_str = ", ".join(f"{k}={v}" for k, v in step.get("params", {}).items())
                print(f"       {i}. {step['action']}({params_str})  — {step['description']}")
        else:
            print("[LLM] 无需执行步骤")


# ── 独立测试 ──────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("LLM_Control 测试")
    print("=" * 50)

    llm = LLM_Control()

    test_cases = [
        "你好",
        "往前走两步",
        "鞠躬",
        "帮我拿红色杯子",
        "看到有人就打招呼",
        "做几个俯卧撑",
        "我有点渴了",
    ]

    for text in test_cases:
        print(f"\n{'─' * 40}")
        result = llm.parse(text, keep_history=False)
        print(f"输入: {text}")
        print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print(f"{'─' * 40}")
