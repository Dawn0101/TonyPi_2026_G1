#!/usr/bin/env python3
# coding=utf8
"""
Demo 总控 — 按序运行全部 8 个演示场景
══════════════════════════════════════════════

用法:
    python3 Demo/run_all.py              # 按序运行所有演示
    python3 Demo/run_all.py --start 3    # 从第 3 个开始
    python3 Demo/run_all.py --only 4     # 只运行第 4 个
    python3 Demo/run_all.py --list       # 列出所有演示

演示顺序:
    1. 离线语音 + Dance
    2. 颜色识别 + 追踪 + 巡线
    3. 智能搬运
    4. 实时目标检测与追踪
    5. 讯飞在线语音 + 动作
    6. 语音驱动视觉查询 (VLM JSON 输出)
    7. DeepSeek 多轮记忆对话
    8. VLM 多帧视觉动作决策
    9. 梦的反思（离线经验回放）

每个演示结束后会提示是否继续下一个。
"""

import os
import sys
import time
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMOS = [
    {
        "id": 1,
        "name": "离线语音 + Dance",
        "script": "demo_01_offline_voice.py",
        "duration": "~30s",
        "tag": "基础",
        "desc": "硬件离线 ASR 关键词识别，重点展示 dance 舞蹈动作",
    },
    {
        "id": 2,
        "name": "颜色识别 + 追踪 + 巡线",
        "script": "demo_02_vision.py",
        "duration": "~35s",
        "tag": "视觉",
        "desc": "颜色检测 → 云台 PID 跟踪 → 黑线巡线自主行走",
    },
    {
        "id": 3,
        "name": "智能搬运",
        "script": "demo_03_transport.py",
        "duration": "~45s",
        "tag": "综合",
        "desc": "颜色识别定位 → 自主靠近 → 抓取 → AprilTag导航 → 放置",
    },
    {
        "id": 4,
        "name": "实时目标检测与追踪",
        "script": "demo_04_object_tracking.py",
        "duration": "~20s",
        "tag": "检测",
        "desc": "MobileNet-SSD 实时检测90种物体，追踪人+头部扫描+TTS播报",
    },
    {
        "id": 5,
        "name": "讯飞在线语音 + 动作",
        "script": "demo_05_online_voice.py",
        "duration": "~30s",
        "tag": "在线AI",
        "desc": "科大讯飞云端 STT，完整句子识别 → 关键词 → 动作",
    },
    {
        "id": 6,
        "name": "语音驱动视觉查询 (VLM JSON 输出)",
        "script": "demo_06_visual_query.py",
        "duration": "~35s",
        "tag": "多模态",
        "desc": "语音自由提问 → 拍照 → GLM-4V → JSON 文件保存 + TTS 摘要播报",
    },
    {
        "id": 7,
        "name": "DeepSeek 多轮记忆对话",
        "script": "demo_07_memory_dialog.py",
        "duration": "~40s",
        "tag": "LLM",
        "desc": "LLM 上下文记忆 + 复合指令 + 指代消解 + 步骤执行",
    },
    {
        "id": 8,
        "name": "VLM 多帧视觉动作决策",
        "script": "demo_08_vlm_decision.py",
        "duration": "~30s/轮",
        "tag": "VLM",
        "desc": "5秒连续拍照 → VLM 分析画面变化 → 自主选动作执行 → 循环，无需 CV 参数",
    },
    {
        "id": 9,
        "name": "梦的反思（离线经验回放）",
        "script": "demo_09_dream_reflector.py",
        "duration": "~10s",
        "tag": "梦",
        "desc": "读取 DreamLogs → DeepSeek 分析 → 生成 ≤50 字梦境反思 → TTS 播报",
    },
]


def print_banner():
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║       TonyPi 智能机器人 — 演示总控           ║")
    print("║       8 大场景 · 逐层递进 · 能力展示          ║")
    print("╚══════════════════════════════════════════════╝")
    print()


def print_list():
    """列出所有演示"""
    print_banner()
    print(f"{'#':<4} {'场景':<28} {'时长':<8} {'标签':<10}")
    print("-" * 52)
    for d in DEMOS:
        print(f"{d['id']:<4} {d['name']:<28} {d['duration']:<8} {d['tag']:<10}")
    print("-" * 52)
    print(f"  总计约 4 分 55 秒")
    print()


def run_demo(demo_info):
    """运行单个演示脚本"""
    script_path = os.path.join(_PROJECT_ROOT, "Demo", demo_info["script"])
    if not os.path.exists(script_path):
        print(f"  ✗ 脚本不存在: {script_path}")
        return False

    print(f"\n{'─' * 56}")
    print(f"  启动: #{demo_info['id']} {demo_info['name']} [{demo_info['tag']}]")
    print(f"  {demo_info['desc']}")
    print(f"{'─' * 56}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=_PROJECT_ROOT,
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\n  [总控] #{demo_info['id']} 被用户中断")
        return False


def ask_continue(current_id):
    """询问是否继续"""
    if current_id >= len(DEMOS):
        return False
    try:
        ans = input(f"\n  继续下一个演示？(Enter=继续, s=跳过, q=退出) > ").strip().lower()
        if ans == "q":
            return False
        elif ans == "s":
            return True  # skip handled by caller
        return True
    except (KeyboardInterrupt, EOFError):
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TonyPi 演示总控")
    parser.add_argument("--list", action="store_true", help="列出所有演示")
    parser.add_argument("--start", type=int, default=1, help="从第 N 个演示开始 (默认: 1)")
    parser.add_argument("--only", type=int, default=0, help="只运行第 N 个演示")
    parser.add_argument("--auto", action="store_true", help="自动连续运行（不询问）")
    args = parser.parse_args()

    if args.list:
        print_list()
        return

    print_banner()

    if args.only > 0:
        # 只运行指定演示
        demo = next((d for d in DEMOS if d["id"] == args.only), None)
        if demo is None:
            print(f"  无效编号: {args.only}，有效范围 1-{len(DEMOS)}")
            return
        print(f"  只运行: #{demo['id']} {demo['name']}")
        run_demo(demo)
        return

    # 按序运行
    start_idx = max(0, args.start - 1)
    if start_idx >= len(DEMOS):
        print(f"  起始编号 {args.start} 超出范围 (1-{len(DEMOS)})")
        return

    print(f"  从 #{args.start} 开始，共 {len(DEMOS) - start_idx} 个演示")
    print()

    for i in range(start_idx, len(DEMOS)):
        demo = DEMOS[i]

        run_demo(demo)

        if i < len(DEMOS) - 1:
            if args.auto:
                print(f"\n  3 秒后自动进入下一个演示...")
                time.sleep(3)
            else:
                cont = ask_continue(i + 1)
                if not cont:
                    print(f"\n  [总控] 演示序列中止于 #{demo['id']}")
                    break

    print(f"\n{'═' * 56}")
    print(f"  演示全部结束！")
    print(f"{'═' * 56}\n")


if __name__ == "__main__":
    main()
