#!/usr/bin/env python3
"""极简 TFLite 输出检查 - 从 TonyPi 根目录运行"""
import sys
sys.path.insert(0, '/home/pi/TonyPi')
import cv2, numpy as np, time
from tflite_runtime.interpreter import Interpreter

interp = Interpreter('/home/pi/TonyPi/models/detect.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

print(f"输入: shape={inp[0]['shape'].tolist()}")
print(f"输出个数: {len(out)}")
for i, d in enumerate(out):
    print(f"  输出[{i}]: shape={d['shape'].tolist()} name={d['name']} dtype={d['dtype']}")

# 模拟输入跑一次
if len(inp[0]['shape']) == 4:
    h, w = inp[0]['shape'][1], inp[0]['shape'][2]
else:
    h, w = 320, 320
dummy = np.zeros((1, h, w, 3), dtype=np.uint8)
interp.set_tensor(inp[0]['index'], dummy)
interp.invoke()

print("\n=== 模拟推理输出 ===")
for i, d in enumerate(out):
    t = interp.get_tensor(d['index'])
    flat = t.flatten()
    print(f"\n输出[{i}] shape={list(t.shape)}")
    print(f"  flat[:20]={flat[:20].tolist()}")
    print(f"  min={float(flat.min()):.6f} max={float(flat.max()):.6f}")
    print(f"  >0.3: {len(flat[flat > 0.3])} 个, >0.5: {len(flat[flat > 0.5])} 个")

# 拍真照片
import hiwonder.Camera as Camera
cam = Camera.Camera()
cam.camera_open()
time.sleep(0.5)
for _ in range(5):
    cam.frame
    time.sleep(0.1)
frame = cam.frame.copy()
cam.camera_close()

input_data = np.expand_dims(cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB).astype(np.uint8), axis=0)
interp.set_tensor(inp[0]['index'], input_data)
interp.invoke()

print("\n=== 真实图像推理输出 ===")
for i, d in enumerate(out):
    t = interp.get_tensor(d['index'])
    flat = t.flatten()
    print(f"\n输出[{i}] shape={list(t.shape)}")
    print(f"  flat[:20]={flat[:20].tolist()}")
    print(f"  min={float(flat.min()):.6f} max={float(flat.max()):.6f}")
    print(f"  >0.3: {len(flat[flat > 0.3])} 个")
    # 尝试用不同方式解析
    if len(t.shape) == 3 and t.shape[-1] >= 4:
        print(f"  尝试解析为 [N, 框+conf+class] 格式:")
        for j in range(min(5, t.shape[1])):
            row = t[0, j]
            if len(row) >= 7:
                print(f"    [{j}] {[round(float(x),4) for x in row[:7]]}")
