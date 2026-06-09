#!/usr/bin/env python3
"""单张照片目标检测测试"""
import sys
sys.path.append('/home/pi/TonyPi/')
import cv2, time
import hiwonder.Camera as Camera
from CustomFunctions.object_detector import ObjectDetector

detector = ObjectDetector(confidence_threshold=0.3)
cam = Camera.Camera()
cam.camera_open()
time.sleep(0.5)

for _ in range(5):
    cam.frame
    time.sleep(0.1)

frame = cam.frame.copy()
cam.camera_close()

t0 = time.time()
results = detector.detect(frame)
elapsed = (time.time() - t0) * 1000

print(f"检测耗时: {elapsed:.0f}ms")
print(f"检测到 {len(results)} 个物体:")
print("-" * 40)
for obj in results:
    print(f"  {obj['label_cn']} ({obj['label']})  conf={obj['confidence']:.2f}  bbox={obj['bbox']}")

frame_out = frame.copy()
for obj in results:
    x1, y1, x2, y2 = obj['bbox']
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_out, f"{obj['label_cn']} {obj['confidence']:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imwrite('/home/pi/TonyPi/detect_result.jpg', frame_out)
print(f"\n结果已保存: /home/pi/TonyPi/detect_result.jpg")