#!/usr/bin/env python3
# coding=utf8
"""
通用目标检测模块 (Object Detector)

双方案:
  A) Caffe: MobileNetSSD_deploy (21类) — 优先
  B) TFLite: detect.tflite / EfficientDet-Lite0 (90类) — 兜底
"""

import os
import sys
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# ── Caffe 21 类标签 ──
CAFFE_CLASSES_EN = {
    0: "background", 1: "airplane", 2: "bicycle", 3: "bird", 4: "boat",
    5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
    10: "cow", 11: "dining table", 12: "dog", 13: "horse",
    14: "motorcycle", 15: "person", 16: "potted plant",
    17: "sheep", 18: "sofa", 19: "train", 20: "tv monitor"
}
CAFFE_CLASSES_CN = {
    0: "背景", 1: "飞机", 2: "自行车", 3: "鸟", 4: "船",
    5: "瓶子", 6: "公共汽车", 7: "汽车", 8: "猫", 9: "椅子",
    10: "奶牛", 11: "餐桌", 12: "狗", 13: "马",
    14: "摩托车", 15: "人", 16: "盆栽植物",
    17: "羊", 18: "沙发", 19: "火车", 20: "电视显示器"
}

# ── TFLite 90 类标签（EfficientDet, 0-indexed, 加偏移1对齐 labelmap.txt）──
TFLITE_LABELS = [
    "???", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "???",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "???", "backpack", "umbrella", "???", "???", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "???", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "???", "dining table", "???", "???",
    "toilet", "???", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "???", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

IMPORTANT_OBJECTS = {
    "person": "人", "bottle": "瓶子", "cat": "猫", "dog": "狗",
    "chair": "椅子", "car": "汽车", "dining table": "餐桌",
    "sofa": "沙发", "tv monitor": "电视", "tv": "电视",
    "laptop": "笔记本电脑", "cell phone": "手机", "book": "书",
    "cup": "杯子", "sports ball": "球", "teddy bear": "玩具熊"
}


class ObjectDetector:
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self._caffe_net = None
        self._tflite_interp = None
        self._anchors = None
        self._load_model()

    def _load_model(self):
        """方案 A: Caffe（优先）"""
        prototxt = os.path.join(_MODELS_DIR, 'MobileNetSSD_deploy.prototxt')
        model = os.path.join(_MODELS_DIR, 'MobileNetSSD_deploy.caffemodel')
        if os.path.exists(prototxt) and os.path.exists(model):
            try:
                self._caffe_net = cv2.dnn.readNetFromCaffe(prototxt, model)
                print(f"[Detector] Caffe 模型加载成功")
            except Exception as e:
                print(f"[Detector] Caffe 加载失败: {e}")

        # 同时加载 TFLite 作为兜底（Caffe 可能加载成功但推理失败）
        tflite_path = os.path.join(_MODELS_DIR, 'detect.tflite')
        if os.path.exists(tflite_path):
            try:
                from tflite_runtime.interpreter import Interpreter
                self._tflite_interp = Interpreter(model_path=tflite_path)
                self._tflite_interp.allocate_tensors()
                inp = self._tflite_interp.get_input_details()
                h, w = inp[0]['shape'][1], inp[0]['shape'][2]
                self._anchors = self._generate_anchors(h, w)
                print(f"[Detector] TFLite 模型加载成功 ({len(self._anchors)} anchors)")
                return
            except Exception as e:
                print(f"[Detector] TFLite 加载失败: {e}")

        print(f"[Detector] ❌ 所有模型加载失败")

    @property
    def available(self):
        return self._caffe_net is not None or self._tflite_interp is not None

    def detect(self, frame):
        # 先尝试 Caffe，如果第一次推理失败就永久切 TFLite
        if self._caffe_net is not None:
            results = self._detect_caffe(frame)
            if results is not None:
                return results
            # Caffe 推理失败，切 TFLite
            print("[Detector] Caffe 推理失败，切换 TFLite 模式")
            self._caffe_net = None  # 关掉 Caffe，后续直接用 TFLite
        if self._tflite_interp is not None:
            return self._detect_tflite(frame)
        return []

    # ── 方案 A: Caffe（你那个脚本的原始逻辑）──

    def _detect_caffe(self, frame):
        img_h, img_w = frame.shape[:2]
        results = []
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5
            )
            self._caffe_net.setInput(blob)
            detections = self._caffe_net.forward()

            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < self.confidence_threshold:
                    continue
                idx = int(detections[0, 0, i, 1])
                if idx < 0 or idx > 20:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
                x1, y1, x2, y2 = box.astype("int")
                x1 = max(0, min(x1, img_w)); y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w)); y2 = max(0, min(y2, img_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                label = CAFFE_CLASSES_EN.get(idx, "unknown")
                results.append({
                    "label": label,
                    "label_cn": IMPORTANT_OBJECTS.get(label, CAFFE_CLASSES_CN.get(idx, label)),
                    "confidence": round(confidence, 3), "class_id": idx,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "area": (x2 - x1) * (y2 - y1),
                })
        except Exception as e:
            print(f"[Detector] ⚠ Caffe 异常: {e}")
            return None  # 通知调用方切 TFLite
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    # ── 方案 B: TFLite (EfficientDet-Lite0: 分类[19206,90] + 回归[19206,4]) ──

    def _detect_tflite(self, frame):
        img_h, img_w = frame.shape[:2]
        results = []
        try:
            inp = self._tflite_interp.get_input_details()
            out = self._tflite_interp.get_output_details()

            h, w = inp[0]['shape'][1], inp[0]['shape'][2]
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
            self._tflite_interp.set_tensor(inp[0]['index'],
                                            np.expand_dims(frame_rgb.astype(np.uint8), axis=0))
            self._tflite_interp.invoke()

            class_scores = self._tflite_interp.get_tensor(out[0]['index'])[0]  # [19206, 90]
            box_offsets = self._tflite_interp.get_tensor(out[1]['index'])[0]   # [19206, 4]

            # 对每个 anchor 取最高分类分数
            max_scores = np.max(class_scores, axis=1)
            best_classes = np.argmax(class_scores, axis=1)

            # 过滤低置信度
            mask = max_scores >= self.confidence_threshold
            idxs = np.where(mask)[0]

            for i in idxs:
                class_id = int(best_classes[i])
                label_idx = class_id + 1  # EfficientDet 0-indexed → labelmap 1-indexed
                if label_idx >= len(TFLITE_LABELS):
                    continue
                label = TFLITE_LABELS[label_idx]
                if label in ('???', 'background', 'unknown'):
                    continue

                # 解码框: 回归值 [cy, cx, h, w] 相对于 anchor
                anc = self._anchors[i]
                raw = box_offsets[i]
                cx = anc[0] + raw[1] * anc[2]
                cy = anc[1] + raw[0] * anc[3]
                bw = anc[2] * np.exp(float(raw[3]))
                bh = anc[3] * np.exp(float(raw[2]))

                x1 = int((cx - bw / 2) * img_w)
                y1 = int((cy - bh / 2) * img_h)
                x2 = int((cx + bw / 2) * img_w)
                y2 = int((cy + bh / 2) * img_h)
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                results.append({
                    "label": label,
                    "label_cn": IMPORTANT_OBJECTS.get(label, label),
                    "confidence": round(float(max_scores[i]), 3),
                    "class_id": class_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "area": (x2 - x1) * (y2 - y1),
                })

            # NMS
            if len(results) > 1:
                boxes_wh = np.array([(r["bbox"][0], r["bbox"][1],
                                      r["bbox"][2] - r["bbox"][0],
                                      r["bbox"][3] - r["bbox"][1])
                                      for r in results], dtype=np.float32)
                scores_arr = np.array([r["confidence"] for r in results], dtype=np.float32)
                keep = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores_arr.tolist(), 0.0, 0.45)
                if len(keep) > 0:
                    # OpenCV 4.4 返回元组或 list
                    if isinstance(keep, tuple):
                        keep = list(keep[0]) if len(keep) > 0 else []
                    elif isinstance(keep, np.ndarray):
                        keep = keep.flatten().tolist()
                    results = [results[int(i)] for i in keep]

        except Exception as e:
            print(f"[Detector] ⚠ TFLite 异常: {e}")

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    @staticmethod
    def _generate_anchors(img_h, img_w):
        """EfficientDet-Lite0 anchor 生成: 5 特征层 × 9 anchors/位置"""
        strides = [8, 16, 32, 64, 128]
        scales = [1.0, 2.0 ** (1/3), 2.0 ** (2/3)]
        aspect_ratios = [1.0, 2.0, 0.5]
        anchors = []
        for stride in strides:
            fh = int(np.ceil(img_h / stride))
            fw = int(np.ceil(img_w / stride))
            for y in range(fh):
                for x in range(fw):
                    cx = (x + 0.5) / fw
                    cy = (y + 0.5) / fh
                    for scale in scales:
                        for ar in aspect_ratios:
                            bw = scale * np.sqrt(1.0 / ar) / fw
                            bh = scale * np.sqrt(ar) / fh
                            anchors.append([cx, cy, bw, bh])
        return np.array(anchors, dtype=np.float32)

    # ── 方向分析 ──

    def analyze_direction(self, frame, target_label="person"):
        results = self.detect(frame)
        targets = [r for r in results if r["label"] == target_label]
        if not targets:
            return {"has_target": False, "direction": "", "distance": "",
                    "target_count": 0, "details": results}
        t = targets[0]
        area_ratio = t["area"] / (frame.shape[0] * frame.shape[1])
        third = frame.shape[1] / 3
        d = "left" if t["center"][0] < third else "right" if t["center"][0] > frame.shape[1] - third else "center"
        dist = "far" if area_ratio < 0.02 else "close" if area_ratio > 0.10 else "medium"
        return {"has_target": True, "direction": d, "distance": dist,
                "target_count": len(targets), "target_confidence": t["confidence"],
                "details": results}

    _FONT = None

    @staticmethod
    def draw_detections(frame, results):
        """绘制检测框+中文标签（Pillow 渲染，解决 OpenCV 不支持中文）"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            # 字体缓存（只加载一次）
            if ObjectDetector._FONT is None:
                for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                           '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']:
                    if os.path.exists(fp):
                        ObjectDetector._FONT = ImageFont.truetype(fp, 18)
                        break
                if ObjectDetector._FONT is None:
                    ObjectDetector._FONT = ImageFont.load_default()

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            for obj in results:
                x1, y1, x2, y2 = obj["bbox"]
                c = (0, 255, 0) if obj["label"] == "person" else (255, 255, 0)
                draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
                label = f"{obj['label']} {obj['confidence']:.2f}"
                bbox = draw.textbbox((0, 0), label, font=ObjectDetector._FONT)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=c)
                draw.text((x1 + 2, y1 - th - 2), label, fill=(0, 0, 0), font=ObjectDetector._FONT)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            # 降级：OpenCV 原生绘制（中文显示为 ???）
            for obj in results:
                x1, y1, x2, y2 = obj["bbox"]
                c = (0, 255, 0) if obj["label"] == "person" else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"{obj['label']} {obj['confidence']:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            return frame
