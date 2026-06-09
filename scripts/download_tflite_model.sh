#!/bin/bash
# ═══════════════════════════════════════════════
# 下载 TFLite 目标检测模型
# ═══════════════════════════════════════════════
# 可选模型（按精度从低到高）:
#   1. mobilenet_v1    老模型，快但精度差 (5MB)
#   2. mobilenet_v2    精度较好，速度快 (6MB)  ← 默认
#   3. efficientdet_l0 精度最好，稍慢   (6MB)
# ═══════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# 默认模型 (efficientdet_l0, 精度最好)
MODEL="${1:-efficientdet_l0}"

# 可用模型（已确认的下载地址）:
#   mobilenet_v1     → 老模型，快但精度差 (5MB)  ✅ 可下载
#   efficientdet_l0  → 精度最好，稍慢   (6MB)   ✅ 可下载  ← 默认

case "$MODEL" in
  mobilenet_v1)
    MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    ZIP_FILE="/tmp/tflite_mobilenet_v1.zip"
    ;;
  efficientdet_l0)
    MODEL_URL="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
    ZIP_FILE=""
    ;;
  *)
    echo "未知模型: $MODEL，可选: mobilenet_v1, efficientdet_l0"
    exit 1
    ;;
esac

echo "=== 下载 TFLite 模型 ==="
echo "模型: $MODEL"
echo "目标: $MODELS_DIR/"
echo ""

if [ "$MODEL" = "efficientdet_l0" ]; then
    # 直接下载 .tflite 文件
    echo "[1/2] 下载 efficientdet_lite0 (~6MB)..."
    wget -O "$MODELS_DIR/efficientdet_lite0.tflite" "$MODEL_URL"
    echo "[2/2] 复制为 detect.tflite"
    cp "$MODELS_DIR/efficientdet_lite0.tflite" "$MODELS_DIR/detect.tflite"
else
    if [ ! -f "$ZIP_FILE" ]; then
        echo "[1/3] 下载模型 (~6MB)..."
        wget -O "$ZIP_FILE" "$MODEL_URL"
    else
        echo "[1/3] ZIP 已存在，跳过下载"
    fi

    echo "[2/3] 解压到 $MODELS_DIR/"
    rm -f "$MODELS_DIR/detect.tflite" "$MODELS_DIR/labelmap.txt"
    unzip -o "$ZIP_FILE" -d "$MODELS_DIR/"
fi

# 统一标签文件
if [ ! -f "$MODELS_DIR/labelmap.txt" ]; then
    echo "[info] 生成默认 COCO labelmap..."
    cat > "$MODELS_DIR/labelmap.txt" << 'LMAP'
???
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
???
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
???
backpack
umbrella
???
???
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
???
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
???
dining table
???
???
toilet
???
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
???
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
LMAP
fi

# 3. 安装 TFLite 运行时
echo "[3/3] 检查 TFLite 运行时..."

ARCH=$(uname -m)
OS=$(uname -s)

if python3 -c "import tflite_runtime" 2>/dev/null || python3 -c "import tensorflow.lite" 2>/dev/null; then
    echo "  ✓ 已安装"
elif [ "$OS" = "Linux" ] && [ "$ARCH" = "aarch64" -o "$ARCH" = "armv7l" ]; then
    echo "  树莓派 → pip3 install tflite-runtime"
    pip3 install tflite-runtime
elif [ "$OS" = "Darwin" ]; then
    echo "  macOS → 跳过安装"
else
    echo "  尝试 pip3 install tensorflow..."
    pip3 install tensorflow 2>/dev/null || echo "  ⚠ 请手动安装"
fi

echo ""
echo "=== 完成 ==="
ls -lh "$MODELS_DIR/detect.tflite" 2>/dev/null || echo "⚠ detect.tflite 未找到"
echo ""
echo "用法:"
echo "  bash scripts/download_tflite_model.sh                    # 默认 efficientdet_l0"
echo "  bash scripts/download_tflite_model.sh efficientdet_l0   # 精度最好 (推荐)"
echo "  bash scripts/download_tflite_model.sh mobilenet_v1      # 老模型(快但差)"
echo ""
echo "运行 demo4: python3 Demo/demo_04_object_tracking.py"
