#!/usr/bin/env bash
# W7：ONNX -> TensorRT engine（Jetson 本机 trtexec）+ 时延基准。
#
# 用法： bash build_trt.sh ckpt/student.onnx [ckpt/student.engine]
#
# 说明：
#   - trtexec 通常在 /usr/src/tensorrt/bin/trtexec（JetPack 自带）。
#   - FP16 在 Orin 上明显更快；学生网络很小，时延应 < 5 ms（目标 < 20ms / 50Hz）。
#   - 部署节点用 onnxruntime 的 TensorrtExecutionProvider 跑同一 .onnx 即可获得加速；
#     独立 .engine 主要用于离线时延基准与极限性能确认。
set -e

ONNX="${1:-ckpt/student.onnx}"
ENGINE="${2:-${ONNX%.onnx}.engine}"

TRTEXEC="$(command -v trtexec || true)"
if [ -z "$TRTEXEC" ]; then
  for c in /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec; do
    [ -x "$c" ] && TRTEXEC="$c" && break
  done
fi
if [ -z "$TRTEXEC" ]; then
  echo "[build_trt] 找不到 trtexec。JetPack 一般在 /usr/src/tensorrt/bin/trtexec"
  exit 1
fi

echo "[build_trt] trtexec = $TRTEXEC"
echo "[build_trt] $ONNX -> $ENGINE (FP16)"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --minShapes=lidar:1x1x108,scalar:1x11 \
  --optShapes=lidar:1x1x108,scalar:1x11 \
  --maxShapes=lidar:8x1x108,scalar:8x11 \
  --workspace=512

echo "[build_trt] 时延基准："
"$TRTEXEC" --loadEngine="$ENGINE" --fp16 --iterations=200 --avgRuns=200 \
  --shapes=lidar:1x1x108,scalar:1x11 2>/dev/null | grep -Ei "mean|median|GPU Compute" || true
echo "[build_trt] 完成 -> $ENGINE"
