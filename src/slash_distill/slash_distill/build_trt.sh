#!/usr/bin/env bash
# W7：ONNX -> TensorRT engine（Jetson 本机 trtexec）+ 时延基准。
#
# 用法： bash build_trt.sh ckpt/student.onnx [ckpt/student.engine] \
#           [--channels C --beams B --scalar S] [--batch-max 8]
#
# 说明：
#   - trtexec 通常在 /usr/src/tensorrt/bin/trtexec（JetPack 自带）。
#   - FP16 在 Orin 上明显更快；学生网络很小，时延应 < 5 ms（目标 < 20ms / 50Hz）。
#   - 部署节点用 onnxruntime 的 TensorrtExecutionProvider 跑同一 .onnx 即可获得加速；
#     独立 .engine 主要用于离线时延基准与极限性能确认。
#   - 输入 shape 不再写死 1x1x108 / 1x11：优先从 ONNX 模型本身读取真实输入维度
#     （lidar[*,C,B] / scalar[*,S]），读不到再回退到同目录 {stem}_meta.npz
#     （n_layers/n_beams/scalar_dim），最后回退默认 C=1 B=108 S=11。
#     这样 3 层实车学生或改了 n_beams/use_opponent 的模型都能正确构建。
set -e

ONNX="${1:-ckpt/student.onnx}"
shift || true
ENGINE=""

# 解析剩余位置参数与 --flags
BATCH_MAX=8
OVR_C="" OVR_B="" OVR_S=""
while [ $# -gt 0 ]; do
  case "$1" in
    --channels) OVR_C="$2"; shift 2;;
    --beams)    OVR_B="$2"; shift 2;;
    --scalar)   OVR_S="$2"; shift 2;;
    --batch-max) BATCH_MAX="$2"; shift 2;;
    *)
      if [ -z "$ENGINE" ]; then ENGINE="$1"; shift;
      else echo "[build_trt] 未知参数: $1" >&2; exit 1; fi ;;
  esac
done
[ -n "$ENGINE" ] || ENGINE="${ONNX%.onnx}.engine}"

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

# ---- 推断输入 shape：C (lidar 通道) / B (n_beams) / S (scalar 维度) ----
META_NPZ="${ONNX%.onnx}_meta.npz"
read -r C B S <<EOF
$(python3 - "$ONNX" "$META_NPZ" "$OVR_C" "$OVR_B" "$OVR_S" <<'PY'
import sys, os
onnx_path, meta_path, oc, ob, os_ = sys.argv[1:6]
C = B = S = None
# 1) 优先从 ONNX 本身读真实输入 shape
try:
    import onnx
    m = onnx.load(onnx_path)
    for inp in m.graph.input:
        sh = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if len(sh) == 3 and C is None:
            C, B = sh[1], sh[2]
        elif len(sh) == 2 and S is None:
            S = sh[1]
except Exception:
    pass
# 2) 回退到 {stem}_meta.npz（n_layers/n_beams/scalar_dim）
if (C is None or B is None or S is None) and os.path.exists(meta_path):
    try:
        import numpy as np
        d = dict(np.load(meta_path))
        if C is None: C = int(d.get("n_layers", d.get("in_channels", 1)))
        if B is None: B = int(d.get("n_beams", 108))
        if S is None: S = int(d.get("scalar_dim", 11))
    except Exception:
        pass
# 3) 默认
if C is None: C = 1
if B is None: B = 108
if S is None: S = 11
# 4) 命令行覆盖优先级最高
if oc: C = int(oc)
if ob: B = int(ob)
if os_: S = int(os_)
print(C, B, S)
PY
)
EOF

echo "[build_trt] trtexec = $TRTEXEC"
echo "[build_trt] $ONNX -> $ENGINE (FP16)"
echo "[build_trt] lidar  shape = (N, ${C}, ${B})"
echo "[build_trt] scalar shape = (N, ${S})"

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --minShapes="lidar:1x${C}x${B},scalar:1x${S}" \
  --optShapes="lidar:1x${C}x${B},scalar:1x${S}" \
  --maxShapes="lidar:${BATCH_MAX}x${C}x${B},scalar:${BATCH_MAX}x${S}" \
  --workspace=512

echo "[build_trt] 时延基准："
"$TRTEXEC" --loadEngine="$ENGINE" --fp16 --iterations=200 --avgRuns=200 \
  --shapes="lidar:1x${C}x${B},scalar:1x${S}" 2>/dev/null | grep -Ei "mean|median|GPU Compute" || true
echo "[build_trt] 完成 -> $ENGINE"
