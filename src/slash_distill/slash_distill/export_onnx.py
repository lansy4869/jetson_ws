"""W7：把 BC/DAgger 学生 ckpt 导出为 ONNX（再用 build_trt.sh 转 TensorRT）。

无状态、无控制流 → 直接 torch.onnx.export，TRT 友好。

用法：
  python3 -m slash_distill.export_onnx --ckpt ckpt/student_best.pt --out ckpt/student.onnx
"""
import argparse
import os
import numpy as np


def main(argv=None):
    import torch
    from .models.student_cnn import StudentCNN

    ap = argparse.ArgumentParser(description="W7 ONNX 导出")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mc = ckpt["model_cfg"]; meta = ckpt["meta"]
    model = StudentCNN(**mc); model.load_state_dict(ckpt["state_dict"]); model.eval()

    C = mc["in_channels"]; B = mc["n_beams"]; S = mc["scalar_dim"]
    out = args.out or (os.path.splitext(args.ckpt)[0] + ".onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    dummy_l = torch.zeros(1, C, B, dtype=torch.float32)
    dummy_s = torch.zeros(1, S, dtype=torch.float32)
    torch.onnx.export(
        model, (dummy_l, dummy_s), out,
        input_names=["lidar", "scalar"], output_names=["action"],
        opset_version=args.opset,
        dynamic_axes={"lidar": {0: "batch"}, "scalar": {0: "batch"}, "action": {0: "batch"}})
    print(f"[export] 写出 {out}  (lidar[1,{C},{B}] scalar[1,{S}] -> action[1,2])")
    # 元数据同写，部署节点据此构造观测维度
    np.savez(os.path.splitext(out)[0] + "_meta.npz", **meta)
    print(f"[export] meta -> {os.path.splitext(out)[0]}_meta.npz : {meta}")

    try:
        import onnx
        m = onnx.load(out); onnx.checker.check_model(m)
        print("[export] onnx.checker 通过")
    except ImportError:
        print("[export] 未装 onnx，跳过 checker（pip install onnx 可校验）")
    # 数值一致性自检（torch vs onnxruntime）
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        l = np.random.randn(1, C, B).astype(np.float32)
        s = np.random.randn(1, S).astype(np.float32)
        o_ort = sess.run(None, {"lidar": l, "scalar": s})[0]
        with torch.no_grad():
            o_t = model(torch.from_numpy(l), torch.from_numpy(s)).numpy()
        err = float(np.max(np.abs(o_ort - o_t)))
        print(f"[export] onnxruntime vs torch 最大误差 = {err:.2e} ({'OK' if err < 1e-4 else '偏大'})")
    except ImportError:
        print("[export] 未装 onnxruntime，跳过数值自检")


if __name__ == "__main__":
    main()
