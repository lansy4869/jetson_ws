"""学生策略封装（离线 torch 版）：加载 ckpt，提供 act(lidar, scalar)->(steer, speed) 原始指令。

部署节点（student_policy_node）用 onnxruntime/TensorRT，不依赖本模块；
本模块给 eval_closed_loop / dagger 用，避免重复反归一化逻辑。
"""
import numpy as np

from .student_cnn import StudentCNN, denormalize_action


class TorchStudentPolicy:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        import torch
        self.torch = torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.meta = ckpt["meta"]
        mc = ckpt["model_cfg"]
        self.model = StudentCNN(**mc)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.model.to(self.device)
        self.steer_max = float(self.meta["steer_max"])
        self.v_max = float(self.meta["v_max"])

    def act(self, lidar: np.ndarray, scalar: np.ndarray):
        torch = self.torch
        with torch.no_grad():
            lt = torch.as_tensor(lidar[None], dtype=torch.float32, device=self.device)
            st = torch.as_tensor(scalar[None], dtype=torch.float32, device=self.device)
            out = self.model(lt, st).cpu().numpy()[0]
        steer, speed = denormalize_action(float(out[0]), float(out[1]), self.steer_max, self.v_max)
        return steer, speed
