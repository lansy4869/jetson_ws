#!/usr/bin/env python3
"""Step 4: MQTT server for online deployment (GPU side)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util

import paho.mqtt.client as mqtt
import torch


def _load_module(module_filename: str, module_name: str):
    root = Path(__file__).resolve().parent
    module_path = root / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_MOD = _load_module("2_train_diffusion.py", "train_diffusion_module")
INFER_MOD = _load_module("3_composition_filter.py", "composition_filter_module")

RSTP_Diffuser = TRAIN_MOD.RSTP_Diffuser
ddim_compose_inference = INFER_MOD.ddim_compose_inference
safety_filter = INFER_MOD.safety_filter


class RSTPMQTTServer:
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        sub_topic: str = "f1tenth/perception",
        pub_topic: str = "f1tenth/control_command",
        n_retry: int = 3,
    ):
        self.root = Path(__file__).resolve().parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_retry = n_retry
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic

        self.model_st = RSTP_Diffuser().to(self.device)
        self.model_dy = RSTP_Diffuser().to(self.device)
        self.model_st.load_state_dict(torch.load(self.root / "weights" / "model_st.pth", map_location=self.device))
        self.model_dy.load_state_dict(torch.load(self.root / "weights" / "model_dy.pth", map_location=self.device))
        self.model_st.eval()
        self.model_dy.eval()

        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker_host, broker_port, 60)

    def on_connect(self, client, userdata, flags, rc):  # type: ignore[no-untyped-def]
        if rc == 0:
            print(f"Connected to MQTT broker. Subscribing: {self.sub_topic}")
            client.subscribe(self.sub_topic)
        else:
            print(f"MQTT connect failed with rc={rc}")

    def _parse_obstacles(self, data: Dict[str, Any]) -> List[Tuple[float, float]]:
        static_pts = data.get("obs_static_points", [])
        dynamic_pts = data.get("obs_dynamic_points", [])
        all_pts = static_pts + dynamic_pts
        return [(float(pt[0]), float(pt[1])) for pt in all_pts if len(pt) >= 2]

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def on_message(self, client, userdata, msg):  # type: ignore[no-untyped-def]
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            obs_st = self._to_tensor(data["obs_static"])
            obs_dy = self._to_tensor(data["obs_dynamic"])

            if obs_st.shape[0] == 1:
                obs_st = obs_st.repeat(8, 1, 1)
            if obs_dy.shape[0] == 1:
                obs_dy = obs_dy.repeat(8, 1, 1)

            obstacles = self._parse_obstacles(data)

            final_traj: Optional[Any] = None
            for _ in range(self.n_retry):
                batch_trajs = ddim_compose_inference(
                    self.model_st,
                    self.model_dy,
                    obs_st,
                    obs_dy,
                    n_filter=8,
                    device=self.device,
                )
                final_traj = safety_filter(batch_trajs, obstacles)
                if final_traj is not None:
                    break

            if final_traj is not None:
                publish_data = {
                    "success": True,
                    "traj_x": final_traj[0].tolist(),
                    "traj_y": final_traj[1].tolist(),
                }
                print("Planning success: published safe trajectory.")
            else:
                publish_data = {"success": False, "traj_x": [], "traj_y": []}
                print("Planning failed: published emergency stop.")

            client.publish(self.pub_topic, json.dumps(publish_data))
        except Exception as exc:
            err = {"success": False, "traj_x": [], "traj_y": [], "error": str(exc)}
            client.publish(self.pub_topic, json.dumps(err))
            print(f"Message handling failed: {exc}")

    def spin(self) -> None:
        print("RSTP MQTT server started, waiting for perception messages...")
        self.client.loop_forever()


def main() -> None:
    server = RSTPMQTTServer()
    server.spin()


if __name__ == "__main__":
    main()
