# RSTP Reproduction Project

This folder contains a runnable reproduction scaffold for:
`Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition`.

## 0) Environment setup

```bash
conda create -n rstp_env python=3.10 -y
conda activate rstp_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install casadi numpy scipy matplotlib paho-mqtt tqdm
```

## 1) Offline MPC dataset generation

```bash
cd src/slash_navigation/roboracer_china_2025/roboracer_china_2025/rstp_project
python3 1_generate_mpc_data.py
```

Saved files:
- `datasets/static/tau.npy`
- `datasets/static/obs.npy`
- `datasets/dynamic/tau.npy`
- `datasets/dynamic/obs.npy`

## 2) Conditional diffusion training (CFG)

```bash
python3 2_train_diffusion.py
```

Saved files:
- `weights/model_st.pth`
- `weights/model_dy.pth`

## 3) Diffusion composition + safety filter

```bash
python3 3_composition_filter.py
```

## 4) Online MQTT server deployment (GPU side)

```bash
python3 4_mqtt_server_node.py
```

Default topics:
- Subscribe: `f1tenth/perception`
- Publish: `f1tenth/control_command`

## Notes

- Trajectory representation uses quaternion components `[qz, qw]` to avoid yaw wrap-around discontinuities.
- DDIM inference uses `8` steps for low latency.
- Composition gains (`nu_st`, `nu_dy`) are sensitive and should be tuned per map/scenario.
