# 直接使用：视频 + 轨迹 + 内参 -> 孪生

## 一条命令

```bash
export zhesuan_API_KEY='...'
python twin.py \
  --video      /绝对路径/room.mp4 \
  --trajectory /绝对路径/trajectory.json \
  --intrinsics /绝对路径/K_pinhole_pipeline.json \
  --out        /绝对路径/runs/room01
```

不需要手工准备 `calibration.json`、`RECON_EVIDENCE.json`、`ARCHITECTURE.json`、`SCENE_PLAN.json`、`BLUEPRINT.json` 或任何 specs；也不要改源码里的路径。`twin.py` 现在直接调用 `pipeline.py`：

1. 复制输入，`inputs_prep.prepare()` 生成 `calibration.json`（轨迹重力对齐/内参）；
2. `ReconScanner` 多轮 API 认知，生成 `RECON_EVIDENCE.json`（物体、材质、结构、门窗/地面证据）；
3. `floorplan_pass.generate_architecture()` 用连续视频帧 + 重力对齐轨迹 + 门窗证据，**由模型生成** `ARCHITECTURE.json` 的 footprint/层高/窗/门；不再读固定 L 形户型；
4. `auto_scene.auto_plan()` 生成 `SCENE_PLAN.json` + `specs_v22.json`，`blueprint.build_blueprint()` 生成重力俯视 `BLUEPRINT.json`；
5. `blender_build.py` 在 Blender 中建 shell（墙/窗/门/地板/天花/灯/相机）+ 所有资产，并渲染同轨迹 `previews/traj_frames/`；
6. 输出 `scene.blend`、`previews/PATH_QA.json` 和可编辑资产。

## 先零成本检查输入

```bash
python twin.py --video ... --trajectory ... --intrinsics ... --out ... --check
```

只检查路径和 JSON，不调用 API、不启动 Blender，不产生费用。

## 输入约定

- `--video`：`.mp4` / `.mov` 室内走查视频。
- `--trajectory`：相机轨迹 JSON，支持 PI3 或任意 c2w pose；`inputs_prep` 会写 `calibration.json`。
- `--intrinsics`：`fx/fy/cx/cy/W/H` 或 `K` 矩阵；`inputs_prep` 会解析 FOV。
- `--out`：输出目录，自动创建；不要求目录预先存在。

## 输出

```text
out/
└── project/
    ├── inputs/              # 复制后的视频/轨迹/内参
    ├── calibration.json     # 自动生成
    ├── reports/recon/       # RECON_EVIDENCE、帧、原始 floorplan 输出
    ├── ARCHITECTURE.json    # 自动生成（LLM floorplan pass）
    ├── SCENE_PLAN.json
    ├── specs_v22.json
    ├── BLUEPRINT.json
    ├── scene.blend
    ├── assets/person_supplied.glb  # 用户提供的人体模型（CadConvert 从 x_t 转出）
    └── previews/            # final_overview/final_showcase/final_top/traj_frames/final_compare.mp4
```

## 可调参数

```text
--preset fast|balanced|high     质量/耗时
--model kimi-k2.6               API 模型
--api-base ...                  API endpoint
--api-key-env zhesuan_API_KEY   key 的环境变量名
--blender /path/to/blender      不写则自动找 PATH/常见安装路径
--frames 60                     同轨迹对比帧数
--res 384                       渲染分辨率
--fps 5                         对比采样率
```

`--evidence` 只是调试/复现旧认知证据的选项，不是正常用户入口；不传时程序自己调用 API 认知。人体默认使用 `assets/person_supplied.glb`（由用户提供的 SolidWorks/Parasolid 模型转换而来），识别到 person 时放入。

## 说明

`auto_scene_cli.py`、`build_shell.py`、`build_assets.py`、`run_traj.py` 是旧 V20 的**内部确定性阶段**，不是用户入口。新入口 `twin.py` -> `pipeline.py` -> `blender_build.py` 不依赖它们，也不写死项目路径。`ARCHITECTURE.json` 由 `floorplan_pass.py` 在每次运行时通过 API 生成；模型输出不可解析时才使用 `fallback_bbox_from_trajectory` 兜底，并明确标记。
