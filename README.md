# EgoNavForge

MMLab 式：**模块注册 + YAML 工作流**。换数据只改 dataset 绑定；换算法只换/加模块。

- 框架设计：[`docs/OPEN_SOURCE_FRAMEWORK.md`](docs/OPEN_SOURCE_FRAMEWORK.md)
- 大模块与 I/O：[`docs/MODULES.md`](docs/MODULES.md)

## 原则

1. 工作流由 YAML 的 stage→modules 列表决定。  
2. 大模块是注册类（`type:`），彼此只通过 `EpisodeContext` 对接。  
3. 用户加新方法：写一个 `@register("MyXxx")` 类，挂进 YAML——不必 fork 整条流水线。  
4. `backends/` 只有权重、third_party，以及可选的 vendored agent（如室内孪生），**不是**工作流脚本目录。

## 布局

```text
configs/
  workflows/pi3_moge_lerobot.yaml   # 共享工作流（含可选 IndoorTwinScene）
  datasets/*.yaml                   # 只绑数据 + 开关
src/ego_nav_forge/
  runner.py / registry.py / context.py
  modules/                          # 用户可见的大模块（含 scene/IndoorTwinScene）
backends/
  checkpoints/                      # Pi3 / MoGe 权重
  indoor_twin_agent/                # 资产库检索孪生 agent（替代 v37）
  office_library_clean/            # 外部资产 USD 与缩略图
  asset_bank/                      # 检索库，由 office_library_clean 生成
docs/MODULES.md
```

## 运行

```bash
cd /mnt/xxxxx/releases/ego_nav_forge_v1
pip install -e .

# 原有：Pi3+MoGe → LeRobot
python -m ego_nav_forge run -c configs/datasets/umi_single_video_imu.yaml
python -m ego_nav_forge run -c configs/datasets/threeeyes_three_video_imu.yaml

# 场景孪生（Pi3+MoGe + 资产库 IndoorTwinScene）
export zhesuan_API_KEY='...'   # 或 MOONSHOT_API_KEY；也可在 YAML 写 api_key
python -m ego_nav_forge run -c configs/datasets/seeker_scene_twin.yaml --max-eps 1

# 冒烟（复用已有 suite，只检查轨迹和内参，不调用模型）
python -m ego_nav_forge run -c configs/datasets/seeker_scene_twin_smoke.yaml --max-eps 1

# 零费用输入检查
python -m ego_nav_forge run -c configs/datasets/seeker_scene_twin_check.yaml --max-eps 1
```

可选环境变量：

| 变量 | 作用 |
|------|------|
| `EGO_NAV_FORGE_PI3_ROOT` | `pi3_moge_lerobot` 根目录（默认同级 sibling） |
| `EGO_NAV_FORGE_TWIN_ROOT` | indoor_twin_agent 根（默认 `backends/indoor_twin_agent`） |
| `EGO_NAV_FORGE_ASSET_BANK` | 检索库目录（默认 `backends/asset_bank`） |
| `zhesuan_API_KEY` / `MOONSHOT_API_KEY` | 孪生 VLM（OpenAI 兼容） |
| `TWIN_API_BASE` / `TWIN_MODEL` | API 地址与模型名默认值 |
| `BLENDER` | Blender 可执行文件 |

## 场景孪生（finalize → `IndoorTwinScene`）

在共享工作流的 **finalize** 阶段增加可选模块 `IndoorTwinScene`（默认 `enabled: false`）。  
这一版用资产库检索 agent 替换原来的 v37。数据集 YAML 用 `dataset.scene_twin` 打开：

```yaml
dataset:
  lerobot_export: false
  scene_twin:
    enabled: true
    api_base: https://xxxxx/v1
    api_key_env: API_KEY
    model: kimi-k2.6
    blender: /path/to/blender
    calib_camera: bleft
    use_pointcloud: true       # 有 PLY 就传给 agent；没有则只用视频、轨迹、内参
    scene_type: lab
    frames: 9                  # 观察帧数，不是渲染帧数
    samples: 40
```

流水线效果：

1. **Pi3XRecon + MoGe**：米制轨迹，并在 `rgb_moge/points_metric_textured.ply` 写出点云。  
2. **IndoorTwinScene**：视频 + 轨迹（OpenCV c2w，时间对齐到视频 PTS）+ 内参，点云可选 → `Scene/IndoorTwin.blend` 和 `Scene/Scene.usda`。

产物目录：`work_dir/episodes/<ep>/scene_twin/`。场景文件在 `Scene/IndoorTwin.blend`。

资产库第一次使用前，用 Blender 从 `backends/office_library_clean` 生成 `backends/asset_bank`：

```bash
blender -b --python backends/indoor_twin_agent/build_asset_bank.py -- \
  --lib backends/office_library_clean --out backends/asset_bank
```

## 配置对照

| 配置 | 用途 |
|------|------|
| `configs/datasets/seeker_scene_twin.yaml` | **用户正式跑**：与 threeeyes 同类数据绑定 + 全量孪生 |
| `configs/datasets/seeker_scene_twin_smoke.yaml` | **冒烟**：复用 suite + `seed_project` 只重建 Blender/USD |
| `configs/datasets/seeker_scene_twin_check.yaml` | 只检查视频/轨迹/内参，不调 API/Blender |
| `configs/datasets/threeeyes_three_video_imu.yaml` | 原 LeRobot 流程（孪生关闭） |
