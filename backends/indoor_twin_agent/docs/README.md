# IndoorTwin Improved

这是在 `IndoorTwin_Prefab_Refinement`（GPT/Kimi 版）代码上的完整改造版，不是 diff 补丁。

## 目录

- `Code/`：完整 Python/Blender 源码
- `AssetBank/`：88 个资产的 GPT Catalog 格式库（`index.sqlite3`、`AT_*/Prototype.blend`、`Metadata.json`、`Thumbnail.jpg`）
- `Cases/<case>/Runs/improved_v2/`：真实运行结果
  - `Scene/Scene.usda`：从评估后的 Blender 场景直接序列化，避免二次 USD 发布丢纹理/改位姿
  - `Scene/AssemblyValidation.json`：穿模/解体/窗尺寸/重复实例检查
  - `Scene/Retrieval.json`：每个 query 的 REUSE/BUILD 及门禁原因
  - `Scene/RunSummary.json`：汇总
  - `Scene/Previews/`：同机位渲染与 overview
- `Reports/ImprovementReport_CN.md`：问题、改动、指标、残余风险
- `Reports/SCENE_METRICS.json`：机器可读指标
- `AssetScripts/`：玩具与办公桌入库、认知目录、AssetBank 构建脚本
- `Reports/SCENE_OVERVIEWS.jpg`：五场景俯视拼图

## 两条运行路径

### 在线（VLM 检索 + layout）

设置 `KIMI_API_KEY`，模型建议 `kimi-k2.6`（`doubao-seed-2-0-pro-260215` 在长 `max_tokens` 下会被网关断连）。

```bash
python3 Code/agent.py \
  --video Cases/lab/Scene/Reference/source_video.mp4 \
  --trajectory Cases/lab/Inputs/trajectory.json \
  --intrinsics Cases/lab/Inputs/intrinsics.json \
  --pointcloud Cases/lab/Inputs/cloud.ply \
  --catalog AssetBank --output Cases/lab/Runs/new \
  --scene-type lab --frames 9 --samples 40 \
  --blender /path/to/blender \
  --endpoint https://ai.zj-computility.com/maas/v1 --model kimi-k2.6
```

### 离线（复现已有观测，不调用检索/layout）

```bash
python3 Code/agent.py \
  ... \
  --resume-observations --offline-retrieval --skip-review
```

`--offline-retrieval` 只使用认知门禁与尺寸门禁选择公共资产；未命中的对象用测量包围盒生成真实场景 mesh。  
`--resume-observations` 要求 `Runs/<job>/Observations.json` 和 `ModelLogs/*observe.response.json` 已存在。

## 资产库

`AssetBank` 由 `asset_lab/build_gpt_asset_bank.py` 从 clean 库生成。  
`Prototype.blend` 的贴图以相对路径引用 `asset_lab/office_library_clean/usd/textures`；交付 AssetBank 时需同时交付 clean 库，或把贴图 pack 进 blend。

## 已知限制

- `AssemblyValidation.json` 的 AABB 相交是粗筛，不是网格级穿模证明；当前状态为 `review`。
- large 样本只有 2 帧真实观察响应（第 3 帧 API 断连），作为稀疏观测鲁棒性样例。
- 窗户几何已修正，但仍由观测 portal 驱动，没有独立点云窗洞检测。
