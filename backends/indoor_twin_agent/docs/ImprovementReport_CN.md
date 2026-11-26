# IndoorTwin Improved — 完整代码与真实实验结果

## 1. 这次做了什么

在 GPT/Kimi 版 `IndoorTwin_Prefab_Refinement` 的代码基础上继续优化，而不是回到 V37/V39 的 topdown 逻辑。保留了 GPT 版的多视角观测、米制抬升、7 维位姿拟合、支撑约束和投影复核，新增了五层约束与资产认知。

### 关键改动

| 问题 | 根因 | 本次实现 |
|---|---|---|
| 位姿穿模、物品解体 | `fit_one` 只约束整体投影框；实例之间没有 AABB/OBB 拒绝；collection 资产可能有多个部件仍被整体实例化 | `build_twin.py` 新增 `AssemblyValidation.json`：逐实例世界 AABB、实例间相交、同资产近距离重复、窗尺寸检查；新增距离 0.38 m 的重复实例裁剪 |
| 绿色瑜伽球/长凳重复 | `general_plan` 的观测合并阈值 `min(d)*0.77` 过小 | 合并阈值改为 `max(.48, min(d)*1.05)`，并在 Blender 层再做同原型近距离去重 |
| 木质工位隔断被蓝色软屏障替换 | `CATS['partition']='barrier'`，`TEMPLATES['barrier']=FabricBarrier`，检索没有 scene/role 负向门禁 | 语义拆成 `partition` / `barrier`；新增 `WoodPartition` 程序化原型；`scene_compatible()` 按 `scene_types + spatial_roles + negative_roles + instance_unit` 硬过滤；office 场景下 `barrier_concrete` 直接拒绝 |
| 窗户建模错误 | 旧 `WindowBay` 实际是一个不透光的 `DaylightPanel + Blind` 代理，不是窗；窗口尺寸没有独立校验 | `WindowBay` 重写为铝框 + 玻璃 + 竖梃 + 横档 + 窗台，本地包围盒 1.49×0.20×2.33 m；`AssemblyValidation` 检查缩放后的窗宽/高 |
| 资产库没有认知内容 | 只有 USD/catalog/缩略图；GPT `Catalog` 只有 `semantic + 尺寸 + 材质` | 新增 `cognitive_catalog.json`（88 条）、别名/场景适用性/空间角色/支撑能力/材质族/负向用途/compound 标记/来源/质量；`catalog.find_candidates()` 用这些字段做硬门禁 |


### v3.3 新增公开成品办公桌（针对视频桌面不一致）

从公开 CC0 源新增 3 个真实尺寸办公桌，并写入名录、认知层、缩略图和 `AssetBank`：

| asset_id | 来源 | 尺寸 (W×D×H, m) | 说明 |
|---|---|---|---:|
| `office_desk_white_a_frame` | 3DAssets.dev, CC0 | 1.80 × 0.80 × 0.735 | 白色层压板台面 + 浅灰 A 型钢腿，最接近 desk 视频的白色长桌 |
| `office_desk_grab3d` | Grab3D, CC0 | 1.60 × 0.8045 × 0.74 | 蜂蜜木台面 + 炭灰雪橇腿 + 薄荷抽屉 |
| `standing_desk_grab3d` | Grab3D, CC0 | 1.40 × 0.7517 × 1.10 | 蜂蜜木台面 + 白色伸缩腿，升降工位 |

同时移除了 `desk → OfficeDesk` 的强制原生命名模板，改为让 desk 进入公共资产检索；`catalog.py` 对 desk 增加 **finish-first** 排序：在尺寸门禁通过后先按材质颜色接近观测白色/灰色，再按尺寸排序，避免蜂蜜木桌仅因尺寸略近而抢走白色桌。

### 资产库

- 清理库现在 **88 个资产**（83 个外部清理资产 + 2 个 CC0 玩具 + 3 个 CC0 办公桌：`office_desk_white_a_frame`、`office_desk_grab3d`、`standing_desk_grab3d`）。
- 每个资产有 USD、catalog、`asset_meta.json`、三视图缩略图、SQLite 索引；新增 `cognitive_catalog.json`。
- `AssetBank/` 是 GPT `Catalog` 所需格式：85 个 `AT_<hash24>/Prototype.blend` + `Asset.usda` + `Metadata.json` + `Thumbnail.jpg` + SQLite。
- 认知字段示例：
  - `scene_types`: office / lab / household / living / large / kids
  - `spatial_roles`: floor / tabletop / wall / divider / seating / supports_tabletop
  - `negative_roles`: 例如蓝色软屏障在 office 场景禁止充当 `office_divider`；window 禁止被 door/monitor/cabinet 替代
  - `instance_unit`: `single` / `compound_set`（如积木套装、抱枕套装）
  - `aliases`: 中英文检索别名（如 `partition` / `工位隔断` / `木隔断`）

## 2. 真实实验结果

五个样本：lab（我们的 `seeker_cropbot`，SHA256 与 GPT case 完全一致）、TUM `room`、`desk`、`household`、`large`（pioneer_slam3）。  
lab/room/desk/household 的观测阶段复用了原 GPT 包中同一视频的真实 Kimi `Observations.json`；large 只拿到前 2 帧的真实观察响应（第 3 帧 API 断连），因此结果标记为 `partial_observations`。

| case | 观测实例 | 最终实例 | REUSE | BUILD | 重复实例 | 窗尺寸异常 | AABB 相交对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lab | 23 | 96 | 7 | 17 | 0 | 0 | 27 |
| room | 35 | 42 | 8 | 22 | 0 | 0 | 5 |
| desk | 29 | 36 | 6 | 22 | 0 | 0 | 4 |
| household | 31 | 43 | 6 | 27 | 0 | 0 | 15 |
| large (partial) | 9 | 26 | 0 | 13 | 0 | 0 | 0 |

五个案例的窗尺寸校验现在全部为 0；room 已用修正阈值重跑，保持原 GPT 的 `office` 场景语义（不是 `room`，否则会被场景门禁误拒绝）。

### 与 GPT 原结果的对照

| case | GPT REUSE/BUILD | 本次 REUSE/BUILD | GPT 报告实例数 | 本次实例数 |
|---|---|---|---|---:|---:|
| lab | 16 / 5 | 7 / 17 | 58 | 96 |
| room | 23 / 8 | 8 / 22 | 44 | 42 |
| desk | 18 / 7 (+1 variant) | 6 / 22 | 33 | 36 |
| household | 25 / 6 (+1 variant) | 6 / 27 | 53 | 43 |

本次 REUSE 比例刻意降低：离线检索只允许通过认知门禁和尺寸门禁的公共资产，不再让 VLM 自由把 `barrier` 映射成蓝色软屏障。**资产复用率不是越高越好**，错误复用（问题 2/3）正是上一版的核心缺陷。需要更高复用率时，启用在线 `match` 复核即可，但门禁仍然生效。

## 3. 交付物

```
Code/                        完整可运行源码（非补丁）
AssetBank/                   88 资产、GPT Catalog 格式（Prototype.blend + Metadata.json + SQLite）
Cases/<case>/Runs/improved_v2/  Scene.usda、IndoorTwin.blend、AssetManifest.json、
                              AssemblyValidation.json、Retrieval.json、RunSummary.json、Previews/
Reports/SCENE_OVERVIEWS.jpg   五场景俯视拼图
Reports/SCENE_METRICS.json    机器可读指标
AssetBankManifest.json        资产库认知字段快照
```

## 4. 运行

```bash
blender=/path/to/blender
python3 Code/agent.py \
  --video Cases/lab/Scene/Reference/source_video.mp4 \
  --trajectory Cases/lab/Inputs/trajectory.json \
  --intrinsics Cases/lab/Inputs/intrinsics.json \
  --pointcloud Cases/lab/Inputs/cloud.ply \
  --catalog AssetBank --output Cases/lab/Runs/new \
  --scene-type lab --frames 9 --samples 40 \
  --blender "$blender" \
  --endpoint https://ai.zj-computility.com/maas/v1 --model kimi-k2.6
```

离线复现（不调用模型做检索/布局，但保留已存观测）：

```bash
python3 Code/agent.py ... --resume-observations --offline-retrieval --skip-review
```

## 5. 仍未解决的问题（不要当成已解决）

1. **AABB 相交仍是粗筛**：lab 27 对、household 15 对。其中多数是连续 `FabricBarrier` 段之间的 0.06 m³ 接触、桌面物体与支撑家具的包含关系、以及 AABB 对斜放家具的高估；但没有做 OBB/网格级窄相位，因此不能宣称“零穿模”。`AssemblyValidation.json` 将这些显式标成 `review`，没有隐藏。
2. **large 只有 2 帧真实观察**：第 3 帧 API 远端断连，结果只能作为管线在稀疏观测下的鲁棒性样例，不能和四案例同口径比较。
3. **窗户仍是观测框驱动**：已修正几何和尺寸校验，但没有从墙面点云独立检测窗洞；遮挡或反光时会退化。
4. **认知库的“是否应入库”仍是规则评分**：当前用 `semantic novelty + 观察帧数 + 尺寸合法 + 许可证 + 场景覆盖` 做建议，不是训练出来的分类器。
5. **纹理**：`Prototype.blend` 以相对路径引用 `asset_lab/office_library_clean/usd/textures`；单独移动 AssetBank 而不带 clean 库会丢失贴图。打包库时两者必须一起交付。
