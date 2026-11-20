# V26：轨迹—户型刚体+尺度配准版

## 核心修复
1. **新增 `registration.py`**：把 PI3 重力对齐的轨迹，用旋转 θ + 统一尺度 s + 平移 (tx,ty) 配准到模型生成的 footprint。评分同时考虑轨迹点是否全在多边形内、最小墙距、首帧相机是否看向房间内部、门锚点距离和尺度惩罚；配准结果写入 `PATH_REGISTRATION.json`，并进入 `BLUEPRINT.json`。
2. **同轨迹渲染真正使用配准后的轨迹**：`pipeline.py` 把 `path` 写入 `SCENE_PLAN.json`，`bs_trajvideo._path()` 优先读取该轨迹；不再在渲染器里把轨迹重新归一化或把相机推到同一个屏障边缘。
3. **相机姿态**：不再把 PI3 旋转后的世界 Z 当俯仰（会出现仰头拍天花板）；只用原采集的 pitch，水平方向由配准后的 yaw 决定。相机眼高取原始 `pos[2]`（1.0–1.8 m），不再出现相机在负 Z 地板下。
4. **计数去重**：`build_index()` 中全局 count 改为权威覆盖，不再用 `max()`；修复全局 `sofa=1` 被 segment 内的 `2` 覆盖导致的重复沙发。
5. **长凳回补**：`O11_Table` 的 A 形腿 + 横梁结构被重新标注为“长凳（A 形腿长桌）”，进入 `bench` 类别；避免把同一视频里的长条凳漏掉。
6. **蓝色屏障数量/位置**：屏障作为“周界/隔断网络”在放置阶段优先保留，8 个面板不再被 free_floor 挤到只剩 2 个；每个 3/3/2 组锚定在配准轨迹的侧向偏移位置，不再一律贴墙。`SCENE_PLAN` 保留 8 个实例。
7. **材质**：沿用 V25 对 desk/table 的表面颜色覆盖，避免整张办公桌被“金属框架”描述判成黑色金属。

## 入口
```bash
export zhesuan_API_KEY=...
python twin.py --video room.mp4 --trajectory trajectory.json \
  --intrinsics K.json --out /runs/room01 --frames 100 --res 384 --fps 5
```
用户只需要视频、轨迹、内参和输出目录；`calibration/SCENE_PLAN/ARCHITECTURE/BLUEPRINT/specs` 全部由管线生成。

## V26 真实实验产物
- `v26/project/PATH_REGISTRATION.json`
- `v26/project/ARCHITECTURE.json`：L 形，5 窗 1 门
- `v26/project/SCENE_PLAN.json`：14 节点 / 21 实例，屏障 8
- `v26/project/RECON_EVIDENCE.json`：沙发 1、长凳 1、屏障 8
- `v26/project/previews/final_top.png`
- `v26/project/previews/final_showcase.png`
- `v26/project/previews/final_compare.mp4`
