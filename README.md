# SDF_Generator

一个用于**三维非球形颗粒数据集生成**的项目：
- 先用球谐参数生成颗粒几何（STL）。
- 再按流动条件（入射角、雷诺数）扩展样本。
- 可选生成体素（voxel）和有符号距离场（SDF）。
- 同时输出 `metadata.csv`，包含几何参数、流动参数、路径和派生物理量（例如阻力系数）。

---

## 1. 这个项目做了什么？

核心入口是 `generate_data.py`，它调用 `pipeline.run_dataset_generation(...)` 启动整套流程。流程包含：

1. **参数采样**：
   - 几何参数：`aspect_ratio`, `d2`, `d9`
   - 流动参数：`incident_angle`, `reynolds_number`
   - 支持 `random / lhs / low_re_dense / grid` 等采样模式。

2. **几何生成**：
   - 用球谐形状生成器（`representation/SHPSG.py` + `representation/geom_generator.py`）构建颗粒表面网格。
   - 先创建基底网格，再根据参数生成具体颗粒。

3. **样本扩展**：
   - 每个几何形状对应多组流动条件。
   - 每组流动条件会对同一几何做旋转，形成不同样本。

4. **数据产物写出**：
   - STL：始终生成。
   - Voxel：通过 `--enable-voxel` 开启。
   - SDF：通过 `--enable-sdf` 开启。

5. **元数据与派生字段**：
   - 每个样本写入一行 `metadata.csv`。
   - 自动计算几何指标（体积、等效直径、球形度等）与阻力相关输出（`Cd_ke`, `Cd_hs`）。

项目内还提供了独立转换脚本：
- `mesh_2_sdf.py`：把已有 STL 批量转为 SDF。
- `mesh_2_voxel.py`：把已有 STL 批量转为体素。

---

## 2. 生成数据依赖于什么？

### 2.1 数学与物理建模依赖

1. **几何表示依赖**
   - 基于球谐系数的形状生成（SHPSG），主要由 `Ar`, `d2`, `d9` 控制。

2. **采样策略依赖**
   - `ParameterSampler` 负责参数空间采样，支持 LHS（拉丁超立方）和低雷诺数加密策略。

3. **派生物理量依赖**
   - 几何量：体积、投影面积、球形度等。
   - 阻力模型：
     - Ke 相关式
     - Hölzer-Sommerfeld 相关式

### 2.2 Python 包依赖

建议至少安装以下包（按代码导入整理）：

- `numpy`
- `pandas`
- `trimesh`
- `pyDOE2`
- `tqdm`

> 说明：
> - 项目中 `config` 通过 `from config import CONFIG` 导入，通常需要你自行准备 `config.py`（可参考 `config_template.py`）。
> - `trimesh` 在某些网格操作场景还可能需要额外后端（如 `rtree` 等）才能获得完整功能。

---

## 3. 目录结构（核心模块）

```text
SDF_Generator/
├─ generate_data.py              # 主入口
├─ config_template.py            # 配置模板（复制为 config.py 使用）
├─ pipeline/
│  ├─ runner.py                  # 主流程编排与并行调度
│  ├─ sample_record.py           # metadata 行构建
│  └─ worker_state.py            # 多进程共享状态
├─ representation/
│  ├─ SHPSG.py                   # 球谐参数形状生成核心
│  ├─ geom_generator.py          # 颗粒几何生成
│  ├─ voxel_generator.py         # STL -> voxel
│  ├─ sdf_generator.py           # STL -> SDF
│  └─ sampling.py                # 参数采样策略
├─ derived_fields/
│  ├─ geometry_metrics.py        # 几何派生量
│  ├─ drag.py                    # 阻力派生量
│  └─ registry.py                # 派生字段注册
├─ physics/
│  ├─ calc_geom_metadata.py      # 几何指标计算
│  └─ drag_coeff.py              # 阻力公式
├─ mesh_2_sdf.py                 # 批量 STL 转 SDF
└─ mesh_2_voxel.py               # 批量 STL 转 voxel
```

---

## 4. 快速开始

### 4.1 准备配置

1. 复制模板配置：

```bash
cp config_template.py config.py
```

2. 按需编辑 `config.py`，重点关注：
- `SAMPLING`：几何数量、每个几何对应流动条件数、采样模式
- `COMPUTATION`：网格精细度、voxel/SDF 分辨率
- `OUTPUT`：输出目录

### 4.2 安装依赖（示例）

```bash
pip install numpy pandas trimesh pyDOE2 tqdm
```

### 4.3 运行数据生成

仅生成 STL + metadata：

```bash
python generate_data.py
```

生成 STL + voxel + metadata：

```bash
python generate_data.py --enable-voxel
```

生成 STL + SDF + metadata：

```bash
python generate_data.py --enable-sdf
```

全部生成（STL + voxel + SDF + metadata）：

```bash
python generate_data.py --enable-voxel --enable-sdf
```

给几何加高斯噪声（可与上面组合）：

```bash
python generate_data.py --noise
```

设置几何 ID 起始值：

```bash
python generate_data.py --geom-id-start 1000
```

---

## 5. 输出结果说明

默认会在 `dataset/` 下输出：

- `dataset/stl/*.stl`
- `dataset/voxel/*.npy`（如果启用 voxel）
- `dataset/sdf/*.npy` 或 `.npy.z`（取决于你使用的脚本）
- `dataset/metadata.csv`

`metadata.csv` 典型字段包括：
- 基础字段：`sample_id`, `geom_id`, `rotate_id`, `aspect_ratio`, `incident_angle`, `Re`, `d2`, `d9`
- 派生字段：`volume`, `equivalent_diameter`, `reference_area`, `sphericity`, `phi_cross`, `Cd_ke`, `Cd_hs`
- 文件路径：`stl_path`, `voxel_path`, `sdf_path`

其中 `sample_id = geom_id * 1000 + rotate_id`。

---

## 6. 注意事项

1. **不要一次性把大规模 STL 全量转 SDF 且分辨率设置过高**，内存压力会非常明显。
2. SDF 与 voxel 分辨率越高，计算时间和内存占用越大。
3. 建议先小规模验证（例如较少几何数 + 较低分辨率），再扩大生成规模。
4. 该项目使用多进程并行，`num_workers`（如果配置中提供）需要根据机器资源合理设置。

---

## 7. 后续可改进方向

- 增加 `requirements.txt` 或 `pyproject.toml` 统一依赖管理。
- 为 `config.py` 增加 schema 校验和更详细错误提示。
- 补充单元测试与最小可复现实验配置。
- 增加数据集可视化与质量检查脚本。

