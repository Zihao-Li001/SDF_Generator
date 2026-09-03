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

### 2.2 Voxel 样本生成主线与数学原理

本节只描述论文中使用的 **voxel 生成流水线**。在该流水线中，SDF 不是主线步骤；项目先采样几何与流动参数，再用 SHPSG 生成球谐系数，将系数映射为三角网格顶点，按入射角旋转后保存 STL，最后把 STL 体素化为固定分辨率的二值 voxel。

#### 2.2.1 参数采样

`ParameterSampler` 从配置文件中的 `GEOM_PARAM_RANGES` 与 `FLOW_PARAM_RANGES` 读取几何参数和流动参数范围。几何参数当前按字典顺序组织为 `aspect_ratio`, `d2`, `d9`，流动参数为 `incident_angle`, `reynolds_number`。采样器支持 `random`, `lhs`, `low_re_dense`, `grid` 四类模式，其中非 `grid` 模式会先在单位超立方体 $[0,1]^p$ 中生成样本，再线性缩放到物理参数范围：

$$
q_i = q_{i,\min} + u_i\left(q_{i,\max}-q_{i,\min}\right),\qquad u_i\in[0,1].
$$

其中 $q_i$ 表示任一几何或流动参数。`lhs` 使用拉丁超立方采样，`low_re_dense` 对几何参数仍使用 LHS，但对流动参数采用低雷诺数加密策略；`grid` 模式用于显式给定网格取值。

#### 2.2.2 SHPSG 如何生成球谐系数

SHPSG 的目标是为三维表面坐标分别生成一组球谐展开系数：

$$
\mathbf{c}_{n,m}=\left(c^x_{n,m}, c^y_{n,m}, c^z_{n,m}\right),\qquad n=0,1,\ldots,N,\quad m=-n,\ldots,n.
$$

当前实现中 $N=15$，因此系数数组长度为 $(N+1)^2=256$，每个系数含 $x,y,z$ 三个复数分量。SHPSG 首先根据长宽比 `Ar` 构造基础椭球。代码令初始主轴为 $(a,b,c)=(Ar,1,1)$，再乘以

$$
s = Ar^{-1/3}
$$

得到归一化主轴

$$
a'=Ar\,s=Ar^{2/3},\qquad b'=c'=s=Ar^{-1/3}.
$$

因此 $a'b'c'=1$，即基础椭球按等体积思想归一化；`Ar` 只改变主轴比例，不改变归一化后的参考体积。随后，代码用一个“单位直径球”对应的低阶系数模板 `fvec_sphere` 构造 $0$ 阶与 $1$ 阶系数块 `Fvec`，并用 $a'$ 缩放 $x$ 方向、用 $b'$ 缩放 $y,z$ 方向，由此得到基础椭球的主形状系数。

在基础椭球之外，`d2` 与 `d9` 控制不规则形变强度。代码先计算基础系数的范数 $d_1$，再把输入参数 `D2_8` 与 `D9_15` 换算为两个谱带的目标幅值：

$$
D_2 = d_{2\_8}\,d_1\left[\sum_{k=2}^{8}\left(\frac{2}{k}\right)^{1.387}\right]^{-1},
$$

$$
D_9 = d_{9\_15}\,d_1\left[\sum_{k=9}^{15}\left(\frac{9}{k}\right)^{1.426}\right]^{-1}.
$$

然后，SHPSG 按幂律衰减把 $D_2$ 分配到低阶形变谱带，把 $D_9$ 分配到高阶形变谱带。低阶谱带代表较长波长的整体起伏，高阶谱带代表较短波长的表面纹理。随机复系数矩阵 $P$ 提供每个阶次和方向的随机相位/方向；最终代码把每一阶随机系数归一化后乘以目标幅值，使该阶球谐描述子的强度等于预设值。

> 论文记号说明：当前代码入口使用 `d2` 和 `d9`；SHPSG 内部参数名是 `D2_8` 与 `D9_15`。如果论文中把高阶纹理参数写作 $d_{9\_18}$，应说明它在本文语义上表示“从第 9 阶开始的高阶纹理谱带强度”；但本仓库当前实现的最高阶为 $N=15$，实际代码名仍是 `d9` / `D9_15`。

#### 2.2.3 如何由球谐系数生成实体几何

几何生成器先构造一个单位球面基底网格：从正二十面体开始，按配置中的 `mesh_level` 做三角面细分，每次把边中点投影回单位球面，然后把所有基底顶点转换为球坐标 $(\theta,\phi)$。这一步只提供采样方向，不直接决定最终颗粒半径。

给定 SHPSG 输出的系数后，实体表面的每个采样方向通过三维球谐展开得到笛卡尔坐标：

$$
x(\theta,\phi)=\sum_{n=0}^{N}\sum_{m=-n}^{n} c^x_{n,m}Y_n^m(\theta,\phi),
$$

$$
y(\theta,\phi)=\sum_{n=0}^{N}\sum_{m=-n}^{n} c^y_{n,m}Y_n^m(\theta,\phi),
$$

$$
z(\theta,\phi)=\sum_{n=0}^{N}\sum_{m=-n}^{n} c^z_{n,m}Y_n^m(\theta,\phi).
$$

代码取上述复数结果的实部作为最终顶点坐标，并沿用基底网格的三角面连接关系形成闭合表面网格。随后，流水线按 `incident_angle` 绕 $y$ 轴旋转该几何，保存为 STL；若启用 `--enable-voxel`，则把 STL 读入 `VoxelProcessor`，先平移到质心，再以 `reference_extent / resolution` 为体素尺寸进行体素化、填充内部、裁剪/补零居中，最后保存为固定尺寸二值 voxel。

#### 2.2.4 三类论文数据集的生成原理

1. **纯椭球数据集：`Ar` 控制形状，`incident_angle` 控制旋转角度**

   纯椭球数据集把不规则扰动关闭，即令 `d2 = 0` 且 `d9 = 0`。此时 SHPSG 只保留由 `Ar` 生成的基础椭球系数：`Ar=1` 时为球形，`Ar>1` 时主轴沿 $x$ 方向拉长，`Ar<1` 时主轴沿 $x$ 方向压缩。由于主轴经过 $Ar^{-1/3}$ 等体积归一化，不同 `Ar` 样本主要体现形状比例差异，而不是整体体积差异。`incident_angle` 不改变几何本体，只在写出样本前绕 $y$ 轴旋转顶点，用于模拟颗粒相对来流的不同姿态。

2. **光滑不规则数据集：`Ar` 控制主尺寸，`d2_8` 控制大尺度变形**

   光滑不规则数据集在基础椭球上打开低阶扰动，通常令 `d9 = 0`，并在给定范围内采样 `d2`（论文中可记为 $d_{2\_8}$）。`d2` 被换算为 $D_2$ 后分配到 2--8 阶球谐系数。低阶球谐函数空间频率较低，对应颗粒表面的长波长变形，因此主要产生整体性的鼓包、凹陷、弯曲或非轴对称轮廓。由于没有高阶纹理项，这类颗粒一般保持相对光滑，适合表示“整体不规则但表面不尖锐”的形状族。

3. **极端纹理不规则数据集：在 `Ar` 与 `d2_8` 之外，用高阶 `d9_18`/`d9` 谱带控制小尺度起伏和尖锐细节**

   极端纹理数据集同时使用基础椭球、低阶大尺度变形和高阶纹理扰动。`Ar` 决定等体积归一化后的主轴比例，`d2` / $d_{2\_8}$ 决定整体轮廓偏离椭球的程度，`d9` 决定第 9 阶以上高阶谱带的强度。高阶球谐函数具有更短的空间波长，因此会在低阶轮廓上叠加局部褶皱、密集凹凸、尖锐突起等细节。当 `d2` 与 `d9` 同时增大时，形状会从光滑不规则进一步过渡到强纹理甚至极端怪异的颗粒。

### 2.3 Python 包依赖

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

