import os
import argparse
import numpy as np
import trimesh
import skimage.measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the project-level runner.
    """
    parser = argparse.ArgumentParser(
        description="Check and export the voxelized geometry and STL mesh"
    )

    parser.add_argument(
        "--path",
        default="sample.npy",
        help="File path to the voxel file",
    )
    parser.add_argument(
        "--mode",
        choices=["surface", "boxes"],
        default="surface",
        help="Display mode: 'surface' (marching cubes) or 'boxes' (raw voxels)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated mesh in an interactive window",
    )
    return parser.parse_args()


# --- Configuration ---
CONTOUR_LEVEL = 0.5  # surface of SDF or density threshold
# 定义三个不同的视角 (elev: 仰角, azim: 方位角)
VIEWS = [
    {"name": "iso", "elev": 30, "azim": 45},  # 等高轴测图
    # {"name": "front", "elev": 0, "azim": 0},  # 正视图
    # {"name": "top", "elev": 90, "azim": 0},  # 俯视图
]
# --- End Configuration ---


def setup_transparent_ax(fig):
    """辅助函数：初始化无背景、无边框的 3D 坐标轴"""
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    return ax


def set_equal_aspect_limits(ax, vertices):
    """
    现代优化版辅助函数：紧凑包围盒 + 真实比例映射
    彻底避免旧版强行填充正方体导致的“巨幅空白/模型显小”问题
    """
    # 1. 计算顶点在 X, Y, Z 三轴上的真实物理跨度 (Extents: [Lx, Ly, Lz])
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    extents = max_bound - min_bound

    # 2. 为坐标轴加上 2% 的微小留白 (Margin)，防止模型边缘贴着图表框
    margin = extents * 0.02
    ax.set_xlim(min_bound[0] - margin[0], max_bound[0] + margin[0])
    ax.set_ylim(min_bound[1] - margin[1], max_bound[1] + margin[1])
    ax.set_zlim(min_bound[2] - margin[2], max_bound[2] + margin[2])

    # 3. 【核心魔法】直接将 3D 视盒的长宽高比例设置为几何真实跨度！
    # 这与体素模式下的 ax.set_box_aspect(filled.shape) 逻辑完全统一
    ax.set_box_aspect(extents)


def export_pdfs(voxels, vertices, faces, mode, base_filename):
    """
    使用 Matplotlib 渲染并导出 Voxel/Marching Cubes 的无背景 PDF 图
    """
    print("正在生成体素/等值面不同角度的无背景 PDF 图片...")

    for view in VIEWS:
        fig = plt.figure(figsize=(8, 8))
        ax = setup_transparent_ax(fig)

        if mode == "surface" and vertices is not None:
            mesh_col = Poly3DCollection(
                vertices[faces],
                alpha=1.0,
                facecolors="#2b8cbe",
                edgecolors="#103648",
                # edgecolors=None,
                linewidths=0.08,
                shade=True,
            )
            # mesh_col.set_antialiased(False)
            ax.add_collection3d(mesh_col)
            set_equal_aspect_limits(ax, vertices)

        else:
            filled = voxels > CONTOUR_LEVEL
            # 8FB7C9
            # 9FC7B5
            # D7B38A
            # B7A9C9

            # 9BB7C9 蓝灰
            # A8C3B0 绿灰
            # CBB79E 米棕
            # B8AEC7 紫灰

            ax.voxels(
                filled,
                # facecolors="#8FB7C9",
                # edgecolors="#6F95A6",
                # facecolors="#9FC7B5",
                # edgecolors="#7EA08F",
                # facecolors="#D7B38A",
                # edgecolors="#B08D67",
                facecolors="#B7A9C9",
                edgecolors="#9183A6",
                # facecolors="#f9f871",
                # edgecolors="#e0dc68",
                # edgecolors=None,
                linewidth=0.18,
                alpha=1.00,
                shade=True,
            )
            nx, ny, nz = filled.shape[2], filled.shape[1], filled.shape[0]
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_zlim(0, nz)

            # 使用与 surface 模式一致的物理比例（假设体素各向同性）
            ax.set_box_aspect((nx, ny, nz))

        ax.view_init(elev=view["elev"], azim=view["azim"])

        output_pdf = f"{base_filename}_{view['name']}.pdf"
        plt.savefig(
            output_pdf,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=300,
        )
        plt.close(fig)
        print(f" -> 已保存: {output_pdf}")


def export_stl_pdfs(stl_mesh, base_filename):
    """
    针对同名 STL 文件，使用相同 VIEWS 渲染并导出无背景 PDF 图
    """
    print("正在生成同名 STL 网格不同角度的无背景 PDF 图片...")

    for view in VIEWS:
        fig = plt.figure(figsize=(8, 8))
        ax = setup_transparent_ax(fig)

        # 提取 STL 的顶点与面片
        mesh_col = Poly3DCollection(
            stl_mesh.vertices[stl_mesh.faces],
            alpha=0.88,
            facecolors="#2b8cbe",  # 保持与表面一样的配色
            edgecolors="#103648",  # 深蓝轮廓线
            linewidths=0.2,
        )
        ax.add_collection3d(mesh_col)

        # 调整包围盒比例与视角
        set_equal_aspect_limits(ax, stl_mesh.vertices)
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # 加上 _stl 后缀防止与 .npy 导出的表面冲突
        output_pdf = f"{base_filename}_stl_{view['name']}.pdf"
        plt.savefig(
            output_pdf,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=300,
        )
        plt.close(fig)
        print(f" -> 已保存: {output_pdf}")


# --- Main Visualization Logic ---
if __name__ == "__main__":
    args = parse_args()
    SDF_FILE_PATH = args.path
    base_name = os.path.splitext(SDF_FILE_PATH)[0]

    # 1. 处理 .npy 体素文件
    if not os.path.exists(SDF_FILE_PATH):
        print(f"Error: File not found at '{SDF_FILE_PATH}'")
    else:
        print(f"Loading {SDF_FILE_PATH}...")
        voxels = np.load(SDF_FILE_PATH)
        print(f"Loaded voxel grid with shape: {voxels.shape}")

        vertices, faces = None, None
        mesh = None

        if args.mode == "surface":
            print("Running Marching Cubes to extract surface...")
            vertices, faces, normals, _ = skimage.measure.marching_cubes(
                voxels, level=CONTOUR_LEVEL
            )
            print(
                f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces."
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            print("Converting directly to voxel boxes...")
            voxel_grid = trimesh.voxel.VoxelGrid(voxels > CONTOUR_LEVEL)
            mesh = voxel_grid.as_boxes()

        # 导出 .npy 对应的 PDF
        export_pdfs(voxels, vertices, faces, args.mode, base_name)

    # 2. 自动检测并处理同名 .stl 文件
    stl_path = f"{base_name}.stl"
    if os.path.exists(stl_path):
        print(f"\n检测到同名 STL 文件: {stl_path}，正在读取...")
        try:
            # 使用 trimesh 加载 STL 文件
            stl_mesh = trimesh.load(stl_path)
            if isinstance(stl_mesh, trimesh.Scene):
                # 如果 STL 被识别为 Scene，将其转换为单一实体 Mesh
                stl_mesh = stl_mesh.dump(concatenate=True)

            print(
                f"STL 加载成功: {len(stl_mesh.vertices)} 顶点, {len(stl_mesh.faces)} 面片。"
            )
            export_stl_pdfs(stl_mesh, base_name)
        except Exception as e:
            print(f"读取或渲染 STL 文件时出错: {e}")
    else:
        print(f"\n未在路径下检测到同名 STL 文件 ({stl_path})，跳过 STL 导出。")

    # 3. 是否显示交互式窗口 (仅展示 .npy 生成的 mesh)
    if args.show and mesh is not None:
        print("\nDisplaying interactive window...")
        mesh.show()
