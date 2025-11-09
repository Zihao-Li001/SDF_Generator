# representation/funcs.py
"""
A collection of functions for generating and manipulating 3D particle geometries,
primarily for use with Spherical Harmonics (SH) expansions.
This is a refactored and corrected version.
"""
import numpy as np
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation as R
import stl
from stl import mesh


def sph2cart_sh(coeffs, phi, theta):
    """
    (Corrected Version)
    Calculates Cartesian coordinates (x, y, z) from spherical harmonic coefficients.
    """
    x, y, z = (
        np.zeros_like(phi, dtype=complex),
        np.zeros_like(phi, dtype=complex),
        np.zeros_like(phi, dtype=complex),
    )
    index = 0
    max_n = int(np.sqrt(len(coeffs)) - 1)

    for n in range(max_n + 1):
        for m in range(-n, n + 1):
            if index < len(coeffs):
                Y_nm = sph_harm(m, n, phi, theta)
                x += coeffs[index, 0] * Y_nm
                y += coeffs[index, 1] * Y_nm
                z += coeffs[index, 2] * Y_nm
                index += 1
    return x.real, y.real, z.real


def calculate_sh_vertices(coeffs, base_sph_coords):
    """
    (New function from splitting sh2stl)
    Calculates final vertex positions of a particle using SH coefficients.
    """
    theta, phi = base_sph_coords[:, 4], base_sph_coords[:, 5]
    x, y, z = sph2cart_sh(coeffs, phi, theta)
    return np.vstack([x, y, z]).T


def save_stl_from_data(filepath, vertices, faces):
    """
    (New function from splitting sh2stl)
    Creates and saves an STL mesh from vertex and face data.
    """
    particle_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        particle_mesh.vectors[i] = vertices[f]
    particle_mesh.name = "particle"
    particle_mesh.save(filepath, mode=stl.Mode.ASCII)


def icosahedron():
    """Defines the vertices and faces of a unit icosahedron."""
    t = (1 + np.sqrt(5)) / 2
    v = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ]
    )
    v = v / np.linalg.norm(v[0, :])
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )
    return v, f


def car2sph(xyz):
    """Converts Cartesian coordinates to spherical coordinates."""
    pts_new = np.zeros((xyz.shape[0], 6))
    pts_new[:, :3] = xyz
    xy_sq = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    pts_new[:, 3] = np.sqrt(xy_sq + xyz[:, 2] ** 2)
    pts_new[:, 4] = np.arctan2(np.sqrt(xy_sq), xyz[:, 2])  # theta (polar)
    pts_new[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])  # phi (azimuthal)
    return pts_new


def subdivided_mesh(vertices, faces):
    """
    (A more robust implementation)
    Subdivides the triangle faces of a mesh, creating a smoother surface.
    This version avoids side effects and is more efficient by caching midpoints.
    """
    midpoint_cache = {}  # 使用字典作为缓存，key是边的元组，value是新顶点的索引
    new_faces = []
    new_vertices = list(vertices)  # 将原始顶点复制到新列表中

    for face in faces:
        v_indices = []
        # 遍历一个面的三条边
        for i in range(3):
            p1_idx = face[i]
            p2_idx = face[(i + 1) % 3]  # 使用 % 确保循环 (v1-v2, v2-v3, v3-v1)

            # --- 核心：缓存逻辑 ---
            edge = tuple(sorted((p1_idx, p2_idx)))  # 标准化边的表示方式
            if edge in midpoint_cache:
                # 如果这个边的中点已经计算过，直接从缓存获取其索引
                midpoint_idx = midpoint_cache[edge]
            else:
                # 如果是新的边，则计算中点
                p1 = vertices[p1_idx]
                p2 = vertices[p2_idx]
                pm = (p1 + p2) / 2.0
                pm = pm / np.linalg.norm(pm)  # 投影到单位球上

                # 将新顶点添加到列表中，并获取其索引
                new_vertices.append(pm)
                midpoint_idx = len(new_vertices) - 1
                midpoint_cache[edge] = midpoint_idx  # 将新中点的索引存入缓存

            v_indices.append(midpoint_idx)

        # 用原始顶点和新的中点创建4个新面
        p1_idx, p2_idx, p3_idx = face
        m1_idx, m2_idx, m3_idx = v_indices[0], v_indices[1], v_indices[2]

        new_faces.extend(
            [
                [p1_idx, m1_idx, m3_idx],
                [p2_idx, m2_idx, m1_idx],
                [p3_idx, m3_idx, m2_idx],
                [m1_idx, m2_idx, m3_idx],
            ]
        )

    return np.array(new_vertices), np.array(new_faces)


def get_midpoint(p1_idx, p2_idx, vertices, cache):
    # Helper for subdivision.
    # Returns index of a midpoint, creating it if needed
    edge = tuple(sorted((p1_idx, p2_idx)))
    if edge in cache:
        return cache[edge][0]

    p1 = vertices[p1_idx]
    p2 = vertices[p2_idx]
    pm = (p1 + p2) / 2
    pm = pm / np.linalg.norm(pm)

    new_idx = len(vertices) + len(cache)
    cache[edge] = (new_idx, pm)
    return new_idx


def cleanmesh(f, v):
    # remove duplicate vertices
    v, AC, TC = np.unique(v, return_index=True, return_inverse=True, axis=0)
    # reassign faces to trimmed vertex list
    for i in range(len(f)):
        for j in range(3):
            f[i, j] = TC[f[i, j]]
    return v, f


def rotate_vertices(vertices, angle_deg, axis="y"):
    """
    Rotates a set of vertices around a specified axis
    Args:
        vertices: vertex data, shape(N, 3)
        angle_deg: rotation angle in degrees
        axis: the axis to rotate, in this paper is around 'y' or 'z'
    """
    angle_rad = np.deg2rad(angle_deg)
    rotation = R.from_euler(axis, angle_rad)
    rotated_vertices = rotation.apply(vertices)
    return rotated_vertices


def add_gaussian_noise(vertices, scale=0.01):
    # Add Gaussian noise to the vertices of mesh
    noise = np.random.normal(0, scale, vertices.shape)
    return vertices + noise
