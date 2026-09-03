from __future__ import annotations

import math
import unittest

import numpy as np
import trimesh

from physics.calc_geom_metadata import projected_area
from recompute_crosswise_sphericity import (
    SOURCE_COMPARE_COLUMNS,
    _patch_phi_column,
    _verify_only_phi_changed,
)


def old_surface_integral_area(mesh: trimesh.Trimesh, direction) -> float:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return float(
        0.5
        * np.sum(
            np.abs(mesh.face_normals @ direction)
            * mesh.area_faces
        )
    )


class ProjectedAreaTests(unittest.TestCase):
    def test_convex_box_matches_analytic_area(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        self.assertAlmostEqual(projected_area(mesh, [1.0, 0.0, 0.0]), 12.0, places=12)
        self.assertAlmostEqual(
            projected_area(mesh, [1.0, 0.0, 0.0]),
            old_surface_integral_area(mesh, [1.0, 0.0, 0.0]),
            places=12,
        )

    def test_generic_direction_uses_orthogonal_basis(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        direction = np.array([1.0, 2.0, 3.0])
        unit = direction / np.linalg.norm(direction)
        expected = 12.0 * abs(unit[0]) + 8.0 * abs(unit[1]) + 6.0 * abs(unit[2])
        self.assertTrue(
            math.isclose(projected_area(mesh, direction), expected, rel_tol=1e-12)
        )

    def test_union_removes_complete_occlusion(self):
        front = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
        back = front.copy()
        back.apply_translation([3.0, 0.0, 0.0])
        mesh = trimesh.util.concatenate([front, back])
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(projected_area(mesh, [1.0, 0.0, 0.0]), 6.0, places=12)
        self.assertAlmostEqual(old_surface_integral_area(mesh, [1.0, 0.0, 0.0]), 12.0)

    def test_degenerate_projection_is_rejected(self):
        mesh = trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        with self.assertRaisesRegex(ValueError, "projection|non-degenerate"):
            projected_area(mesh, [1.0, 0.0, 0.0])

    def test_zero_direction_is_rejected(self):
        mesh = trimesh.creation.box()
        with self.assertRaisesRegex(ValueError, "non-zero"):
            projected_area(mesh, [0.0, 0.0, 0.0])


class ExactCsvUpdateTests(unittest.TestCase):
    def test_only_phi_cross_tokens_change(self):
        original = (
            b"sample_id,reference_area,phi_cross,status\r\n"
            b"1001,1.25,0.8,ok\r\n"
            b"1002,1.50,0.7,keep-this\r\n"
        )
        candidate = _patch_phi_column(
            original,
            {0: "0.81234567890123457", 1: "0.75"},
            expected_rows=2,
        )
        _verify_only_phi_changed(original, candidate)
        self.assertIn(b"1001,1.25,0.81234567890123457,ok\r\n", candidate)
        self.assertIn(b"1002,1.50,0.75,keep-this\r\n", candidate)
        self.assertTrue(candidate.endswith(b"\r\n"))

    def test_missing_replacement_is_rejected(self):
        original = b"sample_id,phi_cross\n1,0.5\n2,0.6\n"
        with self.assertRaisesRegex(ValueError, "missing phi_cross replacement"):
            _patch_phi_column(original, {0: "0.7"}, expected_rows=2)


class MappingIdentityTests(unittest.TestCase):
    def test_target_column_is_not_used_as_geometry_identity(self):
        self.assertNotIn("phi_cross", SOURCE_COMPARE_COLUMNS)
        self.assertIn("reference_area", SOURCE_COMPARE_COLUMNS)
        self.assertIn("volume", SOURCE_COMPARE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
