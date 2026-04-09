import open3d as o3d
import numpy as np
from pathlib import Path
from pipeline.base import Stage


class MeshingStage(Stage):
    """Mesh generation using Poisson Surface Reconstruction."""

    def __init__(self, depth: int = 9, crop_bottom_percentile: float = 20.0, radial_percentile: float = 90.0):
        super().__init__("Meshing")
        self.depth = depth
        self.crop_bottom_percentile = crop_bottom_percentile
        self.radial_percentile = radial_percentile

    def _detect_orientation(self, points: np.ndarray, vert_axis: int = 1) -> bool:
        """Returns True if mesh is inverted (shoulders at top, head at bottom)."""
        y = points[:, vert_axis]
        bottom = points[y < np.percentile(y, 20)]
        top = points[y > np.percentile(y, 80)]

        if len(bottom) < 50 or len(top) < 50:
            return False

        def horizontal_area(pts):
            xz = pts[:, [0, 2]]
            return float(np.prod(xz.max(axis=0) - xz.min(axis=0) + 1e-6))

        def horizontal_radius(pts):
            xz = pts[:, [0, 2]]
            return float(np.percentile(np.linalg.norm(xz - np.median(xz, axis=0), axis=1), 80))

        inverted = horizontal_area(top) > horizontal_area(bottom) and horizontal_radius(top) > horizontal_radius(bottom)
        return inverted

    def run(self, context: dict) -> dict:
        point_cloud_path = Path(context["point_cloud"])
        work_dir = Path(context["work_dir"])

        mesh_dir = work_dir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        raw_mesh_path = mesh_dir / "raw_mesh.ply"

        pcd = o3d.io.read_point_cloud(str(point_cloud_path))
        if len(pcd.points) < 100:
            raise RuntimeError("Point cloud too small for meshing.")

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(pcd.points)

        inverted = self._detect_orientation(points)

        if self.crop_bottom_percentile > 0:
            if inverted:
                threshold = np.percentile(points[:, 1], 100 - self.crop_bottom_percentile)
                mask = points[:, 1] < threshold
            else:
                threshold = np.percentile(points[:, 1], self.crop_bottom_percentile)
                mask = points[:, 1] > threshold

            pcd = pcd.select_by_index(np.where(mask)[0])
            points = np.asarray(pcd.points)
            if len(pcd.points) < 500:
                raise RuntimeError("Point cloud too small after crop.")

        if self.radial_percentile > 0:
            center = np.median(points, axis=0)
            distances = np.linalg.norm(points[:, [0, 2]] - center[[0, 2]], axis=1)
            mask = distances < np.percentile(distances, self.radial_percentile)
            pcd = pcd.select_by_index(np.where(mask)[0])
            points = np.asarray(pcd.points)
            if len(pcd.points) < 500:
                raise RuntimeError("Point cloud too small after radial filter.")

        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(diag * 0.01, 0.005), max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, width=0, scale=1.1, linear_fit=False
        )

        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities < np.percentile(densities, 5))

        if len(mesh.triangles) == 0:
            raise RuntimeError("Mesh became empty after density filtering.")

        mesh = mesh.crop(bbox.scale(1.05, bbox.get_center()))

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        tc, cn, _ = mesh.cluster_connected_triangles()
        tc, cn = np.asarray(tc), np.asarray(cn)
        if len(cn) > 1:
            mesh.remove_triangles_by_mask(tc != cn.argmax())
            mesh.remove_unreferenced_vertices()

        if len(mesh.triangles) == 0:
            raise RuntimeError("Mesh became empty after cleanup.")

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh)

        context["raw_mesh"] = str(raw_mesh_path)
        context["raw_mesh_num_faces"] = len(mesh.triangles)
        context["raw_mesh_num_vertices"] = len(mesh.vertices)
        return context
