import open3d as o3d
import numpy as np
import pymeshlab
from pathlib import Path
from pipeline.base import Stage


class PostProcessStage(Stage):
    """Mesh cleanup, shoulder detection, capping and export."""

    def __init__(self, max_faces: int = 50000, fill_holes: bool = True):
        super().__init__("PostProcess")
        self.max_faces = max_faces
        self.fill_holes = fill_holes

    def _detect_shoulder_cut(self, V: np.ndarray, n_slices: int = 50) -> float:
        """Detect Y level where mesh widens (shoulders)."""
        y = V[:, 1]
        ymin, ymax = y.min(), y.max()
        h = ymax - ymin

        widths = []
        y_centers = []

        for i in range(n_slices):
            y_low = ymin + i * h / n_slices
            y_high = ymin + (i+1) * h / n_slices
            band = V[(y > y_low) & (y < y_high)]
            if len(band) > 10:
                center = np.median(band[:, [0, 2]], axis=0)
                dist = np.linalg.norm(band[:, [0, 2]] - center, axis=1)
                width = np.percentile(dist, 90)
            else:
                width = np.nan
            widths.append(width)
            y_centers.append((y_low + y_high) / 2)

        widths = np.array(widths)
        y_centers = np.array(y_centers)

        valid = ~np.isnan(widths)
        if valid.sum() < 10:
            return ymin + 0.3 * h
        widths = np.interp(np.arange(n_slices), np.where(valid)[0], widths[valid])

        widths_smooth = np.convolve(widths, np.ones(5)/5, mode='same')

        diffs = np.diff(widths_smooth)
        start = int(0.15 * n_slices)
        end = int(0.85 * n_slices)
        shoulder_idx = np.argmax(diffs[start:end]) + start + 1
        y_cut = y_centers[shoulder_idx]

        print(f"[PostProcess] Shoulder detected at Y={y_cut:.3f}")
        return y_cut

    def _cut_at_shoulder(self, mesh: o3d.geometry.TriangleMesh, y_cut: float) -> o3d.geometry.TriangleMesh:
        """Cut mesh at shoulder level, keeping head side."""
        V = np.asarray(mesh.vertices)
        T = np.asarray(mesh.triangles)
        original_faces = len(T)

        above = V[V[:, 1] > y_cut]
        below = V[V[:, 1] < y_cut]

        if len(above) > 10 and len(below) > 10:
            c_above = np.median(above[:, [0, 2]], axis=0)
            c_below = np.median(below[:, [0, 2]], axis=0)
            w_above = np.percentile(np.linalg.norm(above[:, [0, 2]] - c_above, axis=1), 90)
            w_below = np.percentile(np.linalg.norm(below[:, [0, 2]] - c_below, axis=1), 90)
            keep_above = w_above < w_below
        else:
            keep_above = True

        print(f"[PostProcess] Keeping {'above' if keep_above else 'below'} Y={y_cut:.3f}")

        tri_y = V[T].mean(axis=1)[:, 1]
        keep_tri = tri_y > y_cut if keep_above else tri_y < y_cut

        mesh_cut = o3d.geometry.TriangleMesh(mesh)
        mesh_cut.remove_triangles_by_mask(~keep_tri)
        mesh_cut.remove_unreferenced_vertices()

        tc, cn, _ = mesh_cut.cluster_connected_triangles()
        tc, cn = np.asarray(tc), np.asarray(cn)
        if len(cn) == 0:
            print("[PostProcess] No components after cut, reverting.")
            return mesh
        mesh_cut.remove_triangles_by_mask(tc != cn.argmax())
        mesh_cut.remove_unreferenced_vertices()

        print(f"[PostProcess] Faces after shoulder cut: {len(mesh_cut.triangles)}")

        # Fallback: if cut removed more than 60% revert
        if len(mesh_cut.triangles) < 0.4 * original_faces:
            print("[PostProcess] Shoulder cut too aggressive, skipping cut.")
            return mesh

        return mesh_cut

    def _cap_boundary(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Cap the largest boundary loop with a fan triangulation using PCA."""
        V = np.asarray(mesh.vertices)
        T = np.asarray(mesh.triangles)

        edge_count = {}
        for tri in T:
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i+1)%3]]))
                edge_count[e] = edge_count.get(e, 0) + 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            print("[PostProcess] No boundary edges found — already closed")
            return mesh

        adj = {}
        for e in boundary_edges:
            adj.setdefault(e[0], []).append(e[1])
            adj.setdefault(e[1], []).append(e[0])

        visited = set()
        loops = []
        for start in adj:
            if start in visited:
                continue
            loop = []
            cur = start
            prev = None
            while True:
                visited.add(cur)
                loop.append(cur)
                neighbors = [n for n in adj[cur] if n != prev]
                if not neighbors or neighbors[0] in visited:
                    break
                prev = cur
                cur = neighbors[0]
            if len(loop) > 2:
                loops.append(loop)

        print(f"[PostProcess] Found {len(loops)} boundary loops")

        if not loops:
            print("[PostProcess] No valid boundary loops found")
            return mesh
        main_loop = max(loops, key=len)
        print(f"[PostProcess] Capping loop with {len(main_loop)} vertices")

        loop_pts = V[main_loop]
        center = loop_pts.mean(axis=0)

        X = loop_pts - center
        _, _, VT = np.linalg.svd(X, full_matrices=False)
        axis1 = VT[0]
        axis2 = VT[1]

        coords_2d = np.column_stack([X @ axis1, X @ axis2])
        angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
        order = np.argsort(angles)
        ordered_verts = [main_loop[i] for i in order]

        n = len(V)
        center_idx = n
        new_V = np.vstack([V, center.reshape(1, 3)])

        cap_tris = []
        for i in range(len(ordered_verts)):
            v1 = ordered_verts[i]
            v2 = ordered_verts[(i+1) % len(ordered_verts)]
            cap_tris.append([center_idx, v1, v2])

        new_T = np.vstack([T, np.array(cap_tris)])

        mesh_capped = o3d.geometry.TriangleMesh()
        mesh_capped.vertices = o3d.utility.Vector3dVector(new_V)
        mesh_capped.triangles = o3d.utility.Vector3iVector(new_T)

        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            center_color = colors[main_loop].mean(axis=0)
            new_colors = np.vstack([colors, center_color.reshape(1, 3)])
            mesh_capped.vertex_colors = o3d.utility.Vector3dVector(new_colors)

        mesh_capped.remove_degenerate_triangles()
        mesh_capped.remove_duplicated_triangles()
        mesh_capped.remove_duplicated_vertices()
        mesh_capped.remove_unreferenced_vertices()
        mesh_capped.compute_vertex_normals()
        print(f"[PostProcess] Watertight: {mesh_capped.is_watertight()}")
        return mesh_capped

    def run(self, context: dict) -> dict:
        raw_mesh_path = Path(context["raw_mesh"])
        output_path = Path(context["output_path"])

        print(f"[PostProcess] Loading mesh: {raw_mesh_path}")
        mesh = o3d.io.read_triangle_mesh(str(raw_mesh_path))
        print(f"[PostProcess] Faces: {len(mesh.triangles)} | Vertices: {len(mesh.vertices)}")

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        print(f"[PostProcess] Faces after cleanup: {len(mesh.triangles)}")

        tc, cn, _ = mesh.cluster_connected_triangles()
        tc, cn = np.asarray(tc), np.asarray(cn)
        if len(cn) > 0:
            mesh.remove_triangles_by_mask(tc != cn.argmax())
            mesh.remove_unreferenced_vertices()
        print(f"[PostProcess] Faces after largest component: {len(mesh.triangles)}")

        V = np.asarray(mesh.vertices)
        y_cut = self._detect_shoulder_cut(V)
        mesh = self._cut_at_shoulder(mesh, y_cut)

        mesh = self._cap_boundary(mesh)

        if self.fill_holes:
            print("[PostProcess] Filling remaining holes...")
            tmp_path = raw_mesh_path.parent / "tmp_prefill.ply"
            tmp_filled = raw_mesh_path.parent / "tmp_filled.ply"
            o3d.io.write_triangle_mesh(str(tmp_path), mesh)

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(tmp_path))
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_null_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_close_holes(maxholesize=500, newfaceselected=False)
            ms.meshing_repair_non_manifold_edges()
            ms.save_current_mesh(str(tmp_filled))

            mesh = o3d.io.read_triangle_mesh(str(tmp_filled))

            if tmp_path.exists(): tmp_path.unlink()
            if tmp_filled.exists(): tmp_filled.unlink()

        if len(mesh.triangles) > self.max_faces:
            print(f"[PostProcess] Decimating to {self.max_faces} faces...")
            mesh = mesh.simplify_quadric_decimation(self.max_faces)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        # Fix orientation: flip Y (upside down) and rotate 180 around Y (back to front)
        V = np.asarray(mesh.vertices)
        V[:, 1] = -V[:, 1]  # flip up/down
        V[:, 0] = -V[:, 0]  # rotate 180 around Y
        V[:, 2] = -V[:, 2]
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])  # flip normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        if len(mesh.triangles) == 0:
            raise RuntimeError("Mesh is empty after post-processing.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_path), mesh)

        context["final_mesh"] = str(output_path)
        context["final_mesh_num_faces"] = len(mesh.triangles)
        context["final_mesh_num_vertices"] = len(mesh.vertices)

        print(f"[PostProcess] Final mesh: {output_path}")
        print(f"[PostProcess] Faces: {context['final_mesh_num_faces']} | Vertices: {context['final_mesh_num_vertices']}")
        print(f"[PostProcess] Watertight: {mesh.is_watertight()}")
        print("[PostProcess] Done.")
        return context
