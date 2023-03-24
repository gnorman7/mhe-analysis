import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pathlib import Path


if __name__ == "__main__":
    label = 26259
    # prefix = 'F63-slices_280_to_670-min_peak_dist-6'
    # stl_dir_path = Path(f'data/{prefix}_STLs')
    # stl_path = Path(stl_dir_path) / f'{prefix}_{label}.stl'
    stl_path = Path(
        f'data\F63-280_to_670-d6-{label}\{label}-simplified-10_tris.stl')
    if not stl_path.exists():
        raise ValueError(f'File not found: {stl_path}')
    stl_mesh = o3d.io.read_triangle_mesh(str(stl_path))
    # Repair mesh
    stl_mesh.remove_duplicated_vertices()
    # stl_mesh.remove_degenerate_triangles()
    # stl_mesh.remove_duplicated_triangles()
    # stl_mesh.remove_non_manifold_edges()
    print(f'{stl_mesh.is_watertight()=}')
    stl_mesh.compute_triangle_normals()
    stl_mesh.compute_vertex_normals()
    # Color mesh
    # color = [1.0, 0.7, 0.0]
    color = plt.cm.tab10.colors[0]
    stl_mesh.paint_uniform_color(color)
    o3d.visualization.draw_geometries(
        [stl_mesh], mesh_show_wireframe=True, mesh_show_back_face=True,
        width=720, height=720, left=200, top=80,
        # Unpack dictionary copied to clipboard when Ctrl+C pressed in Open3D
		**{
			"front" : [ -0.28710527399055802, -0.57792682317004995, -0.76391828666905226 ],
			"lookat" : [ 6498.0376251764392, 6962.9188992380205, 3387.9976439617676 ],
			"up" : [ 0.022861471423419905, -0.80140045452610331, 0.59769111136916819 ],
			"zoom" : 0.69999999999999996
		}
    )