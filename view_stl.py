import numpy as np
import open3d as o3d
from pathlib import Path


if __name__ == "__main__":
    particleID = 1319
    # stl_dir_location = r'C:\Users\gusb\Research\PSAAP\STL-files\F63_300-465'
    # output_filename_base = 'F63_300-465'
    # fn_suffix = '_1-erosions'
    # n_particles_digits = 5
    # stl_path = Path(stl_dir_location) / (
    #     f'{output_filename_base}'
    #     f'_{str(particleID).zfill(n_particles_digits)}'
    #     f'{fn_suffix}.stl'
    # )
    stl_path = Path(
        r"C:\Users\gusb\Research\PSAAP\STL-files"
        r"\F63-300-465_1204_05-simplified_01-iters_285-tris.stl"
    )
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
    color = [1.0, 0.7, 0.0]
    stl_mesh.paint_uniform_color(color)
    o3d.visualization.draw_geometries(
        [stl_mesh], mesh_show_wireframe=True, mesh_show_back_face=True,
        width=720, height=720, left=200, top=80
    )