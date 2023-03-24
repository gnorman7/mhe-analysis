import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pathlib import Path
import sys
# import mesh_ops as ops
# import segment


# To-do: add "-c" flag option for color string

SHOW_WIREFRAME = False

def handle_args(args):
    try:
        if args[0] == '-i':
            if len(args) == 2:
                meshes = load_stl_meshes(
                    args[1],
                    # fn_prefix='cake_0_',
                    # separate_color='cake_0'
                    # iter_size=100,
                )
            else:
                print('Warning: Only "-i" flag allowed.')
        else:
            print('Specify STL directory after "-i" flag.')
    except IndexError:
        print('You must use the "-i" flag to specify STL directory.')
    finally:
        if len(meshes) != 0:
            o3d.visualization.draw_geometries(
                meshes,
                mesh_show_wireframe=SHOW_WIREFRAME,
                mesh_show_back_face=True,
                width=720, height=720, left=200, top=80,
            )
        else:
            print('No meshes loaded.')

def load_stl_meshes(
        stl_dir_path,
        fn_prefix='',
        fn_suffix='',
        particleIDs=None,
        separate_color=None,
        colors='tab10',
        iter_size=1):
    stl_dir_path = Path(stl_dir_path)
    n_digits = 2
    if colors == 'four':
        colors = [
            (1.0, 0.7, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.7, 0.0),
            (0.0, 0.7, 1.0),
        ]
    elif colors == 'tab10':
        colors = plt.cm.tab10.colors
    if particleIDs is not None:
        stl_paths = [
            (
                f'{stl_dir_path}/{fn_prefix}'
                f'{str(particleID).zfill(n_digits)}{fn_suffix}.stl'
            )
            for i, particleID in enumerate(particleIDs)
            if i % iter_size == 0
        ]
    else:
        print(fn_prefix)
        stl_paths = [
            str(path) for i, path in enumerate(stl_dir_path.glob('*.stl'))
            if path.stem.startswith(fn_prefix)
            and path.stem.endswith(fn_suffix)
        ]
        print(len(stl_paths))
    meshes = []
    for i, path in enumerate(stl_paths):
        if Path(path).exists():
            print(f'Loading mesh: {path}')
            # stl_mesh = segment.postprocess_mesh(
            #     path, smooth_iter=1, simplify_n_tris=250, save_mesh=False,
            #     recursive_simplify=True, return_mesh=True, return_props=False
            # )
            # segment.check_properties(stl_mesh)
            # stl_mesh = segment.repair_mesh(stl_mesh)
            stl_mesh = o3d.io.read_triangle_mesh(str(path))
            stl_mesh.compute_triangle_normals()
            stl_mesh.compute_vertex_normals()
            meshes.append(stl_mesh)
        else:
            raise ValueError(f'Path not found: {path}')
    for i, m in enumerate(meshes):
        if separate_color is not None:
            if separate_color in Path(stl_paths[i]).stem:
                color = colors[0]
            else:
                color = colors[1]
        else:
            color = colors[i % len(colors)]
        m.paint_uniform_color(color)
    return meshes

if __name__ == "__main__":
    # wrap_lines_in_file(sys.argv[-1])
    handle_args(sys.argv[1:])

