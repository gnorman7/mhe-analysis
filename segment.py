# Standard library imports
from concurrent.futures import process
from pathlib import Path
# Third-party imports
# import imagecodecs  # dependency required for loading compressed tif images
import imageio as iio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import napari
import numpy as np
from scipy import ndimage as ndi
from skimage import color, feature, filters, morphology, measure, segmentation, util
from stl import mesh


def load_images(
    img_dir,
    slice_crop=None, 
    row_crop=None, 
    col_crop=None, 
    return_3d_array=False, 
    also_return_names=False,
    also_return_dir_name=False,
    convert_to_float=False,
    file_suffix='tif'
):
    """Load images from path and return as list of 2D arrays. Can also return names of images.

    Parameters
    ----------
    img_dir : str or Path
        Path to directory containing images to be loaded.
    slice_crop : list or None
        Cropping limits of slice dimension (imgs.shape[0]) of 3D array of images. Essentially chooses subset of images from sorted image directory. If None, all images/slices will be loaded. Defaults to None.
    row_crop : str or None
        Cropping limits of row dimension (imgs.shape[1]) of 3D array of images. If None, all rows will be loaded. Defaults to None.
    col_crop : str or None
        Cropping limits of column dimension (imgs.shape[2]) of 3D array of images. If None, all columns will be loaded. Defaults to None.
    return_3d_array : bool, optional
        If True, return loaded images as a 3D numpy array, else return images in list, by default False
    also_return_names : bool, optional
        If True, returns a list of the names of the images in addition to the list of images themselves. Defaults to False.
    also_return_dir_name : bool, optional
        If True, returns a string representing the name of the image directory in addition to the list of images themselves. Defaults to False.
    convert_to_float : bool, optional
        If True, convert loaded images to floating point images, else retain their original dtype. Defaults to False
    file_suffix : str, optional
        File suffix of images that will be loaded from img_dir. Defaults to 'tif'

    Returns
    -------
    list, numpy.ndarray, or tuple
        List of arrays or 3D array representing images (depending on return_3d_array), or if also_return_names is True, list containing names of images from filenames is also returned.
    """
    img_path_list = [
        path for path in Path(img_dir).glob(f'*{file_suffix}')
    ]
    img_path_list.sort()
    if slice_crop is None:
        slice_crop = [0, len(img_path_list)]
    img_path_sublist = [
        img_path for i, img_path in enumerate(img_path_list) 
        if i in list(range(slice_crop[0], slice_crop[1]))
    ]
    img = iio.imread(img_path_sublist[0])
    if row_crop is None:
        row_crop = [0, img.shape[0]]
    if col_crop is None:
        col_crop = [0, img.shape[1]]
    imgs = []
    for img_path in img_path_sublist:
        img = iio.imread(img_path)[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]
        if convert_to_float:
            img = util.img_as_float(img)
        imgs.append(img)
    if return_3d_array:
        imgs = np.stack(imgs)
    if also_return_names and also_return_dir_name:
        return imgs, [img_path.stem for img_path in img_path_list], img_dir.stem
    elif also_return_names:
        return imgs, [img_path.stem for img_path in img_path_list]
    elif also_return_dir_name:
        return imgs, img_dir.stem
    else:
        return imgs

def crop_data(imgs, slice_lims=None, row_lims=None, col_lims=None):
    if slice_lims is None:
        slice_lims = [0, imgs.shape[0]]
    if row_lims is None:
        row_lims = [0, imgs.shape[1]]
    if col_lims is None:
        col_lims = [0, imgs.shape[2]]
    return imgs[
        slice_lims[0]:slice_lims[1], 
        row_lims[0]:row_lims[1], 
        col_lims[0]:col_lims[1]
    ]

def save_images(
    imgs,
    save_dir,
    img_names=None,
    convert_to_16bit=False
):
    """Save images to save_dir.

    Parameters
    ----------
    imgs : numpy.ndarray or list
        Images to save, either as a list or a 3D numpy array (4D array of colored images also works)
    save_dir : str or Path
        Path to new directory to which iamges will be saved. Directory must not already exist to avoid accidental overwriting. 
    img_names : list, optional
        List of strings to be used as image filenames when saved. If not included, images will be names by index. Defaults to None.
    convert_to_16bit : bool, optional
        Save images as 16-bit, by default False
    """
    save_dir = Path(save_dir)
    # Create directory, or raise an error if that directory already exists
    save_dir.mkdir(parents=True, exist_ok=False)
    # If imgs is a numpy array and not a list, convert it to a list of images
    if isinstance(imgs, np.ndarray):
        # If 3D: (slice, row, col)
        if len(imgs.shape) == 3:
            file_suffix = 'tif'
            imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        # If 4D: (slice, row, col, channel) where channel is RGB (color) value
        elif len(imgs.shape) == 4:
            file_suffix = 'png'
            imgs = [
                util.img_as_ubyte(imgs[i, :, :, :]) 
                for i in range(imgs.shape[0])
            ]
    for i, img in enumerate(imgs):
        if convert_to_16bit:
            img = img.astype(np.uint16)
        # if no img_names, use the index of the image
        if img_names is None:
            img_name = str(i).zfill(3)
        else:
            img_name = img_names[i]
        iio.imsave(Path(save_dir / f'{img_name}.{file_suffix}'), img)
    print(f'{len(imgs)} image(s) saved to: {save_dir.resolve()}')

def binarize_3d(
    imgs, 
    thresh_val=0.65, 
    fill_holes=64,
    return_process_dict=False
):
    """Creates binary images from list of images using a threshold value.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing the floating point images to be binarized.
    thresh_val : float, optional
        Value to threshold point images. Defaults to 0.65 for floating point images.
    fill_holes : str or int, optional
        If 'all', all holes will be filled, else if integer, all holes with an area in pixels below that value will be filled in binary array/images. Defaults to 64.
    return_process_dict : bool, optional
        If True, return a dictionary containing all processing steps instead of last step only, defaults to False

    Returns
    -------
    numpy.ndarray or dict
        If return_process_dict is False, a 3D array representing the hole-filled, binary images, else a dictionary is returned with a 3D array for each step in the binarization process.
    """
    smoothed = filters.gaussian(imgs)
    binarized = smoothed > thresh_val
    filled = binarized.copy()
    if fill_holes == 'all':
        for i in range((imgs.shape[0])):
            filled[i, :, :] = ndi.binary_fill_holes(binarized[i, :, :])
    else:
        filled = morphology.remove_small_holes(
            binarized, area_threshold=fill_holes
        )
    if return_process_dict:
        process_dict = {
            'binarized' : binarized,
            'holes-filled' : filled
        }
        return process_dict
    else:
        return filled

def binarize_multiotsu(imgs, n_otsu_classes=2, n_selected_thresholds=1):
    """Binarize stack of images (3D array) using multi-Otsu thresholding algorithm.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    n_otsu_classes : int, optional
        Number of classes to threshold images, by default 2
    n_selected_thresholds : int, optional
        Number of classes to group together (from the back of the thresholded values array returned by multi-Otsu function) to create binary image, by default 1

    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of threshold values used to create binarized images 
    """
    imgs_binarized = np.zeros_like(imgs, dtype=np.float32)
    imgs_flat = imgs.flatten()
    imgs_binarized = np.zeros_like(imgs, dtype=np.float32)
    thresh_vals = filters.threshold_multiotsu(imgs_flat, n_otsu_classes)
    imgs_binarized[imgs > thresh_vals[-n_selected_thresholds]] = 1
    return imgs_binarized, thresh_vals

def segment_3d(
    imgs, 
    thresh_val=0.65, 
    fill_holes=64,
    min_peak_distance=30,
    return_process_dict=False
):
    """Create images with regions segmented and labeled using a watershed segmentation algorithm.

    Parameters
    ----------
    binarized_imgs : numpy.ndarray
        3D array representing binary images to be used in segmentation.
    min_peak_distance : int, optional
        Minimum distance (in pixels) of local maxima to be used to generate seeds for watershed segmentation algorithm. Defaults to 30.

    Returns
    -------
    list
        List of 2-D arrays representing the segmented and labeled images.
    """
    binarize_3d_dict = binarize_3d(
        imgs, 
        thresh_val=thresh_val,
        fill_holes=fill_holes,
        return_process_dict=True
    )
    dist_map = ndi.distance_transform_edt(binarize_3d_dict['holes-filled'])
    # Get Nx2 array of N number of (row, col) coordinates
    maxima = feature.peak_local_max(
        dist_map, 
        min_distance=min_peak_distance,
        exclude_border=False
    )
    # Assign a label to each point to use as seed for watershed seg
    maxima_mask = np.zeros_like(binarize_3d_dict['holes-filled'], dtype=float)
    maxima_mask[tuple(maxima.T)] = 1
    seeds = measure.label(maxima_mask)
    labels = segmentation.watershed(
        -1 * dist_map, seeds, mask=binarize_3d_dict['holes-filled']
    )
    colored_labels = color.label2rgb(labels, bg_label=0)
    if return_process_dict:
        process_dict = {
            'raw' : imgs,
            'binarized' : binarize_3d_dict['binarized'],
            'holes-filled' : binarize_3d_dict['holes-filled'],
            'distance-map' : dist_map,
            'maxima-points' : maxima,
            'maxima-mask' : maxima_mask,
            'seeds' : seeds,
            'integer-labels' : labels,
            'colored-labels' : colored_labels
        }
        return process_dict
    else:
        return labels

def watershed_segment(
    imgs_binarized, 
    min_peak_distance=1,
    return_dict=False
):
    """Create images with regions segmented and labeled using a watershed segmentation algorithm.

    Parameters
    ----------
    binarized_imgs : numpy.ndarray
        3D array representing binary images to be used in segmentation.
    min_peak_distance : int or str, optional
        Minimum distance (in pixels) of local maxima to be used to generate seeds for watershed segmentation algorithm. 'median_radius' can be passed to use the radius of the circle with equivalent area to the median binary region. Defaults to 1.

    Returns
    -------
    segment_dict : dict
        Dictionary of 2-D arrays the segmentation steps and labeled images. Keys for dict: 'binarized', 'distance-map', 'maxima-points', 'maxima-mask', 'seeds', 'integer-labels', 'colored-labels'
    """
    dist_map = ndi.distance_transform_edt(imgs_binarized)
    # If prompted, calculate equivalent median radius
    if min_peak_distance == 'median_radius':
        regions = []
        for i in range(imgs_binarized.shape[0]):
            labels = measure.label(imgs_binarized[0, ...])
            regions += measure.regionprops(labels)
        areas = [region.area for region in regions]
        median_slice_area = np.median(areas)
        # Twice the radius of circle of equivalent area
        min_peak_distance = 2 * int(round(np.sqrt(median_slice_area) // np.pi))
        print(f'Calculated min_peak_distance: {min_peak_distance}')
    # Calculate the local maxima with min_peak_distance separation
    maxima = feature.peak_local_max(
        dist_map, 
        min_distance=min_peak_distance,
        exclude_border=False
    )
    # Assign a label to each point to use as seed for watershed seg
    maxima_mask = np.zeros_like(imgs_binarized, dtype=float)
    maxima_mask[tuple(maxima.T)] = 1
    seeds = measure.label(maxima_mask)
    labels = segmentation.watershed(
        -1 * dist_map, seeds, mask=imgs_binarized
    )
    colored_labels = color.label2rgb(labels, bg_label=0)
    if return_dict:
        segment_dict = {
            'binarized' : imgs_binarized,
            'distance-map' : dist_map,
            'maxima-points' : maxima,
            'maxima-mask' : maxima_mask,
            'seeds' : seeds,
            'integer-labels' : labels,
            'colored-labels' : colored_labels
        }
        return segment_dict
    else:
        return labels

def plot_segment_steps(
    raw_imgs,
    segment_dict, 
    img_idx, 
    keys=['binarized', 'distance-map', 'colored-labels'],
    plot_maxima='distance-map', 
    fig_w=7,
):
    n_axes_h = 1
    # + 1 accounts for raw image in first axis
    n_axes_w = len(keys) + 1
    img_w = raw_imgs.shape[2]
    img_h = raw_imgs.shape[1]
    title_buffer = .5
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w) + title_buffer
    fig, axes = plt.subplots(
        n_axes_h, n_axes_w, dpi=300, figsize=(fig_w, fig_h), 
        constrained_layout=True, facecolor='white',
    )
    ax = axes.ravel()
    ax[0].imshow(
        raw_imgs[img_idx, ...], interpolation='nearest',
        # vmin=0, vmax=1
    )
    ax[0].set_axis_off()
    ax[0].set_title('raw')
    for i, key in enumerate(keys):
        ax[i + 1].imshow(
            segment_dict[key][img_idx, ...], interpolation='nearest'
        )
        ax[i + 1].set_axis_off()
        ax[i + 1].set_title(key)
        if plot_maxima == key:
            # Get x, y for all maxima
            x = segment_dict['maxima-points'][:, 2]
            y = segment_dict['maxima-points'][:, 1]
            # Find the maxima that fall on the current slice (img_idx)
            x_img_idx = x[segment_dict['maxima-points'][:, 0] == img_idx]
            y_img_idx = y[segment_dict['maxima-points'][:, 0] == img_idx]
            ax[i + 1].scatter(x_img_idx, y_img_idx, color='red', s=2)
    return fig, axes
    
def plot_process(img_idx, process_dict):
    fig, axes = plt.subplots(2, 3, dpi=300, constrained_layout=True)
    for i, key in enumerate(['raw', 'binarized', 'holes-filled']):
        axes[0, i].imshow(process_dict[key][img_idx, :, :], interpolation='nearest')
        axes[0, i].set_axis_off()
        axes[0, i].set_title(key)
    for i, key in enumerate(['distance-map', 'integer-labels']):
        axes[1, i].imshow(process_dict[key][img_idx, :, :], interpolation='nearest')
        axes[1, i].set_axis_off()
        axes[1, i].set_title(key)
    axes[1, 2].imshow(
        process_dict['colored-labels'][img_idx, :, :, :], interpolation='nearest'
    )
    axes[1, 2].set_axis_off()
    axes[1, 2].set_title('colored-labels')
    # Get x, y for all maxima
    x = process_dict['maxima-points'][:, 2]
    y = process_dict['maxima-points'][:, 1]
    # Find the maxima that fall on the current slice (img_idx)
    x_img_idx = x[process_dict['maxima-points'][:, 0] == img_idx]
    y_img_idx = y[process_dict['maxima-points'][:, 0] == img_idx]
    axes[1, 0].scatter(x_img_idx, y_img_idx, color='red', s=2)
    return fig, axes

def plot_comparison(img_idx, dict1, dict2, keys='all', fig_w=4, dpi=300):
    if keys == 'all':
        keys = [
            key for key in dict1.keys() 
            if (isinstance(dict1[key], np.ndarray) and len(dict1[key].shape) > 2)
        ]
    nrows = 2
    ncols = len(keys)
    img_w = dict1[keys[0]].shape[2]
    img_h = dict1[keys[0]].shape[1]
    fig_h = fig_w * (img_h / img_w) * (nrows / ncols)
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
    for i, key in enumerate(keys):
        for j, d in enumerate([dict1, dict2]):
            axes[j, i].imshow(d[key][img_idx, :, :], interpolation='nearest')
            axes[j, i].set_axis_off()
    return fig, axes

def count_segmented_voxels(process_dict, exclude_zero=True):
    imgs_seg = process_dict['integer-labels']
    unique, counts = np.unique(imgs_seg, return_counts=True)
    label_counts = dict(zip(unique, counts))
    if exclude_zero:
        del label_counts[0]
    return label_counts

def show_particle_labels(
    segment_dict, 
    img_idx,
    label_color='white',
    label_bg_color=(0, 0, 0, 0),
    key='colored-labels',
    fig_w=7,
):
    n_axes_h = 1
    n_axes_w = 1
    img_w = segment_dict['integer-labels'].shape[2]
    img_h = segment_dict['integer-labels'].shape[1]
    title_buffer = .5
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w) + title_buffer
    fig, ax = plt.subplots(
        n_axes_h, n_axes_w, dpi=300, figsize=(fig_w, fig_h), 
        constrained_layout=True, facecolor='white',
    )
    ax.imshow(
        segment_dict[key][img_idx, ...], interpolation='nearest'
    )
    ax.set_axis_off()
    ax.set_title(key)
    regions = measure.regionprops(segment_dict['integer-labels'][img_idx, ...])
    label_centroid_pairs = [(region.label, region.centroid) for region in regions]
    for label, centroid in label_centroid_pairs:
        ax.text(
            centroid[1], centroid[0], str(label), fontsize='large',
            color=label_color, backgroundcolor=label_bg_color, ha='center', 
            va='center'
        )
    return fig, ax

def isolate_particle(segment_dict, integer_label):
    imgs_single_particle = np.zeros_like(segment_dict['integer-labels'])
    imgs_single_particle[segment_dict['integer-labels'] == integer_label] = 1
    return imgs_single_particle

def plot_particle_slices(imgs_single_particle, n_slices=4, fig_w=7):
    # bounds: (min_slice, min_row, min_col, max_slice, max_row, max_col)
    bounds = measure.regionprops(imgs_single_particle)[0].bbox
    print(bounds)
    # bounds[0] and bounds[3] used for min_slice and max_slice respectively
    slices = [round(i) for i in np.linspace(bounds[0], bounds[3], n_slices)]
    n_axes_h = 1
    n_axes_w = n_slices
    img_w = imgs_single_particle.shape[2]
    img_h = imgs_single_particle.shape[1]
    title_buffer = .5
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w) + title_buffer
    fig, axes = plt.subplots(
        n_axes_h, n_axes_w, dpi=300, figsize=(fig_w, fig_h), 
        constrained_layout=True, facecolor='white',
    )
    if not isinstance(axes, np.ndarray):
        ax = [axes]
    else:
        ax = axes.ravel()
    for i, slice_i in enumerate(slices):
        ax[i].imshow(imgs_single_particle[slice_i, ...], interpolation='nearest')
        ax[i].set_axis_off()
        ax[i].set_title(f'Slice: {slice_i}')
    return fig, ax

def plot_mesh_3D(verts, faces):
    """Plot triangualar mesh with Matplotlib.

    Parameters
    ----------
    verts : array-like
        Array of (x, y, z) vertices indexed with faces to construct triangles.
    faces : array-like
        Array of indices referencing verts that define the triangular faces of the mesh.

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('black')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_xlim(min(verts[:, 0]), max(verts[:, 0]))
    ax.set_ylim(min(verts[:, 1]), max(verts[:, 1]))
    ax.set_zlim(min(verts[:, 2]), max(verts[:, 2]))
    return fig, ax

def save_stl(verts, faces, save_path, suppress_save_message=False):
    """Save triangular mesh defined by vertices and face indices as an STL file.

    Parameters
    ----------
    verts : array-like
        Array of (x, y, z) vertices indexed with faces to construct triangles.
    faces : array-like
        Array of indices referencing verts that define the triangular faces of the mesh.
    save_path : Path or str
        Path at which STL file will be saved. If doesn't end with '.stl', it will be added.
    suppress_save_message : bool, optional
        If True, particle label and STL file path will not be printed. By default False
    """
    if not str(save_path).endswith('.stl'):
        save_path = Path(f'{save_path}.stl')
    if save_path.exists():
        print(f'File already exists: {save_path}')
    else:
        # Convert vertices (verts) and faces to numpy-stl format for saving:
        vertice_count = faces.shape[0]
        stl_mesh = mesh.Mesh(
            np.zeros(vertice_count, dtype=mesh.Mesh.dtype),
            remove_empty_areas=False
        )
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = verts[face[j], :]
        # Write the mesh to STL file
        stl_mesh.save(save_path)
        if not suppress_save_message:
            print(f'STL saved: {save_path}')

def save_each_stl(
    segment_dict, 
    dir_name, 
    save_dir_parent_path,
    voxel_step_size=1,
    n_particle_label_digits=5
):
    """Iterate through particles in segment_dict['integer-labels'] and use marching cubes algorithm to convert

    Parameters
    ----------
    segment_dict : dict
        Dictionary containing segmentation results with the key and value pair 'integer-labels' and 3D array of integers which N unique labels that correspond to N individual particles
    dir_name : str
        Name for directory that will be created to contain STL files.
    save_dir_parent_path : Path or str
        Path to the directory where the STL-containing directory will be created.
    voxel_step_size : int, optional
        Number of voxels to iterate across in marching cubes algorithm. Larger steps yield faster but coarser results. Defaults to 1. 
    n_particle_label_digits : int, optional
        Number of digits to denote particle label. Determines number of leading zeros. Defaults to 5.

    Raises
    ------
    ValueError
        Raise ValueError when directory named dir_name already exists at location save_dir_parent_path
    """
    save_dir_parent_path = Path(save_dir_parent_path)
    save_dir_path = Path(save_dir_parent_path / f'{dir_name}_STLs')
    if save_dir_path.exists():
        raise ValueError(f'Save directory already exists: {save_dir_path}')
    else:
        # Make directory to save STL files
        save_dir_path.mkdir()
    n_particles = np.max(segment_dict['integer-labels'])
    print(f'Saving {n_particles} STL file(s)...')
    # Start iteration at label 1 because 0 is background
    # End iteration at n_particles + 1 to include max label
    for particle_i in range(1, n_particles + 1):
        # Isolate particle with label particle_i
        isolated_voxels = isolate_particle(segment_dict, particle_i)
        # Use marching cubes to obtain the surface mesh of these ellipsoids
        verts, faces, normals, values = measure.marching_cubes(
            isolated_voxels, step_size=voxel_step_size
        ) 
        fn = f'{dir_name}-{str(particle_i).zfill(n_particle_label_digits)}.stl'
        save_path = Path(save_dir_path / fn)
        save_stl(verts, faces, save_path, suppress_save_message=True)
    print(f'{particle_i} STL file(s) saved: {save_dir_path}')

def raw_to_3d_segment(
    img_dir, 
    new_segmented_dir_path,
    thresh_val=0.65,
    fill_holes=64,
    min_peak_distance=30
):
    """Workflow for loading, binarizing, and segmenting example images.

    Parameters
    ----------
    img_dir : str or Path
        Path to images to be binarized and segmented.
    new_segmented_dir_path : str or Path
        Path for new directory to be created to contain the segmented and labeled images that will be created.
    thresh_val : int, optional
        Floating-point grayscale level, to be passed to binarize_3d(), at which images are thresholded above, by default 0.65
    fill_holes : int or 'all', optional 
        Hole area in pixels, to be passed to binarize_3d(), for which any smaller hole will be filled in binary images. If 'all' is passed, image slices will be iterated to fill all holes. Defaults to 64.
    min_peak_distance : int, optional
        Minimum distance in pixels between local maxima of distance map to be passed to segment_3d(), by default 30
    """
    print(f'Loading images...')
    imgs, img_names = load_images(
        img_dir, 
        return_3d_array=True, 
        also_return_names=True,
        convert_to_float=True
    )
    print(f'Segmenting images...')
    process_dict = segment_3d(
        imgs, 
        thresh_val=thresh_val, 
        fill_holes=fill_holes, 
        min_peak_distance=min_peak_distance,
        return_process_dict=True
    )
    print(f'Saving images...')
    save_images(
        process_dict['integer-labels'], 
        new_segmented_dir_path, 
        img_names=img_names,
        convert_to_16bit=True
    )


if __name__ == '__main__':

    raw_to_3d_segment(
        'example-imgs',  # Path to directory containing CT data
        'segmented-integer-labels',  # Path for new dir to contain segmented images
        thresh_val=0.65,  # Floating-point grayscale level at which images are thresholded above 
        fill_holes=64,  # Hole area in pixels for which any smaller hole will be filled in binary images
        min_peak_distance=30  # Minimum distance in pixels between local maxima of distance map
    )

