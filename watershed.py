# Standard library imports
from concurrent.futures import process
from pathlib import Path
# Third-party imports
import imagecodecs  # dependency required for loading compressed tif images
import imageio as iio
import matplotlib.pyplot as plt
import napari
import numpy as np
from scipy import ndimage as ndi
import skimage


def load_images(
    img_dir, 
    slice_subset=None,
    return_3d_array=False, 
    also_return_names=False,
    convert_to_float=False,
    file_suffix='tif'
):
    """Load images from path and return as list of 2D arrays. Can also return names of images.

    Parameters
    ----------
    img_dir : str or Path
        Path to directory containing images to be loaded.
    return_3d_array : bool, optional
        If True, return loaded images as a 3D numpy array, else return images in list, by default False
    also_return_names : bool, optional
        If True, returns a list of the names of the images in addition to the list of images themselves. Defaults to False.
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
        path for path in Path(img_dir).glob(f'*.{file_suffix}')
    ]
    img_path_list.sort()
    imgs = []
    for img_path in img_path_list:
        img = iio.imread(img_path) 
        if convert_to_float:
            img = skimage.util.img_as_float(img)
        imgs.append(img)
    if return_3d_array:
        imgs = np.stack(imgs)
    if also_return_names:
        return imgs, [img_path.stem for img_path in img_path_list]
    else:
        return imgs

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
                skimage.util.img_as_ubyte(imgs[i, :, :, :]) 
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
    fill_hole_area=64,
    return_process_dict=False
):
    """Creates binary images from list of images using a threshold value.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing the floating point images to be binarized.
    thresh_val : float, optional
        Value to threshold point images. Defaults to 0.65 for floating point images.
    fill_hole_area : int, optional
        Size of holes (in pixel area) in binary images that will be filled, defaults to 64
    return_process_dict : bool, optional
        If True, return a dictionary containing all processing steps instead of last step only, defaults to False

    Returns
    -------
    numpy.ndarray or dict
        If return_process_dict is False, a 3D array representing the hole-filled, binary images, else a dictionary is returned with a 3D array for each step in the binarization process.
    """
    smoothed = skimage.filters.gaussian(imgs)
    binarized = smoothed > thresh_val
    filled = skimage.morphology.remove_small_holes(
        binarized, area_threshold=fill_hole_area
    )
    if return_process_dict:
        process_dict = {
            'binarized' : binarized,
            'holes-filled' : filled
        }
        return process_dict
    else:
        return filled

def segment_3d(
    binarized_imgs, 
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
    dist_map = ndi.distance_transform_edt(binarized_imgs)
    # Get Nx2 array of N number of (row, col) coordinates
    maxima = skimage.feature.peak_local_max(
        dist_map, 
        min_distance=min_peak_distance,
        exclude_border=False
    )
    # Assign a label to each point to use as seed for watershed seg
    maxima_mask = np.zeros_like(binarized_imgs, dtype=float)
    maxima_mask[tuple(maxima.T)] = 1
    seeds = skimage.measure.label(maxima_mask)
    labels = skimage.segmentation.watershed(-1 * dist_map, seeds, mask=binarized_imgs)
    colored_labels = skimage.color.label2rgb(labels, bg_label=0)
    if return_process_dict:
        process_dict = {
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

def raw_to_3d_segment(
    img_dir, 
    new_segmented_dir_path,
    thresh_val=0.65,
    fill_hole_area=64,
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
    fill_hole_area : int, optional 
        Hole area in pixels, to be passed to binarize_3d(), for which any smaller hole will be filled in binary images, by default 64
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
    print(f'Binarizing images...')
    binarized_3d_dict = binarize_3d(
        imgs, 
        thresh_val=thresh_val, 
        fill_hole_area=fill_hole_area, 
        return_process_dict=True
    )
    print(f'Segmenting images...')
    segment_3d_dict = segment_3d(
        binarized_3d_dict['holes-filled'], 
        min_peak_distance=min_peak_distance,
        return_process_dict=True
    )
    print(f'Saving images...')
    save_images(
        segment_3d_dict['integer-labels'], 
        new_segmented_dir_path, 
        img_names=img_names,
        convert_to_16bit=True
    )


if __name__ == '__main__':

    raw_to_3d_segment(
        'example-imgs',  # Path to directory containing CT data
        'segmented-integer-labels',  # Path for new dir to contain segmented images
        thresh_val=0.65,  # Floating-point grayscale level at which images are thresholded above 
        fill_hole_area=64,  # Hole area in pixels for which any smaller hole will be filled in binary images
        min_peak_distance=30  # Minimum distance in pixels between local maxima of distance map
    )

