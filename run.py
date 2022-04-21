import os
import cv2
import numpy as np

from deepforest import main
from osgeo import gdal

import tif_util




def export_bbox_tif(dataset, model, region, tone_map_params, 
    return_patches=True, return_plot=False, debug=False, resize=False):
    """
    Example usage:
        dataset = gdal.Open('data/Delivery/Ortho_PS/20OCT10155557-PS-014910772010_01_P001.tif')
        model = main.deepforest()
        model.use_release()
        tone_map_params = tif_util.find_percent_clip_params(dataset)
        bbox, patches, img = export_bbox_tif(dataset, model, (3000,3500,3000,3500), tone_map_params)
    Args: 
        dataset: gdal dataset object
        model: deepforest model object
        region: list of length of 4 (hmin, hmax, wmin, wmax)
        tone_map_params: tone-mapping parameter, see tif_util.py
        return_patches: bool, whether to return cropped numpy arrays
            Be careful that these patches reference the same memory as the original array.
        return_plot: bool, whether to return the input image with bbox drawn over
        debug: bool, if True, saves patches and returned plot to disk for visualization.
    Returns: 
        bbox_list: list of bounding box coordinates
        patches: list of npy array of images
    """

    patch = tif_util.read_patch(
        dataset, 
        region,
        tone_map='percent_clip', 
        tone_map_params=tone_map_params,
        clip=(0, 255),
        scaler=255,
    )
    # Tonemapping: Collapse 8 channels (MS) into 3 (RGB)
    patch_rgb = np.dstack((patch[...,4], patch[...,2], patch[...,1]))

    # Resize so that each pixel in our dataset corresponds to training data for DeepFroest
    if resize: patch_rgb, scale = tif_util.resize(dataset, patch_rgb, pixelsize=0.25)
    else: scale = (1., 1.)
    patch_rgb = patch_rgb.astype(np.float32)

    # Assumes image in 0-255
    df = model.predict_tile(image=patch_rgb, patch_size=600)

    # Take into account rescaling
    df['xmin_orig'] = df['xmin'].apply(lambda x: x/scale[1])
    df['ymin_orig'] = df['ymin'].apply(lambda x: x/scale[0])
    df['xmax_orig'] = df['xmax'].apply(lambda x: x/scale[1])
    df['ymax_orig'] = df['ymax'].apply(lambda x: x/scale[0])

    patches, img = [], None
    if return_patches:
        for index, row in df.iterrows():
            patches.append(patch[round(row['ymin_orig']):round(row['ymax_orig']), round(row['xmin_orig']):round(row['xmax_orig'])])
            if debug:
                _img = patch_rgb[round(row['ymin']):round(row['ymax']), round(row['xmin']):round(row['xmax'])]
                # cv2.imwrite(f"temp/{index}.png", _img[...,::-1])

    if return_plot:
        # The returned img is in cv2 BGR format.
        img = model.predict_tile(image=patch_rgb, return_plot=True, patch_size=600)
        if debug:
            cv2.imwrite(f"temp/bbox.png", img)

    return df, patches, img


if __name__ == "__main__":
    dataset = gdal.Open('data/Delivery/Ortho_PS/20OCT10155557-PS-014910772010_01_P001.tif')
    model = main.deepforest()
    model.use_release()
    tone_map_params = tif_util.find_percent_clip_params(dataset)
    bbox, patches, img = export_bbox_tif(
        dataset, model, (15000,16000,15000,16000), tone_map_params, 
        debug=True, return_plot=True
    )



# m = main.deepforest()
# m.use_release()

# tiffile = 'deepforest/data/resize.tif'
# df = m.predict_file(
#     tiffile,
#     os.path.dirname(tiffile)
# )

# df = m.predict_image(
#     path=tiffile,
#     return_plot=True,
# )

# cv2.imwrite('output.png', df)

# df = m.predict_tile(
#     path=tiffile,
#     patch_size=200,
#     return_plot=True,
# )
