import os
import cv2
import numpy as np

from deepforest import main
from osgeo import gdal
import pandas as pd
import time

import tif_util



def export_bbox_tif(dataset, model, region, tone_map_params, return_patches=True, return_plot=False, debug=False):
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
        # bands=[4, 2, 1],        # Input is in RGB format
        # bands=[1, 2, 4],        # Input is in RGB format
        tone_map='percent_clip', 
        tone_map_params=tone_map_params,
        clip=(0, 255),
        scaler=255,
    )
    # Collapse 8 channels (MS) into 3 (RGB)
    patch_rgb = np.dstack((patch[...,1], patch[...,2], patch[...,4]))
    # Resize so that each pixel in our dataset corresponds to training data for DeepFroest
    patch_rgb, scale = tif_util.resize(dataset, patch_rgb, pixelsize=0.25)
    patch_rgb = patch_rgb.astype(np.float32)
    save_img(cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR), 0 , 0 , 0 , 0)

    # Assumes image in 0-255
    df = model.predict_image(image=patch_rgb)
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
                cv2.imwrite(f"temp/{index}.png", _img[...,::-1])
                cv2.imwrite(f"temp/{index}_origres.png", np.dstack([patches[-1][..., 4], patches[-1][..., 2], patches[-1][..., 1]]))

    if return_plot:
        # The returned img is in cv2 BGR format.
        img = model.predict_image(image=patch_rgb, return_plot=True)
        if debug:
            cv2.imwrite(f"temp/bbox.png", img)

    return df, patches, img

def get_img(dataset, region, tone_map_params):
    patch = tif_util.read_patch(
        dataset, 
        region,
        # bands=[4, 2, 1],        # Input is in RGB format
        # bands=[1, 2, 4],        # Input is in RGB format
        tone_map='percent_clip', 
        tone_map_params=tone_map_params,
        clip=(0, 255),
        scaler=255,
    )
    # Collapse 8 channels (MS) into 3 (RGB)
    patch_rgb = np.dstack((patch[...,1], patch[...,2], patch[...,4]))
    # Resize so that each pixel in our dataset corresponds to training data for DeepFroest
    patch_rgb, scale = tif_util.resize(dataset, patch_rgb, pixelsize=0.25)
    patch_rgb = patch_rgb.astype(np.float32)
    # patch = tif_util.resize(dataset, patch, pixelsize=0.25)[0]
    # patch_rgb = patch.astype(np.float32)
    return cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)


def save_img(img, hmin, hmax, wmin, wmax):
    path = "data/SavedImgs"
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    cv2.imwrite(f"{hmin}_{hmax}_{wmin}_{wmax}_region.png", img)
    os.chdir("../..")

def saveNImages(n = 10, hRange = 500, wRange = 1000):
    H = 42953
    W = 18318
    startH = 20000
    startW = 1200
    dataset = gdal.Open('data/Delivery/Ortho_PS/20OCT10155557-PS-014910772010_01_P001.tif')
    tone_map_params = tif_util.find_percent_clip_params(dataset)
    for _ in range(n):
        bgr_patch = get_img(dataset, (startH, startH + hRange, startW, startW + wRange), tone_map_params)
        save_img(bgr_patch, startH, startH + hRange, startW, startW + wRange)
        startW += wRange
        if startW + wRange > W:
            startH += hRange
            startW = 1200

def split_data(path, percent = 0.75):
    df = pd.read_csv(path)
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= percent

    train = df[msk]
    valid = df[~msk]
    #drop the split column
    train.drop(train.columns[len(train.columns)-1], axis=1, inplace=True)
    valid.drop(valid.columns[len(valid.columns)-1], axis=1, inplace=True)

    train.to_csv("data/train.csv", index=False)
    valid.to_csv("data/valid.csv", index=False)


def format_csv():
    data = pd.read_csv('data/annotations.csv')
    keep_col = ['filename','region_shape_attributes']
    new_f = data[keep_col]
    xMin = []
    yMin = []
    xMax = []
    yMax = []
    for _, row in new_f.iterrows():
        row_str: str = row['region_shape_attributes'][1:-1]
        attrs = row_str.split(",")
        for i in range(1, len(attrs), 1):
            key_val_pairs = attrs[i].split(":")
            if i == 1:
                xMin.append(int(key_val_pairs[1]))
            elif i == 2:
                yMin.append(int(key_val_pairs[1]))
            elif i == 3:
                xMax.append(xMin[-1] + int(key_val_pairs[1]))
            else:
                yMax.append(yMin[-1] + int(key_val_pairs[1]))
    new_f = new_f['filename']
    new_f = new_f.str.rstrip(".jpg")
    new_f = new_f.astype(str) + '.png'
    new_f = pd.DataFrame(new_f)
    new_f = new_f.rename(columns={'filename' : 'image_path'})
    # new_f.rename(columns = {'f':'new_col1', 'old_col2':'new_col2'}, inplace = True)
    new_f['xmin'] = xMin
    new_f['xmax'] = xMax
    new_f['ymin'] = yMin
    new_f['ymax'] = yMax
    new_f['label'] = "tree"
    new_f.to_csv("data/annotations_final.csv", index=False)


def train_model(m, train_file, valid_file):
    m = main.deepforest()
    m.config['gpus'] = '-1' #move to GPU and use all the GPU resources
    m.config["train"]["csv_file"] = train_file
    m.config["train"]["root_dir"] = os.path.dirname(train_file)
    #We might want to remove this, as this only keeps high quality boxes. It's there to cut traiing time
    m.config["score_thresh"] = 0.4
    m.config["train"]['epochs'] = 2
    m.config["validation"]["csv_file"] = valid_file
    m.config["validation"]["root_dir"] = os.path.dirname(valid_file)
    #create a pytorch lighting trainer used to training 
    m.create_trainer()
    #load the lastest release model 
    m.use_release()

    start_time = time.time()
    m.trainer.fit(m)
    print(f"--- Training on GPU: {(time.time() - start_time):.2f} seconds ---")

    save_dir = os.path.join(os.getcwd(),'pred_result')
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    results = m.evaluate(train_file, os.path.dirname(train_file), iou_threshold = 0.4, savedir= save_dir)
    print(results)



if __name__ == "__main__":
    # dataset = gdal.Open('data/Delivery/Ortho_PS/20OCT10155557-PS-014910772010_01_P001.tif')
    model = main.deepforest()
    # model.use_release()
    # tone_map_params = tif_util.find_percent_clip_params(dataset)
    # bbox, patches, img = export_bbox_tif(
    #     dataset, model, (15000,15500,15000,16000), tone_map_params, 
    #     debug=True, return_plot=True
    # )

    train_model(model, "data/training_data_folder/train.csv", "data/train_data_folder/valid.csv")
   

   

# These are only used to create and split the training data
# saveNImages(n=5)
# format_csv()
# split_data(path='data/annotations_final.csv')



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
