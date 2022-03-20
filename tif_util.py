from osgeo import gdal
import numpy as np
import imageio, cv2
import math
import os

def gps2pixel(dataset):
    xform = dataset.GetGeoTransform()   # x, x-pixel-size, x-rotation, y, y-rotation, y-pixel-size
    # todo

def resize(dataset, img, pixelsize=0.25):
    """
    Args: pixelsize: real world distance (unit: meters) per pixel
    """
    transform = dataset.GetGeoTransform()
    w_scale = pixelsize / abs(transform[1])    # todo: check if this is h or w
    h_scale = pixelsize / abs(transform[5])
    h, w = img.shape[:2]
    dsize = round(h * h_scale), round(w * w_scale)
    ret_img = cv2.resize(img, dsize)

    return ret_img


def find_percent_clip_params(dataset, t_min=10, t_max=90, patchsize=128, patches=16):
    B, H, W = dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize
    _min = [None for _ in range(B)]
    _max = [None for _ in range(B)]
    for b in range(B):
        if H*W < (patchsize**2 * patches**2):
            temp = dataset.GetRasterBand(b+1).ReadAsArray().flatten()
        else:
            # load random patches
            h_skip, w_skip = math.floor((H-patchsize)/patches), math.floor((W-patchsize)/patches)
            temp = []
            for h in range(0, H-patchsize, h_skip):
                for w in range(0, W-patchsize, w_skip):
                    temp.append(dataset.GetRasterBand(b+1).ReadAsArray(w, h, patchsize, patchsize))
            temp = np.stack(temp).flatten()
        _min[b] = np.percentile(temp, t_min)
        _max[b] = np.percentile(temp, t_max)

    return _min, _max


def read_patch(dataset, reg, bands=None, tone_map='none', tone_map_params=None, clip=None, scaler=1.):
    """
    This is the core of the dataloader.
    Args:
        dataset: gdal dataset object
        reg: (hmin, hmax, wmin, wmax) pixel coordinates of the rectangular patch
        bands: specific bands to load, leave as None to load all bands
        tone_map: tone mapping strategy, string
        tone_map: if the particular tone mapping function is parameterized (e.g. percent_clip), supply parameters here
        clip: whether to clamp results, leave as None to not clip, otherwise input two numbers (low, high)
        scalar: whether to scale the data (e.g. to 0 - 255)
    """
    B, H, W = dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize
    h0, h1, w0, w1 = reg
    assert (h0 >= 0) and (w0 >= 0) and ((h1-h0) > 0) and ((w1-w0) > 0) and (h1 <= H) and (w1 <= W), f"{h0}, {h1}, {w0}, {w1}"
    
    bands = range(B) if bands is None else bands

    arr = []
    for b in bands:
        temp = dataset.GetRasterBand(b+1).ReadAsArray(w0, h0, w1-w0, h1-h0)

        if tone_map == "sRGB": temp = (temp / (temp + 1)) * scaler
        if tone_map == 'max': temp = (temp / temp.max()) * scaler
        if tone_map == 'percent_clip':
            _min, _max = tone_map_params
            temp = (temp - _min[b]) / (_max[b] - _min[b]) * scaler

        if not (clip is None): temp = np.clip(temp, clip[0], clip[1])
        arr.append(temp)

    patch = np.dstack(arr)

    return patch


def ms2rgb(dataset, regions, out_filename=None, save_intermediate=False):
    """
    Experiental function. Not working yet.
    Converts the 8-channel multispectral TIFF into RGB TIFF file.
    Make sure each entry in regions is small enough to fit into memory.
    Args:
        dataset: gdal dataset object
        regions: list of list, each sublist is a list of bounding box coordinates (hmin, hmax, wmin, wmax)
    """
    B, H, W = dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize
    assert B == 8, "Only 8 band is supported right now."
    out_filename = str(dataset.GetDescription()).replace(".tif", "_rgb.tif") if out_filename is None else out_filename
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(out_filename, 5000, 5000, 3, gdal.GDT_Float32)
    outdata.SetGeoTransform(dataset.GetGeoTransform())   #sets same geotransform as input
    outdata.SetProjection(dataset.GetProjection())       #sets same projection as input
    
    # Tone map is needed to map unbounded sensor values to 0 - 255 (or 0 - 1)
    tone_map_params = find_percent_clip_params(dataset)
    
    ind = 0
    for reg in regions:

        patch = read_patch(
            dataset, 
            reg,
            bands=[4, 2, 1],
            tone_map='percent_clip', 
            tone_map_params=tone_map_params,
            clip=(0, 255),
            scaler=255,
        )
        patch = resize(dataset, patch)

        for i in range(3):
            outdata.GetRasterBand(i+1).WriteArray(patch[i].astype(np.float32), reg[0], reg[2])
        outdata.FlushCache()

        if save_intermediate: imageio.imwrite(os.path.join(os.path.dirname(out_filename), f"{ind}.png"), patch.astype(np.uint8))
        ind += 1


if __name__ == "__main__":
    # dataset = gdal.Open('test.tif')
    dataset = gdal.Open('data/Delivery/Ortho_PS/20OCT10155557-PS-014910772010_01_P001.tif')

    # ms2rgb(dataset, [(0,3500,0,3500)], save_intermediate=True)
