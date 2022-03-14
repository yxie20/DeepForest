from osgeo import gdal
import numpy as np
import imageio



def read_patch(dataset, h0, h1, w0, w1):
    B = dataset.RasterCount
    H, W = dataset.RasterYSize, dataset.RasterXSize
    assert (h0 >= 0) and (w0 >= 0) and ((h1-h0) > 0) and ((w1-w0) > 0) and (h1 <= H) and (w1 <= W), f"{h0}, {h1}, {w0}, {w1}"

    arr = []
    for b in range(B):
        temp = dataset.GetRasterBand(b+1).ReadAsArray(w0, h0, w1-w0, h1-h0)
        print(temp.mean(), temp.min(), temp.max())

        # temp = (temp / (temp + 1)) * 255
        temp = (temp / temp.max()) * 255 * 1
        temp = np.clip(temp, 0, 255)
        arr.append(temp)
        print(arr[-1].mean(), arr[-1].min(), arr[-1].max())

    img = np.dstack(arr)[...,-3:]
    # img = np.dstack(arr)[...,:3]
    img = img[...,::-1]
    print(img.shape)
    print(np.mean(img), np.max(img), np.min(img))
    imageio.imwrite("test.png", img)



if __name__ == "__main__":
    # dataset = gdal.Open('test.tif')
    dataset = gdal.Open('data/Delivery/Ortho_PS/20OCT10155557-PS-014910772010_01_P001.tif')
    B = dataset.RasterCount
    H, W = dataset.RasterYSize, dataset.RasterXSize
    print("Shape of dataset:", H, W)
    print("Number of bands:", B)


    # bands = []
    # for b in range(B):
    #     bands.append(dataset.GetRasterBand(b+1)) # Red channel
    # arr = []
    # for b in range(B):
    #     arr.append(bands[b].ReadAsArray())

    # img = np.dstack(arr)
    # print(img.shape)
    # imageio.imwrite("test.png", img)


    read_patch(dataset, 3000, 4000, 2000, 3000)