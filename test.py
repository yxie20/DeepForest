from deepforest import main
import os
import cv2


m = main.deepforest()
m.use_release()

tiffile = 'deepforest/data/resize.tif'
# df = m.predict_file(
#     tiffile,
#     os.path.dirname(tiffile)
# )

df = m.predict_image(
    path=tiffile,
    return_plot=True,
)

cv2.imwrite('output.png', df)