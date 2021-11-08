import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage.io import imread

from struct_classes import ImageItem
from app_config import (outputPath, originalImage, headlineCSV,
                        illustrationCSV, titleCSV, textCSV)


for index in range(len(originalImage)):
    im = ImageItem(
        originalImage[int(index)],
        headlineCSV[int(index)],
        illustrationCSV[int(index)],
        titleCSV[int(index)],
        textCSV[int(index)]
    )

    im.Normalize()
    # im.ApplyGaussianFilter()
    im.ApplyLowLimit()
    im.CreateColorLayout()
    im.ColorLayoutsToImages()
    im.AllLayoutsOnImage()
    im.OverlayOnImage()
    # im.ApplyBorders()
    print("EoI: "+ str(index))
    break

image = img_as_ubyte(
    imread(outputPath + "_TEXT_242_0_2638730_0_5D1327B5.tif", as_gray=True)
)

thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

cleared = clear_border(bw)

label_image = label(image)
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    if region.area >= 1000:
        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red',
            linewidth=2
        )

        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()

print("EoP")