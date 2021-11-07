from struct_classes import *

for index in range(originalImage.__len__()):
    im=ImageItem(originalImage[int(index)], headlineCSV[int(index)], illustrationCSV[int(index)], titleCSV[int(index)], textCSV[int(index)])
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


# image = data.coins()[50:-50, 50:-50]
# image = Image.open(imagePath+originalImage[0])


image = img_as_ubyte(imread(outputPath+"_TEXT_242_0_2638730_0_5D1327B5.tif", as_gray=True))


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

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()



print("EoP")