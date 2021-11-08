from struct_classes import *

for index in range(originalImage.__len__()):
    im=ImageItem(originalImage[index], headlineCSV[index], illustrationCSV[index], titleCSV[index], textCSV[index])
    im.Normalize()
    # im.ApplyGaussianFilter()
    im.ApplyLowLimit()
    im.CreateColorLayout()
    im.ColorLayoutsToImages()
    im.AllLayoutsOnImage()
    im.OverlayOnImage()
    im.ApplyBorders()
    print("EoI: "+ str(index))

print("EoP")