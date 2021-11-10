from PIL import Image, ImageColor
from sklearn import preprocessing
import numpy as np
import scipy as sp
import cv2
from app_config import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage import io, morphology

def applyOnLayout(pathToLayoutImage=""):
    image = cv2.imread(pathToLayoutImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5500:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return close

class ImageItem:
    def __init__(self, originalImageName="", headlineCSVName="",
                 illustrationCSVName="", titleCSVName="", textCSVName=""):
        
        # save the original name of the image
        self.OriginalName = originalImageName

        # get image dimensions by reading one related csv
        headlineFile = open(csvPath + headlineCSVName, "r")
        self.rows = -1
        self.cols = -1
        for line in headlineFile:
            line = line.split(",")
            self.cols = line.__len__()
            self.rows += 1
        headlineFile.close()

        # process the csv which contains the pixel probabilities for the headline class
        headlineFile = open(csvPath + headlineCSVName, "r")
        self.headline = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.headlineColor = [[0 for i in range(self.cols)] for j in range(self.rows)]

        # convert headline probability matrix from string to float
        for i in range(self.rows):
            row = headlineFile.readline()
            row = row.strip()
            row = row.split(",")
            for j in range(self.cols):
                self.headline[i][j] = float(row[j])

        # close the headline probability csv file
        headlineFile.close()

        # process the csv which contains the pixel probabilities for the illustration class
        illustrationFile = open(csvPath+illustrationCSVName, "r")
        self.illustration = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.illustrationColor = [[0 for i in range(self.cols)] for j in range(self.rows)]

        #  convert illustration probability matrix from string to float
        for i in range(self.rows):
            row = illustrationFile.readline()
            row = row.strip()
            row = row.split(",")
            for j in range(self.cols):
                self.illustration[i][j] = float(row[j])
        
        # close the illustration probability csv file
        illustrationFile.close()

        # process the csv which contains the pixel probabilities for the title csv
        titleFile = open(csvPath+titleCSVName,"r")
        self.title = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.titleColor = [[0 for i in range(self.cols)] for j in range(self.rows)]

        # convert title probability matrix from string to float
        for i in range(self.rows):
            row = titleFile.readline()
            row = row.strip()
            row = row.split(",")
            for j in range(self.cols):
                self.title[i][j] = float(row[j])
        
        # close the title probability csv file
        titleFile.close()

        # process the csv which contains the pixel probabilities for the text csv
        textFile = open(csvPath+textCSVName,"r")
        self.text = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.textColor = [[0 for i in range(self.cols)] for j in range(self.rows)]
        
        # convert text probability matrix from string to float
        for i in range(self.rows):
            row = textFile.readline()
            row = row.strip()
            row = row.split(",")
            for j in range(self.cols):
                self.text[i][j] = float(row[j])

        # close the text probability csv file
        textFile.close()

        # open original image
        image = Image.open(imagePath+originalImageName)
        
        # resize the original image to fit the dimensions of the csvs pixel matrix
        resizedImage = image.resize((self.cols, self.rows))

        # save the resized image
        resizedImage.save(resizedPath + "_RESIZED_" + originalImageName)

        # the input image will be the resized one in order to identify the vertices of the polygons
        self.OriginalImage = Image.open(resizedPath + "_RESIZED_" + originalImageName)

    # normalize the class matrices using MinMaxScaler
    def Normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        self.illustration = min_max_scaler.fit_transform(self.illustration)
        self.headline = min_max_scaler.fit_transform(self.headline)
        self.title = min_max_scaler.fit_transform(self.title)
        self.text = min_max_scaler.fit_transform(self.text)

    # process the matrices and setup threshold to lowLimit = 0.7
    def ApplyLowLimit(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.illustration[row][col] <= lowLimit:
                    self.illustration[row][col]=0
                if self.headline[row][col] <= lowLimit:
                    self.headline[row][col]=0.0
                if self.title[row][col] <= lowLimit:
                    self.title[row][col]=0
                if self.text[row][col] <= lowLimit:
                    self.text[row][col]=0

    # Create matrix for each class with 0 and 1 values   
    def CreateColorLayout(self):
        for row in range(self.rows):
            for col in range(self.cols):
                maxValue = max(0, self.illustration[row][col],
                                  self.headline[row][col],
                                  self.title[row][col],
                                  self.text[row][col])
                if maxValue == 0:
                    self.illustrationColor[row][col] = 0
                    self.headlineColor[row][col] = 0
                    self.titleColor[row][col] = 0
                    self.textColor[row][col] = 0
                elif maxValue == self.illustration[row][col]:
                    self.illustrationColor[row][col] = 1
                elif maxValue == self.headline[row][col]:
                    self.headlineColor[row][col] = 1
                elif maxValue == self.title[row][col]:
                    self.titleColor[row][col] = 1
                elif maxValue == self.text[row][col]:
                    self.textColor[row][col] = 1
    
    # color each matrix with specific color
    def ColorLayoutsToImages(self):
        self.illustrationImage = Image.new(
            mode='RGB',
            size=(self.cols, self.rows)
        )
        self.headlineImage = Image.new(
            mode='RGB',
            size=(self.cols, self.rows)
        )
        self.titleImage = Image.new(
            mode='RGB',
            size=(self.cols, self.rows)
        )
        self.textImage = Image.new(
            mode='RGB',
            size=(self.cols, self.rows)
        )
        for i in range(self.rows):
            for j in range(self.cols):
                if self.illustrationColor[i][j] == 1:
                    self.illustrationImage.putpixel(
                        (j, i),
                        ImageColor.getcolor(illustrationColor, 'RGB')
                    )
                if self.headlineColor[i][j] == 1:
                    self.headlineImage.putpixel(
                        (j, i),
                        ImageColor.getcolor(headlineColor, 'RGB')
                    )
                if self.titleColor[i][j] == 1:
                    self.titleImage.putpixel(
                        (j, i),
                        ImageColor.getcolor(titleColor, 'RGB')
                    )
                if self.textColor[i][j] == 1:
                    self.textImage.putpixel(
                        (j, i),
                        ImageColor.getcolor(textColor, 'RGB')
                    )
        
        self.illustrationImage.save(outputPath + "_ILLUSTRATION_" + self.OriginalName)
        self.headlineImage.save(outputPath + "_HEADLINE_" + self.OriginalName)
        self.titleImage.save(outputPath + "_TITLE_" + self.OriginalName)
        self.textImage.save(outputPath + "_TEXT_" + self.OriginalName)
    
    # combine all matrices in one image
    def AllLayoutsOnImage(self):
        self.AllLayouts = Image.new(mode='RGB', size=(self.cols, self.rows))
        for i in range(self.rows):
            for j in range(self.cols):
                if self.illustrationColor[i][j] == 1:
                    self.AllLayouts.putpixel(
                        (j,i),
                        ImageColor.getcolor(illustrationColor, 'RGB')
                    )
                if self.headlineColor[i][j] == 1:
                    self.AllLayouts.putpixel(
                        (j,i),
                        ImageColor.getcolor(headlineColor, 'RGB')
                    )
                if self.titleColor[i][j] == 1:
                    self.AllLayouts.putpixel(
                        (j,i),
                        ImageColor.getcolor(titleColor, 'RGB')
                    )
                if self.textColor[i][j] == 1:
                    self.AllLayouts.putpixel(
                        (j,i),
                        ImageColor.getcolor(textColor, 'RGB')
                    )
        self.AllLayouts.save(outputPath + "_ALL_" + self.OriginalName)

    # overlay resulted matrix from previous function on original resized image
    def OverlayOnImage(self):
        self.OriginalImage = self.OriginalImage.convert("RGBA")
        self.AllLayouts = self.AllLayouts.convert("RGBA")
        resultedImage = Image.blend(self.OriginalImage, self.AllLayouts, 0.5)
        resultedImage.save(outputPath + "__OVERLAY_" + self.OriginalName)

    # remove small objects
    def RemoveSmallObjects(self):
        self.illustrationImage = Image.new(mode = 'RGB', size = (self.cols, self.rows))
        self.headlineImage = Image.new(mode = 'RGB', size = (self.cols, self.rows))
        self.titleImage = Image.new(mode = 'RGB', size = (self.cols, self.rows))
        self.textImage = Image.new(mode = 'RGB', size = (self.cols, self.rows))


        ilustrationIMG=applyOnLayout(outputPath+"_ILLUSTRATION_"+self.OriginalName)
        headlineIMG=applyOnLayout(outputPath+"_HEADLINE_"+self.OriginalName)
        titleIMG=applyOnLayout(outputPath+"_TITLE_"+self.OriginalName)
        textIMG=applyOnLayout(outputPath+"_TEXT_"+self.OriginalName)

        for i in range(self.rows):
            for j in range(self.cols):
                if ilustrationIMG[i][j] > 0:
                    self.illustrationImage.putpixel((j,i), ImageColor.getcolor(illustrationColor, 'RGB'))
                if headlineIMG[i][j] > 0:
                    self.headlineImage.putpixel((j,i), ImageColor.getcolor(headlineColor, 'RGB'))
                if titleIMG[i][j] > 0:
                    self.titleImage.putpixel((j,i), ImageColor.getcolor(titleColor, 'RGB'))
                if textIMG[i][j] > 0:
                    self.textImage.putpixel((j,i), ImageColor.getcolor(textColor, 'RGB'))


        self.illustrationImage.save(outputPath+"_ILLUSTRATION_"+self.OriginalName)
        self.headlineImage.save(outputPath+"_HEADLINE_"+self.OriginalName)
        self.titleImage.save(outputPath+"_TITLE_"+self.OriginalName)
        self.textImage.save(outputPath+"_TEXT_"+self.OriginalName)

    # draw rectangles on image
    def ApplyBorders(self):
        self.OriginalUbyte = img_as_ubyte(imread(resizedPath + "_RESIZED_" + self.OriginalName, as_gray=False))
        self.IllustrationUbyte = img_as_ubyte(imread(outputPath+"_ILLUSTRATION_"+self.OriginalName, as_gray=True))
        self.HeadlineUbyte = img_as_ubyte(imread(outputPath+"_HEADLINE_"+self.OriginalName, as_gray=True))
        self.TitleUbyte = img_as_ubyte(imread(outputPath+"_TITLE_"+self.OriginalName, as_gray=True))
        self.TextUbyte = img_as_ubyte(imread(outputPath+"_TEXT_"+self.OriginalName, as_gray=True))
        self.AllUbyte = img_as_ubyte(imread(outputPath+"_ALL_"+self.OriginalName, as_gray=True))
        
        illustration_label=label(self.IllustrationUbyte)
        headline_label=label(self.HeadlineUbyte)
        title_label=label(self.TitleUbyte)
        text_label=label(self.TextUbyte)

        fig, ax = plt.subplots(figsize=(self.cols/100, self.rows/100))
        ax.imshow(self.OriginalUbyte)

        for region in regionprops(illustration_label):
            if region.area >= regionArea:
                minr, minc, maxr, maxc = region.bbox

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor=illustrationColor, linewidth=2)
                ax.add_patch(rect)
        
        for region in regionprops(headline_label):
            if region.area >= regionArea:
                minr, minc, maxr, maxc = region.bbox

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor=headlineColor, linewidth=2)
                ax.add_patch(rect)

        for region in regionprops(title_label):
            if region.area >= regionArea:
                minr, minc, maxr, maxc = region.bbox

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor=titleColor, linewidth=2)
                ax.add_patch(rect)

        for region in regionprops(text_label):
            if region.area >= regionArea:
                minr, minc, maxr, maxc = region.bbox

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor=textColor, linewidth=2)
                ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(outputPath+"__BORDER_"+self.OriginalName)