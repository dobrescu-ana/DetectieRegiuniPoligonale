from PIL import Image, ImageColor
from sklearn import preprocessing
import numpy as np
import scipy as sp
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

class ImageItem:
    def __init__(self, originalImageName="", headlineCSVName="", illustrationCSVName="", titleCSVName="", textCSVName=""):

        # save the original name of the image
        self.OriginalName = originalImageName

        # get image dimensions by reading one related csv
        headlineFile = open(csvPath+headlineCSVName,"r")
        self.rows = -1
        self.cols = -1

        for line in headlineFile:
            line = line.split(",")
            self.cols=line.__len__()
            self.rows += 1
        headlineFile.close()

        # process the csv which contains the pixel probabilities for the headline class
        headlineFile = open(csvPath+headlineCSV,"r")
        self.headline = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.headlineColor = [[0 for i in range(self.cols)] for j in range(self.rows)]

        # convert headline probability matrix from string to float
        for i in range(self.rows):
            row = headlineFile.readline()
            row = row.strip()
            row = row.split(",")
            for j in range(self.cols):
                self.headline[i][j] = float(row[j])
        
        #close the headline probability csv file
        headlineFile.close()

        # process the csv which contains the pixel probabilities for the illustration class
        illustrationFile = open(csvPath+illustrationCSVName,"r")
        self.illustration = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.illustrationColor = [[0 for i in range(self.cols)] for j in range(self.rows)]

        # convert illustration probability matrix from string to float
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
                self.title[i][j]=float(row[j])
        
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
                self.text[i][j]=float(row[j])
        
        # close the text probability csv file
        textFile.close()

        # open original image
        image = Image.open(imagePath+originalImageName)

        # resize the original image to fit the dimensions of the csvs pixel matrix
        resizedImage = image.resize((self.cols, self.rows))

        # save the resized image
        resizedImage.save(resizedImage + "_RESIZED_" + originalImageName)

        # the input image will be the resized one in order to identify the vertices of the polygons
        self.OriginalImage = Image.open(resizedImage + "_RESIZED_" + originalImageName)

    def Normalize(self):
        min_max_scalar = preprocessing.MinMaxScaler()
        self.illustration = min_max_scalar.fit_transform(self.illustration)
        self.headline = min_max_scalar.fit_transform(self.headline)
        self.title = min_max_scalar.fit_transform(self.title)
        self.text = min_max_scalar.fit_transform(self.text)