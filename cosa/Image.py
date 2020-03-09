# from matplotlib.pyplot import imread, imsave
from sklearn.cluster import KMeans
import warnings
import os
import sys
from random import randint
from skimage.io import imread, imsave

from .transform import k_representative_pallette,elastic_transform


class Image:

    def __init__(self):
        
        self.img = None
        self.transformed = None
    
    def read(self, filename = 'unprocessed/*.jpg'):

        self.img = imread(filename)

    def transform(self, fun = None, **args):

        functions = {
            'k_rep':k_representative_pallette, 
            'elastic':elastic_transform
            }

        if not fun:
            funcs = ['k_rep', 'elastic']
            fun = funcs[randint(0,len(funcs) - 1)]

        self.transformed = functions[fun](img = self.img, **args)

    def save(self, directory = 'processed'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        imsave(directory + '/processed.jpg', self.transformed)
