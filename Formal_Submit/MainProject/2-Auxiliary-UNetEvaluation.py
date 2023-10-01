import numpy as np               # numerical array manipulation
import pandas as pd
import os
import time
from collections import defaultdict
from functools import reduce
from PIL import Image
import rasterio
#import fiona     # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape
# import fiona
import cv2
from tqdm import tqdm

from tensorflow.keras.models import load_model

from core.UNet import UNet
from core.losses_FTL import focalTversky,accuracy,dice_coef,dice_loss,true_positives,false_positives,true_negatives,false_negatives,sensitivity,specificity,PA,IoU_Pos,IoU_Neg,mIoU,F1_Score
from core.optimizers import adaDelta
#, adagrad, adam, nadam
from core.frame_info import FrameInfo
from core.dataset_generator import DataGenerator
from core.split_frames import split_dataset
from core.visualize import display_images

#%matplotlib inline
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches

import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

#reload_ext autoreload
#%autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import tensorflow as tf
print(tf.__version__)


# Initialize the data related variables used in the notebook

# For reading thepan and annotated images generated in the step  1
base_dir = r'D:\ACM\ModelResult0930_Final\PreDataSetFinal'
model_dir = r'D:\ACM\ModelResult0930_Final'
image_type = '.png'
annotation_weight_type = '.tif'
pan_fn = 'pan'
annotation_fn = 'annotation'
weight_fn = 'boundary'

# For testing, images are divided into sequential patches
patch_generation_stratergy = 'sequential'
patch_size = (512,512,5)
BATCH_SIZE = 8 # Model is evaluated in batches; See https://keras.io/models/model/#evaluate

# # When stratergy == sequential
step_size = (512,512)


# The data has four channels
# The order is [ PAN, ANNOTATION, WEIGHT]
input_shape = (512,512,3)
input_image_channel = [0,1,2]
input_label_channel = [3]
input_weight_channel = [4]

OPTIMIZER = adaDelta
LOSS = focalTversky

#Only for the name of the model in the very end
OPTIMIZER_NAME = 'adaDelta'
LOSS_NAME = 'weightmap_focalTversky'

modelToEvaluate =r'D:\ACM\ModelResult0930_Final\trees_20230930-2149_AdaDelta_weightmap_focalTversky_0123_512.h5'

#File path for final report
timestr = time.strftime("%Y%m%d-%H%M")
chf = input_image_channel + input_label_channel
chs = reduce(lambda a,b: a+str(b), chf, '')

evaluation_report_path = model_path = model_dir
if not os.path.exists(evaluation_report_path):
    os.makedirs(evaluation_report_path)
evaluation_report_filename = os.path.join(evaluation_report_path,'evaluation_per_pixel{}_{}.csv'.format(timestr,chs))

# Read all images/frames into memory
frames = []

all_files = os.listdir(base_dir)
all_files_pan = [fn for fn in all_files if fn.startswith(pan_fn) and fn.endswith(image_type)]
len(all_files_pan)
#dtype = {'F': np.float32, 'L': np.uint8}[pil_img.mode]

for i, fn in enumerate(tqdm(all_files_pan)):
    pan_img = rasterio.open(os.path.join(base_dir, fn))
    read_pan_img = pan_img.read()
    annotation_im = Image.open(os.path.join(base_dir, fn.replace(pan_fn,annotation_fn).replace(image_type,annotation_weight_type)))
    annotation = np.array(annotation_im)
    weight_im = Image.open(os.path.join(base_dir, fn.replace(pan_fn,weight_fn).replace(image_type,annotation_weight_type)))
    weight = np.array(weight_im)
    f = FrameInfo(read_pan_img, annotation, weight)
    frames.append(f)

# For testing on all frames. All sequential frames are kept in memory and this may create memory related errors in some cases.
testing_frames  = list(range(len(frames)))

annotation_channels = input_label_channel + input_weight_channel
test_generator = DataGenerator(input_image_channel, patch_size, testing_frames, frames, annotation_channels)

# Sequential generate all patches from the all frames
test_patches = test_generator.all_sequential_patches(step_size)
print('Total patches to evaluate the model on: ' + str(len(test_patches[0])))

#Display the some of the test images
numberOfImagesToDisplay = 10

train_images, real_label = test_patches[0][:numberOfImagesToDisplay], test_patches[1][:numberOfImagesToDisplay]
display_images(np.concatenate((train_images,real_label), axis = -1))

#Evaluate model
def evaluate_model(model_path, evaluation_report_filename):
    print(model_path, evaluation_report_filename)
    #model = UNet([config.BATCH_SIZE, *config.input_shape], config.input_label_channel)
    print("1")
    model = load_model(model_path, custom_objects={'focalTversky': focalTversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss,\
                                                   'true_positives':true_positives, 'false_positives':false_positives,'true_negatives':true_negatives, 'false_negatives':false_negatives,\
                                                   'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity, 'PA':PA, 'IoU_Pos':IoU_Pos, 'IoU_Neg':IoU_Neg,\
                                                   'mIoU':mIoU, 'F1_Score':F1_Score}, compile=False)
    print('2')
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[accuracy,dice_coef,dice_loss,true_positives,false_positives,true_negatives,false_negatives,sensitivity,specificity,PA,IoU_Pos,IoU_Neg,mIoU,F1_Score])
    print('Evaluating model now!')
    ev = model.evaluate(x=test_patches[0], y=test_patches[1], batch_size=8 ,verbose=1, use_multiprocessing=True)
    print('3')
    report  = dict(zip(model.metrics_names, ev))
    print('4')
    report['model_path'] =  model_path
    report['test_frame_dir']= base_dir
    report['total_patch_count']= len(test_patches[0])
    return report

report = evaluate_model(modelToEvaluate, evaluation_report_filename)


def transform_contours_to_xy(contours, transform):
    tp = []
    for cnt in contours:
        pl = cnt[:, 0, :]
        cols, rows = zip(*pl)
        x, y = rasterio.transform.xy(transform, rows, cols)
        tl = [list(i) for i in zip(x, y)]
        tp.append(tl)
    return (tp)


def mask_to_polygons(mask, transform, th=0.5):
    # first, find contours with cv2: it's much faster than shapely and returns hierarchy
    mask[mask < th] = 0
    mask[mask >= th] = 1
    mask = ((mask) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours from image coordinate to xy coordinate (world coordinates)
    contours = transform_contours_to_xy(contours, transform)

    if not contours:  # TODO: Raise an error maybe
        print('Warning: No contours/polygons detected!!')
        return [Polygon()]  # [Polygon()]

    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area/hole (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours:  # and cv2.contourArea(cnt) >= min_area: #Do we need to check for min_area??
            try:
                poly = Polygon(
                    shell=cnt,
                    holes=[c for c in cnt_children.get(idx, [])])
                # if cv2.contourArea(c) >= min_area]) #Do we need to check for min_area??
                all_polygons.append(poly)
            except Exception as e:
                #                 print(e)
                pass
    #     print(len(all_polygons))
    return (all_polygons)

