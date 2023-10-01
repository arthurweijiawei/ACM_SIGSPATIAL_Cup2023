import sys
print(sys.version)
print(sys.executable)
#sys.setdefaultencoding('utf-8')
import tensorflow as tf
import numpy as np
from PIL import Image
import rasterio
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import imageio
import os
import time
import rasterio.warp             # Reproject raster samples
from functools import reduce
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


#from core.UNet import UNet
from core.UNetAttention import UNet
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from core.losses_FTL import focalTversky,accuracy,dice_coef,dice_loss,true_positives,false_positives,true_negatives,false_negatives,sensitivity,specificity,PA,IoU_Pos,IoU_Neg,mIoU,F1_Score
from core.optimizers import adaDelta
#, adagrad, adam, nadam #学习率优化过时
# from tensorflow.keras.optimizers import legacy

from core.frame_info import FrameInfo
from core.dataset_generator import DataGenerator
from core.split_frames import split_dataset
from core.visualize import display_images

import json
from sklearn.model_selection import train_test_split

#%matplotlib inline
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一块GPU（从0开始）


import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

#%reload_ext autoreload
#%autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# ——————
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

config1 = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
config1.gpu_options.allow_growth = True
session=tf.compat.v1.Session(config=config1)
#
# config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
#
# config.gpu_options.allow_growth = True
#
# session = tf.Session(config=config)
#
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# gpu_options =tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.6)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True, log_device_placement=False))
#gpu_options.allow_growth = True
#——————

print(tf.__version__)
# Required configurations (including the input and output paths) are stored in a separate file (such as config/UNetTraining.py)
# Please provide required info in the file before continuing with this notebook.

from config import UNetTraining

# In case you are using a different folder name such as configLargeCluster, then you should import from the respective folder
# Eg. from configLargeCluster import UNetTraining
config = UNetTraining.Configuration()
import tensorflow as tf
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

frames = []

all_files = os.listdir(config.base_dir)
#all_files=all_files[:len(all_files)//2]
all_files_pan = [fn for fn in all_files if fn.startswith(config.pan_fn) and fn.endswith(config.image_type)]
for i, fn in enumerate(all_files_pan):
    pan_img = rasterio.open(os.path.join(config.base_dir, fn))
    #pan_img = rasterio.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.pan_fn)))
    #read_ndvi_img = ndvi_img.read()
    read_pan_img = pan_img.read()
    # print(read_pan_img.shape)
    # (3, 524, 524)
    #comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
    #comb_img = np.transpose(pan_img, axes=(1,2,0)) #Channel at the end
    #print(fn)
    annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.pan_fn,config.annotation_fn)[:-4]+".tif"))
    annotation = np.array(annotation_im)
    # print(annotation.shape)
    # (524, 524)
    weight_im = Image.open(os.path.join(config.base_dir, fn.replace(config.pan_fn,config.weight_fn)[:-4]+".tif"))
    weight = np.array(weight_im)
    # print(weight.shape)
    # (524, 524)
    f = FrameInfo(read_pan_img, annotation, weight)
    #print(f)
    frames.append(f)
# print(len(frames))

training_frames, validation_frames, testing_frames  = split_dataset(frames, config.frames_json, config.patch_dir)
# training_frames = validation_frames = testing_frames  = list(range(len(frames)))
# print(len(training_frames))
# print(len(validation_frames))
# print(len(testing_frames))
annotation_channels = config.input_label_channel + config.input_weight_channel
train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)
val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
OPTIMIZER = adaDelta
# LOSS = tversky
LOSS = focalTversky
#Only for the name of the model in the very end
OPTIMIZER_NAME = 'AdaDelta'
# LOSS_NAME = 'weightmap_tversky'
LOSS_NAME = 'weightmap_focalTversky'

# Declare the path to the final model
# If you want to retrain an exising model then change the cell where model is declared.
# This path is for storing a model after training.

timestr = time.strftime("%Y%m%d-%H%M")
chf = config.input_image_channel + config.input_label_channel
chs = reduce(lambda a,b: a+str(b), chf, '')

print("0")
if not os.path.exists(config.model_path):
    os.makedirs(config.model_path)
model_path = os.path.join(config.model_path,'trees_{}_{}_{}_{}_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0]))

# The weights without the model architecture can also be saved. Just saving the weights is more efficent.

# weight_path="./saved_weights/UNet/{}/".format(timestr)
# if not os.path.exists(weight_path):
#     os.makedirs(weight_path)
# weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
# print(weight_path)

# Define the model and compile it

print("1")
print(config.BATCH_SIZE)
print(config.input_shape)
print(config.input_label_channel)
model = UNet([config.BATCH_SIZE, *config.input_shape],config.input_label_channel)
#model = model.cuda()
print("1.5")
#focalTversky,accuracy,dice_coef,dice_loss,true_positives,false_positives,true_negatives,false_negatives,sensitivity,specificity,PA,IoU_Pos,IoU_Neg,mIoU,F1_Score
#model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[accuracy,dice_coef,dice_loss,true_positives,false_positives,true_negatives,false_negatives,sensitivity,specificity,PA,IoU_Pos,IoU_Neg,mIoU,F1_Score])
print("2")
# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard


checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = False)
print("3")
#reduceonplatea; It can be useful when using adam as optimizer
#Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
#cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
#reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                   # patience=4, verbose=1, mode='min',
                                   # min_delta=0.0001, cooldown=4, min_lr=1e-16)

#early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)

log_dir = os.path.join(config.model_path,'UNet_{}_{}_{}_{}_{}'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, config.input_shape[0]))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

reduceLROnPlat =ReduceLROnPlateau(monitor='val_loss',factor=0.33,patience=5,verbose=1,mode='min',min_delta=0.001,cooldown=4,min_lr=1e-16)
early = EarlyStopping(monitor="val_loss",mode="min",verbose=2,patience=50)


callbacks_list = [checkpoint, tensorboard,early] #reduceLROnPlat is not required with adaDelta


#with tf.device('/gpu:0'):
loss_history = [model.fit(train_generator,
                         steps_per_epoch=config.MAX_TRAIN_STEPS,
                         epochs=config.NB_EPOCHS,
                         validation_data=val_generator,
                         validation_steps=config.VALID_IMG_COUNT,
                         callbacks=callbacks_list,
                         workers=1,
                          )]
