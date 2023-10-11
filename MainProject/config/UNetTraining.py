import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the ndvi, pan and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        self.base_dir = r'E:\ACM\PreprocessingResult\Data0927'
        #self.base_dir = 'D:\SelfStudy\ACM_contest2023\O_Data\Result_0911'
        self.image_type = '.tif'
        self.pan_fn = 'pan'
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.dem_fn = 'dem'
        
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        #self.patch_size = (256,256,5) # Height * Width * (Input + Output) channels
        self.patch_size = (512, 512, 6)  # Height * Width * (Input + Output) channels
        # # When stratergy == sequential, then you need the step_size as well
        # step_size = (128,128)
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio. 
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 0.4

        
        # The split of training areas into training, validation and testing set, is cached in patch_dir.
        self.patch_dir = r'D:\SelfStudy\ACM_contest2023\ankitkariryaa-An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel-eeeb76f\notebooks\patches128\patches{}'.format(self.patch_size[0])
        self.frames_json = os.path.join(self.patch_dir,'frames_list.json')


        # Shape of the input data, height*width*channel; Here channels are NVDI and Pan
        self.input_shape = (512,512,4)
        self.input_image_channel = [0,1,2,3]
        self.input_label_channel = [4]
        self.input_weight_channel = [5]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 6
        self.NB_EPOCHS = 80

        # number of validation images to use
        self.VALID_IMG_COUNT = 200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 50
        self.model_path = r'D:\SelfStudy\ACM_contest2023\ModelResult\U0916\\'

