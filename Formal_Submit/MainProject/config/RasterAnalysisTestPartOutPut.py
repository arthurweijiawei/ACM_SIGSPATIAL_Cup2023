# Configuration of the parameters for the 3-FinalRasterAnalysis.ipynb notebook
class Configuration:
    '''
    Configuration for the notebook where objects are predicted in the image.
    Copy the configTemplate folder and define the paths to input and output data.
    '''

    def __init__(self):
        # Input related variables
        self.input_image_dir = r'E:\ACM\Final_ChongCi\FinalOutPut\TestPartResult0930_0825_New'
        self.input_image_type = '.tif'
        self.pan_fn_st = 'pan_'
        self.trained_model_path = r'D:\SelfStudy\ACM_contest2023\Formal_Submit\Training_ModelFile\trees_20230930-2149_AdaDelta_weightmap_focalTversky_0123_512.h5'

        # Output related variables
        self.output_dir = 'E:\ACM\Final_ChongCi\FinalOutPut\FinalResult_0825'
        self.output_image_type = '.tif'
        self.output_prefix = 'det_'
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files = True
        self.output_dtype = 'uint8'

        # Variables related to batches and model
        self.BATCH_SIZE = 8  # Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH = 512  # Should be same as the WIDTH used for training the model
        self.HEIGHT = 512  # Should be same as the HEIGHT used for training the model
        self.STRIDE = 512  # 224 or 196   # STRIDE = WIDTH means no overlap, STRIDE = WIDTH/2 means 50 % overlap in prediction
