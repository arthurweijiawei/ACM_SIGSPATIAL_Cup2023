
import os

# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons

        self.Preprocessing_count_label="333"   # "000" "111" "222" "333"
        self.training_base_dir = r'E:\ACM\OriginalAreaPolygon'

        # self.study_area_fn = 'Area_0603.gpkg'
        # self.training_polygon_fn = 'Lake_0603.gpkg'
        #
        # self.study_area_fn = 'Area_0619.gpkg'
        # self.training_polygon_fn = 'Lake_0619.gpkg'

        # self.study_area_fn = 'Area_0731.gpkg'
        # self.training_polygon_fn = 'Lake_0731.gpkg'
        #
        self.study_area_fn = 'Area_0825.gpkg'
        self.training_polygon_fn = 'Lake_0825.gpkg'

        self.bands = [0,1,2] # img contain 3 bands
        self.raw_image_base_dir = 'D:\SelfStudy\ACM_contest2023\All_TraningData_Prepare\Image'
        self.raw_image_file_type = '.tif'
        # self.raw_pan_image_prefix = 'Greenland26X_22W_Sentinel2_2019-06-03_05'
        # self.raw_pan_image_prefix = 'Greenland26X_22W_Sentinel2_2019-06-19_20'
        # self.raw_pan_image_prefix = 'Greenland26X_22W_Sentinel2_2019-07-31_25-004'
        self.raw_pan_image_prefix = 'Greenland26X_22W_Sentinel2_2019-08-25_29-001'
        #self.raw_dem_image_prefix = 'DEM'

        self.fishnet_base_dir = r'E:\ACM\PreprocessingResult\FishNet'

        # For writing the extracted images and their corresponding annotations and boundary file
        self.path_to_write = r'E:\ACM\Final2\AllinitialData'
        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.tif'
        #self.extracted_ndvi_filename = 'ndsi'
        self.extracted_pan_filename = 'pan'
        self.extracted_annotation_filename = 'annotation'
        self.extracted_boundary_filename = 'boundary'
        #self.extracted_dem_filename = 'dem'

        # Path to write should be a valid directory
        assert os.path.exists(self.path_to_write)

        if not len(os.listdir(self.path_to_write)) == 0:
            print('Warning: path_to_write is not empty! The old files in the directory may not be overwritten!!')