This repository contains the neural network model (UNet) and other essential codes for segmenting lakes in Greenland.

## Code environment
See environment.yml &  requirements.txt

## Structure
The code is structured in Jupyter notebooks available in the MainProject/ folder. 
Each notebook contains a considerable part of the pipeline and they are supported with core libraries available in the notebooks/core directory. 
Input, output paths and other configurations for each notebook must be declared in the notebooks/config/ directory. 
### Step 1: Data preparation TrainingPart - [MainProject/1-Preprocessing-ForTrainingPart-Pan-AddRandomFishnet]
The Original data has two main components, the satellite images and the label of lakes in those images. 

Once the dataset is ready, start by declaring the relevant paths in the configuration. 
Declare the input paths and other relevant configurations in MainProject/config/Preprocessing.py file.
After declaring the required paths, this step try to split these images into smaller images by fishnet one by one. 
Lakes are splitted into 2-valued (1 and nodata) annotation maps based on the extent of fishnet.
We also generate a weighted boundary for each lake.
The splitted images training area and the corresponding annotation and boundary weights are then written to separate files.
All images have a resolution of 512*512

!!!!!--------!!!!!
Attention! Four images' training areas overlap with each other.In order to avoid conflict, we need to execute the preprocessing script four times. Each time for one images.

### Step 2: Model training - [MainProject/2-UNetTraining_AddAttention.ipynb]
Train the UNet model with the splitted images using the -UNetTraining_AddAttention.ipynb notebook
Declare the relevant configuration in MainProject/config/UNetTraining.py.
Models will be output as .h5 files.
#Step 2-Extra: Model evaluation
In case we use 2-Auxiliary-UNetEvaluation.ipynb to evaluate the performance of the model. 

### Step 3: Data preparation TestPart - [MainProject/3-Preprocessing-ForTestPart.ipynb]
Declare the relevant configuration in MainProject/config/PreprocessingTestPart.py. 
The Test Part images  also need to split by fishnet.
This step is quite similar as step1, except that there is no Lake data.

### Step 4: Model predict test part - [MainProject/4-RasterAnalysis-ForTestPart.ipynb]

Declare the relevant configuration in MainProject/config/RasterAnalysis.py
Next, use the trained model to analyze test part images.
The output file will be 2-valued (1 and nodata) imgs. Lakes and non-lakes

Similar to Step1, this part needs to be executed four times, each time corresponding to a image.

### Step 5: Data Post-Processing - [MainProject/5-1-Postprocessing-FillHole-DelArea.ipynb & MainProject/5-2-Postprocessing-Filter-Valid-Lake]

The relevant input and output path should be given in this two script.
Finally, the imgs will be convert into vector in this step.
We need to make some shape judgments on these elements. 
For example,
(1)Areas less than 0.1km² and aspect ratios that do not meet the requirements should be removed.
(2)The vectors should also not have internal holes. 
(3)Lakes in non-glacial areas and slush should not be included in the annotaton result.
    The RBG values of the images of these areas and the lake we need will show some differences. 
    Therefore they can be removed by correlation band analysis.
In short, in this step we deal with vectors that do not meet the requirements of the question.
Executed four times in this Step, each time corresponding to a image. 