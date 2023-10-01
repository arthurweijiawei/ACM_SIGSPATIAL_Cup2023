import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask

#load gpkg
vector_gdf = gpd.read_file(r'E:\ACM\Final_ChongCi\FinalOutPut\PredictLake_0619.gpkg')
outputpackage=r'E:\ACM\Final_ChongCi\FinalOutPut\PredictLake_0619_JudgeSnow_Selected_Final_Lake.gpkg'

#load tif
#tif_file = 'D:\SelfStudy\ACM_contest2023\All_TraningData_Prepare\Image\Greenland26X_22W_Sentinel2_2019-06-03_05.tif'
tif_file = 'D:\SelfStudy\ACM_contest2023\All_TraningData_Prepare\Image\Greenland26X_22W_Sentinel2_2019-06-19_20.tif'
#tif_file = 'D:\SelfStudy\ACM_contest2023\All_TraningData_Prepare\Image\Greenland26X_22W_Sentinel2_2019-07-31_25-004.tif'
#tif_file = 'D:\SelfStudy\ACM_contest2023\All_TraningData_Prepare\Image\Greenland26X_22W_Sentinel2_2019-08-25_29-001.tif'


src = rasterio.open(tif_file)


# Define a function to retrieve RGB values at specified coordinates
def get_rgb_values(x, y):
    # Use Rasterio to read band values at the specified coordinates
    tif_data = src.sample([(x, y)])
    tif_values = next(tif_data)
    r = tif_values[0]
    g = tif_values[1]
    b = tif_values[2]
    return r, g, b


# Read the geodataframe and tif image
vector_gdf = gpd.read_file('your_geodataframe.geojson')
tif_path = 'your_image.tif'

# Define a list to store the indices of features to delete
to_delete = []

# Open the tif image
with rasterio.open(tif_path) as src:
    # Loop through each feature
    for index, row in vector_gdf.iterrows():
        # Get the geometry of the feature
        geom = row['geometry']

        # Buffer the geometry by 100 meters
        buffered_geom = geom.buffer(100)

        # Simplify the boundary of the buffered geometry with a tolerance of 10 (you can adjust as needed)
        simplified_geom = buffered_geom.simplify(10)

        # Calculate the average RGB values for all points within the simplified geometry
        r_sum = 0.0
        g_sum = 0.0
        b_sum = 0.0
        ndsi_sum = 0.0
        point_count = 0

        for x, y in simplified_geom.exterior.coords:
            r, g, b = get_rgb_values(x, y)
            r_sum += r
            g_sum += g
            b_sum += b
            ndsi_sum += (b - r) / (b + r)
            point_count += 1

        if point_count > 0:
            r_avg = r_sum / point_count
            g_avg = g_sum / point_count
            b_avg = b_sum / point_count
            ndsi_avg = ndsi_sum / point_count
        else:
            r_avg, g_avg, b_avg, ndsi_avg = 0, 0, 0, 0

        # Update the attributes of the vector feature
        vector_gdf.at[index, 'R_buffer'] = r_avg
        vector_gdf.at[index, 'G_buffer'] = g_avg
        vector_gdf.at[index, 'B_buffer'] = b_avg
        vector_gdf.at[index, 'NDSI_Single_buffer'] = ndsi_avg

# Loop through features again
for idx, row in vector_gdf.iterrows():
    # Get the geometry of the feature
    geom = row['geometry']

    # Simplify the boundary of the geometry with a tolerance of 10 (you can adjust as needed)
    boundary_geom = geom.simplify(10)

    # Calculate the average RGB values for all points within the boundary geometry
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0
    ndsi_sum = 0.0
    point_count = 0

    for x, y in boundary_geom.exterior.coords:
        r, g, b = get_rgb_values(x, y)
        r_sum += r
        g_sum += g
        b_sum += b
        point_count += 1

    if point_count > 0:
        r_avg = r_sum / point_count
        g_avg = g_sum / point_count
        b_avg = b_sum / point_count
    else:
        r_avg, g_avg, b_avg = 0, 0, 0

    # Update the attributes of the vector feature
    vector_gdf.at[idx, 'R_boundary'] = r_avg
    vector_gdf.at[idx, 'G_boundary'] = g_avg
    vector_gdf.at[idx, 'B_boundary'] = b_avg

# Define a list to store the indices of features to delete
to_delete = []

# Open the tif image
with rasterio.open(tif_file) as src:
    # Loop through each feature
    for index, row in vector_gdf.iterrows():
        # Get the geometry of the feature
        geom = row['geometry']

        # Use rasterio's mask function to extract image data within the feature's boundary
        out_image, out_transform = mask(src, [geom], crop=True)

        # Calculate the mean values for each band
        r_band = out_image[0]  # R band
        g_band = out_image[1]  # G band
        b_band = out_image[2]  # B band

        r_mean = np.mean(r_band)
        g_mean = np.mean(g_band)
        b_mean = np.mean(b_band)
        r_std = np.std(r_band)
        g_std = np.std(g_band)
        b_std = np.std(b_band)

        r_cv = r_std / np.mean(r_band)
        g_cv = g_std / np.mean(g_band)
        b_cv = b_std / np.mean(b_band)

        # Update the attributes of the vector feature
        vector_gdf.at[index, 'R_Inter_Std'] = r_std
        vector_gdf.at[index, 'G_Inter_Std'] = g_std
        vector_gdf.at[index, 'B_Inter_Std'] = b_std
        vector_gdf.at[index, 'R_Inter_CV'] = r_cv
        vector_gdf.at[index, 'G_Inter_CV'] = g_cv
        vector_gdf.at[index, 'B_Inter_CV'] = b_cv

        # Update the attributes of the vector feature with mean values
        vector_gdf.at[index, 'R_Inter_Mean'] = r_mean
        vector_gdf.at[index, 'G_Inter_Mean'] = g_mean
        vector_gdf.at[index, 'B_Inter_Mean'] = b_mean

        # Check conditions and add indices of features to delete to the list
        if (r_cv > 0.8 and g_cv > 0.8 and b_cv > 0.8) and (
                np.mean(r_band) > 100 and np.mean(g_band) > 100 and np.mean(b_band) > 100):
            to_delete.append(index)

# Delete features based on the indices in the to_delete list
vector_gdf.drop(to_delete, inplace=True)

# Filter conditions
condition = (vector_gdf['NDSI_Single_buffer'] < 0.5)  # (vector_gdf['NDSI_Single_buffer'] > 0.02) &
filtered_vector_gdf = vector_gdf[condition]

condition = (filtered_vector_gdf['R_boundary'] < 150) | (filtered_vector_gdf['G_boundary'] < 150) | (
            filtered_vector_gdf['B_boundary'] < 150)

# Apply filter conditions and create a new GeoDataFrame
filtered_vector_gdf2 = filtered_vector_gdf[~condition]

filtered_vector_gdf.to_file(outputpackage, driver='GPKG')