import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the shapefile
shapefile_path = '/path/shapefile'
gdf = gpd.read_file(shapefile_path)

# Filter the GeoDataFrame for provinces in China
china_gdf = gdf[gdf['admin'] == 'China']

# Assign a unique number to each province
china_gdf = china_gdf.reset_index(drop=True)  # Ensure the index is reset for consistent numbering
china_gdf['province_id'] = range(1, len(china_gdf) + 1)
num_provinces = len(china_gdf)

# Define the spatial resolution and extent
x_min, y_min, x_max, y_max = china_gdf.total_bounds
pixel_size = 0.1  # Define the pixel size in degrees

width = int((x_max - x_min) / pixel_size)
height = int((y_max - y_min) / pixel_size)
transform = from_origin(x_min, y_max, pixel_size, pixel_size)

# Function to retrieve province ID
def get_province_id(province):
    return province['province_id']

# Rasterize the GeoDataFrame
shapes = ((geom, value) for geom, value in zip(china_gdf.geometry, china_gdf.province_id))
raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int32')

# Create a custom colormap
viridis = plt.cm.get_cmap('viridis', num_provinces + 1)
newcolors = viridis(np.linspace(0, 1, num_provinces + 1))
newcolors[0] = [1, 1, 1, 1]  # Set the first color to white (R=1, G=1, B=1, A=1)
custom_cmap = ListedColormap(newcolors)

# Save the raster array as an image
plt.imshow(raster, extent=[x_min, x_max, y_min, y_max], cmap=custom_cmap)
plt.colorbar(label='Province ID')
plt.title('Provinces of China with Unique IDs')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Save the x, y coordinates and province ID array
x_coords = np.linspace(x_min, x_max, width)
y_coords = np.linspace(y_min, y_max, height)

# Flatten the arrays for saving or further processing
x_coords_flat = np.tile(x_coords, height)
y_coords_flat = np.repeat(y_coords, width)
province_ids_flat = raster.flatten()

# Combine x, y, and province IDs into a single array for saving
result_array = np.column_stack((x_coords_flat, y_coords_flat, province_ids_flat))

# Save the result array to a file or use it for further processing
# np.save('province_ids.npy', result_array)

raster2 = raster.copy()
raster2[-22:] = 30
plt.imshow(raster2, extent=[x_min, x_max, y_min, y_max], cmap=custom_cmap)
plt.colorbar(label='Province ID')
plt.title('Provinces of China with Unique IDs')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

raster3 = raster.copy()
raster3 = raster3[:-22]
plt.imshow(raster3, extent=[x_min, x_max, 17.98086606170566, y_max], cmap=custom_cmap)
plt.colorbar(label='Province ID')
plt.title('Provinces of China with Unique IDs')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Parameters
N = 32  # Number of divisions

# Calculate the step sizes for the grid lines
x_step = (x_max - x_min) / N
y_step = (y_max - 17.98086606170566) / N

# Plot the image
plt.imshow(raster3, extent=[x_min, x_max, 17.98086606170566, y_max], cmap=custom_cmap)
plt.colorbar(label='Province ID')
plt.title('Provinces of China with Unique IDs')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add vertical grid lines
for i in range(1, N):
    plt.axvline(x_min + i * x_step, color='red', linestyle='--', linewidth=0.5)

# Add horizontal grid lines
for j in range(1, N):
    plt.axhline(17.98086606170566 + j * y_step, color='red', linestyle='--', linewidth=0.5)

plt.show()

 Parameters
N = 32  # Number of divisions

arr_x_max = 611
arr_x_min = 0
arr_y_max = 355
arr_y_min = 0

# Calculate the step sizes for the grid lines
x_step = (arr_x_max - arr_x_min) / N
y_step = (arr_y_max - arr_y_min) / N

# Initialize the array to store the most frequent colors
grid_colors = np.zeros((N, N), dtype=int)

# Loop through each grid cell
for i in range(N):
    for j in range(N):
        # Define the bounds of the current grid cell
        x_start = arr_x_min + i * x_step
        x_end = arr_x_min + (i + 1) * x_step
        y_start = arr_y_min + j * y_step
        y_end = arr_y_min + (j + 1) * y_step
        
        x_start = math.ceil(x_start)
        y_start = math.ceil(y_start)
        x_end = math.floor(x_end)
        y_end = math.floor(y_end)
        cell_pixels = raster3[y_start:y_end,x_start:x_end]
        
        # Determine the most frequent color in the current cell
        if len(cell_pixels) > 0:
            unique, counts = np.unique(cell_pixels, return_counts=True)
            most_frequent_color = unique[np.argmax(counts)]
            grid_colors[j, i] = most_frequent_color

# Plot the grid
plt.imshow(raster3, extent=[x_min, x_max, y_min, y_max], cmap=custom_cmap)
plt.colorbar(label='Province ID')
plt.title('Provinces of China with Unique IDs')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add vertical grid lines
for i in range(1, N):
    plt.axvline(x_min + i * x_step, color='red', linestyle='--', linewidth=0.5)

# Add horizontal grid lines
for j in range(1, N):
    plt.axhline(y_min + j * y_step, color='red', linestyle='--', linewidth=0.5)

plt.show()
