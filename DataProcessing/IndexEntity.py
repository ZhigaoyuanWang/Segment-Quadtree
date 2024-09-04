import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Load the shapefile
shapefile_path = '/data/zgyw/co2emission/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp'
gdf = gpd.read_file(shapefile_path)

# List of European countries
european_countries = [
    'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
    'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany',
    'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania',
    'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland',
    'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland',
    'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'
]

# Filter the GeoDataFrame for European countries
europe_gdf = gdf[gdf['admin'].isin(european_countries)]

# Dissolve by 'admin' to get country boundaries instead of provinces
europe_gdf_dissolved = europe_gdf.dissolve(by='admin')

# Create a custom colormap that transitions from light green to white
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#90EE90'])

# Assign a value to represent the current states (1 represents current states, 0 means total white)
value = np.ones(len(europe_gdf_dissolved)) * 0.1  # Replace with your actual data

# Normalize the values to range [0, 1]
norm = mcolors.Normalize(vmin=0, vmax=1)

# Plot the countries with filled areas and colored by the value
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Set the longitude and latitude limits for Western Europe
minx, miny, maxx, maxy = -10, 35, 25, 70  # Approximate Western Europe bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Plot the dissolved GeoDataFrame with filled areas colored by the value
europe_gdf_dissolved.plot(ax=ax, column=value, cmap=cmap, norm=norm, edgecolor='grey', linewidth=0.5)

# Remove the title and scale labels
ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Define grid line spacing
x_spacing = 2  # Distance between vertical grid lines
y_spacing = 2  # Distance between horizontal grid lines

# Draw vertical grid lines
for x in np.arange(minx, maxx, x_spacing):
    ax.plot([x, x], [miny, maxy], color='red', linestyle='-', linewidth=1)

# Draw horizontal grid lines
for y in np.arange(miny, maxy, y_spacing):
    ax.plot([minx, maxx], [y, y], color='red', linestyle='-', linewidth=1)

plt.show()
