# **1. Temperate Forests - Dark Green(#2E8B57)**
#    Temperature Range: -15°C ~ 30°C
#    Altitude Range: 0m ~ 1500m
#    Humidity Range: 60% ~ 80%
    
# **2. Tropical Rainforests - Bright Green(#32CD32)**
#    Temperature Range: 20°C ~ 35°C
#    Altitude Range: 0m ~ 1000m
#    Humidity Range: 75% ~ 100%
    
# **3. Arid Deserts - Sandy Yellow(#EDC9AF)**
#    Temperature Range: 10°C ~ 50°C
#    Altitude Range: -50m ~ 1000m
#    Humidity Range: 10% ~ 30%
    
# **4. Rocky Deserts - Gray Brown(#C2B280)**
#    Temperature Range: 5°C ~ 45°C
#    Altitude Range: 0m ~ 1500m
#    Humidity Range: 15% ~ 35%
    
# **5. Prairies - Grass Green(#7CFC00)**
#    Temperature Range: -10°C ~ 30°C
#    Altitude Range: 0m ~ 1500m
#    Humidity Range: 30% ~ 60%
    
# **6. Hilly Grasslands - Olive Green(#556B2F)**
#    Temperature Range: -5°C ~ 25°C
#    Altitude Range: 500m ~ 2000m
#    Humidity Range: 35% ~ 65%
    
# **7. Snow-capped Mountains - Ice Blue(#E0FFFF)**
#    Temperature Range: -40°C ~ 0°C
#    Altitude Range: 2000m ~ 6000m
#    Humidity Range: 30% ~ 60%
    
# **8. High Land - Light Brown(#D2B48C)**
#    Temperature Range: -15°C ~ 20°C
#    Altitude Range: 1000m ~ 4000m
#    Humidity Range: 20% ~ 50%
    
# **9. Freshwater Lakes - Light Blue(#87CEEB)**
#    Temperature Range: 5°C ~ 25°C
#    Altitude Range: 0m ~ 3000m
#    Humidity Range: 50% ~ 80%
    
# **10. Rivers - Blue(#46822B4)**
#    Temperature Range: 0°C ~ 25°C
#    Altitude Range: 0m ~ 3500m
#    Humidity Range: 60% ~ 90%
    
# **11. Oceans - Dark Blue(#000CD)**
#    Temperature Range: -2°C ~ 30°C
#    Altitude Range: 0m ~ 11000m
#    Humidity Range: 70% ~ 100%
    
# **12. Swamps - Dark Green-Brown(Mix of #556B2f & #6B8E23)**
#    Temperature Range: 10°C ~ 30°C
#    Altitude Range: 0m ~ 500m
#    Humidity Range: 80% ~ 100%
    
# **13. Active Volcanoes - Fiery Red(#FF4500)**
#    Temperature Range: 20°C ~ 600°C
#    Altitude Range: 500m ~ 3000m
#    Humidity Range: 20% ~ 60%
    
# **14. Dormant Volcanoes - Dark Gray(#696969)**
#    Temperature Range: -10°C ~ 30°C
#    Altitude Range: 500m ~ 3000m
#    Humidity Range: 20% ~ 60%
    
# **15. Volcanic Plains - Dark Brown(#2F4F4F)**
#    Temperature Range: 15°C ~ 40°C
#    Altitude Range: 0m ~ 1000m
#    Humidity Range: 20% ~ 40%
    
import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter, label, binary_dilation
import random
from collections import deque

# Set simulation parameters
width = 1000      # Width of the map in grid cells
height = 600      # Height of the map in grid cells
scale = 200.0     # Scale factor for Perlin noise (increase to make terrain smoother)
octaves = 6       # Number of noise octaves
persistence = 0.5 # Persistence value for Perlin noise
lacunarity = 2.0  # Lacunarity value for Perlin noise
seed = 37         # Seed for random number generation to ensure reproducibility

# Define river width in pixels
river_width = 1  # You can adjust this value as needed, e.g., 1, 5, 10, 15, etc.

# Define flow accumulation threshold to determine river presence
flow_threshold = 100  # Adjust this value based on desired river density, e.g., 50, 100, 200, etc.

# Initialize random seeds to ensure reproducible results
np.random.seed(seed)
random.seed(seed)

# Function to generate the elevation map using Perlin noise
def generate_elevation_map(width, height, scale, octaves, persistence, lacunarity, seed=0):
    elevation = np.zeros((height, width))
    repeatx = int(width / scale)  # Set repeatx to ensure seamless horizontal wrapping
    repeaty = 1024                 # Set repeaty to a large number as vertical wrapping is not needed
    for i in range(height):
        for j in range(width):
            elevation[i][j] = pnoise2(j / scale, 
                                      i / scale, 
                                      octaves=octaves, 
                                      persistence=persistence, 
                                      lacunarity=lacunarity, 
                                      repeatx=repeatx, 
                                      repeaty=repeaty, 
                                      base=seed)
    # Normalize elevation values to the range [0, 1]
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    return elevation

# Generate the elevation map based on defined parameters
elevation = generate_elevation_map(width, height, scale, octaves, persistence, lacunarity, seed)

# Calculate the sea level threshold to achieve the desired ocean coverage percentage
desired_ocean_percentage = 0.60  # Target ocean coverage as 60%
sea_level = np.percentile(elevation, desired_ocean_percentage * 100)
print(f"Sea level threshold for {desired_ocean_percentage*100}% ocean coverage: {sea_level:.4f}")

# Create a mask for ocean regions where elevation is below or equal to sea level
ocean_mask = elevation <= sea_level

# Function to identify lakes within the elevation map
def identify_lakes(elevation, sea_level, min_size=100):
    # Consider only areas above sea level
    land = elevation > sea_level
    # Label all connected land regions
    labeled, num_features = label(land)
    lakes = np.zeros_like(elevation, dtype=bool)
    
    for region in range(1, num_features + 1):
        mask = labeled == region
        # Find the lowest elevation point within the region
        min_elev = elevation[mask].min()
        # Calculate the average elevation of the surrounding area
        surrounding = np.copy(elevation)
        surrounding[mask] = np.nan
        avg_surrounding = np.nanmean(surrounding)
        # Determine if the region qualifies as a lake based on elevation and size
        if min_elev < avg_surrounding - 0.05 and np.sum(mask) > min_size:
            lakes = lakes | mask
    return lakes

# Identify lakes based on elevation and surrounding terrain
lakes_mask = identify_lakes(elevation, sea_level)

# Function to compute the flow direction for each cell based on elevation
def compute_flow_direction(elevation):
    flow_dir = np.full(elevation.shape, -1, dtype=int)  # Initialize with -1 indicating no flow direction
    # Direction encoding: 0=North, 1=Northeast, 2=East, 3=Southeast, 4=South, 5=Southwest, 6=West, 7=Northwest
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    for i in range(1, elevation.shape[0]-1):
        for j in range(1, elevation.shape[1]-1):
            min_elev = elevation[i][j]
            min_dir = -1
            for d, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                if elevation[ni][nj] < min_elev:
                    min_elev = elevation[ni][nj]
                    min_dir = d
            flow_dir[i][j] = min_dir
    return flow_dir

# Compute the flow direction for the entire elevation map
flow_dir = compute_flow_direction(elevation)

# Function to compute flow accumulation using the D8 algorithm
def compute_flow_accumulation(flow_dir):
    accumulation = np.ones(flow_dir.shape, dtype=int)  # Initialize accumulation with 1 for each cell
    in_degree = np.zeros(flow_dir.shape, dtype=int)
    
    # Direction encoding consistent with compute_flow_direction
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # Calculate the number of cells flowing into each cell (in-degree)
    for i in range(flow_dir.shape[0]):
        for j in range(flow_dir.shape[1]):
            dir = flow_dir[i][j]
            if dir != -1:
                di, dj = directions[dir]
                ni, nj = i + di, j + dj
                if 0 <= ni < flow_dir.shape[0] and 0 <= nj < flow_dir.shape[1]:
                    in_degree[ni][nj] += 1
    
    # Initialize a queue with cells that have no incoming flows
    queue = deque()
    for i in range(flow_dir.shape[0]):
        for j in range(flow_dir.shape[1]):
            if in_degree[i][j] == 0:
                queue.append((i, j))
    
    # Process the queue to accumulate flow
    while queue:
        i, j = queue.popleft()
        dir = flow_dir[i][j]
        if dir != -1:
            di, dj = directions[dir]
            ni, nj = i + di, j + dj
            if 0 <= ni < flow_dir.shape[0] and 0 <= nj < flow_dir.shape[1]:
                accumulation[ni][nj] += accumulation[i][j]
                in_degree[ni][nj] -= 1
                if in_degree[ni][nj] == 0:
                    queue.append((ni, nj))
    
    return accumulation

# Compute the flow accumulation based on flow directions
accumulation = compute_flow_accumulation(flow_dir)

# Optional: Function to visualize the flow accumulation map for debugging purposes
def plot_flow_accumulation(accumulation):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log1p(accumulation), cmap='Blues')
    plt.colorbar(label='Log Flow Accumulation')
    plt.title("Flow Accumulation (Log Scale)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# Uncomment the following line to visualize flow accumulation
# plot_flow_accumulation(accumulation)

# Function to extract the river network based on flow accumulation threshold
def extract_river_network(accumulation, flow_threshold):
    river_mask = accumulation >= flow_threshold
    return river_mask

# Extract rivers from the flow accumulation map
rivers_mask = extract_river_network(accumulation, flow_threshold)

# Function to ensure rivers are connected to existing water bodies (oceans or lakes)
def connect_rivers_to_water(rivers_mask, ocean_mask, lakes_mask):
    water_mask = ocean_mask | lakes_mask
    # Use binary dilation to ensure connectivity between rivers and water bodies
    structure = np.ones((3,3))
    rivers_connected = binary_dilation(rivers_mask, structure=structure)
    rivers_connected = rivers_connected & ~water_mask  # Keep rivers and water bodies separate
    return rivers_connected

# Ensure that rivers are properly connected to oceans or lakes
rivers_mask = connect_rivers_to_water(rivers_mask, ocean_mask, lakes_mask)

# Combine masks to represent all water bodies: oceans, lakes, and rivers
water_mask = ocean_mask | lakes_mask | rivers_mask

# Create a layered color image to represent different map features
# 1. Create the base map with oceans, lakes, and land
map_base = np.zeros((height, width, 3))
colors = {
    'ocean': [0, 0, 205],          # Ocean color (RGB)
    'lake': [70, 130, 180],        # Lake color (RGB)
    'land': [0, 0, 0]               # Land color (black)
}
map_base[ocean_mask] = np.array(colors['ocean']) / 255
map_base[lakes_mask] = np.array(colors['lake']) / 255
map_base[~water_mask] = np.array(colors['land']) / 255

# 2. Apply Gaussian blur to smooth the base map
map_base_blurred = gaussian_filter(map_base, sigma=1)

# 3. Create a river layer without applying blur
map_rivers = np.zeros((height, width, 3))
map_rivers[rivers_mask] = np.array([25, 25, 112]) / 255  # River color (RGB)

# 4. Merge the river layer into the blurred base map
map_final = map_base_blurred.copy()
map_final[rivers_mask] = map_rivers[rivers_mask]

# 5. Adjust the river width based on the defined river_width variable
if river_width > 1:
    # Use morphological operations to dilate rivers and achieve the desired width
    structure_size = river_width  # Diameter of the structuring element
    structure = np.zeros((structure_size, structure_size))
    radius = structure_size // 2
    y, x = np.ogrid[:structure_size, :structure_size]
    mask = (x - radius)**2 + (y - radius)**2 <= radius**2
    structure[mask] = 1
    
    # Dilate the river mask using the structuring element
    rivers_mask_dilated = binary_dilation(rivers_mask, structure=structure)
    
    # Create a dilated river layer with the updated mask
    map_rivers_dilated = np.zeros((height, width, 3))
    map_rivers_dilated[rivers_mask_dilated] = np.array([25, 25, 112]) / 255  # River color (RGB)
    
    # Merge the dilated river layer into the blurred base map
    map_final = map_base_blurred.copy()
    map_final[rivers_mask_dilated] = map_rivers_dilated[rivers_mask_dilated]
else:
    # If river_width is 1, keep rivers thin without dilation
    map_final = map_final.copy()  # Rivers are already drawn

# Create a legend for the map visualization
import matplotlib.patches as mpatches

legend_elements = [
    mpatches.Patch(color=np.array(colors['ocean'])/255, label='Ocean'),
    mpatches.Patch(color=np.array(colors['lake'])/255, label='Lake'),
    mpatches.Patch(color=np.array([25, 25, 112])/255, label='River'),
    mpatches.Patch(color=np.array(colors['land'])/255, label='Land')
]

# Visualize the final generated map with oceans, rivers, and lakes
plt.figure(figsize=(20, 12))
plt.imshow(map_final, origin='lower')
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1), title="Features", fontsize='medium')
plt.title("Generated Map with 60% Oceans, Rivers, and Lakes", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.tight_layout()
plt.show()

# Display the actual ocean coverage percentage achieved
actual_ocean_percentage = np.sum(ocean_mask) / (height * width) * 100
print(f"Actual ocean coverage: {actual_ocean_percentage:.2f}%")
