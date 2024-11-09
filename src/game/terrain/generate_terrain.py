#**1. Temperate Forests - Dark Green(#2E8B57)**
#    Temperature Range: -15c ~ 30c
#    Altitude Range: 0m ~ 1500m
#    Humidity Rrange: 60% ~ 80%

#**2. Tropical Rainforests - Bright Green(#32CD32)**
#    Temperature Range: 20c ~ 35c
#    Altitude Range: 0m ~ 1000m
#    Humidity Rrange: 75% ~ 100%

#**3. Arid Deserts - Sandy Yellow(#EDC9AF)**
#    Temperature Range: 10c ~ 50c
#    Altitude Range: -50m ~ 1000m
#    Humidity Rrange: 10% ~ 30%

#**4. Rocky Deserts - Gray Brown(#C2B280)**
#    Temperature Range: 5c ~ 45c
#    Altitude Range: 0m ~ 1500m
#    Humidity Rrange: 15% ~ 35%

#**5. Prairies - Grass Green(#7CFC00)**
#    Temperature Range: -10c ~ 30c
#    Altitude Range: 0m ~ 1500m
#    Humidity Rrange: 30% ~ 60%

#**6. Hilly Grasslands - Olive Green(#556B2F**
#    Temperature Range: -5c ~ 25c
#    Altitude Range: 500m ~ 2000m
#    Humidity Rrange: 35% ~ 65%

#**7. Snow-capped Mountains - Ice Blue(#E0FFFF)**
#    Temperature Range: -40c ~ 0c
#    Altitude Range: 2000m ~ 6000m
#    Humidity Rrange: 30% ~ 60%

#**8. High Land - Light Brown(#D2B48C)**
#    Temperature Range: -15c ~ 20c
#    Altitude Range: 1000m ~ 4000m
#    Humidity Rrange: 20% ~ 50%

#**9. Freshwater Lakes - Light Blue(#87CEEB)**
#    Temperature Range: 5c ~ 25c
#    Altitude Range: 0m ~ 3000m
#    Humidity Rrange: 50% ~ 80%

#**10. Rivers - Blue(#46822B4)**
#    Temperature Range: 0c ~ 25c
#    Altitude Range: 0m ~ 3500m
#    Humidity Rrange: 60% ~ 90%

#**11. Oceans - Dark Blue(#000CD)**
#    Temperature Range: -2c ~ 30c
#    Altitude Range: 0m ~ 11000m
#    Humidity Rrange: 70% ~ 100%

#**12. Swamps - Dark Green-Brown(Mix of #556B2f & #6B8E23)**
#    Temperature Range: 10c ~ 30c
#    Altitude Range: 0m ~ 500m
#    Humidity Rrange: 80% ~ 100%

#**13. Active Volcanoes - Fiery Red(#FF4500)**
#    Temperature Range: 20c ~ 600c
#    Altitude Range: 500m ~ 3000m
#    Humidity Rrange: 20% ~ 60%

#**14. Dormant Vocanoes - Dark Gray(#696969)**
#    Temperature Range: -10c ~ 30c
#    Altitude Range: 500m ~ 3000m
#    Humidity Rrange: 20% ~ 60%

#**15. Volcanic Plains - Dark Brown(#2F4F4F)**
#    Temperature Range: 15c ~ 40c
#    Altitude Range: 0m ~ 1000m
#    Humidity Rrange: 20% ~ 40%

import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter, label, binary_dilation
import random
from collections import deque

# 设置参数
width = 1000      # 地图宽度（栅格数量）
height = 600      # 地图高度（栅格数量）
scale = 200.0     # Perlin 噪声的缩放比例（增大scale使地形更平缓）
octaves = 6
persistence = 0.5
lacunarity = 2.0
seed = 42         # 随机种子

# 河流宽度（像素）
river_width = 1  # 您可以根据需要调整此值，例如1、5、10、15等

# 流量积累阈值（用于确定河流）
flow_threshold = 100  # 您可以根据需要调整此值，例如50、100、200等

# 设置随机种子以确保结果可重复
np.random.seed(seed)
random.seed(seed)

# 生成高程图
def generate_elevation_map(width, height, scale, octaves, persistence, lacunarity, seed=0):
    elevation = np.zeros((height, width))
    repeatx = int(width / scale)  # 设置repeatx以确保左右边界无缝连接
    repeaty = 1024                 # 纵向不需要无缝连接，可以设为一个较大的数
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
    # 标准化到0-1范围
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    return elevation

elevation = generate_elevation_map(width, height, scale, octaves, persistence, lacunarity, seed)

# 计算海平面阈值以达到60%的海洋覆盖率
desired_ocean_percentage = 0.60
sea_level = np.percentile(elevation, desired_ocean_percentage * 100)
print(f"Sea level threshold for {desired_ocean_percentage*100}% ocean coverage: {sea_level:.4f}")

# 标记海洋
ocean_mask = elevation <= sea_level

# 识别湖泊：低洼区域，高于海平面，并且周围有水
def identify_lakes(elevation, sea_level, min_size=100):
    # 只考虑高于海平面的区域
    land = elevation > sea_level
    # 标记所有高于海平面的连通区域
    labeled, num_features = label(land)
    lakes = np.zeros_like(elevation, dtype=bool)
    
    for region in range(1, num_features + 1):
        mask = labeled == region
        # 计算区域内的最低点
        min_elev = elevation[mask].min()
        # 计算该区域的平均周围高程
        surrounding = np.copy(elevation)
        surrounding[mask] = np.nan
        avg_surrounding = np.nanmean(surrounding)
        # 如果区域内最低点显著低于周围平均高程，且区域足够大，则认为是湖泊
        if min_elev < avg_surrounding - 0.05 and np.sum(mask) > min_size:
            lakes = lakes | mask
    return lakes

lakes_mask = identify_lakes(elevation, sea_level)

# 计算流向
def compute_flow_direction(elevation):
    flow_dir = np.full(elevation.shape, -1, dtype=int)  # -1表示无流向
    # 方向编码：0=北, 1=东北, 2=东, 3=东南, 4=南, 5=西南, 6=西, 7=西北
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

flow_dir = compute_flow_direction(elevation)

# 计算流量积累（基于 D8 算法）
def compute_flow_accumulation(flow_dir):
    accumulation = np.ones(flow_dir.shape, dtype=int)  # 每个单元格至少有自身的流量
    in_degree = np.zeros(flow_dir.shape, dtype=int)
    
    # 定义方向编码
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # 计算每个单元格的入度
    for i in range(flow_dir.shape[0]):
        for j in range(flow_dir.shape[1]):
            dir = flow_dir[i][j]
            if dir != -1:
                di, dj = directions[dir]
                ni, nj = i + di, j + dj
                if 0 <= ni < flow_dir.shape[0] and 0 <= nj < flow_dir.shape[1]:
                    in_degree[ni][nj] += 1
    
    # 初始化队列，将入度为0的单元格加入队列
    queue = deque()
    for i in range(flow_dir.shape[0]):
        for j in range(flow_dir.shape[1]):
            if in_degree[i][j] == 0:
                queue.append((i, j))
    
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

accumulation = compute_flow_accumulation(flow_dir)

# 可选：可视化流量积累图以调试
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

# 提取河流网络
def extract_river_network(accumulation, flow_threshold):
    river_mask = accumulation >= flow_threshold
    return river_mask

rivers_mask = extract_river_network(accumulation, flow_threshold)

# 确保河流连贯（连接到海洋或湖泊）
def connect_rivers_to_water(rivers_mask, ocean_mask, lakes_mask):
    water_mask = ocean_mask | lakes_mask
    # 使用二值化膨胀确保河流与水体连通
    structure = np.ones((3,3))
    rivers_connected = binary_dilation(rivers_mask, structure=structure)
    rivers_connected = rivers_connected & ~water_mask  # 保留河流与水体的独立性
    return rivers_connected

rivers_mask = connect_rivers_to_water(rivers_mask, ocean_mask, lakes_mask)

# 标记湖泊和河流
water_mask = ocean_mask | lakes_mask | rivers_mask

# 创建彩色图像分层
# 1. 创建基础地图（海洋、湖泊、陆地）
map_base = np.zeros((height, width, 3))
colors = {
    'ocean': [0, 0, 205],          # 海洋
    'lake': [70, 130, 180],        # 湖泊
    'land': [0, 0, 0]               # 黑色
}
map_base[ocean_mask] = np.array(colors['ocean']) / 255
map_base[lakes_mask] = np.array(colors['lake']) / 255
map_base[~water_mask] = np.array(colors['land']) / 255

# 2. 应用高斯模糊平滑基础地图
map_base_blurred = gaussian_filter(map_base, sigma=1)

# 3. 创建河流图层，不应用模糊
map_rivers = np.zeros((height, width, 3))
map_rivers[rivers_mask] = np.array([25, 25, 112]) / 255  # 河流颜色

# 4. 合并河流图层到基础地图
map_final = map_base_blurred.copy()
map_final[rivers_mask] = map_rivers[rivers_mask]

# 5. 调整河流宽度（可通过 river_width 变量设置）
if river_width > 1:
    # 使用形态学操作膨胀河流，以达到设定的河流宽度
    structure_size = river_width  # 使用river_width变量
    structure = np.zeros((structure_size, structure_size))
    radius = structure_size // 2
    y, x = np.ogrid[:structure_size, :structure_size]
    mask = (x - radius)**2 + (y - radius)**2 <= radius**2
    structure[mask] = 1
    
    # 膨胀河流掩膜
    rivers_mask_dilated = binary_dilation(rivers_mask, structure=structure)
    
    # 创建河流图层（膨胀后的河流）
    map_rivers_dilated = np.zeros((height, width, 3))
    map_rivers_dilated[rivers_mask_dilated] = np.array([25, 25, 112]) / 255  # 河流颜色
    
    # 合并膨胀后的河流图层到基础地图
    map_final = map_base_blurred.copy()
    map_final[rivers_mask_dilated] = map_rivers_dilated[rivers_mask_dilated]
else:
    # 如果 river_width 为1，不进行膨胀，保持河流细长
    map_final = map_final.copy()  # 已经绘制过河流

# 创建图例
import matplotlib.patches as mpatches

legend_elements = [
    mpatches.Patch(color=np.array(colors['ocean'])/255, label='Ocean'),
    mpatches.Patch(color=np.array(colors['lake'])/255, label='Lake'),
    mpatches.Patch(color=np.array([25, 25, 112])/255, label='River'),
    mpatches.Patch(color=np.array(colors['land'])/255, label='Land')
]

# 可视化地图
plt.figure(figsize=(20, 12))
plt.imshow(map_final, origin='lower')
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1), title="Features", fontsize='medium')
plt.title("Generated Map with 60% Oceans, Rivers, and Lakes", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.tight_layout()
plt.show()

# 显示实际的海洋覆盖率
actual_ocean_percentage = np.sum(ocean_mask) / (height * width) * 100
print(f"Actual ocean coverage: {actual_ocean_percentage:.2f}%")
