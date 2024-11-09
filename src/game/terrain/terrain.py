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
import matplotlib as mpl

# 设置参数
width = 100  # 平面图的宽度
height = 100  # 平面图的高度
scale = 0.1  # 控制 Perlin 噪声的平滑度
amplitude = 30  # 控制温度的最大波动范围
base_temperature = 15  # 基础温度

# 生成 Perlin 噪声平面图用于温度、高度和湿度
perlin_noise_temp = np.zeros((width, height))
perlin_noise_altitude = np.zeros((width, height))
perlin_noise_humidity = np.zeros((width, height))
for i in range(width):
    for j in range(height):
        perlin_noise_temp[i][j] = pnoise2(i * scale, j * scale, repeatx=1024, repeaty=1024)
        perlin_noise_altitude[i][j] = pnoise2((i + 1000) * scale, (j + 1000) * scale, repeatx=1024, repeaty=1024)
        perlin_noise_humidity[i][j] = pnoise2((i + 2000) * scale, (j + 2000) * scale, repeatx=1024, repeaty=1024)

# 将 Perlin 噪声映射到温度范围
min_temp = -10  # 温度下限
max_temp = 40   # 温度上限
temperature_distribution = base_temperature + amplitude * perlin_noise_temp
temperature_distribution = np.clip(temperature_distribution, min_temp, max_temp)

# 将 Perlin 噪声映射到高度范围
min_altitude = 0  # 高度下限
max_altitude = 5000  # 高度上限
altitude_distribution = (perlin_noise_altitude + 1) / 2 * (max_altitude - min_altitude) + min_altitude

# 将 Perlin 噪声映射到湿度范围
min_humidity = 0  # 湿度下限
max_humidity = 100  # 湿度上限
humidity_distribution = (perlin_noise_humidity + 1) / 2 * (max_humidity - min_humidity) + min_humidity

# 定义群系颜色和条件
biome_colors = {
    "Temperate Forests": [46, 139, 87],
    "Tropical Rainforests": [50, 205, 50],
    "Arid Deserts": [237, 201, 175],
    "Rocky Deserts": [194, 178, 128],
    "Prairies": [124, 252, 0],
    "Hilly Grasslands": [85, 107, 47],
    "Snow-capped Mountains": [224, 255, 255],
    "High Land": [210, 180, 140],
    "Freshwater Lakes": [135, 206, 235],
    "Rivers": [70, 130, 180],
    "Oceans": [0, 0, 205],
    "Swamps": [107, 142, 35],
    "Active Volcanoes": [255, 69, 0],
    "Dormant Volcanoes": [105, 105, 105],
    "Volcanic Plains": [47, 79, 79]
}

biome_map = np.zeros((width, height, 3))
biome_names = np.empty((width, height), dtype=object)

for i in range(width):
    for j in range(height):
        temp = temperature_distribution[i][j]
        altitude = altitude_distribution[i][j]
        humidity = humidity_distribution[i][j]

        if -15 <= temp <= 30 and 0 <= altitude <= 1500 and 60 <= humidity <= 80:
            biome_map[i, j] = np.array(biome_colors["Temperate Forests"]) / 255  # Temperate Forests
            biome_names[i, j] = "Temperate Forests"
        elif 20 <= temp <= 35 and 0 <= altitude <= 1000 and 75 <= humidity <= 100:
            biome_map[i, j] = np.array(biome_colors["Tropical Rainforests"]) / 255  # Tropical Rainforests
            biome_names[i, j] = "Tropical Rainforests"
        elif 10 <= temp <= 50 and -50 <= altitude <= 1000 and 10 <= humidity <= 30:
            biome_map[i, j] = np.array(biome_colors["Arid Deserts"]) / 255  # Arid Deserts
            biome_names[i, j] = "Arid Deserts"
        elif 5 <= temp <= 45 and 0 <= altitude <= 1500 and 15 <= humidity <= 35:
            biome_map[i, j] = np.array(biome_colors["Rocky Deserts"]) / 255  # Rocky Deserts
            biome_names[i, j] = "Rocky Deserts"
        elif -10 <= temp <= 30 and 0 <= altitude <= 1500 and 30 <= humidity <= 60:
            biome_map[i, j] = np.array(biome_colors["Prairies"]) / 255  # Prairies
            biome_names[i, j] = "Prairies"
        elif -5 <= temp <= 25 and 500 <= altitude <= 2000 and 35 <= humidity <= 65:
            biome_map[i, j] = np.array(biome_colors["Hilly Grasslands"]) / 255  # Hilly Grasslands
            biome_names[i, j] = "Hilly Grasslands"
        elif -40 <= temp <= 0 and 2000 <= altitude <= 6000 and 30 <= humidity <= 60:
            biome_map[i, j] = np.array(biome_colors["Snow-capped Mountains"]) / 255  # Snow-capped Mountains
            biome_names[i, j] = "Snow-capped Mountains"
        elif -15 <= temp <= 20 and 1000 <= altitude <= 4000 and 20 <= humidity <= 50:
            biome_map[i, j] = np.array(biome_colors["High Land"]) / 255  # High Land
            biome_names[i, j] = "High Land"
        elif 5 <= temp <= 25 and 0 <= altitude <= 3000 and 50 <= humidity <= 80:
            biome_map[i, j] = np.array(biome_colors["Freshwater Lakes"]) / 255  # Freshwater Lakes
            biome_names[i, j] = "Freshwater Lakes"
        elif 0 <= temp <= 25 and 0 <= altitude <= 3500 and 60 <= humidity <= 90:
            if np.random.rand() > 0.95:  # 增加条件以减少河流的覆盖面积
                biome_map[i, j] = np.array(biome_colors["Rivers"]) / 255  # Rivers
                biome_names[i, j] = "Rivers"
        elif -2 <= temp <= 30 and 0 <= altitude <= 11000 and 70 <= humidity <= 100:
            biome_map[i, j] = np.array(biome_colors["Oceans"]) / 255  # Oceans
            biome_names[i, j] = "Oceans"
        elif 10 <= temp <= 30 and 0 <= altitude <= 500 and 80 <= humidity <= 100:
            biome_map[i, j] = np.array(biome_colors["Swamps"]) / 255  # Swamps
            biome_names[i, j] = "Swamps"
        elif 20 <= temp <= 600 and 500 <= altitude <= 3000 and 20 <= humidity <= 60:
            biome_map[i, j] = np.array(biome_colors["Active Volcanoes"]) / 255  # Active Volcanoes
            biome_names[i, j] = "Active Volcanoes"
        elif -10 <= temp <= 30 and 500 <= altitude <= 3000 and 20 <= humidity <= 60:
            biome_map[i, j] = np.array(biome_colors["Dormant Volcanoes"]) / 255  # Dormant Volcanoes
            biome_names[i, j] = "Dormant Volcanoes"
        elif 15 <= temp <= 40 and 0 <= altitude <= 1000 and 20 <= humidity <= 40:
            biome_map[i, j] = np.array(biome_colors["Volcanic Plains"]) / 255  # Volcanic Plains
            biome_names[i, j] = "Volcanic Plains"

# 绘制最终群系图
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(biome_map, origin='lower')

# 创建图例
from matplotlib.patches import Patch

legend_elements = []
for biome, color in biome_colors.items():
    legend_elements.append(Patch(facecolor=np.array(color)/255, edgecolor='k', label=biome))

# 由于生物群系较多，图例可能会很长，使用分栏显示
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1),
            title="Biomes", ncol=1, fontsize='small')

ax.set_title("Biome Map")
ax.set_xlabel("Width")
ax.set_ylabel("Height")

plt.tight_layout()
plt.show()
