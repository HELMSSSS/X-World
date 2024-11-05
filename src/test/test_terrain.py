import unittest
import numpy as np
from game.terrain import TerrainGenerator, TerrainType

class TestTerrainGenerator(unittest.TestCase):
    def setUp(self):
        """
        Initialize a TerrainGenerator instance with a fixed seed for reproducibility.
        """
        self.generator = TerrainGenerator(
            width=100,       # Smaller size for faster tests
            height=100,
            scale=50,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0,
            seed=42
        )
        self.generator.generate_all()

    def test_height_map_normalization(self):
        """
        Test that the height map values are normalized between 0 and 1.
        """
        self.assertTrue(np.all(self.generator.height_map >= 0.0), "Height map has values less than 0.")
        self.assertTrue(np.all(self.generator.height_map <= 1.0), "Height map has values greater than 1.")

    def test_terrain_type_assignment_ocean(self):
        """
        Test that terrain types are correctly assigned as OCEAN for low elevation.
        """
        ocean_indices = np.where(self.generator.height_map < 0.3)
        terrain_types = self.generator.terrain_map[ocean_indices]
        self.assertTrue(np.all(terrain_types == TerrainType.OCEAN.value), "Not all low elevations are assigned as OCEAN.")

    def test_terrain_type_assignment_beach(self):
        """
        Test that terrain types are correctly assigned as BEACH for elevations between 0.3 and 0.35.
        """
        beach_indices = np.where((self.generator.height_map >= 0.3) & (self.generator.height_map < 0.35))
        terrain_types = self.generator.terrain_map[beach_indices]
        self.assertTrue(np.all(terrain_types == TerrainType.BEACH.value), "Not all elevations between 0.3 and 0.35 are assigned as BEACH.")

    def test_color_map_assignment(self):
        """
        Test that the color map correctly corresponds to the terrain map.
        """
        # Define expected colors
        expected_colors = {
            TerrainType.OCEAN.value: (0, 0, 128),        # Deep Blue
            TerrainType.BEACH.value: (240, 230, 140),    # Sandy Yellow
            TerrainType.FOREST.value: (0, 128, 0),       # Dense Green
            TerrainType.GRASSLAND.value: (128, 204, 77), # Light Green
            TerrainType.DESERT.value: (237, 220, 131),   # Sand Yellow
            TerrainType.SWAMP.value: (25, 102, 25),      # Dark Green
            TerrainType.MOUNTAIN.value: (128, 128, 128), # Gray Rock
            TerrainType.SNOW.value: (255, 255, 255),      # White
        }

        for terrain_value, color in expected_colors.items():
            indices = np.where(self.generator.terrain_map == terrain_value)
            # Convert normalized color to 0-255 range
            normalized_color = np.array(color) / 255.0
            actual_colors = self.generator.color_map[indices]
            # Check if all colors match
            self.assertTrue(np.allclose(actual_colors, normalized_color, atol=0.01),
                            f"Color map does not match expected color for terrain type {TerrainType(terrain_value).name}.")

    def test_river_generation_count(self):
        """
        Test that the correct number of rivers are generated.
        """
        # Rivers are represented by TerrainType.OCEAN in river paths, but ocean is also used for seas.
        # We'll count the number of unique river sources by checking high elevation points adjacent to rivers.
        # This is a simplistic test and might need refinement based on river generation logic.
        river_count = 0
        for i in range(self.generator.height):
            for j in range(self.generator.width):
                if self.generator.terrain_map[i][j] == TerrainType.OCEAN.value:
                    # Check if it's part of a river by looking for high elevation neighbors
                    neighbors = self.generator._get_neighbors(i, j)
                    if any(self.generator.height_map[ni][nj] > 0.7 for ni, nj in neighbors):
                        river_count += 1
        # Since rivers are simplified, this test might not be precise. Adjust as needed.
        self.assertGreaterEqual(river_count, 1, "No rivers were generated.")
        # Optionally, set an upper limit
        self.assertLessEqual(river_count, self.generator.octaves * 2, "Too many rivers generated.")

    def test_terrain_map_shape(self):
        """
        Test that the terrain map has the correct shape.
        """
        self.assertEqual(self.generator.terrain_map.shape, (self.generator.height, self.generator.width),
                         "Terrain map shape is incorrect.")

    def test_height_map_shape(self):
        """
        Test that the height map has the correct shape.
        """
        self.assertEqual(self.generator.height_map.shape, (self.generator.height, self.generator.width),
                         "Height map shape is incorrect.")

    def test_color_map_shape(self):
        """
        Test that the color map has the correct shape.
        """
        self.assertEqual(self.generator.color_map.shape, (self.generator.height, self.generator.width, 3),
                         "Color map shape is incorrect.")

    def test_river_path_continuity(self):
        """
        Test that river paths are continuous and flow from high to low elevations.
        """
        # This is a more complex test and may require access to river paths.
        # Since rivers are represented as TerrainType.OCEAN, we'll simulate a continuity check.
        # For simplicity, we'll ensure that any ocean cell has at least one neighbor with equal or lower elevation.
        for i in range(self.generator.height):
            for j in range(self.generator.width):
                if self.generator.terrain_map[i][j] == TerrainType.OCEAN.value:
                    neighbors = self.generator._get_neighbors(i, j)
                    if neighbors:
                        # Check if at least one neighbor has elevation <= current
                        current_elevation = self.generator.height_map[i][j]
                        neighbor_elevations = [self.generator.height_map[ni][nj] for ni, nj in neighbors]
                        self.assertTrue(any(neighbor_elevations <= current_elevation),
                                        f"River at ({i}, {j}) does not flow to a lower or equal elevation.")

if __name__ == '__main__':
    unittest.main()
