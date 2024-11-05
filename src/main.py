import pygame
import sys
from game.terrain import TerrainGenerator

# Define constants for the game
WINDOW_WIDTH = 800    # Width of the game window
WINDOW_HEIGHT = 600   # Height of the game window
TILE_SIZE = 4         # Size of each terrain tile in pixels

# Define the frames per second
FPS = 60

def main():
    """
    Main function to initialize the game, generate the terrain, and handle the game loop.
    """
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Terrain Generation Demo")
    
    # Create the game window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Create a clock object to manage the frame rate
    clock = pygame.time.Clock()
    
    # Initialize the TerrainGenerator with desired parameters
    generator = TerrainGenerator(
        width=512,       # Width of the terrain map
        height=512,      # Height of the terrain map
        scale=100,       # Scale factor for Perlin Noise
        octaves=6,       # Number of noise octaves
        persistence=0.5, # Persistence of the noise
        lacunarity=2.0,  # Lacunarity of the noise
        seed=42          # Seed for reproducibility
    )
    
    # Generate the entire terrain
    generator.generate_all()
    
    # Convert the color map to a Pygame Surface for rendering
    terrain_surface = pygame.Surface((generator.width, generator.height))
    # Loop through each pixel and set the color based on the terrain type
    for i in range(generator.height):
        for j in range(generator.width):
            color = tuple(int(c * 255) for c in generator.color_map[i][j])
            terrain_surface.set_at((j, i), color)
    
    # Scale the terrain surface to fit the game window
    scaled_surface = pygame.transform.scale(terrain_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle other events like key presses or mouse clicks here
                
        # Render the scaled terrain
        screen.blit(scaled_surface, (0, 0))
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    # Quit Pygame gracefully
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
