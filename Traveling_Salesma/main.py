import pygame
import numpy as np
import random
import math

# ============================================
# Configuration
# ============================================
WIDTH, HEIGHT = 800, 800
CANVAS_WIDTH, CANVAS_HEIGHT = 160, 160  # Low-res pixel art canvas
FPS = 60
PADDING = 10  # Padding on the low-res canvas

# Colors (Retro Pixel Art Palette)
BG_COLOR = (45, 27, 46)  # Purple-ish dark gray
NODE_COLOR = (255, 230, 80)  # Bright yellow
PATH_COLOR = (255, 50, 150)  # Neon Magenta/Pink
PATH_ALT_COLOR = (50, 255, 200)  # Cyan (alternative)
FLASH_COLOR = (255, 255, 255)  # White flash on improvement
TEXT_COLOR = (150, 150, 150)

# TSP Parameters
INITIAL_TEMP = 100.0
COOLING_RATE = 0.9995
MIN_TEMP = 0.01

# Audio Parameters
SAMPLE_RATE = 22050
DURATION = 0.12  # Short chiptune beep

# Flash effect
FLASH_DURATION = 3  # frames


# ============================================
# City and TSP Logic
# ============================================
class TSPSolver:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.cities = []
        self.current_route = []
        self.best_route = []
        self.best_distance = float('inf')
        self.current_distance = float('inf')
        self.temperature = INITIAL_TEMP
        self.paused = False
        self.flash_timer = 0
        self.generate_cities()
        
    def generate_cities(self):
        """Generate random cities with padding from edges (on low-res canvas)."""
        self.cities = [
            (
                random.randint(PADDING, CANVAS_WIDTH - PADDING),
                random.randint(PADDING, CANVAS_HEIGHT - PADDING)
            )
            for _ in range(self.num_cities)
        ]
        # Start with random route
        self.current_route = list(range(self.num_cities))
        random.shuffle(self.current_route)
        self.best_route = self.current_route[:]
        self.current_distance = self.calculate_distance(self.current_route)
        self.best_distance = self.current_distance
        self.temperature = INITIAL_TEMP
        self.flash_timer = 0
        
    def calculate_distance(self, route):
        """Calculate total distance of a route."""
        total = 0
        for i in range(len(route)):
            city_a = self.cities[route[i]]
            city_b = self.cities[route[(i + 1) % len(route)]]
            total += math.sqrt((city_a[0] - city_b[0])**2 + (city_a[1] - city_b[1])**2)
        return total
    
    def two_opt_swap(self, route, i, j):
        """Perform 2-opt swap: reverse the segment between i and j."""
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return new_route
    
    def step(self):
        """Perform one optimization step using Simulated Annealing + 2-Opt."""
        if self.paused or self.temperature < MIN_TEMP:
            return False
        
        # Pick two random indices for 2-Opt
        i = random.randint(0, self.num_cities - 2)
        j = random.randint(i + 1, self.num_cities - 1)
        
        # Generate candidate route
        candidate_route = self.two_opt_swap(self.current_route, i, j)
        candidate_distance = self.calculate_distance(candidate_route)
        
        # Simulated Annealing acceptance criteria
        delta = candidate_distance - self.current_distance
        if delta < 0 or random.random() < math.exp(-delta / self.temperature):
            self.current_route = candidate_route
            self.current_distance = candidate_distance
            
            # Update best if improved
            if self.current_distance < self.best_distance:
                self.best_route = self.current_route[:]
                self.best_distance = self.current_distance
                self.flash_timer = FLASH_DURATION
                return True  # Signal improvement (for audio)
        
        # Cool down
        self.temperature *= COOLING_RATE
        return False
    
    def update_flash(self):
        """Update flash timer."""
        if self.flash_timer > 0:
            self.flash_timer -= 1


# ============================================
# Audio System (Chiptune Style)
# ============================================
class AudioSystem:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
        self.min_distance = 500
        self.max_distance = 5000
        
    def play_improvement_sound(self, distance):
        """Play a chiptune-style beep when solution improves.
        Higher pitch = shorter distance (better solution)."""
        # Map distance to frequency (shorter distance = higher pitch)
        # Range: 300 Hz (long) to 1400 Hz (short)
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        frequency = 1400 - (normalized * 1100)  # Inverse: shorter = higher
        
        # Generate square wave (chiptune style)
        samples = int(SAMPLE_RATE * DURATION)
        t = np.linspace(0, DURATION, samples, False)
        
        # Square wave: sign of sine wave
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
        
        # Add some harmonics for richer sound
        wave += 0.3 * np.sign(np.sin(2 * np.pi * frequency * 2 * t))
        wave = wave / np.max(np.abs(wave))  # Normalize
        
        # Envelope (quick attack, fast decay for punchy sound)
        envelope = np.ones_like(t)
        attack_samples = int(samples * 0.02)
        decay_samples = samples - attack_samples
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:] = np.linspace(1, 0, decay_samples) ** 2  # Exponential decay
        
        # Apply envelope
        wave = wave * envelope * 0.25  # Volume control
        
        # Convert to 16-bit audio
        wave = (wave * 32767).astype(np.int16)
        
        # Play
        sound = pygame.mixer.Sound(buffer=wave)
        sound.play()


# ============================================
# Rendering (Pixel Art Pipeline)
# ============================================
def draw_route(canvas, solver, route, color):
    """Draw a route on the low-res canvas."""
    if len(route) < 2:
        return
    
    for i in range(len(route)):
        city_a = solver.cities[route[i]]
        city_b = solver.cities[route[(i + 1) % len(route)]]
        pygame.draw.line(canvas, color, city_a, city_b, 1)  # 1 pixel width on canvas


def draw_cities(canvas, solver):
    """Draw city nodes as 2x2 pixel dots."""
    for city in solver.cities:
        # Draw 2x2 pixel block
        pygame.draw.rect(canvas, NODE_COLOR, (city[0] - 1, city[1] - 1, 2, 2))


def draw_ui(canvas, solver, font):
    """Draw minimal UI (very small, unobtrusive)."""
    # Only show distance in corner, very small
    distance_text = f"{int(solver.best_distance)}"
    text_surface = font.render(distance_text, True, TEXT_COLOR)
    canvas.blit(text_surface, (2, 2))


# ============================================
# Main Loop
# ============================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TSP Visualizer - Pixel Art Edition")
    clock = pygame.time.Clock()
    
    # Low-res canvas for pixel art rendering
    canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    
    # Tiny pixel font
    font = pygame.font.Font(None, 8)
    
    # Initialize systems
    solver = TSPSolver(num_cities=60)
    audio = AudioSystem()
    
    # City count rotation
    city_counts = [40, 60, 80]
    current_count_index = 1
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    solver.paused = not solver.paused
                elif event.key == pygame.K_r:
                    solver.generate_cities()
                elif event.key == pygame.K_c:
                    # Cycle through city counts
                    current_count_index = (current_count_index + 1) % len(city_counts)
                    solver.num_cities = city_counts[current_count_index]
                    solver.generate_cities()
        
        # Update
        if not solver.paused:
            # Perform multiple steps per frame for faster convergence
            for _ in range(10):
                improved = solver.step()
                if improved:
                    audio.play_improvement_sound(solver.best_distance)
        
        # Update flash effect
        solver.update_flash()
        
        # Render on low-res canvas
        canvas.fill(BG_COLOR)
        
        # Determine path color (flash white on improvement)
        path_color = FLASH_COLOR if solver.flash_timer > 0 else PATH_COLOR
        
        # Draw best route
        draw_route(canvas, solver, solver.best_route, path_color)
        
        # Draw cities
        draw_cities(canvas, solver)
        
        # Draw minimal UI
        draw_ui(canvas, solver, font)
        
        # Scale up canvas to screen (pixel art effect)
        scaled_surface = pygame.transform.scale(canvas, (WIDTH, HEIGHT))
        screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()
