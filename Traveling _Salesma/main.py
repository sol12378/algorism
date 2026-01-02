import pygame
import numpy as np
import random
import math

# ============================================
# Configuration
# ============================================
WIDTH, HEIGHT = 800, 800
FPS = 60
PADDING = 50

# Colors (Neon Aesthetics)
BG_COLOR = (20, 20, 25)
NODE_COLOR = (255, 255, 255)
PATH_COLOR = (0, 255, 150)  # Neon Green
CANDIDATE_COLOR = (80, 80, 100, 60)  # Semi-transparent for candidate routes
TEXT_COLOR = (200, 200, 200)

# TSP Parameters
INITIAL_TEMP = 100.0
COOLING_RATE = 0.9995
MIN_TEMP = 0.01

# Audio Parameters
SAMPLE_RATE = 22050
DURATION = 0.15  # Short, satisfying beep


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
        self.generate_cities()
        
    def generate_cities(self):
        """Generate random cities with padding from edges."""
        self.cities = [
            (
                random.randint(PADDING, WIDTH - PADDING),
                random.randint(PADDING, HEIGHT - PADDING)
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
                return True  # Signal improvement (for audio)
        
        # Cool down
        self.temperature *= COOLING_RATE
        return False


# ============================================
# Audio System
# ============================================
class AudioSystem:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
        self.min_distance = 10000
        self.max_distance = 50000
        
    def play_improvement_sound(self, distance):
        """Play a satisfying beep when solution improves.
        Higher pitch = shorter distance (better solution)."""
        # Map distance to frequency (shorter distance = higher pitch)
        # Typical range: 300 Hz (long) to 1200 Hz (short)
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        frequency = 1200 - (normalized * 900)  # Inverse: shorter = higher
        
        # Generate sine wave with envelope
        samples = int(SAMPLE_RATE * DURATION)
        t = np.linspace(0, DURATION, samples, False)
        
        # Sine wave
        wave = np.sin(2 * np.pi * frequency * t)
        
        # ADSR-like envelope (quick attack, gentle decay)
        envelope = np.ones_like(t)
        attack_samples = int(samples * 0.05)
        decay_samples = samples - attack_samples
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:] = np.linspace(1, 0, decay_samples)
        
        # Apply envelope
        wave = wave * envelope * 0.3  # Volume control
        
        # Convert to 16-bit audio
        wave = (wave * 32767).astype(np.int16)
        
        # Play
        sound = pygame.mixer.Sound(buffer=wave)
        sound.play()


# ============================================
# Rendering
# ============================================
def draw_route(surface, solver, route, color, width=2, alpha=255):
    """Draw a route with specified color and width."""
    if len(route) < 2:
        return
    
    if alpha < 255:
        # Create transparent surface for semi-transparent routes
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i in range(len(route)):
            city_a = solver.cities[route[i]]
            city_b = solver.cities[route[(i + 1) % len(route)]]
            pygame.draw.line(temp_surface, (*color, alpha), city_a, city_b, width)
        surface.blit(temp_surface, (0, 0))
    else:
        for i in range(len(route)):
            city_a = solver.cities[route[i]]
            city_b = solver.cities[route[(i + 1) % len(route)]]
            pygame.draw.line(surface, color, city_a, city_b, width)


def draw_cities(surface, solver):
    """Draw city nodes as white glowing dots."""
    for city in solver.cities:
        # Outer glow
        pygame.draw.circle(surface, (150, 150, 150), city, 5)
        # Inner bright core
        pygame.draw.circle(surface, NODE_COLOR, city, 3)


def draw_ui(surface, solver, font):
    """Draw UI information."""
    distance_text = f"Distance: {solver.best_distance:.1f}"
    temp_text = f"Temperature: {solver.temperature:.2f}"
    cities_text = f"Cities: {solver.num_cities}"
    
    text_surface = font.render(distance_text, True, TEXT_COLOR)
    surface.blit(text_surface, (20, 20))
    
    text_surface = font.render(temp_text, True, TEXT_COLOR)
    surface.blit(text_surface, (20, 50))
    
    text_surface = font.render(cities_text, True, TEXT_COLOR)
    surface.blit(text_surface, (20, 80))
    
    # Controls
    controls = [
        "SPACE: Pause/Resume",
        "R: Reset",
        "C: Change City Count"
    ]
    y_offset = HEIGHT - 100
    for control in controls:
        text_surface = font.render(control, True, (120, 120, 120))
        surface.blit(text_surface, (20, y_offset))
        y_offset += 25
    
    if solver.paused:
        pause_font = pygame.font.Font(None, 72)
        pause_text = pause_font.render("PAUSED", True, (255, 100, 100))
        text_rect = pause_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        surface.blit(pause_text, text_rect)


# ============================================
# Main Loop
# ============================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TSP Visualizer - Untangling the Knot")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    
    # Initialize systems
    solver = TSPSolver(num_cities=80)
    audio = AudioSystem()
    
    # City count rotation
    city_counts = [50, 100, 200]
    current_count_index = 1  # Start with 80 (not in list, but close to 100)
    
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
        
        # Render
        screen.fill(BG_COLOR)
        
        # Draw best route (neon green)
        draw_route(screen, solver, solver.best_route, PATH_COLOR, width=3)
        
        # Draw cities
        draw_cities(screen, solver)
        
        # Draw UI
        draw_ui(screen, solver, font)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()
