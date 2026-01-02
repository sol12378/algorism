import pygame
import numpy as np
import random
import math

# ============================================
# Configuration
# ============================================
WINDOW_SIZE = 800
POPULATION_SIZE = 250
LIFESPAN = 400
TARGET_POS = pygame.Vector2(400, 50)
TARGET_RADIUS = 20
START_POS = pygame.Vector2(400, 750)
MUTATION_RATE = 0.01
SAMPLE_RATE = 44100

# ============================================
# Audio Engine
# ============================================
class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
        pygame.mixer.init()
        self.crash_sound = None
        self.goal_sound = None
        self.sweep_sound = None
        self.generate_sounds()
    
    def generate_sounds(self):
        """Pre-generate all sound effects"""
        # Crash sound: Low frequency noise burst
        duration = 0.05
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        noise = np.random.uniform(-1, 1, samples).astype(np.float32)
        envelope = np.exp(-t * 30)
        wave = noise * envelope * 0.15
        wave = (wave * 32767).astype(np.int16)
        self.crash_sound = pygame.mixer.Sound(buffer=wave.tobytes())
        
        # Goal sound: High pitched bell
        duration = 0.3
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        freq = 880  # A5
        wave = np.sin(2 * np.pi * freq * t)
        wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        envelope = np.exp(-t * 8)
        wave = wave * envelope * 0.3
        wave = (wave * 32767).astype(np.int16)
        self.goal_sound = pygame.mixer.Sound(buffer=wave.tobytes())
        
        # Sweep sound: Frequency sweep
        duration = 0.8
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        freq_start = 200
        freq_end = 600
        freq = freq_start + (freq_end - freq_start) * (t / duration)
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        wave = np.sin(phase)
        envelope = np.exp(-t * 2) * (1 - np.exp(-t * 15))
        wave = wave * envelope * 0.2
        wave = (wave * 32767).astype(np.int16)
        self.sweep_sound = pygame.mixer.Sound(buffer=wave.tobytes())
    
    def play_crash(self):
        if self.crash_sound and pygame.mixer.get_busy() < 3:
            self.crash_sound.play()
    
    def play_goal(self):
        if self.goal_sound:
            self.goal_sound.play()
    
    def play_sweep(self):
        if self.sweep_sound:
            self.sweep_sound.play()


# ============================================
# DNA (Genome)
# ============================================
class DNA:
    def __init__(self, genes=None):
        if genes is None:
            # Random force vectors for each frame
            self.genes = [pygame.Vector2(
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ).normalize() * random.uniform(0, 0.3) for _ in range(LIFESPAN)]
        else:
            self.genes = genes
    
    def crossover(self, partner):
        """Create child DNA from two parents"""
        midpoint = random.randint(0, len(self.genes))
        new_genes = self.genes[:midpoint] + partner.genes[midpoint:]
        return DNA(new_genes)
    
    def mutate(self):
        """Randomly mutate genes"""
        for i in range(len(self.genes)):
            if random.random() < MUTATION_RATE:
                self.genes[i] = pygame.Vector2(
                    random.uniform(-1, 1),
                    random.uniform(-1, 1)
                ).normalize() * random.uniform(0, 0.3)


# ============================================
# Obstacle
# ============================================
class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.dragging = False
        self.offset = pygame.Vector2(0, 0)
    
    def draw(self, screen, alpha=255):
        s = pygame.Surface((self.rect.width, self.rect.height))
        s.set_alpha(alpha)
        s.fill((60, 60, 60))
        screen.blit(s, self.rect.topleft)
        pygame.draw.rect(screen, (150, 150, 150), self.rect, 2)
    
    def contains_point(self, pos):
        return self.rect.collidepoint(pos)
    
    def start_drag(self, mouse_pos):
        self.dragging = True
        self.offset = pygame.Vector2(mouse_pos) - pygame.Vector2(self.rect.topleft)
    
    def update_drag(self, mouse_pos):
        if self.dragging:
            new_pos = pygame.Vector2(mouse_pos) - self.offset
            self.rect.topleft = (int(new_pos.x), int(new_pos.y))
    
    def stop_drag(self):
        self.dragging = False


# ============================================
# Rocket
# ============================================
class Rocket:
    def __init__(self, dna=None):
        self.dna = dna if dna else DNA()
        self.pos = START_POS.copy()
        self.vel = pygame.Vector2(0, 0)
        self.acc = pygame.Vector2(0, 0)
        self.completed = False
        self.crashed = False
        self.frame_count = 0
        self.fitness = 0
        self.completion_time = LIFESPAN
    
    def apply_force(self, force):
        self.acc += force
    
    def update(self, obstacles):
        if self.completed or self.crashed:
            return
        
        # Apply DNA force for this frame
        if self.frame_count < len(self.dna.genes):
            self.apply_force(self.dna.genes[self.frame_count])
        
        # Physics update
        self.vel += self.acc
        self.vel = self.vel.clamp_magnitude(4)  # Max speed
        self.pos += self.vel
        self.acc *= 0  # Reset acceleration
        
        # Check boundaries
        if (self.pos.x < 0 or self.pos.x > WINDOW_SIZE or
            self.pos.y < 0 or self.pos.y > WINDOW_SIZE):
            self.crashed = True
        
        # Check obstacles
        for obs in obstacles:
            if obs.rect.collidepoint(self.pos.x, self.pos.y):
                self.crashed = True
        
        # Check target
        if self.pos.distance_to(TARGET_POS) < TARGET_RADIUS:
            self.completed = True
            self.completion_time = self.frame_count
        
        self.frame_count += 1
    
    def calculate_fitness(self):
        """Calculate fitness score"""
        distance = self.pos.distance_to(TARGET_POS)
        
        # Base fitness: inverse of distance (closer = better)
        self.fitness = 1 / (distance + 1)
        
        # Huge bonus for completing
        if self.completed:
            self.fitness *= 10
            # Extra bonus for completing faster
            self.fitness *= (LIFESPAN / (self.completion_time + 1))
        
        # Penalty for crashing
        if self.crashed:
            self.fitness *= 0.1
        
        return self.fitness
    
    def draw(self, screen, is_elite=False):
        if self.crashed:
            return
        
        # Color
        if is_elite:
            color = (0, 255, 100, 200)
        else:
            color = (255, 255, 255, 120)
        
        # Draw triangle pointing in direction of velocity
        if self.vel.length() > 0.1:
            angle = math.atan2(self.vel.y, self.vel.x)
            size = 8 if is_elite else 5
            
            # Triangle vertices
            points = [
                (self.pos.x + size * math.cos(angle),
                 self.pos.y + size * math.sin(angle)),
                (self.pos.x + size * math.cos(angle + 2.5),
                 self.pos.y + size * math.sin(angle + 2.5)),
                (self.pos.x + size * math.cos(angle - 2.5),
                 self.pos.y + size * math.sin(angle - 2.5))
            ]
            
            # Draw to surface with alpha
            s = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(s, color, points)
            screen.blit(s, (0, 0))
        else:
            # Stationary: draw circle
            s = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(self.pos.x), int(self.pos.y)), 3)
            screen.blit(s, (0, 0))


# ============================================
# Population
# ============================================
class Population:
    def __init__(self):
        self.rockets = [Rocket() for _ in range(POPULATION_SIZE)]
        self.generation = 1
        self.frame_count = 0
        self.best_rocket = None
        self.mating_pool = []
    
    def update(self, obstacles):
        for rocket in self.rockets:
            rocket.update(obstacles)
        self.frame_count += 1
    
    def draw(self, screen):
        # Find elite rockets (top 10%)
        sorted_rockets = sorted(self.rockets, key=lambda r: r.fitness, reverse=True)
        elite_count = max(1, POPULATION_SIZE // 10)
        elite_rockets = set(sorted_rockets[:elite_count])
        
        # Draw non-elite first
        for rocket in self.rockets:
            if rocket not in elite_rockets:
                rocket.draw(screen)
        
        # Draw elite on top
        for rocket in elite_rockets:
            rocket.draw(screen, is_elite=True)
    
    def evaluate(self):
        """Calculate fitness for all rockets"""
        max_fitness = 0
        for rocket in self.rockets:
            rocket.calculate_fitness()
            if rocket.fitness > max_fitness:
                max_fitness = rocket.fitness
                self.best_rocket = rocket
    
    def selection(self):
        """Create mating pool based on fitness"""
        self.mating_pool = []
        
        # Normalize fitness
        max_fitness = max(r.fitness for r in self.rockets)
        if max_fitness == 0:
            max_fitness = 1
        
        for rocket in self.rockets:
            # Number of times this rocket appears in pool (weighted by fitness)
            n = int(rocket.fitness / max_fitness * 100) + 1
            self.mating_pool.extend([rocket] * n)
    
    def reproduction(self):
        """Create next generation"""
        new_rockets = []
        
        for _ in range(POPULATION_SIZE):
            # Select two parents
            parent_a = random.choice(self.mating_pool)
            parent_b = random.choice(self.mating_pool)
            
            # Crossover
            child_dna = parent_a.dna.crossover(parent_b.dna)
            
            # Mutate
            child_dna.mutate()
            
            # Create new rocket
            new_rockets.append(Rocket(child_dna))
        
        self.rockets = new_rockets
        self.generation += 1
        self.frame_count = 0
    
    def is_generation_complete(self):
        """Check if generation is finished"""
        return self.frame_count >= LIFESPAN


# ============================================
# Main Application
# ============================================
class SmartRocketsApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Smart Rockets - Genetic Algorithm")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        
        self.audio = AudioEngine()
        self.population = Population()
        
        # Obstacles
        self.obstacles = [
            Obstacle(200, 400, 400, 20)  # Central barrier
        ]
        self.dragged_obstacle = None
        
        # State
        self.paused = False
        self.prev_crashes = 0
        self.prev_completions = 0
    
    def reset(self):
        """Reset simulation to generation 1"""
        self.population = Population()
        self.prev_crashes = 0
        self.prev_completions = 0
    
    def update(self):
        if self.paused:
            return
        
        # Update population
        self.population.update(self.obstacles)
        
        # Count events for audio
        crashes = sum(1 for r in self.population.rockets if r.crashed)
        completions = sum(1 for r in self.population.rockets if r.completed)
        
        # Play sounds for new events
        if crashes > self.prev_crashes:
            self.audio.play_crash()
        if completions > self.prev_completions:
            self.audio.play_goal()
        
        self.prev_crashes = crashes
        self.prev_completions = completions
        
        # Check for generation completion
        if self.population.is_generation_complete():
            self.population.evaluate()
            self.population.selection()
            self.population.reproduction()
            self.audio.play_sweep()
            self.prev_crashes = 0
            self.prev_completions = 0
    
    def draw(self):
        # Background
        self.screen.fill((10, 10, 15))
        
        # Target with glow
        glow_alpha = int(100 + 50 * math.sin(pygame.time.get_ticks() * 0.005))
        for r in range(TARGET_RADIUS + 10, TARGET_RADIUS, -2):
            s = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            alpha = int(glow_alpha * (1 - (r - TARGET_RADIUS) / 10))
            pygame.draw.circle(s, (0, 255, 100, alpha), 
                             (int(TARGET_POS.x), int(TARGET_POS.y)), r)
            self.screen.blit(s, (0, 0))
        
        pygame.draw.circle(self.screen, (0, 255, 100), 
                         (int(TARGET_POS.x), int(TARGET_POS.y)), TARGET_RADIUS)
        
        # Obstacles
        for obs in self.obstacles:
            obs.draw(self.screen)
        
        # Rockets
        self.population.draw(self.screen)
        
        # UI
        gen_text = self.font.render(f"Generation: {self.population.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (20, 20))
        
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 200, 200))
            text_rect = pause_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            self.screen.blit(pause_text, text_rect)
        
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_ESCAPE:
                    return False
            
            # Mouse drag for obstacles
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for obs in self.obstacles:
                    if obs.contains_point(mouse_pos):
                        obs.start_drag(mouse_pos)
                        self.dragged_obstacle = obs
                        break
                else:
                    # Create new obstacle
                    new_obs = Obstacle(mouse_pos[0] - 50, mouse_pos[1] - 10, 100, 20)
                    self.obstacles.append(new_obs)
                    new_obs.start_drag(mouse_pos)
                    self.dragged_obstacle = new_obs
            
            if event.type == pygame.MOUSEMOTION:
                if self.dragged_obstacle:
                    self.dragged_obstacle.update_drag(pygame.mouse.get_pos())
            
            if event.type == pygame.MOUSEBUTTONUP:
                if self.dragged_obstacle:
                    self.dragged_obstacle.stop_drag()
                    self.dragged_obstacle = None
        
        return True
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    app = SmartRocketsApp()
    app.run()
