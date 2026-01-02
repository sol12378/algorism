"""
Maze Generation Battle Visualizer
Two algorithms compete: Prim's Algorithm (Lightning) vs Randomized DFS (Snake)
"""

import pygame
import numpy as np
import random
from collections import deque

# Constants
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 80
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 60
STEPS_PER_FRAME = 4  # Adjust for 20-30 second completion

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ELECTRIC_CYAN = (0, 255, 255)
NEON_GREEN = (50, 255, 50)
BRIGHT_YELLOW = (255, 255, 100)
DARK_CYAN = (0, 150, 150)
DARK_GREEN = (25, 150, 25)

# Audio settings
SAMPLE_RATE = 22050
pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 2048)


class AudioEngine:
    """Real-time audio synthesis using NumPy - Pleasant Sound Theory"""
    
    def __init__(self):
        pygame.mixer.init()
        self.channel = pygame.mixer.Channel(0)
        
        # C Major Pentatonic Scale frequencies (C4-A5)
        # C, D, E, G, A in multiple octaves for variety
        self.pentatonic_high = [
            523.25,  # C5
            587.33,  # D5
            659.25,  # E5
            783.99,  # G5
            880.00,  # A5
            1046.50, # C6
        ]
        
        self.pentatonic_low = [
            261.63,  # C4
            293.66,  # D4
            329.63,  # E4
            392.00,  # G4
            440.00,  # A4
        ]
    
    def apply_fade(self, wave, fade_ms=10):
        """Apply fade-in and fade-out to eliminate clicking/popping noise."""
        samples = len(wave)
        fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
        fade_samples = min(fade_samples, samples // 4)  # Max 25% of wave
        
        if fade_samples > 0:
            # Fade in at start
            fade_in = np.linspace(0, 1, fade_samples)
            wave[:fade_samples] *= fade_in
            
            # Fade out at end
            fade_out = np.linspace(1, 0, fade_samples)
            wave[-fade_samples:] *= fade_out
        
        return wave
    
    def play_lightning_crack(self):
        """Ethereal Chimes - Transparent, chorus-laden high frequencies"""
        duration = 0.15
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Pick random frequency from high pentatonic scale
        base_freq = random.choice(self.pentatonic_high)
        
        # Create chorus effect: layer 3 detuned sine waves
        wave = np.zeros(samples, dtype=np.float32)
        
        # Main tone
        wave += np.sin(2 * np.pi * base_freq * t)
        
        # Slightly detuned voices for warmth and thickness
        wave += np.sin(2 * np.pi * (base_freq * 1.005) * t) * 0.8
        wave += np.sin(2 * np.pi * (base_freq * 0.995) * t) * 0.8
        
        # Normalize
        wave /= 2.6
        
        # Envelope: Fast attack, gentle release (like a chime)
        attack = int(samples * 0.05)
        decay = samples - attack
        envelope = np.ones(samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:] = np.exp(-np.linspace(0, 5, decay))
        
        wave = wave * envelope * 0.35  # Gentle volume
        
        # Apply fade to eliminate clicking
        wave = self.apply_fade(wave, fade_ms=5)
        
        stereo = np.column_stack([wave, wave])
        sound_array = (stereo * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        self.channel.play(sound)
    
    def play_snake_tick(self):
        """Organic Marimba - Warm wooden percussion with pitch drop"""
        duration = 0.08
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Pick random frequency from low-mid pentatonic scale
        base_freq = random.choice(self.pentatonic_low)
        
        # Pitch envelope: slight pitch drop (natural percussion behavior)
        pitch_envelope = 1.0 - (t / duration) * 0.15  # Drop by 15%
        instantaneous_freq = base_freq * pitch_envelope
        
        # Sine wave with pitch modulation
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / SAMPLE_RATE
        wave = np.sin(phase).astype(np.float32)
        
        # Sharp attack, quick decay (staccato)
        envelope = np.exp(-t * 40)
        wave = wave * envelope * 0.4
        
        # Apply fade to eliminate clicking
        wave = self.apply_fade(wave, fade_ms=5)
        
        stereo = np.column_stack([wave, wave])
        sound_array = (stereo * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        self.channel.play(sound)
    
    def play_completion_chord(self):
        """Resolution chord on completion"""
        duration = 1.5
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Major chord: C4, E4, G4, C5
        frequencies = [261.63, 329.63, 392.00, 523.25]
        wave = np.zeros(samples, dtype=np.float32)
        
        for freq in frequencies:
            wave += np.sin(2 * np.pi * freq * t)
        
        wave /= len(frequencies)
        
        # Fade out
        envelope = np.exp(-t * 2)
        wave = wave * envelope * 0.3
        
        # Apply fade to eliminate clicking
        wave = self.apply_fade(wave, fade_ms=10)
        
        stereo = np.column_stack([wave, wave])
        sound_array = (stereo * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        self.channel.play(sound)
    
    def stop(self):
        """Stop all sounds"""
        self.channel.stop()


class PrimMaze:
    """Prim's Algorithm - The Lightning ⚡️"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        self.grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.frontiers = set()
        
        # Start from center
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        self.grid[start_y][start_x] = True
        
        # Add initial frontiers
        self._add_frontiers(start_x, start_y)
        self.completed = False
    
    def _add_frontiers(self, x, y):
        """Add neighboring cells to frontier"""
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if not self.grid[ny][nx]:
                    self.frontiers.add((nx, ny))
    
    def step(self):
        """Execute one step of Prim's algorithm"""
        if not self.frontiers:
            self.completed = True
            return None
        
        # Pick random frontier (creates chaotic spread)
        x, y = random.choice(list(self.frontiers))
        self.frontiers.remove((x, y))
        
        # Mark as passage
        self.grid[y][x] = True
        
        # Add new frontiers
        self._add_frontiers(x, y)
        
        return (x, y)
    
    def get_frontiers(self):
        return list(self.frontiers)


class DFSMaze:
    """Randomized DFS - The Snake 🐍"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        self.grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.stack = []
        
        # Start from center
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        self.grid[start_y][start_x] = True
        self.stack.append((start_x, start_y))
        self.completed = False
    
    def step(self):
        """Execute one step of DFS"""
        if not self.stack:
            self.completed = True
            return None
        
        # Current position (snake head)
        x, y = self.stack[-1]
        
        # Get unvisited neighbors
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if not self.grid[ny][nx]:
                    neighbors.append((nx, ny))
        
        if neighbors:
            # Choose random unvisited neighbor
            nx, ny = random.choice(neighbors)
            self.grid[ny][nx] = True
            self.stack.append((nx, ny))
            return (nx, ny)
        else:
            # Backtrack
            self.stack.pop()
            return None
    
    def get_head(self):
        """Get current position of snake head"""
        return self.stack[-1] if self.stack else None


class MazeBattle:
    """Main application"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.audio = AudioEngine()
        
        self.mode = "PRIM"  # or "DFS"
        self.paused = False
        self.flash_timer = 0
        self.flash_duration = 10  # frames
        
        self.prim = PrimMaze(GRID_SIZE)
        self.dfs = DFSMaze(GRID_SIZE)
        
        self.update_title()
    
    def update_title(self):
        if self.mode == "PRIM":
            pygame.display.set_caption("Maze Battle - Mode: The Lightning ⚡️ (Prim's Algorithm)")
        else:
            pygame.display.set_caption("Maze Battle - Mode: The Snake 🐍 (Recursive DFS)")
    
    def reset(self):
        """Reset current algorithm"""
        self.audio.stop()
        if self.mode == "PRIM":
            self.prim.reset()
        else:
            self.dfs.reset()
        self.flash_timer = 0
    
    def toggle_mode(self):
        """Switch between algorithms"""
        self.mode = "DFS" if self.mode == "PRIM" else "PRIM"
        self.update_title()
        self.reset()
    
    def update(self):
        """Update simulation"""
        if self.paused:
            return
        
        if self.flash_timer > 0:
            self.flash_timer -= 1
            return
        
        current = self.prim if self.mode == "PRIM" else self.dfs
        
        if current.completed:
            return
        
        # Execute multiple steps per frame for speed
        for _ in range(STEPS_PER_FRAME):
            result = current.step()
            
            # Play sound
            if result is not None:
                if self.mode == "PRIM":
                    if random.random() < 0.3:  # Don't play every single step
                        self.audio.play_lightning_crack()
                else:
                    if random.random() < 0.4:
                        self.audio.play_snake_tick()
            
            if current.completed:
                # Trigger completion effect
                self.flash_timer = self.flash_duration
                self.audio.stop()
                self.audio.play_completion_chord()
                break
    
    def draw(self):
        """Render the maze"""
        # Flash effect
        if self.flash_timer == self.flash_duration:
            self.screen.fill(WHITE)
            pygame.display.flip()
            return
        
        # Background (wall)
        self.screen.fill(BLACK)
        
        current = self.prim if self.mode == "PRIM" else self.dfs
        
        # Determine colors
        if self.mode == "PRIM":
            path_color = ELECTRIC_CYAN if not current.completed else DARK_CYAN
            frontier_color = WHITE
        else:
            path_color = NEON_GREEN if not current.completed else DARK_GREEN
            head_color = BRIGHT_YELLOW
        
        # Draw passages
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if current.grid[y][x]:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, path_color, rect)
        
        # Draw special markers
        if not current.completed:
            if self.mode == "PRIM":
                # Draw frontiers (white flashes)
                frontiers = self.prim.get_frontiers()
                for x, y in frontiers:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, frontier_color, rect)
            else:
                # Draw snake head
                head = self.dfs.get_head()
                if head:
                    x, y = head
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, head_color, rect)
        
        pygame.display.flip()
    
    def run(self):
        """Main loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_m:
                        self.toggle_mode()
            
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()


if __name__ == "__main__":
    app = MazeBattle()
    app.run()
