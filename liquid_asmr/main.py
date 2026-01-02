#!/usr/bin/env python3
"""
ASMR Physics Simulation: Sand & Water
A satisfying cellular automata simulation with real-time audio synthesis.
"""

import pygame
import numpy as np
from enum import Enum
import random
import math

# ============================================================================
# Constants
# ============================================================================

WINDOW_SIZE = 800
GRID_SIZE = 80
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

FPS = 60
SPAWN_RATE = 3  # particles per frame - slow and satisfying

# Colors
BG_COLOR = (20, 20, 25)
WALL_COLOR = (60, 60, 70)
SAND_BASE = np.array([255, 200, 50])
WATER_BASE = np.array([0, 200, 255])

# Audio
SAMPLE_RATE = 22050
AUDIO_CHUNK_SIZE = 1024


# ============================================================================
# Enums
# ============================================================================

class Mode(Enum):
    SAND = 0
    WATER = 1


class CellType(Enum):
    EMPTY = 0
    SAND = 1
    WATER = 2
    WALL = 3


# ============================================================================
# Audio Engine
# ============================================================================

class AudioEngine:
    """Real-time audio synthesis for ASMR effects."""
    
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=2048)
        self.sample_rate = SAMPLE_RATE
        
        # Pink noise filter state
        self.pink_state = np.zeros(7)
        
        # Water bubble parameters
        self.pentatonic = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]  # C major pentatonic
        
        # Movement tracking for volume
        self.movement_energy = 0
        self.energy_decay = 0.98  # Slower decay for smoother transitions
        
        # Background sound channel
        self.bg_channel = pygame.mixer.Channel(0)
        self.bg_sound = None
    
    def apply_fade(self, wave, fade_samples=500):
        """Apply fade-in and fade-out to eliminate clicking/popping noise."""
        samples = len(wave)
        fade_samples = min(fade_samples, samples // 4)  # Max 25% of wave
        
        if fade_samples > 0:
            # Fade in at start
            fade_in = np.linspace(0, 1, fade_samples)
            wave[:fade_samples] *= fade_in
            
            # Fade out at end
            fade_out = np.linspace(1, 0, fade_samples)
            wave[-fade_samples:] *= fade_out
        
        return wave
        
    def generate_brown_noise(self, duration=0.5, volume=0.08):
        """Generate heavily filtered brown noise for ultra-soft ASMR sound."""
        samples = int(self.sample_rate * duration)
        
        # Generate white noise
        white = np.random.randn(samples)
        
        # Brown noise is the cumulative sum of white noise
        brown = np.cumsum(white)
        
        # Normalize to prevent overflow
        brown = brown / np.max(np.abs(brown))
        
        # Apply aggressive low-pass filter using moving average
        # This removes all harsh high frequencies
        window_size = 50  # Larger window = more smoothing
        window = np.ones(window_size) / window_size
        brown_filtered = np.convolve(brown, window, mode='same')
        
        # Normalize again after filtering
        brown_filtered = brown_filtered / np.max(np.abs(brown_filtered))
        
        # Apply very gentle volume (half of previous)
        output = brown_filtered * volume
        
        # Apply fade to eliminate clicking
        output = self.apply_fade(output, fade_samples=500)
        
        output = (output * 32767).astype(np.int16)
        
        # Convert to stereo (2D array)
        return np.column_stack((output, output))
    
    def generate_bubble_sound(self, base_freq=None):
        """Generate a single bubble/water droplet sound."""
        if base_freq is None:
            base_freq = random.choice(self.pentatonic)
        
        duration = 0.15
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Pitch envelope: rapid downward sweep
        freq_envelope = base_freq * np.exp(-8 * t)
        
        # Amplitude envelope: quick attack, exponential decay
        amp_envelope = np.exp(-6 * t)
        
        # Generate sine wave with frequency modulation
        phase = 2 * np.pi * np.cumsum(freq_envelope) / self.sample_rate
        wave = np.sin(phase) * amp_envelope
        
        # Add harmonics for richer sound
        wave += 0.3 * np.sin(2 * phase) * amp_envelope
        wave += 0.15 * np.sin(3 * phase) * amp_envelope
        
        # Normalize and convert
        volume = 0.15
        wave = wave / np.max(np.abs(wave)) * volume
        
        # Apply fade to eliminate clicking
        wave = self.apply_fade(wave, fade_samples=500)
        
        wave = (wave * 32767).astype(np.int16)
        # Convert to stereo (2D array)
        return np.column_stack((wave, wave))
    
    def generate_chime_sound(self):
        """Generate a crystalline wind chime sound for completion."""
        duration = 2.0
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Multiple harmonically related frequencies
        freqs = [523.25, 659.25, 783.99, 1046.50]  # C, E, G, C (major chord)
        wave = np.zeros(samples)
        
        for i, freq in enumerate(freqs):
            delay = i * 0.05  # Slight delay between tones
            env = np.exp(-2 * np.maximum(0, t - delay))
            phase = 2 * np.pi * freq * np.maximum(0, t - delay)
            wave += np.sin(phase) * env
        
        # Normalize
        volume = 0.4
        wave = wave / np.max(np.abs(wave)) * volume
        
        # Apply fade to eliminate clicking
        wave = self.apply_fade(wave, fade_samples=500)
        
        wave = (wave * 32767).astype(np.int16)
        # Convert to stereo (2D array)
        return np.column_stack((wave, wave))
    
    def play_sand_ambient(self, energy):
        """Play continuous filtered brown noise based on movement energy."""
        if energy > 0.01:
            volume = min(0.12, energy * 0.2)  # Very low volume for ambient background
            sound_array = self.generate_brown_noise(duration=0.1, volume=volume)
            sound = pygame.sndarray.make_sound(sound_array)
            if not self.bg_channel.get_busy():
                self.bg_channel.play(sound)
    
    def play_water_bubble(self):
        """Play a random bubble sound."""
        sound_array = self.generate_bubble_sound()
        sound = pygame.sndarray.make_sound(sound_array)
        sound.play()
    
    def play_completion_chime(self):
        """Play completion sound effect."""
        sound_array = self.generate_chime_sound()
        sound = pygame.sndarray.make_sound(sound_array)
        sound.play()
    
    def update_movement_energy(self, movements):
        """Update movement energy based on particle movements."""
        self.movement_energy = self.movement_energy * self.energy_decay + movements * 0.01
        self.movement_energy = min(1.0, self.movement_energy)


# ============================================================================
# Physics Engine (Cellular Automata)
# ============================================================================

class PhysicsEngine:
    """Cellular automata physics for sand and water."""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.colors = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        self.movement_count = 0
        
    def get_cell(self, x, y):
        """Get cell type at position."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return CellType(self.grid[y, x])
        return CellType.WALL
    
    def set_cell(self, x, y, cell_type, color=None):
        """Set cell type and color at position."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = cell_type.value
            if color is not None:
                self.colors[y, x] = color
    
    def is_empty(self, x, y):
        """Check if cell is empty."""
        return self.get_cell(x, y) == CellType.EMPTY
    
    def spawn_particle(self, x, y, mode):
        """Spawn a particle at position."""
        if self.is_empty(x, y):
            if mode == Mode.SAND:
                color = SAND_BASE + np.random.randint(-20, 20, 3)
                color = np.clip(color, 0, 255)
                self.set_cell(x, y, CellType.SAND, color)
            else:
                color = WATER_BASE + np.random.randint(-30, 30, 3)
                color = np.clip(color, 0, 255)
                self.set_cell(x, y, CellType.WATER, color)
            return True
        return False
    
    def update_sand(self):
        """Update sand particles using cellular automata."""
        self.movement_count = 0
        new_grid = self.grid.copy()
        new_colors = self.colors.copy()
        
        # Process from bottom to top, randomize left-right order
        for y in range(self.grid_size - 2, -1, -1):
            x_order = list(range(self.grid_size))
            random.shuffle(x_order)
            
            for x in x_order:
                if self.grid[y, x] != CellType.SAND.value:
                    continue
                
                color = self.colors[y, x]
                
                # Try to fall down
                if y + 1 < self.grid_size and new_grid[y + 1, x] == CellType.EMPTY.value:
                    new_grid[y, x] = CellType.EMPTY.value
                    new_grid[y + 1, x] = CellType.SAND.value
                    new_colors[y + 1, x] = color
                    self.movement_count += 1
                # Try diagonal fall
                else:
                    dirs = [(-1, 1), (1, 1)]
                    random.shuffle(dirs)
                    moved = False
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                            new_grid[ny, nx] == CellType.EMPTY.value):
                            new_grid[y, x] = CellType.EMPTY.value
                            new_grid[ny, nx] = CellType.SAND.value
                            new_colors[ny, nx] = color
                            self.movement_count += 1
                            moved = True
                            break
                    
                    if not moved:
                        new_grid[y, x] = CellType.SAND.value
                        new_colors[y, x] = color
        
        self.grid = new_grid
        self.colors = new_colors
    
    def update_water(self):
        """Update water particles with spreading behavior."""
        self.movement_count = 0
        new_grid = self.grid.copy()
        new_colors = self.colors.copy()
        
        # Process from bottom to top, randomize order
        for y in range(self.grid_size - 2, -1, -1):
            x_order = list(range(self.grid_size))
            random.shuffle(x_order)
            
            for x in x_order:
                if self.grid[y, x] != CellType.WATER.value:
                    continue
                
                color = self.colors[y, x]
                
                # Try to fall down
                if y + 1 < self.grid_size and new_grid[y + 1, x] == CellType.EMPTY.value:
                    new_grid[y, x] = CellType.EMPTY.value
                    new_grid[y + 1, x] = CellType.WATER.value
                    new_colors[y + 1, x] = color
                    self.movement_count += 1
                # Try diagonal fall
                else:
                    dirs = [(-1, 1), (1, 1)]
                    random.shuffle(dirs)
                    moved = False
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                            new_grid[ny, nx] == CellType.EMPTY.value):
                            new_grid[y, x] = CellType.EMPTY.value
                            new_grid[ny, nx] = CellType.WATER.value
                            new_colors[ny, nx] = color
                            self.movement_count += 1
                            moved = True
                            break
                    
                    # If can't fall, try to spread horizontally
                    if not moved:
                        h_dirs = [(-1, 0), (1, 0)]
                        random.shuffle(h_dirs)
                        for dx, dy in h_dirs:
                            nx, ny = x + dx, y
                            if (0 <= nx < self.grid_size and
                                new_grid[ny, nx] == CellType.EMPTY.value):
                                new_grid[y, x] = CellType.EMPTY.value
                                new_grid[ny, nx] = CellType.WATER.value
                                new_colors[ny, nx] = color
                                self.movement_count += 1
                                moved = True
                                break
                    
                    if not moved:
                        new_grid[y, x] = CellType.WATER.value
                        new_colors[y, x] = color
        
        self.grid = new_grid
        self.colors = new_colors
    
    def clear_particles(self):
        """Clear all particles (keep walls)."""
        mask = (self.grid == CellType.WALL.value)
        self.grid[~mask] = CellType.EMPTY.value
        self.colors[~mask] = 0


# ============================================================================
# Main Application
# ============================================================================

class ASMRSimulation:
    """Main application class."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("ASMR Physics: Sand & Water")
        self.clock = pygame.time.Clock()
        
        # Core systems
        self.physics = PhysicsEngine(GRID_SIZE)
        self.audio = AudioEngine()
        
        # State
        self.mode = Mode.SAND
        self.running = True
        self.paused = False
        self.completed = False
        self.completion_timer = 0
        
        # Overflow detection
        self.overflow_frames = 0
        self.overflow_threshold = 200  # consecutive blocked spawns before completion
        
        # Mouse state
        self.mouse_down = False
        
        # Water bubble timing
        self.water_bubble_timer = 0
        self.water_bubble_interval = 0.1  # seconds
        
        # Flash effect
        self.flash_alpha = 0
        
        # Initialize container
        self._create_container()
    
    def _create_container(self):
        """Create U-shaped container in lower half."""
        # Bottom
        for x in range(GRID_SIZE):
            self.physics.set_cell(x, GRID_SIZE - 1, CellType.WALL, WALL_COLOR)
            self.physics.set_cell(x, GRID_SIZE - 2, CellType.WALL, WALL_COLOR)
        
        # Left wall (from bottom to middle)
        wall_height = 30
        for y in range(GRID_SIZE - wall_height, GRID_SIZE):
            self.physics.set_cell(10, y, CellType.WALL, WALL_COLOR)
            self.physics.set_cell(11, y, CellType.WALL, WALL_COLOR)
        
        # Right wall
        for y in range(GRID_SIZE - wall_height, GRID_SIZE):
            self.physics.set_cell(GRID_SIZE - 11, y, CellType.WALL, WALL_COLOR)
            self.physics.set_cell(GRID_SIZE - 12, y, CellType.WALL, WALL_COLOR)
    
    def spawn_particles(self):
        """Spawn particles from top center. Returns True if successful, False if blocked."""
        if self.completed:
            return True
        
        center_x = GRID_SIZE // 2
        spread = 10  # Wide spread for even filling and natural overflow
        
        spawned_any = False
        for _ in range(SPAWN_RATE):
            x = center_x + random.randint(-spread, spread)
            y = 1
            if self.physics.spawn_particle(x, y, self.mode):
                spawned_any = True
        
        # Check for overflow (spawn area blocked)
        if not spawned_any:
            self.overflow_frames += 1
            if self.overflow_frames >= self.overflow_threshold:
                self.trigger_completion()
        else:
            self.overflow_frames = 0
        
        return spawned_any
    
    def check_completion(self):
        """Check completion - now handled by overflow detection in spawn_particles."""
        # Completion is now triggered only by overflow (consecutive spawn failures)
        # No height-based early completion
        pass
    
    def trigger_completion(self):
        """Trigger completion effect."""
        if not self.completed:
            self.completed = True
            self.completion_timer = pygame.time.get_ticks()
            self.flash_alpha = 255
            self.audio.play_completion_chime()
    
    def handle_events(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                elif event.key == pygame.K_r:
                    self.reset()
                
                elif event.key == pygame.K_m:
                    self.mode = Mode.WATER if self.mode == Mode.SAND else Mode.SAND
                    self.reset()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_down = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
        
        # Handle mouse dragging to draw walls
        if self.mouse_down:
            mx, my = pygame.mouse.get_pos()
            gx = mx // CELL_SIZE
            gy = my // CELL_SIZE
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                self.physics.set_cell(gx, gy, CellType.WALL, WALL_COLOR)
    
    def update(self, dt):
        """Update simulation."""
        if self.paused:
            return
        
        # Spawn particles
        self.spawn_particles()
        
        # Update physics
        if self.mode == Mode.SAND:
            self.physics.update_sand()
        else:
            self.physics.update_water()
        
        # Update audio
        self.audio.update_movement_energy(self.physics.movement_count)
        
        if self.mode == Mode.SAND:
            self.audio.play_sand_ambient(self.audio.movement_energy)
        else:
            # Water bubbles on movement
            self.water_bubble_timer += dt
            if self.physics.movement_count > 5 and self.water_bubble_timer >= self.water_bubble_interval:
                self.audio.play_water_bubble()
                self.water_bubble_timer = 0
        
        # Check completion
        self.check_completion()
        
        # Update flash
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 5)
    
    def render(self):
        """Render the simulation."""
        self.screen.fill(BG_COLOR)
        
        # Render grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell_type = CellType(self.physics.grid[y, x])
                if cell_type != CellType.EMPTY:
                    color = self.physics.colors[y, x]
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
        
        # Flash effect
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            flash_surface.fill((255, 255, 255))
            flash_surface.set_alpha(self.flash_alpha)
            self.screen.blit(flash_surface, (0, 0))
        
        pygame.display.flip()
    
    def reset(self):
        """Reset simulation."""
        self.physics.clear_particles()
        self.completed = False
        self.flash_alpha = 0
        self.water_bubble_timer = 0
        self.overflow_frames = 0
    
    def run(self):
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            
            self.handle_events()
            self.update(dt)
            self.render()
        
        pygame.quit()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    app = ASMRSimulation()
    app.run()
