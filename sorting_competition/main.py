import pygame
import numpy as np
import random
import colorsys
from enum import Enum

# ============================================================================
# Constants
# ============================================================================
WINDOW_SIZE = 800
VIRTUAL_SIZE = 80
PIXEL_SIZE = WINDOW_SIZE // VIRTUAL_SIZE
FPS = 60

# Audio settings
SAMPLE_RATE = 22050
CHUNK_SIZE = SAMPLE_RATE // FPS

class SortMode(Enum):
    BUBBLE = "Bubble Sort (The Tortoise 🐢)"
    RADIX = "Radix Sort (The Laser ⚡)"

# ============================================================================
# Audio Generation
# ============================================================================
def generate_bubble_sound(t, intensity=0.5):
    """Low, muddy drum-like noise for bubble sort"""
    freq = 60 + intensity * 20  # Low frequency
    wave = np.sin(2 * np.pi * freq * t)
    noise = np.random.uniform(-0.3, 0.3, len(t))
    sound = wave * 0.3 + noise * 0.7
    sound *= intensity * 0.3
    return sound

def generate_laser_sound(t, pitch_factor=1.0, intensity=0.8):
    """High-pitched square wave laser sound"""
    freq = 800 * pitch_factor  # High frequency
    phase = 2 * np.pi * freq * t
    square_wave = np.sign(np.sin(phase))
    sound = square_wave * intensity * 0.2
    return sound

def generate_complete_sound(t):
    """Sparkle sound when radix sort completes"""
    freq = 1200
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-3 * t)
    return wave * 0.3

# ============================================================================
# Pixel Data Management
# ============================================================================
class PixelData:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Create shuffled hue gradient"""
        self.data = []
        num_pixels = VIRTUAL_SIZE * VIRTUAL_SIZE
        for i in range(num_pixels):
            hue = (i / num_pixels) * 360
            self.data.append(hue)
        random.shuffle(self.data)
        self.width = VIRTUAL_SIZE
        self.height = VIRTUAL_SIZE
    
    def get_color(self, hue):
        """Convert hue to RGB"""
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def is_sorted(self):
        """Check if data is fully sorted"""
        for i in range(len(self.data) - 1):
            if self.data[i] > self.data[i + 1]:
                return False
        return True

# ============================================================================
# Bubble Sort (The Tortoise)
# ============================================================================
class BubbleSorter:
    def __init__(self, pixel_data):
        self.data = pixel_data.data
        self.i = 0
        self.j = 0
        self.n = len(self.data)
        self.completed = False
        self.comparisons_per_frame = 75  # Intentionally slow
        self.last_swap = False
    
    def step(self):
        """Execute one frame of bubble sort"""
        if self.completed:
            return False
        
        comparisons = 0
        swapped_this_frame = False
        
        while comparisons < self.comparisons_per_frame and not self.completed:
            if self.j < self.n - self.i - 1:
                if self.data[self.j] > self.data[self.j + 1]:
                    self.data[self.j], self.data[self.j + 1] = \
                        self.data[self.j + 1], self.data[self.j]
                    swapped_this_frame = True
                self.j += 1
                comparisons += 1
            else:
                self.j = 0
                self.i += 1
                if self.i >= self.n - 1:
                    self.completed = True
                    break
        
        self.last_swap = swapped_this_frame
        return swapped_this_frame

# ============================================================================
# Radix Sort (The Laser)
# ============================================================================
class RadixSorter:
    def __init__(self, pixel_data):
        self.data = pixel_data.data
        self.completed = False
        self.pass_number = 0
        self.total_passes = 4  # Will complete in 4 passes
        self.frames_per_pass = 30  # Each pass takes 0.5 seconds at 60fps
        self.current_frame = 0
        self.temp_data = self.data.copy()
        self.is_scanning = False
    
    def step(self):
        """Execute one frame of radix sort"""
        if self.completed:
            return False
        
        self.current_frame += 1
        
        # Start a new pass
        if self.current_frame == 1:
            self.is_scanning = True
            self._execute_pass()
            return True
        
        # Complete the pass after animation
        if self.current_frame >= self.frames_per_pass:
            self.current_frame = 0
            self.pass_number += 1
            self.is_scanning = False
            
            if self.pass_number >= self.total_passes:
                self.completed = True
                return False
        
        return self.is_scanning
    
    def _execute_pass(self):
        """Execute one radix sort pass"""
        if self.pass_number >= self.total_passes:
            return
        
        # Use radix sort logic (LSD)
        max_val = max(self.data)
        exp = 10 ** self.pass_number
        
        if exp > max_val:
            self.data.sort()  # Final polish
            return
        
        # Counting sort by digit
        output = [0] * len(self.data)
        count = [0] * 10
        
        for val in self.data:
            digit = int((val / exp) % 10)
            count[digit] += 1
        
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        for i in range(len(self.data) - 1, -1, -1):
            val = self.data[i]
            digit = int((val / exp) % 10)
            output[count[digit] - 1] = val
            count[digit] -= 1
        
        self.data[:] = output

# ============================================================================
# Main Application
# ============================================================================
class SortVisualizer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=CHUNK_SIZE)
        
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Virtual screen for pixel art
        self.virtual_screen = pygame.Surface((VIRTUAL_SIZE, VIRTUAL_SIZE))
        
        # Data and sorter
        self.pixel_data = PixelData()
        self.mode = SortMode.BUBBLE
        self.sorter = None
        self.reset_sorter()
        
        # Audio
        self.audio_time = 0
        self.completion_sound_played = False
        
        self.update_title()
    
    def update_title(self):
        pygame.display.set_caption(f"Sort Race - {self.mode.value}")
    
    def reset_sorter(self):
        """Initialize sorter based on current mode"""
        if self.mode == SortMode.BUBBLE:
            self.sorter = BubbleSorter(self.pixel_data)
        else:
            self.sorter = RadixSorter(self.pixel_data)
        self.completion_sound_played = False
    
    def reset_data(self):
        """Reset and reshuffle data"""
        self.pixel_data.reset()
        self.reset_sorter()
    
    def toggle_mode(self):
        """Switch between bubble and radix sort"""
        if self.mode == SortMode.BUBBLE:
            self.mode = SortMode.RADIX
        else:
            self.mode = SortMode.BUBBLE
        self.update_title()
        self.reset_data()
    
    def render(self):
        """Render the virtual screen and scale up"""
        self.virtual_screen.fill((0, 0, 0))
        
        # Draw each pixel
        for i, hue in enumerate(self.pixel_data.data):
            x = i % VIRTUAL_SIZE
            y = i // VIRTUAL_SIZE
            color = self.pixel_data.get_color(hue)
            self.virtual_screen.set_at((x, y), color)
        
        # Scale up to window size
        scaled = pygame.transform.scale(
            self.virtual_screen,
            (WINDOW_SIZE, WINDOW_SIZE)
        )
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()
    
    def generate_audio(self):
        """Generate audio based on current state"""
        t = np.linspace(
            self.audio_time,
            self.audio_time + CHUNK_SIZE / SAMPLE_RATE,
            CHUNK_SIZE
        )
        self.audio_time += CHUNK_SIZE / SAMPLE_RATE
        
        if self.sorter.completed and not self.completion_sound_played:
            if self.mode == SortMode.RADIX:
                sound = generate_complete_sound(t - (self.audio_time - CHUNK_SIZE / SAMPLE_RATE))
                self.completion_sound_played = True
                return sound
            else:
                return np.zeros(CHUNK_SIZE)
        
        if self.sorter.completed or self.paused:
            return np.zeros(CHUNK_SIZE)
        
        if self.mode == SortMode.BUBBLE:
            if hasattr(self.sorter, 'last_swap') and self.sorter.last_swap:
                intensity = 0.6
            else:
                intensity = 0.2
            sound = generate_bubble_sound(t, intensity)
        else:  # Radix
            if self.sorter.is_scanning:
                pitch_factor = 1.0 + (self.sorter.pass_number * 0.3)
                sound = generate_laser_sound(t, pitch_factor, 0.9)
            else:
                sound = np.zeros(CHUNK_SIZE)
        
        return sound
    
    def play_audio(self):
        """Queue audio to pygame mixer"""
        if pygame.mixer.get_busy():
            return
        
        audio_data = self.generate_audio()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        try:
            # Check if mixer is in stereo mode
            channels = pygame.mixer.get_init()[2]
            if channels == 2:
                # Convert mono to stereo by duplicating channels
                audio_int16 = np.repeat(audio_int16[:, np.newaxis], 2, axis=1)
        except Exception:
            pass
            
        sound = pygame.sndarray.make_sound(audio_int16)
        sound.play()
    
    def handle_events(self):
        """Process user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.toggle_mode()
                elif event.key == pygame.K_r:
                    self.reset_data()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            
            if not self.paused and not self.sorter.completed:
                self.sorter.step()
            
            self.render()
            self.play_audio()
            self.clock.tick(FPS)
        
        pygame.quit()

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    app = SortVisualizer()
    app.run()
