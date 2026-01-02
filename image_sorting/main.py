import pygame
import numpy as np
import random
import math
import os

# ============================================
# Configuration
# ============================================
WINDOW_SIZE = 800
PIXEL_SIZE = 160  # 80x80 pixels for the image
CELL_SIZE = WINDOW_SIZE // PIXEL_SIZE
SAMPLE_RATE = 44100
AUDIO_BUFFER = 512

# ============================================
# Audio Synthesis
# ============================================
class AudioSynthesizer:
    def __init__(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, AUDIO_BUFFER)
        pygame.mixer.init()
        self.current_sound = None
        
    def generate_tone(self, frequency, duration=0.05, volume=0.3, attack=0.01, decay=0.02):
        """Generate a soft marimba-like tone"""
        num_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        
        # Marimba-like tone: fundamental + harmonics with decay
        wave = np.sin(2 * np.pi * frequency * t)
        wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 20)
        wave += 0.25 * np.sin(2 * np.pi * frequency * 4 * t) * np.exp(-t * 30)
        
        # ADSR envelope
        envelope = np.ones(num_samples, dtype=np.float32)
        attack_samples = int(SAMPLE_RATE * attack)
        decay_samples = int(SAMPLE_RATE * decay)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        release_start = int(num_samples * 0.6)
        envelope[release_start:] = np.linspace(1, 0, num_samples - release_start)
        
        wave = wave * envelope * volume
        wave = np.clip(wave, -1, 1)
        wave = (wave * 32767).astype(np.int16)
        
        return pygame.sndobj.Sound(buffer=wave.tobytes()) if hasattr(pygame, 'sndobj') else pygame.mixer.Sound(buffer=wave.tobytes())
    
    def play_progress_tone(self, progress, total):
        """Play tone based on sorting progress (low to high)"""
        # Map progress to frequency (C3 to C6 range)
        base_freq = 130.81  # C3
        freq_range = 4  # 4 octaves
        freq = base_freq * (2 ** (freq_range * progress / total))
        freq = min(freq, 2093)  # Cap at C7
        
        sound = self.generate_tone(freq, duration=0.04, volume=0.2)
        sound.play()
        
    def play_swap_tone(self, index, total):
        """Play tone for random swaps (shell sort)"""
        # Pentatonic scale for pleasant random sounds
        pentatonic = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25]
        freq = pentatonic[index % len(pentatonic)]
        freq *= (1 + (index / total) * 0.5)
        
        sound = self.generate_tone(freq, duration=0.03, volume=0.15)
        sound.play()

    def play_fanfare(self):
        """Play completion fanfare (major chord arpeggio)"""
        # C major chord: C4, E4, G4, C5
        frequencies = [261.63, 329.63, 392.00, 523.25]
        duration = 0.3
        total_samples = int(SAMPLE_RATE * duration * 2)
        
        wave = np.zeros(total_samples, dtype=np.float32)
        
        for i, freq in enumerate(frequencies):
            delay_samples = int(SAMPLE_RATE * 0.1 * i)
            note_samples = int(SAMPLE_RATE * duration)
            t = np.linspace(0, duration, note_samples, dtype=np.float32)
            
            # Rich harmonic tone
            note = np.sin(2 * np.pi * freq * t)
            note += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
            note += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
            
            # Envelope
            envelope = np.exp(-t * 3) * (1 - np.exp(-t * 50))
            note = note * envelope * 0.3
            
            end_idx = min(delay_samples + note_samples, total_samples)
            wave[delay_samples:end_idx] += note[:end_idx - delay_samples]
        
        wave = np.clip(wave, -1, 1)
        wave = (wave * 32767).astype(np.int16)
        
        sound = pygame.mixer.Sound(buffer=wave.tobytes())
        sound.play()


# ============================================
# Image Handler
# ============================================
class ImageHandler:
    def __init__(self, size):
        self.size = size
        self.pixels = []
        
    def load_image(self, filename):
        """Load and resize image using pygame to avoid PIL dependency"""
        try:
            # Get absolute path relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, filename)
            
            if not os.path.exists(path):
                return False
                
            # Use pygame to load the image
            img = pygame.image.load(path).convert()
            # Resize to the internal processing resolution (e.g., 80x80)
            img = pygame.transform.smoothscale(img, (self.size, self.size))
            
            self.pixels = []
            for y in range(self.size):
                for x in range(self.size):
                    color = img.get_at((x, y))
                    # Store as (R, G, B) and original index
                    index = y * self.size + x
                    self.pixels.append({'color': (color.r, color.g, color.b), 'index': index})
            return True
        except Exception as e:
            print(f"Loading error: {e}")
            return False
    
    def generate_gradient(self):
        """Generate colorful gradient pattern"""
        self.pixels = []
        for y in range(self.size):
            for x in range(self.size):
                # Create beautiful gradient pattern
                r = int(128 + 127 * math.sin(x * 0.1 + y * 0.05))
                g = int(128 + 127 * math.sin(y * 0.1 + x * 0.03 + 2))
                b = int(128 + 127 * math.sin((x + y) * 0.08 + 4))
                
                index = y * self.size + x
                self.pixels.append({'color': (r, g, b), 'index': index})
    
    def shuffle(self):
        """Randomly shuffle all pixels"""
        random.shuffle(self.pixels)
    
    def is_sorted(self):
        """Check if all pixels are in correct position"""
        for i, pixel in enumerate(self.pixels):
            if pixel['index'] != i:
                return False
        return True


# ============================================
# Sorting Visualizer
# ============================================
class SortingVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Image Sorting Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.audio = AudioSynthesizer()
        self.image = ImageHandler(PIXEL_SIZE)
        
        # Try to load target image
        for ext in ['jpg', 'jpeg', 'png', 'JPG', 'PNG']:
            if self.image.load_image(f'target.{ext}'):
                break
        else:
            self.image.generate_gradient()
        
        # Store original for comparison
        self.original_pixels = [p.copy() for p in self.image.pixels]
        
        # Shuffle pixels
        self.image.shuffle()
        
        # State
        self.paused = False
        self.sorting_complete = False
        self.algorithm = 0  # 0: Selection Sort, 1: Shell Sort
        self.algorithm_names = ["Selection Sort", "Shell Sort"]
        
        # Sorting state
        self.reset_sort_state()
        
        # Victory effect
        self.flash_alpha = 0
        self.victory_time = 0
        self.glow_phase = 0
        
        # Speed control
        self.operations_per_frame = 3
        self.frame_count = 0
        self.sound_cooldown = 0
        
    def reset_sort_state(self):
        """Reset sorting algorithm state"""
        self.current_index = 0
        self.shell_gap = PIXEL_SIZE * PIXEL_SIZE // 2
        self.sorting_complete = False
        self.flash_alpha = 0
        self.victory_time = 0
        
    def reset(self):
        """Reset to shuffled state"""
        self.image.pixels = [p.copy() for p in self.original_pixels]
        self.image.shuffle()
        self.reset_sort_state()
        self.paused = False
        
    def selection_sort_step(self):
        """Perform one step of selection sort"""
        n = len(self.image.pixels)
        
        if self.current_index >= n:
            return False
            
        # Find the pixel that should be at current_index
        min_idx = self.current_index
        for j in range(self.current_index, n):
            if self.image.pixels[j]['index'] == self.current_index:
                min_idx = j
                break
        
        # Swap
        if min_idx != self.current_index:
            self.image.pixels[self.current_index], self.image.pixels[min_idx] = \
                self.image.pixels[min_idx], self.image.pixels[self.current_index]
        
        # Play sound
        if self.sound_cooldown <= 0:
            self.audio.play_progress_tone(self.current_index, n)
            self.sound_cooldown = 2
        
        self.current_index += 1
        return self.current_index < n
    
    def shell_sort_step(self):
        """
        Denoising Sort (Randomized Cycle Resolution).
        Picks a random wrong pixel and places it in its correct home.
        This guarantees progress and looks like 'denoising' or 'developing'.
        """
        n = len(self.image.pixels)
        performed_swap = False
        
        # Try a few times to find a random misplaced pixel
        # This gives the "random pop-in" effect
        for _ in range(20):
            i = random.randint(0, n - 1)
            target_idx = self.image.pixels[i]['index']
            
            # If pixel is not at its correct home
            if target_idx != i:
                # Swap it to its correct home!
                # Note: The pixel currently at target_idx might be wrong too,
                # but at least 'i' (the one we are moving) will become correct.
                self.image.pixels[i], self.image.pixels[target_idx] = \
                    self.image.pixels[target_idx], self.image.pixels[i]
                
                performed_swap = True
                
                # Sound effect
                if self.sound_cooldown <= 0:
                    self.audio.play_swap_tone(target_idx, n)
                    self.sound_cooldown = 1  # Fast cooldown for this mode
                
                # We only need one successful "pop" per call to show animation,
                # but the main loop calls this multiple times.
                break
        
        # If random sampling failed to find a swap (late stage), scan linearly
        if not performed_swap:
            start_scan = random.randint(0, n - 1)
            for k in range(n):
                i = (start_scan + k) % n
                target_idx = self.image.pixels[i]['index']
                if target_idx != i:
                    self.image.pixels[i], self.image.pixels[target_idx] = \
                        self.image.pixels[target_idx], self.image.pixels[i]
                    performed_swap = True
                    break

        return performed_swap
    
    def update(self):
        """Update sorting state"""
        if self.paused or self.sorting_complete:
            # Still update audio or effects if needed
            self.sound_cooldown -= 1
            return
            
        self.sound_cooldown -= 1
        
        # Speed regulation
        if self.algorithm == 0:
            # Selection sort needs less ops per frame to be visible/satisfying
            ops = self.operations_per_frame * 10 
            for _ in range(ops):
                if not self.selection_sort_step():
                    self.sorting_complete = True
                    break
        else:
            # Denoising can handle many ops per frame for a nice "fizz" effect
            ops = self.operations_per_frame * 15
            for _ in range(ops):
                if not self.shell_sort_step():
                    self.sorting_complete = True
                    break
        
        # Check completion
        # Double check if sorted (sometimes the step returns False early)
        if self.sorting_complete or self.image.is_sorted():
            self.sorting_complete = True
            if self.victory_time == 0:
                self.victory_time = pygame.time.get_ticks()
                self.flash_alpha = 255
                self.audio.play_fanfare()
    
    def draw(self):
        """Draw the current state"""
        self.screen.fill((20, 20, 20))
        
        # Draw pixels
        for i, pixel in enumerate(self.image.pixels):
            x = (i % PIXEL_SIZE) * CELL_SIZE
            y = (i // PIXEL_SIZE) * CELL_SIZE
            color = pixel['color']
            
            # Glow effect for completed pixels in selection sort
            if self.algorithm == 0 and i < self.current_index:
                # Slight brightness boost for sorted area
                color = tuple(min(255, c + 10) for c in color)
            
            pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            
            # Add subtle pixel border for retro look
            if CELL_SIZE > 4:
                border_color = tuple(max(0, c - 30) for c in color)
                pygame.draw.rect(self.screen, border_color, (x, y, CELL_SIZE, CELL_SIZE), 1)
        
        # Draw scan line for selection sort
        if self.algorithm == 0 and not self.sorting_complete and self.current_index < len(self.image.pixels):
            scan_y = (self.current_index // PIXEL_SIZE) * CELL_SIZE
            pygame.draw.rect(self.screen, (255, 255, 255, 128), 
                           (0, scan_y, WINDOW_SIZE, 2))
        
        # Victory flash effect
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            flash_surface.fill((255, 255, 255))
            flash_surface.set_alpha(int(self.flash_alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.flash_alpha = max(0, self.flash_alpha - 8)
        
        # Victory glow effect
        if self.sorting_complete and self.victory_time > 0:
            self.glow_phase += 0.1
            glow_alpha = int(20 + 15 * math.sin(self.glow_phase))
            glow_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            glow_surface.fill((255, 255, 200))
            glow_surface.set_alpha(glow_alpha)
            self.screen.blit(glow_surface, (0, 0))
        
        # Draw UI overlay
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_ui(self):
        """Draw UI elements"""
        # User requested no text overlay, so this is left empty.
        pass
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                elif event.key == pygame.K_r:
                    self.reset()
                
                elif event.key == pygame.K_m:
                    self.algorithm = (self.algorithm + 1) % len(self.algorithm_names)
                    self.reset()
                
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def run(self):
        """Main loop"""
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
    app = SortingVisualizer()
    app.run()
