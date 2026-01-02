import pygame
import numpy as np
import random
import math
from enum import Enum
from typing import Tuple, List

# ==================== Constants ====================
WINDOW_SIZE = 800
GRID_SIZE = 100  # 100x100 = 10,000 dots
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
FPS = 60

# Audio
SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
BUFFER_SIZE = 1024

# C Major Pentatonic Scale (frequencies in Hz)
C_MAJOR_PENTATONIC = {
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'G3': 196.00, 'A3': 220.00,
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'G4': 392.00, 'A4': 440.00,
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'G5': 783.99, 'A5': 880.00,
    'C6': 1046.50, 'D6': 1174.66, 'E6': 1318.51, 'G6': 1567.98, 'A6': 1760.00,
}
PENTATONIC_FREQS = sorted(C_MAJOR_PENTATONIC.values())


class SearchMode(Enum):
    LINEAR = "Linear Search 🐢"
    BINARY = "Binary Search 🐇"


# ==================== Color Palette ====================
class ColorPalette:
    def __init__(self):
        self.randomize()
    
    def randomize(self):
        # Background base hue
        base_hue = random.choice([200, 240, 280, 140, 320])
        self.bg_color = self.hsv_to_rgb(base_hue, 0.3, 0.15)
        self.unexplored_color = self.hsv_to_rgb(base_hue, 0.4, 0.25)
        self.current_color = (255, 255, 255)  # Bright white
        self.target_color = (255, 50, 70)  # Vibrant red
        self.binary_range_color = (0, 255, 255, 80)  # Semi-transparent cyan
    
    @staticmethod
    def hsv_to_rgb(h, s, v):
        """Convert HSV to RGB (h: 0-360, s: 0-1, v: 0-1)"""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    @staticmethod
    def get_rainbow_color(progress):
        """Get rainbow color based on progress (0-1)"""
        # Full rainbow: 0° (red) -> 360° (red again)
        hue = int(progress * 360) % 360
        return ColorPalette.hsv_to_rgb(hue, 0.8, 0.9)


# ==================== Audio Engine ====================
class AudioEngine:
    def __init__(self):
        pygame.mixer.init(SAMPLE_RATE, -16, AUDIO_CHANNELS, BUFFER_SIZE)
        self.base_freq = PENTATONIC_FREQS[0]
    
    def apply_fade(self, samples, fade_ms=5):
        """Apply fade in/out to eliminate clicks"""
        fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
        if len(samples) <= fade_samples * 2:
            return samples
        
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        samples[:fade_samples] *= fade_in
        samples[-fade_samples:] *= fade_out
        return samples
    
    def quantize_to_pentatonic(self, freq):
        """Snap frequency to nearest C Major Pentatonic note"""
        closest = min(PENTATONIC_FREQS, key=lambda x: abs(x - freq))
        return closest
    
    def play_linear_step(self, progress):
        """Marimba-like wood block sound with rising pitch (C3 -> C6)"""
        duration_ms = 60
        samples = int(SAMPLE_RATE * duration_ms / 1000)
        
        # Map progress (0 to 1) to frequency range (C3 to C6)
        min_freq = C_MAJOR_PENTATONIC['C3']
        max_freq = C_MAJOR_PENTATONIC['C6']
        freq = min_freq + (max_freq - min_freq) * progress
        freq = self.quantize_to_pentatonic(freq)
        
        t = np.linspace(0, duration_ms / 1000, samples)
        
        # Marimba: mix of sine and square-ish for woody tone
        wave = 0.6 * np.sin(2 * np.pi * freq * t)
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
        wave += 0.1 * self.triangle_wave(freq * 3, t)   # 3rd harmonic
        
        # Percussive envelope (fast attack, medium decay)
        envelope = np.exp(-10 * t)
        wave *= envelope
        
        # Apply fade
        wave = self.apply_fade(wave, fade_ms=3)
        
        # Normalize and convert
        wave = np.clip(wave * 0.25, -1, 1)
        stereo_wave = np.column_stack([wave, wave])
        audio_data = (stereo_wave * 32767).astype(np.int16)
        
        sound = pygame.sndarray.make_sound(audio_data)
        sound.play()
    
    def play_binary_step(self, step_count):
        """Glass harp / synth bell sound with octave jumps"""
        duration_ms = 200
        samples = int(SAMPLE_RATE * duration_ms / 1000)
        
        # Octave jumps: use different base notes and add octaves
        base_notes = ['C4', 'E4', 'G4', 'C5', 'E5', 'G5', 'C6']
        note = base_notes[step_count % len(base_notes)]
        freq = C_MAJOR_PENTATONIC[note]
        
        t = np.linspace(0, duration_ms / 1000, samples)
        
        # Glass bell: pure sine with rich harmonics
        wave = 0.4 * np.sin(2 * np.pi * freq * t)
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
        wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic
        wave += 0.1 * np.sin(2 * np.pi * freq * 4 * t)  # 4th harmonic
        
        # Bell envelope (slow attack, long decay)
        attack = np.linspace(0, 1, samples // 10)
        sustain = np.ones(samples - len(attack))
        envelope = np.concatenate([attack, sustain])
        envelope *= np.exp(-3 * t)
        wave *= envelope
        
        # Apply fade
        wave = self.apply_fade(wave, fade_ms=5)
        
        # Normalize
        wave = np.clip(wave * 0.2, -1, 1)
        stereo_wave = np.column_stack([wave, wave])
        audio_data = (stereo_wave * 32767).astype(np.int16)
        
        sound = pygame.sndarray.make_sound(audio_data)
        sound.play()
    
    def play_victory_chord(self):
        """Victory chord: C Major with sparkle"""
        duration_ms = 800
        samples = int(SAMPLE_RATE * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, samples)
        
        # C Major chord: C5 + E5 + G5 + C6
        freqs = [
            C_MAJOR_PENTATONIC['C5'],
            C_MAJOR_PENTATONIC['E5'],
            C_MAJOR_PENTATONIC['G5'],
            C_MAJOR_PENTATONIC['C6']
        ]
        
        wave = np.zeros(samples)
        for i, freq in enumerate(freqs):
            # Stagger the start slightly for arpeggio effect
            delay = int(samples * i * 0.02)
            t_delayed = t.copy()
            note = np.sin(2 * np.pi * freq * t_delayed)
            
            # Rich envelope
            envelope = np.exp(-2 * t_delayed)
            note *= envelope
            
            # Add with delay
            if delay < samples:
                wave[delay:] += note[:samples-delay] * (0.25 / len(freqs))
        
        # Add sparkle (high frequency shimmer)
        sparkle_freq = C_MAJOR_PENTATONIC['C6'] * 2
        sparkle = 0.15 * np.sin(2 * np.pi * sparkle_freq * t) * np.exp(-5 * t)
        wave += sparkle
        
        # Apply fade
        wave = self.apply_fade(wave, fade_ms=10)
        
        # Normalize
        wave = np.clip(wave, -1, 1)
        stereo_wave = np.column_stack([wave, wave])
        audio_data = (stereo_wave * 32767).astype(np.int16)
        
        sound = pygame.sndarray.make_sound(audio_data)
        sound.play()
    
    @staticmethod
    def triangle_wave(freq, t):
        """Generate triangle wave"""
        return 2 * np.abs(2 * (freq * t - np.floor(freq * t + 0.5))) - 1


# ==================== Search Visualizer ====================
class SearchVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Linear 🐢 vs Binary 🐇 - Press M to Switch | SPACE for TURBO!")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)
        
        self.palette = ColorPalette()
        self.audio = AudioEngine()
        
        self.mode = SearchMode.LINEAR
        self.paused = False
        
        # Screen shake
        self.shake_amount = 0
        self.shake_duration = 0
        
        # Target pulse animation
        self.pulse_phase = 0
        
        self.reset()
    
    def reset(self):
        """Reset the search with new random target and colors"""
        self.target_index = random.randint(0, GRID_SIZE * GRID_SIZE - 1)
        self.palette.randomize()
        
        # Linear search state
        self.linear_current = 0
        self.linear_found = False
        self.linear_step_counter = 0
        self.rainbow_trail = {}  # index -> color
        
        # Binary search state
        self.binary_low = 0
        self.binary_high = GRID_SIZE * GRID_SIZE - 1
        self.binary_current = -1
        self.binary_found = False
        self.binary_step_counter = 0
        self.binary_range = (0, GRID_SIZE * GRID_SIZE - 1)
        self.binary_frame_counter = 0  # For slowing down binary
        
        # Visual state
        self.discarded = set()
        
        # Reset shake
        self.shake_amount = 0
        self.shake_duration = 0
        
        # Speed control
        if self.mode == SearchMode.LINEAR:
            # Slow linear search: ~3 steps per frame at normal speed
            self.steps_per_frame = 3
        else:
            # Binary: one step every ~60 frames (1 second) to last 8+ seconds
            self.binary_frames_per_step = 60
    
    def toggle_mode(self):
        """Switch between Linear and Binary search"""
        self.mode = SearchMode.BINARY if self.mode == SearchMode.LINEAR else SearchMode.LINEAR
        self.reset()
    
    def coord_to_index(self, x, y):
        """Convert grid coordinates to 1D index"""
        return y * GRID_SIZE + x
    
    def index_to_coord(self, idx):
        """Convert 1D index to grid coordinates"""
        return (idx % GRID_SIZE, idx // GRID_SIZE)
    
    def update_linear(self, turbo_mode=False):
        """Update linear search (turtle)"""
        if self.linear_found:
            return
        
        # Turbo mode: 20x speed
        steps = self.steps_per_frame * (20 if turbo_mode else 1)
        
        for _ in range(steps):
            if self.linear_current >= GRID_SIZE * GRID_SIZE:
                break
            
            if self.linear_current == self.target_index:
                self.linear_found = True
                self.audio.play_victory_chord()
                self.trigger_screen_shake()
                break
            else:
                # Add to rainbow trail
                progress = self.linear_current / (GRID_SIZE * GRID_SIZE)
                self.rainbow_trail[self.linear_current] = self.palette.get_rainbow_color(progress)
                
                self.linear_current += 1
                self.linear_step_counter += 1
                
                # Play sound every few steps (arpeggio rhythm)
                if self.linear_step_counter % 2 == 0 and not turbo_mode:
                    progress = self.linear_current / (GRID_SIZE * GRID_SIZE)
                    self.audio.play_linear_step(progress)
    
    def update_binary(self):
        """Update binary search (rabbit) - slowed to minimum 8 seconds"""
        if self.binary_found:
            return
        
        if self.binary_low > self.binary_high:
            return
        
        # Slow down: only advance every N frames
        self.binary_frame_counter += 1
        if self.binary_frame_counter < self.binary_frames_per_step:
            return
        
        self.binary_frame_counter = 0
        
        # Calculate mid point
        mid = (self.binary_low + self.binary_high) // 2
        self.binary_current = mid
        
        # Update range for visualization
        self.binary_range = (self.binary_low, self.binary_high)
        
        # Play sound
        self.audio.play_binary_step(self.binary_step_counter)
        self.binary_step_counter += 1
        
        # Check if found
        if mid == self.target_index:
            self.binary_found = True
            self.audio.play_victory_chord()
            self.trigger_screen_shake()
        elif mid < self.target_index:
            # Discard lower half
            for i in range(self.binary_low, mid + 1):
                self.discarded.add(i)
            self.binary_low = mid + 1
        else:
            # Discard upper half
            for i in range(mid, self.binary_high + 1):
                self.discarded.add(i)
            self.binary_high = mid - 1
    
    def trigger_screen_shake(self):
        """Start screen shake effect"""
        self.shake_amount = 10
        self.shake_duration = 30  # frames
    
    def update_screen_shake(self):
        """Update screen shake effect"""
        if self.shake_duration > 0:
            self.shake_duration -= 1
            if self.shake_duration == 0:
                self.shake_amount = 0
    
    def get_shake_offset(self):
        """Get current shake offset"""
        if self.shake_amount > 0:
            return (
                random.randint(-self.shake_amount, self.shake_amount),
                random.randint(-self.shake_amount, self.shake_amount)
            )
        return (0, 0)
    
    def draw(self):
        """Render the visualization"""
        # Get shake offset
        shake_x, shake_y = self.get_shake_offset()
        
        # Clear screen
        self.screen.fill(self.palette.bg_color)
        
        # Create a surface for the grid (to apply shake)
        grid_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        grid_surface.fill(self.palette.bg_color)
        
        # Draw grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx = self.coord_to_index(x, y)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Determine color
                if idx == self.target_index:
                    # Pulsing red target
                    self.pulse_phase += 0.1
                    pulse = 0.7 + 0.3 * math.sin(self.pulse_phase)
                    color = (
                        int(self.palette.target_color[0] * pulse),
                        int(self.palette.target_color[1] * pulse),
                        int(self.palette.target_color[2] * pulse)
                    )
                elif self.mode == SearchMode.LINEAR and idx in self.rainbow_trail:
                    # Rainbow trail
                    color = self.rainbow_trail[idx]
                elif idx in self.discarded:
                    # Discarded (for binary)
                    color = self.palette.hsv_to_rgb(0, 0, 0.15)
                else:
                    color = self.palette.unexplored_color
                
                pygame.draw.rect(grid_surface, color, rect)
                
                # Highlight current position
                if self.mode == SearchMode.LINEAR and idx == self.linear_current:
                    pygame.draw.rect(grid_surface, self.palette.current_color, rect)
                elif self.mode == SearchMode.BINARY and idx == self.binary_current:
                    pygame.draw.rect(grid_surface, self.palette.current_color, rect)
        
        # Draw binary range highlight (semi-transparent cyan)
        if self.mode == SearchMode.BINARY and not self.binary_found:
            range_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            for idx in range(self.binary_range[0], self.binary_range[1] + 1):
                x, y = self.index_to_coord(idx)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(range_surface, self.palette.binary_range_color, rect)
            grid_surface.blit(range_surface, (0, 0))
        
        # Blit grid with shake
        self.screen.blit(grid_surface, (shake_x, shake_y))
        
        # Draw UI (not affected by shake)
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_ui(self):
        """Draw UI elements - ALL TEXT REMOVED FOR CLEAN VISUALS"""
        pass  # No UI elements displayed
    
    def get_rainbow_color_animated(self):
        """Get animated rainbow color for victory text"""
        hue = (pygame.time.get_ticks() // 10) % 360
        return self.palette.hsv_to_rgb(hue, 0.9, 1.0)
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.toggle_mode()
                    elif event.key == pygame.K_r:
                        self.reset()
            
            # Check for turbo mode (SPACE held down)
            keys = pygame.key.get_pressed()
            turbo_mode = keys[pygame.K_SPACE]
            
            # Update
            if not self.paused:
                if self.mode == SearchMode.LINEAR:
                    self.update_linear(turbo_mode)
                else:
                    self.update_binary()
                
                self.update_screen_shake()
            
            # Draw
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    visualizer = SearchVisualizer()
    visualizer.run()
