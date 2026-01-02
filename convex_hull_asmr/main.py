import pygame
import numpy as np
import math
import random
import time

# --- Constants & Configuration ---
WINDOW_SIZE = 800
V_SIZE = 160 # Improved pixel density for ripples
SCALE = WINDOW_SIZE // V_SIZE
FPS = 60

# Palettes (BG, Primary, Secondary, Highlight)
PALETTES = [
    # Deep Ocean (Singing Bowl Theme)
    ((5, 15, 30), (0, 255, 230), (50, 100, 150), (200, 255, 255)),
    # Zen Garden
    ((20, 15, 10), (150, 255, 100), (100, 100, 80), (240, 230, 200)),
    # Twilight
    ((15, 10, 25), (255, 100, 150), (80, 50, 100), (255, 200, 220)),
    # Midnight Cyber
    ((5, 5, 10), (0, 255, 255), (0, 100, 100), (255, 255, 255)),
]

# Audio Settings
SAMPLE_RATE = 44100
MAX_CHANNELS = 32

# --- Audio Engine (Singing Bowl Synthesis) ---
class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(MAX_CHANNELS)
        self.channels = [pygame.mixer.Channel(i) for i in range(MAX_CHANNELS)]
        
        # C Major Pentatonic (Harmonious, Healing)
        # C3, D3, E3, G3, A3 ...
        self.scale = [
             130.81, 146.83, 164.81, 196.00, 220.00, # Octave 3
             261.63, 293.66, 329.63, 392.00, 440.00, # Octave 4
             523.25, 587.33, 659.25, 783.99, 880.00  # Octave 5
        ]
        self.last_sound_time = 0
        self.cooldown = 0.08 # 80ms min between sounds

    def _get_channel(self):
        for c in self.channels:
            if not c.get_busy():
                return c
        return self.channels[0]

    def can_play(self):
        now = time.time()
        if now - self.last_sound_time > self.cooldown:
            self.last_sound_time = now
            return True
        return False

    def generate_singing_bowl(self, freq, duration, vol=0.5):
        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # 1/f noise-like fluctuation in freq
        fluct = np.sin(2 * np.pi * 6.0 * t) * 0.002 # Slow vibrato
        
        # Singing Bowl / Glass Harp Synthesis
        # Fundamental + Harmonics (slightly detuned for realism)
        w1 = 0.50 * np.sin(2 * np.pi * freq * (1 + fluct) * t)
        w2 = 0.25 * np.sin(2 * np.pi * (freq * 2.02) * t) # Slightly detuned 2nd harmonic
        w3 = 0.15 * np.sin(2 * np.pi * (freq * 3.01) * t)
        
        wave = w1 + w2 + w3
        
        # Soft Envelope (Bell-like)
        attack = int(0.05 * SAMPLE_RATE) # 50ms attack
        decay = n_samples - attack
        
        env = np.ones_like(t)
        env[:attack] = np.linspace(0, 1, attack)
        env[attack:] = np.exp(-np.linspace(0, 4, decay)) # Exp decay
        
        final_wave = wave * env * vol
        
        # Stereo Width (Phase offset for right channel)
        right_channel = np.roll(final_wave, int(SAMPLE_RATE * 0.002)) # 2ms delay
        stereo_wave = np.column_stack((final_wave, right_channel))
        
        return pygame.sndarray.make_sound((stereo_wave * 32767).astype(np.int16))

    def generate_woodblock(self, vol=0.3):
        # Short, woody sound for Pop
        duration = 0.15
        freq = 300
        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Pitch drop
        inst_freq = freq * np.exp(-t * 10) 
        phase = np.cumsum(inst_freq) * 2 * np.pi / SAMPLE_RATE
        
        wave = np.sin(phase) * 0.5
        wave += np.sin(phase * 2.4) * 0.2 # Inharmonic
        
        env = np.exp(-t * 25)
        final_wave = wave * env * vol
        stereo_wave = np.column_stack((final_wave, final_wave))
        return pygame.sndarray.make_sound((stereo_wave * 32767).astype(np.int16))

    def generate_wind(self, duration=0.1, vol=0.0):
        # Filtered noise for scanning
        # Just pure white noise with low volume for "texture"
        # Only if user wants wind. User said "Silent or extremely small".
        # Let's do silence for scanning to be safe from noise.
        # But if we want "Wind", we can use random noise.
        pass

    def play_scan(self):
        # Validated: Scan should be SILENT or barely audible. 
        # Making it silent to reduce noise as requested.
        pass

    def play_lock(self, y_pos):
        if not self.can_play(): return
        
        norm = 1.0 - (y_pos / V_SIZE)
        idx = int(norm * (len(self.scale) - 1))
        idx = max(0, min(idx, len(self.scale) - 1))
        freq = self.scale[idx]
        
        sound = self.generate_singing_bowl(freq, 2.5, vol=0.4)
        c = self._get_channel()
        c.set_volume(1.0)
        c.play(sound)

    def play_snap(self):
        # Woodblock for "Pop"
        # Allow pops to happen slightly faster?
        sound = self.generate_woodblock(vol=0.3)
        self._get_channel().play(sound)
    
    def play_add(self):
        # "Ting" for adding point in Graham
        if not self.can_play(): return
        freq = random.choice(self.scale[10:]) # High pitch
        sound = self.generate_singing_bowl(freq, 0.5, vol=0.2)
        self._get_channel().play(sound)

    def play_completion(self):
        # Chord
        # C Major Chord spread
        indices = [0, 4, 7, 12, 14]
        for i in indices:
            if i < len(self.scale):
                sound = self.generate_singing_bowl(self.scale[i], 4.0, vol=0.2)
                self.channels[i % MAX_CHANNELS].play(sound)


# --- Visual Effects ---
# Ripple removed as requested

# --- Geometry ---
def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

class ConvexHullViz:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.canvas = pygame.Surface((V_SIZE, V_SIZE))
        pygame.display.set_caption("Convex Hull ASMR")
        self.clock = pygame.time.Clock()
        self.audio = AudioEngine()
        self.font = pygame.font.SysFont("courier", 12)
        self.ui_font = pygame.font.SysFont("courier", 20) 
        
        self.mode = "JARVIS" 
        self.show_ui = True
        self.reset()

    def reset(self):
        self.points = []
        self.pal = random.choice(PALETTES)
        self.bg_color, self.prim_color, self.sec_color, self.hi_color = self.pal
        
        padding = 10
        count = random.randint(40, 60)
        for _ in range(count):
            x = random.randint(padding, V_SIZE - padding)
            y = random.randint(padding, V_SIZE - padding)
            self.points.append((x, y))
            
        self.completed = False
        self.paused = False
        self.hull = []
        self.scan_line = None
        self.focus_point = None
        
        if self.mode == "JARVIS":
            self.algo_generator = self.jarvis_march_gen()
        else:
            self.algo_generator = self.graham_scan_gen()

    def jarvis_march_gen(self):
        start_point = min(self.points, key=lambda p: p[0])
        self.hull = [start_point]
        self.audio.play_lock(start_point[1])
        yield 

        current_point = start_point
        while True:
            candidates = [p for p in self.points if p != current_point]
            candidates.sort(key=lambda p: math.atan2(p[1] - current_point[1], p[0] - current_point[0]))

            best_candidate = candidates[0]
            
            for i, candidate in enumerate(candidates):
                self.focus_point = candidate
                self.scan_line = (current_point, candidate)
                # NO SOUND on scan for Jarvis
                
                val = cross_product(current_point, best_candidate, candidate)
                if val > 0:
                    best_candidate = candidate
                elif val == 0:
                    if dist_sq(current_point, candidate) > dist_sq(current_point, best_candidate):
                        best_candidate = candidate
                
                yield 

            next_point = best_candidate
            self.scan_line = (current_point, next_point)
            
            # Lock Event
            self.audio.play_lock(next_point[1])
            
            # Pause slightly on lock
            for _ in range(5): yield

            if next_point == start_point:
                break
                
            self.hull.append(next_point)
            current_point = next_point
            yield

        self.hull.append(start_point)
        self.completed = True
        self.audio.play_completion()


    def graham_scan_gen(self):
        start_point = max(self.points, key=lambda p: (p[1], p[0]))
        self.hull = [start_point]
        self.audio.play_lock(start_point[1])
        yield

        def polar_angle(p):
            return math.atan2(p[1] - start_point[1], p[0] - start_point[0])
        
        sorted_points = sorted([p for p in self.points if p != start_point], key=polar_angle, reverse=True)
        
        self.hull.append(sorted_points[0])
        self.audio.play_add()
        yield

        for point in sorted_points[1:]:
            self.focus_point = point
            self.scan_line = (self.hull[-1], point)
            
            for _ in range(5): yield 

            while len(self.hull) > 1:
                last = self.hull[-1]
                second_last = self.hull[-2]
                self.scan_line = (second_last, point) 
                
                val = cross_product(second_last, last, point)
                if val <= 0: # Left Turn (Good)
                    break 
                else: 
                    # Right Turn (Bad - Concave)
                    popped = self.hull.pop()
                    self.audio.play_snap() 
                    for _ in range(10): yield 
            
            self.hull.append(point)
            self.audio.play_add()
            for _ in range(10): yield 

        self.hull.append(start_point)
        self.completed = True
        self.scan_line = None
        self.audio.play_completion()


    def update(self):
        if not self.paused and not self.completed and self.algo_generator:
            try:
                next(self.algo_generator)
            except StopIteration:
                self.completed = True

    def draw(self):
        self.canvas.fill(self.bg_color)
        
        # Draw Points
        for p in self.points:
            color = self.sec_color
            if p in self.hull: color = self.prim_color
            if p == self.focus_point: color = self.hi_color
            
            self.canvas.set_at((int(p[0]), int(p[1])), color)

        # Draw Hull
        if len(self.hull) > 1:
            pygame.draw.lines(self.canvas, self.prim_color, False, self.hull, 1)

        # Scan Line
        if self.scan_line:
            pygame.draw.line(self.canvas, self.hi_color, self.scan_line[0], self.scan_line[1], 1)

        # Completion Flash/Fill
        if self.completed and len(self.hull) > 2:
             if int(time.time() * 5) % 2 == 0: 
                 pygame.draw.polygon(self.canvas, self.sec_color, self.hull, 0)
             pygame.draw.lines(self.canvas, self.hi_color, True, self.hull, 1)

        # Upscale
        scaled_surf = pygame.transform.scale(self.canvas, (WINDOW_SIZE, WINDOW_SIZE))
        self.screen.blit(scaled_surf, (0, 0))

        # UI
        if self.show_ui:
            ui_texts = [
                f"Mode: {self.mode}", 
                f"Points: {len(self.points)}",
                "R: Reset | M: Switch",
                "U: Hide UI"
            ]
            for i, t in enumerate(ui_texts):
                label = self.ui_font.render(t, True, self.hi_color)
                shadow = self.ui_font.render(t, True, (0,0,0))
                self.screen.blit(shadow, (12, 12 + i * 25))
                self.screen.blit(label, (10, 10 + i * 25))

        pygame.display.flip()

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    elif event.key == pygame.K_r: self.reset()
                    elif event.key == pygame.K_m: 
                        self.mode = "GRAHAM" if self.mode == "JARVIS" else "JARVIS"
                        self.reset()
                    elif event.key == pygame.K_u: self.show_ui = not self.show_ui

            self.update()
            self.draw()

if __name__ == "__main__":
    app = ConvexHullViz()
    app.run()
