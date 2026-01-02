"""
ASMR Langton's Ant: Healing Frequency Edition
=============================================
A generative sound therapy tool.
Visualizes order emerging from chaos with harmonic singing bowl sounds.

Controls:
- U: Toggle UI (Hide text for recording)
- M: Toggle Turbo Mode (Fast Forward + Mute)
- SPACE: Pause / Resume
- S: Toggle Screen Size (Shorts / Wide)
- R: Reset
- ESC: Quit
"""

import pygame
import numpy as np
from collections import defaultdict, deque
import random
import math

# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Display
    SHORTS_MODE = (540, 960)   # 9:16
    ASMR_MODE = (960, 540)     # 16:9
    FPS = 60
    
    # Simulation
    CELL_SIZE = 8
    NORMAL_STEPS_PER_FRAME = 2 # Slow enough to enjoy the audio
    TURBO_STEPS_PER_FRAME = 200
    WALL_DISTANCE = 120
    
    # Audio
    SAMPLE_RATE = 44100
    
    # Colors (Deep Relaxation Palette)
    BG_COLOR = (10, 10, 25)    # Deep Indigo
    TILE_COLOR = (240, 230, 140)# Pale Gold (Khaki)
    ANT_COLOR = (224, 255, 255)# Light Cyan
    
    # Scale: C Major Pentatonic (Low/Mid)
    # C3, D3, E3, G3, A3, C4
    SCALE = [130.81, 146.83, 164.81, 196.00, 220.00, 261.63]

# =============================================================================
# Healing Audio Engine
# =============================================================================
class HealingAudioEngine:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.muted = False
        
        # Pre-generate singing bowl sounds for the scale
        self.buffers = {}
        for i, freq in enumerate(Config.SCALE):
            self.buffers[i] = self._create_singing_bowl(freq)
            
        # Heartbeat sound for Highway
        self.heartbeat = self._create_heartbeat()
        
        # Timing State (1/f Fluctuation)
        self.timer = 0
        self.next_interval = 0.5
        self.highway_timer = 0
        self.beat_interval = 0.6 # ~100 BPM
        
    def _to_sound(self, wave):
        wave = np.clip(wave, -1.0, 1.0)
        # Apply global fade to ensure start/end is zero
        wave[0:100] *= np.linspace(0, 1, 100)
        wave[-100:] *= np.linspace(1, 0, 100)
        
        stereo = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

    def _create_singing_bowl(self, freq):
        """
        Synthesize a Singing Bowl tone using additive harmonics.
        Deep, resonant, lingering.
        """
        duration = 2.5 # Long decay
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Fundamental
        wave = np.sin(2 * np.pi * freq * t) * 0.5
        
        # Harmonics (Inharmonic for metallic bowl feel, or just harmonic for pleasantness)
        # Let's stick to strict harmonics for "Healing" (Pythagorean)
        wave += np.sin(2 * np.pi * (freq * 2.0) * t) * 0.25 # 2nd Harmonic
        wave += np.sin(2 * np.pi * (freq * 3.0) * t) * 0.12 # 3rd Harmonic
        wave += np.sin(2 * np.pi * (freq * 4.0) * t) * 0.05 # 4th Harmonic
        
        # Slight Detune/Beating (Alpha wave simulation?)
        # Add a very close frequency to create a slow beat (e.g. 4Hz diff)
        wave += np.sin(2 * np.pi * (freq + 4) * t) * 0.1
        
        # Envelope (Soft Bell)
        # Slow attack (no click)
        attack_time = 0.1
        decay_const = 2.0 # Exponential decay
        
        att_samples = int(attack_time * self.sample_rate)
        env = np.exp(-t * 2.0)
        env[:att_samples] = np.linspace(0, 1, att_samples) * env[:att_samples]
        
        wave *= env
        return self._to_sound(wave * 0.5)

    def _create_heartbeat(self):
        """Two low thuds"""
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Kick 1
        k1 = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
        # Kick 2 (delayed)
        shift = int(0.15 * self.sample_rate)
        k2 = np.zeros_like(t)
        if len(t) > shift:
            k2[shift:] = np.sin(2 * np.pi * 50 * t[:-shift]) * np.exp(-t[:-shift] * 20)
            
        wave = (k1 + k2 * 0.6)
        
        # Low pass filter approximation (smooth out click)
        # Just use envelope
        wave[:100] *= np.linspace(0, 1, 100)
        
        return self._to_sound(wave * 0.6)

    def update(self, dt, is_highway, ant_y):
        if self.muted:
            return
            
        if is_highway:
            # Heartbeat Mode (Rhythmic, Grounding)
            self.highway_timer += dt
            if self.highway_timer >= self.beat_interval:
                self.highway_timer -= self.beat_interval
                self.heartbeat.play()
                
                # Slowly drift tempo for "Organic" feel (1/f lite)
                swing = random.uniform(-0.02, 0.02)
                self.beat_interval = max(0.5, min(0.7, 0.6 + swing))
        else:
            # Chaos Mode (Wind Chimes)
            self.timer += dt
            if self.timer >= self.next_interval:
                self.timer = 0
                
                # Determine next interval (1/f Fluctuation logic)
                # Instead of pure random, we drift from current
                # Or just standard Pink Noise approximation? 
                # Simple: Random between 0.3 and 1.2
                self.next_interval = random.uniform(0.3, 1.2)
                
                # Play a note based on Ant Y (Pitch)
                # Map Y to scale index
                # Screen height ~ ±270 (shorts) or ±500 scale
                # Let's say range ±100 maps to scale indices
                idx = int((ant_y + 120) / 40)
                idx = max(0, min(len(Config.SCALE) - 1, idx))
                
                # Variation: sometimes play 5th (idx + 4) or 3rd
                note = self.buffers[idx]
                
                # Random Pan later? Mixer is stereo already.
                note.set_volume(random.uniform(0.3, 0.6))
                note.play()

    def set_turbo(self, active):
        self.muted = active
        if active:
             pygame.mixer.stop()

# =============================================================================
# Logic: Classic Ant
# =============================================================================
class Ant:
    DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # N, E, S, W
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = 0
        self.y = 0
        self.dir = 0
        self.grid = defaultdict(bool)
        self.steps = 0
        self.recent = deque(maxlen=200)
        
    def step(self):
        # Classic Langton's Ant
        is_black = self.grid[(self.x, self.y)]
        
        if is_black:
            self.dir = (self.dir - 1) % 4
        else:
            self.dir = (self.dir + 1) % 4
            
        self.grid[(self.x, self.y)] = not is_black
        
        dx, dy = self.DIRS[self.dir]
        self.x += dx
        self.y += dy
        self.steps += 1
        
        # Wall Collision
        if abs(self.x) > Config.WALL_DISTANCE or abs(self.y) > Config.WALL_DISTANCE:
            self.dir = (self.dir + 2) % 4
            dx, dy = self.DIRS[self.dir]
            self.x += dx
            self.y += dy
            
    def check_highway(self):
        # Simple heuristic: Steps > 10500 usually implies highway in this wall-setup
        # Or check displacement
        self.recent.append((self.x, self.y))
        if len(self.recent) < 200: return False
        
        # If displacement is high consistently?
        # Actually, simpler check: Langton's ant highway starts around step 10212.
        # But walls break it. So it only lasts until wall.
        # So we can just check if we satisfy a displacement condition
        
        start = self.recent[0]
        end = self.recent[-1]
        dist_sq = (end[0]-start[0])**2 + (end[1]-start[1])**2
        return dist_sq > 400 # Moving away fast

# =============================================================================
# Camera
# =============================================================================
class Camera:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.x = 0
        self.y = 0
        
    def update(self, tx, ty):
        # Very slow, smooth drift (Healing vibe)
        self.x += (tx - self.x) * 0.05
        self.y += (ty - self.y) * 0.05
        
    def to_screen(self, wx, wy):
        sx = (wx - self.x) * Config.CELL_SIZE + self.width/2
        sy = (wy - self.y) * Config.CELL_SIZE + self.height/2
        return int(sx), int(sy)
        
    def resize(self, w, h):
        self.width = w
        self.height = h

# =============================================================================
# Renderer
# =============================================================================
class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 24)
        self.show_ui = True
        
    def render(self, ant, camera, turbo, is_highway):
        self.screen.fill(Config.BG_COLOR)
        w, h = self.screen.get_size()
        
        # Draw visible cells
        # Just iterate, optimization usually not needed for simple ant < 20k steps
        # But with wall bounce, it loops forever. Cell count increases.
        # Let's strictly cull.
        
        cols = int(w / Config.CELL_SIZE) + 4
        rows = int(h / Config.CELL_SIZE) + 4
        
        # Iterating only existing grid is usually faster than XY loop if grid is sparse.
        # If grid is dense (chaos), XY loop is safer but Python is slow.
        # Stick to grid items for now.
        
        for (gx, gy), is_black in ant.grid.items():
            if not is_black: continue
            
            sx, sy = camera.to_screen(gx, gy)
            if -Config.CELL_SIZE < sx < w and -Config.CELL_SIZE < sy < h:
                # Soft Glow Tile
                # Main rect
                rect = (sx+1, sy+1, Config.CELL_SIZE-2, Config.CELL_SIZE-2)
                pygame.draw.rect(self.screen, Config.TILE_COLOR, rect)
                
                # Glow (Simulated by drawing larger alpha rect? Pygame Surface slow)
                # Just stick to clean aesthetics for performance.
                
        # Draw Ant (Glowing Orb)
        ax, ay = camera.to_screen(ant.x, ant.y)
        cx, cy = ax + Config.CELL_SIZE//2, ay + Config.CELL_SIZE//2
        
        # Outer glow
        pygame.draw.circle(self.screen, (50, 50, 100), (cx, cy), Config.CELL_SIZE)
        pygame.draw.circle(self.screen, Config.ANT_COLOR, (cx, cy), Config.CELL_SIZE//2 + 1)
        
        if self.show_ui:
            self._draw_ui(ant, turbo, is_highway)
            
    def _draw_ui(self, ant, turbo, is_highway):
        status = "Chaos (Wind Chimes)"
        if is_highway: status = "Order (Heartbeat)"
        if turbo: status = "TURBO (Muted)"
        
        txt = f"Steps: {ant.steps:,} | Mode: {status}"
        surf = self.font.render(txt, True, (150, 150, 180))
        self.screen.blit(surf, (20, 20))
        
        ctrl = "M:Turbo  U:HideUI  Space:Pause  R:Reset"
        surf2 = self.font.render(ctrl, True, (100, 100, 120))
        self.screen.blit(surf2, (20, self.screen.get_height() - 40))

# =============================================================================
# App
# =============================================================================
class App:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=Config.SAMPLE_RATE, size=-16, channels=2, buffer=512)
        
        self.screen_size = Config.ASMR_MODE
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("ASMR Langton's Ant")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.turbo = False
        
        self.ant = Ant()
        self.audio = HealingAudioEngine()
        self.camera = Camera(*self.screen_size)
        self.renderer = Renderer(self.screen)
        
    def handle_input(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_m:
                    self.turbo = not self.turbo
                    self.audio.set_turbo(self.turbo)
                elif e.key == pygame.K_u:
                    self.renderer.show_ui = not self.renderer.show_ui
                elif e.key == pygame.K_r:
                    self.ant.reset()
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_s:
                    if self.screen_size == Config.SHORTS_MODE:
                        self.screen_size = Config.ASMR_MODE
                    else:
                        self.screen_size = Config.SHORTS_MODE
                    self.screen = pygame.display.set_mode(self.screen_size)
                    self.camera.resize(*self.screen_size)
                elif e.key == pygame.K_ESCAPE:
                    self.running = False

    def run(self):
        while self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0
            self.handle_input()
            
            if not self.paused:
                steps = Config.TURBO_STEPS_PER_FRAME if self.turbo else Config.NORMAL_STEPS_PER_FRAME
                is_highway = self.ant.check_highway()
                
                for _ in range(steps):
                    self.ant.step()
                    
                # Audio
                self.audio.update(dt, is_highway, self.ant.y)
                
                self.camera.update(self.ant.x, self.ant.y)
            
            is_highway = self.ant.check_highway() # Check again for UI
            self.renderer.render(self.ant, self.camera, self.turbo, is_highway)
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    App().run()
