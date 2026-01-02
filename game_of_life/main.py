import pygame
import numpy as np
import random

WINDOW_SIZE = 800
GRID_SIZE = 80
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
BG_COLOR = (20, 20, 20)
SAMPLE_RATE = 44100

C_MAJOR_PENTATONIC = [261.63, 293.66, 329.63, 392.00, 440.00,
                      523.25, 587.33, 659.25, 783.99, 880.00]


class SoundEngine:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
        self.sounds = {}
        self._generate_sounds()

    def _generate_sounds(self):
        duration = 0.8
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
        decay = np.exp(-t * 5)
        
        for freq in C_MAJOR_PENTATONIC:
            wave = np.sin(2 * np.pi * freq * t) * 0.6
            wave += np.sin(2 * np.pi * freq * 2 * t) * 0.2
            wave += np.sin(2 * np.pi * freq * 3 * t) * 0.1
            wave += np.sin(2 * np.pi * freq * 4 * t) * 0.05
            wave *= decay
            wave = (wave * 32767 * 0.3).astype(np.int16)
            stereo = np.column_stack((wave, wave))
            sound = pygame.sndarray.make_sound(stereo)
            self.sounds[freq] = sound

    def play_note(self, y_pos):
        idx = int((y_pos / GRID_SIZE) * len(C_MAJOR_PENTATONIC))
        idx = max(0, min(idx, len(C_MAJOR_PENTATONIC) - 1))
        idx = len(C_MAJOR_PENTATONIC) - 1 - idx
        freq = C_MAJOR_PENTATONIC[idx]
        self.sounds[freq].play()


class GameOfLife:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.ages = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.fade = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.randomize()

    def randomize(self, density=0.2):
        self.grid = (np.random.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.int8)
        self.ages = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.fade = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def clear(self):
        self.grid.fill(0)
        self.ages.fill(0)

    def set_cell(self, x, y, alive=True):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            if alive and self.grid[y, x] == 0:
                self.grid[y, x] = 1
                self.ages[y, x] = 0

    def step(self):
        neighbors = sum(
            np.roll(np.roll(self.grid, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0)
        )
        
        new_grid = ((self.grid == 1) & ((neighbors == 2) | (neighbors == 3))) | \
                   ((self.grid == 0) & (neighbors == 3))
        new_grid = new_grid.astype(np.int8)
        
        births = (new_grid == 1) & (self.grid == 0)
        birth_positions = list(zip(*np.where(births)))
        
        self.fade = np.where(
            (self.grid == 1) & (new_grid == 0),
            1.0,
            self.fade
        )
        
        self.ages = np.where(new_grid == 1, self.ages + 1, 0)
        self.ages = np.where(births, 0, self.ages)
        
        self.grid = new_grid
        
        return birth_positions


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Conway's Game of Life")
        self.clock = pygame.time.Clock()
        
        self.game = GameOfLife()
        self.sound = SoundEngine()
        
        self.paused = False
        self.running = True
        self.mouse_down = False
        
        self.screen.fill(BG_COLOR)
        self.prev_grid = np.zeros_like(self.game.grid)
        self.prev_ages = np.zeros_like(self.game.ages)
        self.force_redraw = True

    def _get_cell_color(self, age):
        if age == 0:
            return (255, 255, 220)
        elif age <= 10:
            t = age / 10.0
            r = int(255 * (1 - t) + 0 * t)
            g = int(255 * (1 - t) + 230 * t)
            b = int(100 * (1 - t) + 180 * t)
            return (r, g, b)
        else:
            t = min((age - 10) / 20.0, 1.0)
            r = int(0 * (1 - t) + 100 * t)
            g = int(230 * (1 - t) + 80 * t)
            b = int(180 * (1 - t) + 200 * t)
            return (r, g, b)

    def _draw_gem_cell(self, x, y, color):
        px = x * CELL_SIZE
        py = y * CELL_SIZE
        margin = 1
        
        darker = tuple(max(0, c - 60) for c in color)
        pygame.draw.rect(self.screen, darker, 
                        (px + margin, py + margin, CELL_SIZE - margin * 2, CELL_SIZE - margin * 2),
                        border_radius=2)
        
        inner_margin = 2
        pygame.draw.rect(self.screen, color,
                        (px + inner_margin, py + inner_margin, 
                         CELL_SIZE - inner_margin * 2 - 1, CELL_SIZE - inner_margin * 2 - 1),
                        border_radius=2)
        
        highlight = tuple(min(255, c + 80) for c in color)
        highlight_size = max(2, CELL_SIZE // 4)
        pygame.draw.rect(self.screen, highlight,
                        (px + inner_margin + 1, py + inner_margin + 1, highlight_size, highlight_size),
                        border_radius=1)

    def _draw_fade_cell(self, x, y, fade_value):
        if fade_value <= 0:
            return
        
        px = x * CELL_SIZE
        py = y * CELL_SIZE
        
        intensity = int(fade_value * 80)
        color = (intensity, intensity, intensity)
        
        margin = 1
        pygame.draw.rect(self.screen, color,
                        (px + margin, py + margin, CELL_SIZE - margin * 2, CELL_SIZE - margin * 2),
                        border_radius=2)

    def _clear_cell(self, x, y):
        px = x * CELL_SIZE
        py = y * CELL_SIZE
        pygame.draw.rect(self.screen, BG_COLOR, (px, py, CELL_SIZE, CELL_SIZE))

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.game.randomize()
                    self.force_redraw = True
                elif event.key == pygame.K_c:
                    self.game.clear()
                    self.force_redraw = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
        
        if self.mouse_down:
            mx, my = pygame.mouse.get_pos()
            cx = mx // CELL_SIZE
            cy = my // CELL_SIZE
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    self.game.set_cell(cx + dx, cy + dy, True)
            self.force_redraw = True

    def _update(self):
        if not self.paused:
            births = self.game.step()
            
            if births:
                chosen = random.choice(births)
                self.sound.play_note(chosen[0])
            
            self.game.fade = np.maximum(0, self.game.fade - 0.15)

    def _render(self):
        if self.force_redraw:
            self.screen.fill(BG_COLOR)
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if self.game.grid[y, x] == 1:
                        color = self._get_cell_color(self.game.ages[y, x])
                        self._draw_gem_cell(x, y, color)
            self.prev_grid = self.game.grid.copy()
            self.prev_ages = self.game.ages.copy()
            self.force_redraw = False
        else:
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    curr_alive = self.game.grid[y, x]
                    prev_alive = self.prev_grid[y, x]
                    curr_age = self.game.ages[y, x]
                    prev_age = self.prev_ages[y, x]
                    fade_val = self.game.fade[y, x]
                    
                    if curr_alive != prev_alive or curr_age != prev_age or fade_val > 0:
                        self._clear_cell(x, y)
                        
                        if fade_val > 0 and curr_alive == 0:
                            self._draw_fade_cell(x, y, fade_val)
                        
                        if curr_alive == 1:
                            color = self._get_cell_color(curr_age)
                            self._draw_gem_cell(x, y, color)
            
            self.prev_grid = self.game.grid.copy()
            self.prev_ages = self.game.ages.copy()
        
        pygame.display.flip()

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(15)
        
        pygame.quit()


if __name__ == "__main__":
    app = App()
    app.run()
