import random
from collections import deque

import numpy as np
import pygame


class Maze:
    def __init__(self, size=40):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.generate()

    def generate(self):
        self.tile_count = self.size * 2 + 1
        self.grid = np.ones((self.tile_count, self.tile_count), dtype=np.uint8)
        for row in range(self.size):
            for col in range(self.size):
                self.grid[2 * row + 1][2 * col + 1] = 0

        self.adjacency = {(row, col): set() for row in range(self.size) for col in range(self.size)}
        visited = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            current = stack[-1]
            row, col = current
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                    neighbors.append((nr, nc))

            if neighbors:
                next_cell = random.choice(neighbors)
                self._link_cells(current, next_cell)
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

    def _link_cells(self, src, dst):
        self.adjacency[src].add(dst)
        self.adjacency[dst].add(src)
        r, c = src
        nr, nc = dst
        wall_r = 2 * r + 1 + (nr - r)
        wall_c = 2 * c + 1 + (nc - c)
        self.grid[wall_r][wall_c] = 0

    def get_neighbors(self, cell):
        return self.adjacency.get(cell, [])


class Solver:
    def __init__(self, maze, mode="BFS"):
        self.maze = maze
        self.mode = mode
        self.reset()

    def reset(self):
        self.frontier = deque([self.maze.start])
        self.visited = {self.maze.start}
        self.came_from = {}
        self.current = self.maze.start
        self.goal_reached = False
        self.path = []

    def set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        self.reset()

    def step(self, steps=1):
        for _ in range(steps):
            if self.goal_reached or not self.frontier:
                return

            if self.mode == "BFS":
                cell = self.frontier.popleft()
            else:
                cell = self.frontier.pop()

            self.current = cell
            if cell == self.maze.goal:
                self.goal_reached = True
                self.path = self._reconstruct_path(cell)
                return

            neighbors = list(self.maze.get_neighbors(cell))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor in self.visited:
                    continue
                self.visited.add(neighbor)
                self.came_from[neighbor] = cell
                self.frontier.append(neighbor)

    def _reconstruct_path(self, end):
        path = []
        current = end
        while current and current != self.maze.start:
            path.append(current)
            current = self.came_from.get(current)
        if current:
            path.append(current)
        return list(reversed(path))

    def explored_ratio(self):
        total = max(1, self.maze.size * self.maze.size)
        return min(1.0, len(self.visited) / total)


class SoundEngine:
    def __init__(self, min_freq=220, max_freq=800):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.rate = 44100
        self.duration = 0.12
        self.chunk = max(256, int(self.rate * self.duration))
        self.channel = None
        self.last_freq = None
        self.enabled = False
        self._init_mixer()

    def _init_mixer(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(self.rate, -16, 1, 512)
            self.channel = pygame.mixer.Channel(0)
            self.enabled = True
        except pygame.error:
            self.enabled = False

    def update(self, progress_ratio):
        if not self.enabled:
            return
        ratio = max(0.0, min(1.0, progress_ratio))
        freq = int(self.min_freq + (self.max_freq - self.min_freq) * ratio)
        if freq == self.last_freq and self.channel.get_busy():
            return

        wave = self._generate_wave(freq)
        try:
            sound = pygame.sndarray.make_sound(wave)
            self.channel.play(sound)
            self.last_freq = freq
        except pygame.error:
            self.enabled = False

    def _generate_wave(self, freq):
        t = np.linspace(0, self.duration, self.chunk, False)
        waveform = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
        return np.int16(waveform * 32767)


class App:
    BACKGROUND = (20, 20, 20)
    WALL = (70, 70, 70)
    UNEXPLORED = (30, 30, 30)
    VISITED = (0, 180, 255)
    HEAD = (255, 220, 70)
    START = (34, 255, 91)
    GOAL = (255, 71, 71)
    PATH = (255, 175, 64)

    def __init__(self, window_size=600):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()
        self.maze = Maze()
        self.solver = Solver(self.maze)
        self.sound_engine = SoundEngine()
        self.searching = False
        self.running = True
        self.steps_per_frame = 3
        self._update_title()

    def run(self):
        try:
            while self.running:
                self._handle_events()
                if self.searching and not self.solver.goal_reached:
                    self.solver.step(self.steps_per_frame)
                ratio = self.solver.explored_ratio()
                self.sound_engine.update(ratio)
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.searching = not self.searching
                elif event.key == pygame.K_r:
                    self._reset_maze()
                elif event.key == pygame.K_m:
                    self._toggle_mode()

    def _reset_maze(self):
        self.maze.reset()
        mode = self.solver.mode
        self.solver = Solver(self.maze, mode=mode)
        self.searching = False
        self._update_title()

    def _toggle_mode(self):
        new_mode = "DFS" if self.solver.mode == "BFS" else "BFS"
        self.solver.set_mode(new_mode)
        self.searching = False
        self._update_title()

    def _update_title(self):
        pygame.display.set_caption(f"Satisfying Maze - {self.solver.mode}")

    def _draw(self):
        self.screen.fill(self.BACKGROUND)
        tile_count = self.maze.grid.shape[0]
        tile_size = self.window_size / tile_count

        for row in range(tile_count):
            for col in range(tile_count):
                color = self.WALL if self.maze.grid[row][col] else self.UNEXPLORED
                self._draw_tile(row, col, color, tile_size)

        for cell in self.solver.visited:
            self._draw_cell(cell, self.VISITED, tile_size)

        if self.solver.goal_reached and self.solver.path:
            for cell in self.solver.path:
                self._draw_cell(cell, self.PATH, tile_size)

        self._draw_cell(self.maze.start, self.START, tile_size)
        self._draw_cell(self.maze.goal, self.GOAL, tile_size)

        if self.solver.current:
            self._draw_cell(self.solver.current, self.HEAD, tile_size)

    def _draw_tile(self, map_row, map_col, color, tile_size):
        x = int(map_col * tile_size)
        y = int(map_row * tile_size)
        width = max(1, int((map_col + 1) * tile_size) - x)
        height = max(1, int((map_row + 1) * tile_size) - y)
        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, width, height))

    def _draw_cell(self, cell, color, tile_size):
        map_coord = (2 * cell[0] + 1, 2 * cell[1] + 1)
        self._draw_tile(map_coord[0], map_coord[1], color, tile_size)


if __name__ == "__main__":
    App().run()
