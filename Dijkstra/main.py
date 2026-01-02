import pygame
import numpy as np
import random
import heapq
import math
import sys

# --- Configuration ---
WIDTH, HEIGHT = 800, 800
FPS = 60
# High Resolution: 80x80 = 6400 cells
GRID_COLS = 80
GRID_ROWS = 80
CELL_SIZE = WIDTH // GRID_COLS  # 10px

# Simulation Speed
# 6400 cells / (60 fps * 5 steps) = ~21.3 seconds
STEPS_PER_FRAME = 5

# Colors
BG_COLOR = (20, 20, 25)
WALL_COLOR = (50, 50, 60)

DIJKSTRA_TRAIL = (0, 180, 220)
DIJKSTRA_HEAD = (180, 255, 255)

ASTAR_TRAIL = (220, 20, 147)
ASTAR_HEAD = (255, 180, 220)

PATH_COLOR = (255, 215, 0)
PATH_GLOW = (255, 255, 255)

SAMPLE_RATE = 44100


class SoundEngine:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            self.enabled = True
        except:
            self.enabled = False
        self.channel = pygame.mixer.Channel(0) if self.enabled else None

    def play_step(self, progress):
        if not self.enabled:
            return
        # Pitch rises from 220Hz to 880Hz over the course of the full maze
        freq = 220 + 660 * (progress ** 2)
        n = int(SAMPLE_RATE * 0.04)
        t = np.linspace(0, 0.04, n, False)
        wave = np.sin(2 * np.pi * freq * t) * 0.25
        att, dec = int(n * 0.1), int(n * 0.15)
        sus = n - att - dec
        env = np.concatenate([np.linspace(0, 1, att), np.ones(sus), np.linspace(1, 0, dec)])
        wave *= env
        audio = (wave * 32767).astype(np.int16)
        audio = np.column_stack((audio, audio))
        snd = pygame.sndarray.make_sound(audio)
        if not self.channel.get_busy():
            self.channel.play(snd)

    def stop_all(self):
        if self.enabled:
            pygame.mixer.stop()

    def play_fanfare(self):
        if not self.enabled:
            return
        freqs = [523, 659, 783, 1046]
        wave = np.zeros(0)
        for f in freqs:
            n = int(SAMPLE_RATE * 0.12)
            t = np.linspace(0, 0.12, n, False)
            wave = np.concatenate([wave, np.sin(2 * np.pi * f * t) * 0.4])
        n = int(SAMPLE_RATE * 1.2)
        t = np.linspace(0, 1.2, n, False)
        chord = sum(np.sin(2 * np.pi * f * t) * 0.12 for f in freqs)
        wave = np.concatenate([wave, chord])
        wave *= np.linspace(1, 0, len(wave))
        audio = (wave * 32767).astype(np.int16)
        audio = np.column_stack((audio, audio))
        pygame.sndarray.make_sound(audio).play()


class Maze:
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.grid = [[1] * cols for _ in range(rows)]
        self.start = (1, 1)
        self.end = (cols - 2, rows - 2)
        self._generate()

    def _generate(self):
        # Step 1: Recursive Backtracker
        stack = [(1, 1)]
        self.grid[1][1] = 0
        while stack:
            cx, cy = stack[-1]
            nbs = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < self.cols - 1 and 1 <= ny < self.rows - 1 and self.grid[ny][nx] == 1:
                    nbs.append((nx, ny, dx // 2, dy // 2))
            if nbs:
                nx, ny, wx, wy = random.choice(nbs)
                self.grid[cy + wy][cx + wx] = 0
                self.grid[ny][nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        # Step 2: Create loops (8% removal)
        walls_removed = 0
        target = int(self.cols * self.rows * 0.08)
        attempts = 0
        while walls_removed < target and attempts < target * 10:
            attempts += 1
            rx = random.randint(2, self.cols - 3)
            ry = random.randint(2, self.rows - 3)
            if self.grid[ry][rx] == 1:
                # Check neighbors to ensure we connect two open spaces
                adj_open = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                               if 0 <= rx + dx < self.cols and 0 <= ry + dy < self.rows
                               and self.grid[ry + dy][rx + dx] == 0)
                if adj_open >= 2:
                    self.grid[ry][rx] = 0
                    walls_removed += 1

        # Step 3: Safety clearing around Start/End
        sx, sy = self.start
        ex, ey = self.end
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = sx + dx, sy + dy
                if 0 < nx < self.cols - 1 and 0 < ny < self.rows - 1:
                    self.grid[ny][nx] = 0
                nx, ny = ex + dx, ey + dy
                if 0 < nx < self.cols - 1 and 0 < ny < self.rows - 1:
                    self.grid[ny][nx] = 0

        self.grid[sy][sx] = 0
        self.grid[ey][ex] = 0


class Solver:
    def __init__(self, maze, algo):
        self.maze = maze
        self.algo = algo
        self.open_set = []
        self.open_hash = set()
        self.closed = set()
        self.came_from = {}
        self.g = {maze.start: 0}
        f = self._h(maze.start)
        heapq.heappush(self.open_set, (f, 0, maze.start))
        self.open_hash.add(maze.start)
        self.visited = set()
        self.path = []
        self.goal_reached = False
        self.count = 0
        self.current = maze.start

    def _h(self, node):
        if self.algo == 'dijkstra':
            return 0
        return abs(node[0] - self.maze.end[0]) + abs(node[1] - self.maze.end[1])

    def step(self):
        if not self.open_set or self.goal_reached:
            return False
        _, _, cur = heapq.heappop(self.open_set)
        self.open_hash.discard(cur)
        if cur in self.closed:
            return True
        self.current = cur
        if cur == self.maze.end:
            self._rebuild_path()
            self.goal_reached = True
            return False
        self.closed.add(cur)
        self.visited.add(cur)
        cx, cy = cur
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            nb = (nx, ny)
            if not (0 <= nx < self.maze.cols and 0 <= ny < self.maze.rows):
                continue
            if self.maze.grid[ny][nx] == 1 or nb in self.closed:
                continue
            tg = self.g[cur] + 1
            if tg < self.g.get(nb, float('inf')):
                self.came_from[nb] = cur
                self.g[nb] = tg
                if nb not in self.open_hash:
                    self.count += 1
                    heapq.heappush(self.open_set, (tg + self._h(nb), self.count, nb))
                    self.open_hash.add(nb)
                    self.visited.add(nb)
        return True

    def _rebuild_path(self):
        c = self.maze.end
        while c in self.came_from:
            self.path.append(c)
            c = self.came_from[c]
        self.path.append(self.maze.start)
        self.path.reverse()

    def progress(self):
        sx, sy = self.maze.start
        cx, cy = self.current
        d = math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
        mx = math.sqrt(self.maze.cols ** 2 + self.maze.rows ** 2)
        return min(d / mx, 1.0)


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Algorithm Visualizer (High Res)")
        self.clock = pygame.time.Clock()
        self.sound = SoundEngine()
        self.algo = 'dijkstra'
        self.paused = True
        self.fanfare_done = False
        self._reset()

    def _reset(self):
        self.maze = Maze(GRID_COLS, GRID_ROWS)
        self.solver = Solver(self.maze, self.algo)
        self.fanfare_done = False
        print(f"Mode: {self.algo.upper()}")

    def _toggle(self):
        self.algo = 'astar' if self.algo == 'dijkstra' else 'dijkstra'
        self._reset()

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_m:
                        self._toggle()
                    elif ev.key == pygame.K_r:
                        self._reset()

            if not self.paused and not self.solver.goal_reached:
                for _ in range(STEPS_PER_FRAME):
                    if not self.solver.step():
                        break
                # Reduce sound probability slightly for high speed
                if not self.solver.goal_reached and random.random() < 0.2:
                    self.sound.play_step(self.solver.progress())

            if self.solver.goal_reached and not self.fanfare_done:
                self.sound.stop_all()
                self.sound.play_fanfare()
                self.fanfare_done = True

            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _draw(self):
        self.screen.fill(BG_COLOR)
        trail = DIJKSTRA_TRAIL if self.algo == 'dijkstra' else ASTAR_TRAIL
        head = DIJKSTRA_HEAD if self.algo == 'dijkstra' else ASTAR_HEAD

        if self.solver.goal_reached:
            # Darker trails on finish
            dim = tuple(max(c // 6, 5) for c in trail)
        else:
            dim = trail

        # Draw Walls (Small Rects)
        # Optimization: Blit a pre-surface? For 6400 rects, direct draw is okay 
        # but frame drop is possible. Let's stick to direct draw for simplicity.
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.grid[r][c] == 1:
                    pygame.draw.rect(self.screen, WALL_COLOR, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE + 1, CELL_SIZE + 1))

        # Draw Visited
        for node in self.solver.visited:
            x, y = node
            pygame.draw.rect(self.screen, dim, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE + 1, CELL_SIZE + 1))

        # Draw Head
        if not self.solver.goal_reached:
            for _, _, node in self.solver.open_set:
                x, y = node
                pygame.draw.rect(self.screen, head, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE + 1, CELL_SIZE + 1))

        # Pulse Path
        if self.solver.goal_reached and len(self.solver.path) > 1:
            t = pygame.time.get_ticks()
            pulse = (math.sin(t * 0.02) + 1) / 2
            r = int(PATH_COLOR[0] + (255 - PATH_COLOR[0]) * pulse)
            g = int(PATH_COLOR[1] + (255 - PATH_COLOR[1]) * pulse)
            b = int(PATH_COLOR[2] + (255 - PATH_COLOR[2]) * pulse)
            
            # Using narrower lines for high res
            pts = [(n[0] * CELL_SIZE + CELL_SIZE // 2, n[1] * CELL_SIZE + CELL_SIZE // 2) for n in self.solver.path]
            pygame.draw.lines(self.screen, (r, g, b), False, pts, 4)
            pygame.draw.lines(self.screen, PATH_GLOW, False, pts, 1)
            
            # Sparkles
            for i, n in enumerate(self.solver.path):
                if (i + int(t * 0.1)) % 10 < 2:
                    pygame.draw.rect(self.screen, (255, 255, 255), (n[0] * CELL_SIZE, n[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Start/End
        sx, sy = self.maze.start
        ex, ey = self.maze.end
        pygame.draw.rect(self.screen, (0, 255, 0), (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0), (ex * CELL_SIZE, ey * CELL_SIZE, CELL_SIZE, CELL_SIZE))


if __name__ == "__main__":
    App().run()
