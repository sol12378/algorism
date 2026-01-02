
import pygame
import numpy as np
import random
import heapq
import math
import sys

# --- Configuration & Constants ---
# Window Size Constraint: Height <= 900px
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 860  # fits comfortably
FPS = 60

# Layout Calculation
# Split vertically into 2 equal sections + Middle UI Bar
SECTION_H = 400
UI_BAR_H = 60
# Total H = 400 + 60 + 400 = 860

# Grid
GRID_COLS = 32
# Calculate Cell Size based on Width
CELL_SIZE = WINDOW_WIDTH // GRID_COLS
# Calculate Rows based on Section Height
GRID_ROWS = SECTION_H // CELL_SIZE

# Colors (Neon/Dark Theme)
BG_COLOR = (20, 20, 25)
WALL_COLOR = (50, 50, 60)
UI_BG_COLOR = (30, 30, 35)

# Dijkstra Colors (Cyan Base)
DIJKSTRA_ACCENT = (0, 255, 255)
DIJKSTRA_TRAIL = (0, 100, 255)

# A* Colors (Pink Base)
ASTAR_ACCENT = (255, 20, 147)
ASTAR_TRAIL = (148, 0, 211)

PATH_COLOR_WIN = (255, 215, 0)     # Gold
PATH_GLOW_WIN = (255, 255, 200)
PATH_COLOR_LOSE = (192, 192, 192)  # Silver
PATH_GLOW_LOSE = (200, 200, 200)

# Audio Config
SAMPLE_RATE = 44100

# --- Sound Engine ---
class SoundEngine:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8) # Ensure enough channels
            self.enabled = True
        except Exception as e:
            print(f"Audio init failed: {e}")
            self.enabled = False
            
        self.channel_dijkstra = pygame.mixer.Channel(0) if self.enabled else None
        self.channel_astar = pygame.mixer.Channel(1) if self.enabled else None
        self.channel_fx = pygame.mixer.Channel(2) if self.enabled else None
        
    def get_frequency(self, progress, min_freq=220, max_freq=880):
        # Quadratic scaling for satisfying rising pitch
        return min_freq + (max_freq - min_freq) * (progress ** 2)

    def play_step(self, algo_type, progress):
        if not self.enabled: return
        
        freq = self.get_frequency(progress)
        n_samples = int(SAMPLE_RATE * 0.05) # 50ms
        t = np.linspace(0, 0.05, n_samples, False)
        
        # Waveform & Panning
        if algo_type == 'dijkstra':
            # Sine wave (Smooth)
            wave = np.sin(2 * np.pi * freq * t) * 0.3
            channel = self.channel_dijkstra
            # Pan LEFT (Mostly Left, little Right)
            left_vol, right_vol = 0.8, 0.2
        else:
            # Triangle-ish (Edgy)
            wave = np.arcsin(np.sin(2 * np.pi * freq * t)) * (2/np.pi) * 0.3
            channel = self.channel_astar
            # Pan RIGHT
            left_vol, right_vol = 0.2, 0.8

        # Envelope
        attack = int(n_samples * 0.1)
        decay = int(n_samples * 0.1)
        sustain = n_samples - attack - decay
        envelope = np.concatenate([
            np.linspace(0, 1, attack),
            np.ones(sustain),
            np.linspace(1, 0, decay)
        ])
        wave = wave * envelope
        
        # Create Stereo Buffer based on Pan
        audio_left = (wave * left_vol * 32767).astype(np.int16)
        audio_right = (wave * right_vol * 32767).astype(np.int16)
        audio_stereo = np.column_stack((audio_left, audio_right))
        
        sound = pygame.sndarray.make_sound(audio_stereo)
        
        if not channel.get_busy():
            channel.play(sound)

    def play_fanfare(self, winner_algo):
        if not self.enabled: return
        freqs = [523, 659, 783, 1046] # C Major
        
        full_wave_l = np.zeros(0)
        full_wave_r = np.zeros(0)
        t_len = 0.1
        
        # Panning for fanfare based on winner
        if winner_algo == 'dijkstra':
            l_v, r_v = 1.0, 0.3
        else:
            l_v, r_v = 0.3, 1.0
            
        for f in freqs:
            n_samples = int(SAMPLE_RATE * t_len)
            t = np.linspace(0, t_len, n_samples, False)
            base = np.sin(2 * np.pi * f * t) * 0.5
            full_wave_l = np.concatenate([full_wave_l, base * l_v])
            full_wave_r = np.concatenate([full_wave_r, base * r_v])
            
        # Sustain Chord
        n_samples = int(SAMPLE_RATE * 1.5)
        t = np.linspace(0, 1.5, n_samples, False)
        chord = np.zeros(n_samples)
        for f in freqs:
            chord += np.sin(2 * np.pi * f * t) * 0.15
            
        # Add vibrato/beating
        full_wave_l = np.concatenate([full_wave_l, chord * l_v])
        full_wave_r = np.concatenate([full_wave_r, chord * r_v])
        
        # Master Envelope to fade out
        env = np.linspace(1, 0, len(full_wave_l))
        full_wave_l *= env
        full_wave_r *= env
        
        audio = np.column_stack((
            (full_wave_l * 32767).astype(np.int16),
            (full_wave_r * 32767).astype(np.int16)
        ))
        
        sound = pygame.sndarray.make_sound(audio)
        self.channel_fx.play(sound)

# --- Maze Generation ---
class Maze:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.grid = [[1 for _ in range(cols)] for _ in range(rows)]
        self.start = (1, 1)
        self.end = (cols - 2, rows - 2)
        self.generate()
        
    def generate(self):
        # Recursive Backtracker
        stack = []
        start_x, start_y = 1, 1
        self.grid[start_y][start_x] = 0
        stack.append((start_x, start_y))
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < self.cols - 1 and 1 <= ny < self.rows - 1 and self.grid[ny][nx] == 1:
                    neighbors.append((nx, ny, dx // 2, dy // 2))
            
            if neighbors:
                nx, ny, wx, wy = random.choice(neighbors)
                self.grid[cy + wy][cx + wx] = 0
                self.grid[ny][nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Post-processing: Make it looser (Loops allow A* to shine)
        for _ in range(int(self.cols * self.rows * 0.1)):
            rx = random.randint(1, self.cols - 2)
            ry = random.randint(1, self.rows - 2)
            if self.grid[ry][rx] == 1:
                open_neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if self.grid[ry+dy][rx+dx] == 0:
                        open_neighbors += 1
                if open_neighbors >= 2:
                     self.grid[ry][rx] = 0
        
        self.grid[self.start[1]][self.start[0]] = 0
        self.grid[self.end[1]][self.end[0]] = 0

# --- Pathfinder ---
class Solver:
    def __init__(self, maze, algo_type):
        self.maze = maze
        self.algo = algo_type
        
        # Deep reset
        self.open_set = []
        self.open_set_hash = set()  # Track nodes in open_set efficiently
        self.closed_set = set()  # Nodes already explored
        self.came_from = {}
        self.g_score = {}
        
        start = maze.start
        self.g_score[start] = 0
        f_start = self.heuristic(start, maze.end)
        
        heapq.heappush(self.open_set, (f_start, 0, start))
        self.open_set_hash.add(start)
        
        self.visited = set()  # For visualization
        self.path = []
        self.finished = False
        self.count = 0
        self.current = start
        
        self.max_dist_guess = math.sqrt(maze.cols**2 + maze.rows**2)
        
        # Metrics
        self.nodes_visited = 0
        self.finish_time = None # Frames

    def heuristic(self, a, b):
        if self.algo == 'dijkstra':
            return 0
        # Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self):
        if not self.open_set or self.finished:
            return False

        _, _, current = heapq.heappop(self.open_set)
        self.open_set_hash.discard(current)
        
        # Skip if already visited
        if current in self.closed_set:
            return True
            
        self.current = current
        
        # Check if we reached the goal
        if current == self.maze.end:
            self.reconstruct_path()
            self.finished = True
            return True
        
        # Mark as explored
        self.closed_set.add(current)
        self.visited.add(current)
        self.nodes_visited += 1
        
        cx, cy = current
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            
            # Bounds check
            if not (0 <= nx < self.maze.cols and 0 <= ny < self.maze.rows):
                continue
                
            # Wall check
            if self.maze.grid[ny][nx] == 1:
                continue
            
            # Skip if already explored
            if neighbor in self.closed_set:
                continue
                
            tentative_g = self.g_score[current] + 1
            
            # If this path is better
            if tentative_g < self.g_score.get(neighbor, float('inf')):
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                f = tentative_g + self.heuristic(neighbor, self.maze.end)
                
                # Add to open set if not already there
                if neighbor not in self.open_set_hash:
                    self.count += 1
                    heapq.heappush(self.open_set, (f, self.count, neighbor))
                    self.open_set_hash.add(neighbor)
                    self.visited.add(neighbor)  # For visualization
        
        return True

    def reconstruct_path(self):
        curr = self.maze.end
        while curr in self.came_from:
            self.path.append(curr)
            curr = self.came_from[curr]
        self.path.append(self.maze.start)
        self.path.reverse()
        
    def get_progress(self):
        sx, sy = self.maze.start
        cx, cy = self.current
        dist = math.sqrt((cx-sx)**2 + (cy-sy)**2)
        return min(dist / self.max_dist_guess, 1.0)
    
    def get_progress_pct(self):
        # For progress bar
        return self.get_progress()

# --- Application ---
class RunApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Algorithm Race: Dijkstra vs A*")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_big = pygame.font.SysFont("Arial", 60, bold=True)
        self.font_ui = pygame.font.SysFont("Courier New", 14, bold=True)
        
        self.sound_engine = SoundEngine()
        
        self.reset_race()
        
    def reset_race(self):
        # 1. Generate Maze (Shared)
        self.maze = Maze(GRID_COLS, GRID_ROWS)
        
        # 2. Solvers
        self.dijkstra = Solver(self.maze, 'dijkstra')
        self.astar = Solver(self.maze, 'astar')
        
        # State
        self.winner = None
        self.running = True # Start immediately
        self.finished_all = False
        self.fanfare_played = False
        
    def draw_solver_view(self, solver, offset_y, title, color_accent, color_trail):
        ox = 0
        oy = offset_y
        
        # Draw Maze Walls
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.grid[r][c] == 1:
                    rect = (ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE + 1, CELL_SIZE + 1)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                    
        # Visited
        # Dim if finished to show path
        if solver.finished:
            r, g, b = color_trail
            draw_trail = (max(r//3, 10), max(g//3, 10), max(b//3, 10))
        else:
            draw_trail = color_trail
            
        for node in solver.visited:
            c, r = node
            rect = (ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE + 1, CELL_SIZE + 1)
            pygame.draw.rect(self.screen, draw_trail, rect)
            
        # Head (only if running)
        if not solver.finished:
            for _, _, node in solver.open_set:
                c, r = node
                rect = (ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE + 1, CELL_SIZE + 1)
                pygame.draw.rect(self.screen, color_accent, rect, border_radius=2)

        # Path
        if solver.finished and len(solver.path) > 1:
            points = []
            for node in solver.path:
                nx = ox + node[0] * CELL_SIZE + CELL_SIZE/2
                ny = oy + node[1] * CELL_SIZE + CELL_SIZE/2
                points.append((nx, ny))
            
            # Determine Path Color (Win/Lose)
            if self.winner == solver.algo:
                base_color = PATH_COLOR_WIN
                glow_color = PATH_GLOW_WIN
                
                # Strobing logic for winner
                t = pygame.time.get_ticks()
                pulse = (math.sin(t * 0.015) + 1) / 2
                
                r = int(base_color[0] + (255 - base_color[0]) * pulse)
                g = int(base_color[1] + (255 - base_color[1]) * pulse)
                b = int(base_color[2] + (255 - base_color[2]) * pulse)
                final_color = (r, g, b)
                width = 6
                
                # Sparkles
                for i, node in enumerate(solver.path):
                    if (i + int(t * 0.05)) % 12 < 3:
                       c, r = node
                       rect = (ox + c*CELL_SIZE, oy+r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                       pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=2)
                       
            else:
                final_color = PATH_COLOR_LOSE
                glow_color = (100, 100, 100)
                width = 3
            
            pygame.draw.lines(self.screen, final_color, False, points, width)

        # Start/End
        sx, sy = self.maze.start
        ex, ey = self.maze.end
        pygame.draw.rect(self.screen, (0, 255, 0), (ox+sx*CELL_SIZE, oy+sy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0), (ox+ex*CELL_SIZE, oy+ey*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Title Label
        label = self.font_title.render(title, True, color_accent)
        self.screen.blit(label, (10, oy + 10))
        
        # Winner Label
        if self.winner == solver.algo:
            win_text = self.font_big.render("WINNER!", True, color_accent)
            # Center of section
            cx = WINDOW_WIDTH // 2 - win_text.get_width() // 2
            cy = oy + SECTION_H // 2 - win_text.get_height() // 2
            # Add shadow
            s_text = self.font_big.render("WINNER!", True, (0,0,0))
            self.screen.blit(s_text, (cx+2, cy+2))
            self.screen.blit(win_text, (cx, cy))

    def draw_ui_bar(self):
        # Middle bar
        y_start = SECTION_H
        rect = (0, y_start, WINDOW_WIDTH, UI_BAR_H)
        pygame.draw.rect(self.screen, UI_BG_COLOR, rect)
        
        # Progress Bars
        bar_w = int(WINDOW_WIDTH * 0.4)
        bar_h = 10
        gap = 20
        
        # Left: Dijkstra Progress
        d_pct = self.dijkstra.get_progress_pct()
        pygame.draw.rect(self.screen, (50, 50, 50), (gap, y_start + 25, bar_w, bar_h))
        pygame.draw.rect(self.screen, DIJKSTRA_ACCENT, (gap, y_start + 25, int(bar_w * d_pct), bar_h))
        
        # Right: A* Progress
        a_pct = self.astar.get_progress_pct()
        pygame.draw.rect(self.screen, (50, 50, 50), (WINDOW_WIDTH - gap - bar_w, y_start + 25, bar_w, bar_h))
        pygame.draw.rect(self.screen, ASTAR_ACCENT, (WINDOW_WIDTH - gap - bar_w, y_start + 25, int(bar_w * a_pct), bar_h))
        
        # Center Text
        if self.finished_all:
            msg = "Hit SPACE to Re-Race!"
            col = (255, 255, 255)
        else:
            msg = "RACING..."
            col = (150, 150, 150)
            
        txt = self.font_ui.render(msg, True, col)
        self.screen.blit(txt, (WINDOW_WIDTH//2 - txt.get_width()//2, y_start + 20))
        
        # Names
        name_d = self.font_ui.render(f"Dijkstra ({self.dijkstra.nodes_visited} nodes)", True, DIJKSTRA_ACCENT)
        self.screen.blit(name_d, (gap, y_start + 5))
        
        name_a = self.font_ui.render(f"A* ({self.astar.nodes_visited} nodes)", True, ASTAR_ACCENT)
        self.screen.blit(name_a, (WINDOW_WIDTH - gap - name_a.get_width(), y_start + 5))

    def run(self):
        while True:
            # Event Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.finished_all:
                        self.reset_race()
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            # Logic Update - Run multiple steps for speed? Let's stick to 1-2 per frame for smooth race visibility
            steps = 2
            
            if self.running and not self.finished_all:
                for _ in range(steps):
                    # Step Dijkstra
                    if not self.dijkstra.finished:
                        self.dijkstra.step()
                        if random.random() < 0.3:
                            self.sound_engine.play_step('dijkstra', self.dijkstra.get_progress())
                        if self.dijkstra.finished and self.winner is None:
                            self.winner = 'dijkstra'
                    
                    # Step A*
                    if not self.astar.finished:
                        self.astar.step()
                        if random.random() < 0.3:
                            self.sound_engine.play_step('astar', self.astar.get_progress())
                        if self.astar.finished and self.winner is None:
                            self.winner = 'astar'
            
            # Check completion
            if self.dijkstra.finished and self.astar.finished:
                self.finished_all = True
                if not self.fanfare_played:
                    self.sound_engine.play_fanfare(self.winner)
                    self.fanfare_played = True
            
            # Draw
            self.screen.fill(BG_COLOR)
            
            # Top Section (Dijkstra)
            self.draw_solver_view(self.dijkstra, 0, "Dijkstra - The Perfectionist 🐢", DIJKSTRA_ACCENT, DIJKSTRA_TRAIL)
            
            # Menu Bar
            self.draw_ui_bar()
            
            # Bottom Section (A*)
            self.draw_solver_view(self.astar, SECTION_H + UI_BAR_H, "A* - The Speedster 🚀", ASTAR_ACCENT, ASTAR_TRAIL)
            
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    RunApp().run()
