"""
Sudoku Solver Visualization - NAIVE vs SMART Comparison
Backtracking algorithm with MRV heuristic vs Brute Force
"""

import pygame
import numpy as np
import random
from typing import List, Tuple, Optional, Generator

# ============================================================================
# Constants & Configuration
# ============================================================================

WINDOW_SIZE = 800
GRID_SIZE = 9
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
FPS = 60

# Mode-specific speeds
STEPS_PER_FRAME_NAIVE = 1  # Slow and painful
STEPS_PER_FRAME_SMART = 50  # Lightning fast

# Colors - Cyberpunk Theme
COLOR_BG = (0, 0, 0)  # Pure black
COLOR_GRID_THIN = (0, 80, 0)  # Dark green for thin lines
COLOR_GRID_THICK = (0, 150, 0)  # Brighter green for 3x3 boundaries
COLOR_INITIAL = (255, 255, 255)  # White for given numbers
COLOR_SOLVING = (0, 255, 65)  # Matrix neon green for AI numbers
COLOR_SOLVED = (255, 215, 0)  # Gold for victory wave
COLOR_BACKTRACK = (255, 0, 0)  # Red flash for backtracking
COLOR_BACKTRACK_BG = (255, 0, 0, 120)  # Semi-transparent red for background
COLOR_TEXT = (0, 200, 100)  # UI text color
COLOR_VICTORY = (0, 255, 200)  # Cyan for victory text
COLOR_MODE_NAIVE = (255, 100, 100)  # Reddish for naive mode
COLOR_MODE_SMART = (100, 255, 255)  # Cyan for smart mode

# Audio Configuration
SAMPLE_RATE = 22050
AUDIO_BUFFER_SIZE = 512

# ============================================================================
# Audio Engine
# ============================================================================

class AudioEngine:
    """Real-time audio synthesis using NumPy"""
    
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=AUDIO_BUFFER_SIZE)
        self.channel = pygame.mixer.Channel(0)
        self.victory_channel = pygame.mixer.Channel(1)
        
    def play_forward_naive(self):
        """Slow, deliberate typing sound for naive algorithm"""
        duration = 0.02
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Lower frequency for "slower thinking"
        freq = 1500 + random.randint(-100, 100)
        wave = np.sin(2 * np.pi * freq * t)
        
        # Add noise
        noise = np.random.uniform(-0.3, 0.3, samples)
        wave = wave * 0.6 + noise * 0.4
        
        # Envelope
        envelope = np.exp(-t * 80)
        wave = wave * envelope
        
        # Convert to audio (stereo)
        wave = np.clip(wave * 32767 * 0.25, -32767, 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        self.channel.play(sound)
    
    def play_forward_smart(self):
        """Machine gun rapid-fire typing for smart algorithm"""
        duration = 0.008  # Much shorter
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Higher frequencies for "rapid thinking"
        freq1 = 3000 + random.randint(-200, 200)
        freq2 = 4500 + random.randint(-300, 300)
        
        wave1 = np.sin(2 * np.pi * freq1 * t) * 0.5
        wave2 = np.sin(2 * np.pi * freq2 * t) * 0.3
        wave = wave1 + wave2
        
        # Sharp noise burst
        noise = np.random.uniform(-0.4, 0.4, samples)
        wave = wave * 0.5 + noise * 0.5
        
        # Very sharp envelope
        envelope = np.exp(-t * 250)
        wave = wave * envelope
        
        # Convert to audio (stereo)
        wave = np.clip(wave * 32767 * 0.2, -32767, 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        self.channel.play(sound)
    
    def play_backward(self):
        """Low frequency rewind sound - emphasized for naive mode"""
        duration = 0.04
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Descending frequency for "unwinding" effect
        freq_start = 500
        freq_end = 100
        freq = np.linspace(freq_start, freq_end, samples)
        
        # Create sweeping sine wave
        phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
        wave = np.sin(phase)
        
        # Add noise
        noise = np.random.uniform(-0.2, 0.2, samples)
        wave = wave * 0.7 + noise * 0.3
        
        # Envelope
        envelope = np.exp(-t * 50)
        wave = wave * envelope
        
        # Convert to audio (stereo)
        wave = np.clip(wave * 32767 * 0.28, -32767, 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        self.channel.play(sound)
    
    def play_victory_note(self, note_index: int):
        """Play a single note in the victory arpeggio"""
        duration = 0.08
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Ascending pentatonic scale
        base_freq = 523.25  # C5
        scale = [1.0, 1.125, 1.25, 1.5, 1.875, 2.0, 2.25, 2.5, 3.0]
        freq = base_freq * scale[note_index % len(scale)]
        
        # Clean sine wave with harmonics
        wave = np.sin(2 * np.pi * freq * t) * 0.6
        wave += np.sin(2 * np.pi * freq * 2 * t) * 0.2
        wave += np.sin(2 * np.pi * freq * 3 * t) * 0.1
        
        # Bell-like envelope
        attack = np.linspace(0, 1, samples // 10)
        decay = np.exp(-t[samples // 10:] * 15)
        envelope = np.concatenate([attack, decay[:len(wave) - len(attack)]])
        wave = wave * envelope
        
        # Convert to audio (stereo)
        wave = np.clip(wave * 32767 * 0.4, -32767, 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        self.victory_channel.play(sound)

# ============================================================================
# Sudoku Generator
# ============================================================================

class SudokuGenerator:
    """Generate challenging Sudoku puzzles"""
    
    @staticmethod
    def is_valid(board: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in [board[r][col] for r in range(9)]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        
        return True
    
    @staticmethod
    def solve_complete(board: List[List[int]]) -> bool:
        """Fill the board completely (for generation)"""
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)
                    for num in numbers:
                        if SudokuGenerator.is_valid(board, row, col, num):
                            board[row][col] = num
                            if SudokuGenerator.solve_complete(board):
                                return True
                            board[row][col] = 0
                    return False
        return True
    
    @staticmethod
    def generate_puzzle(difficulty: int = 45) -> Tuple[List[List[int]], List[List[bool]]]:
        """Generate a Sudoku puzzle"""
        # Create a complete valid board
        board = [[0] * 9 for _ in range(9)]
        SudokuGenerator.solve_complete(board)
        
        # Remove cells to create puzzle
        cells_to_remove = difficulty
        removed = 0
        attempts = 0
        max_attempts = 100
        
        while removed < cells_to_remove and attempts < max_attempts:
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            
            if board[row][col] != 0:
                board[row][col] = 0
                removed += 1
            
            attempts += 1
        
        # Create mask for initial numbers
        initial_mask = [[board[r][c] != 0 for c in range(9)] for r in range(9)]
        
        return board, initial_mask

# Challenging puzzle (between difficulty 40 and hardest)
CHALLENGING_PUZZLE = [
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0]
]

# ============================================================================
# Sudoku Solver with Visualization
# ============================================================================

class SudokuSolver:
    """Backtracking solver with NAIVE vs SMART modes"""
    
    def __init__(self, board: List[List[int]], initial_mask: List[List[bool]], mode: str = "naive"):
        self.board = [row[:] for row in board]  # Deep copy
        self.initial_mask = initial_mask
        self.mode = mode  # "naive" or "smart"
        self.solved = False
        
    def is_valid(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in self.board[row]:
            return False
        
        # Check column
        if num in [self.board[r][col] for r in range(9)]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.board[r][c] == num:
                    return False
        
        return True
    
    def count_valid_numbers(self, row: int, col: int) -> int:
        """Count how many valid numbers can be placed at (row, col)"""
        if self.board[row][col] != 0:
            return 9  # Already filled
        
        count = 0
        for num in range(1, 10):
            if self.is_valid(row, col, num):
                count += 1
        return count
    
    def get_next_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Get next empty cell based on mode"""
        if self.mode == "naive":
            # NAIVE: Simple left-to-right, top-to-bottom scan
            for row in range(9):
                for col in range(9):
                    if self.board[row][col] == 0:
                        return (row, col)
            return None
        
        else:  # "smart"
            # SMART: MRV - Find cell with minimum remaining values
            best_cell = None
            min_count = 10
            
            for row in range(9):
                for col in range(9):
                    if self.board[row][col] == 0:
                        count = self.count_valid_numbers(row, col)
                        if count < min_count:
                            min_count = count
                            best_cell = (row, col)
            
            return best_cell
    
    def solve_generator(self) -> Generator[Tuple[str, int, int, int], None, bool]:
        """
        Generator that yields each step of the solving process
        Yields: (action, row, col, number)
        """
        def backtrack() -> Generator[Tuple[str, int, int, int], None, bool]:
            # Get next empty cell based on mode
            cell = self.get_next_empty_cell()
            
            if cell is None:
                # All cells filled - solved!
                return True
            
            row, col = cell
            
            # Try numbers 1-9
            for num in range(1, 10):
                if self.is_valid(row, col, num):
                    # Place number
                    self.board[row][col] = num
                    yield ('place', row, col, num)
                    
                    # Recursively solve
                    result = yield from backtrack()
                    if result:
                        return True
                    
                    # Backtrack - remove number
                    self.board[row][col] = 0
                    yield ('remove', row, col, num)
            
            # No valid number found
            return False
        
        self.solved = yield from backtrack()
        return self.solved

# ============================================================================
# Visualization
# ============================================================================

class SudokuVisualizer:
    """Main visualization class"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Sudoku: NAIVE vs SMART Comparison")
        self.clock = pygame.time.Clock()
        
        # Font - antialiasing OFF for pixel art look
        self.font_large = pygame.font.SysFont('Courier New', 50, bold=True)
        self.font_victory = pygame.font.SysFont('Courier New', 70, bold=True)
        self.font_mode = pygame.font.SysFont('Courier New', 32, bold=True)
        self.font_small = pygame.font.SysFont('Courier New', 16)
        
        # Audio
        self.audio = AudioEngine()
        
        # Game state
        self.mode = "naive"  # "naive" or "smart"
        self.paused = False
        self.reset_game()
        
    def reset_game(self):
        """Start new puzzle"""
        # Use challenging puzzle (difficulty between 40 and hardest)
        if random.random() < 0.5:
            self.board = [row[:] for row in CHALLENGING_PUZZLE]
            self.initial_mask = [[self.board[r][c] != 0 for c in range(9)] for r in range(9)]
        else:
            self.board, self.initial_mask = SudokuGenerator.generate_puzzle(difficulty=45)
        
        self.solver = SudokuSolver(self.board, self.initial_mask, self.mode)
        self.solve_gen = self.solver.solve_generator()
        self.current_board = [row[:] for row in self.board]
        self.cell_flash = [[0 for _ in range(9)] for _ in range(9)]
        self.cell_bg_flash = [[0 for _ in range(9)] for _ in range(9)]
        self.solving = True
        self.solved = False
        
        # Victory animation state
        self.victory_animation = False
        self.victory_wave_cells = []
        self.victory_current_index = 0
        self.victory_frame_counter = 0
        self.victory_text_blink = 0
        
        # Stats
        self.backtracks = 0
        self.steps = 0
        
    def toggle_mode(self):
        """Toggle between naive and smart modes"""
        self.mode = "smart" if self.mode == "naive" else "naive"
        self.reset_game()
        
    def prepare_victory_animation(self):
        """Prepare the wave animation cells"""
        cells = []
        for sum_val in range(18):
            for row in range(9):
                col = sum_val - row
                if 0 <= col < 9:
                    cells.append((row, col))
        self.victory_wave_cells = cells
        self.victory_current_index = 0
        self.victory_frame_counter = 0
        self.victory_animation = True
        self.victory_text_blink = 0
        
    def draw_grid(self):
        """Draw the Sudoku grid"""
        for i in range(10):
            thickness = 4 if i % 3 == 0 else 1
            color = COLOR_GRID_THICK if i % 3 == 0 else COLOR_GRID_THIN
            
            # Horizontal lines
            pygame.draw.line(
                self.screen, color,
                (0, i * CELL_SIZE),
                (WINDOW_SIZE, i * CELL_SIZE),
                thickness
            )
            
            # Vertical lines
            pygame.draw.line(
                self.screen, color,
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, WINDOW_SIZE),
                thickness
            )
    
    def draw_mode_indicator(self):
        """Draw current mode at top of screen"""
        if self.mode == "naive":
            mode_text = "MODE: BRUTE FORCE (DUMB 🐢)"
            color = COLOR_MODE_NAIVE
        else:
            mode_text = "MODE: SMART AI (GENIUS 🧠)"
            color = COLOR_MODE_SMART
        
        # Draw semi-transparent background
        overlay = pygame.Surface((WINDOW_SIZE, 50), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Draw mode text
        text = self.font_mode.render(mode_text, False, color)
        text_rect = text.get_rect(center=(WINDOW_SIZE // 2, 25))
        self.screen.blit(text, text_rect)
    
    def draw_numbers(self):
        """Draw all numbers on the board"""
        for row in range(9):
            for col in range(9):
                num = self.current_board[row][col]
                if num != 0:
                    # Draw background flash for backtracking
                    if self.cell_bg_flash[row][col] > 0:
                        surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        alpha = int(120 * (self.cell_bg_flash[row][col] / 5))
                        surf.fill((255, 0, 0, alpha))
                        self.screen.blit(surf, (col * CELL_SIZE, row * CELL_SIZE))
                        self.cell_bg_flash[row][col] -= 1
                    
                    # Choose color
                    if self.cell_flash[row][col] > 0:
                        color = COLOR_BACKTRACK
                        self.cell_flash[row][col] -= 1
                    elif self.initial_mask[row][col]:
                        color = COLOR_INITIAL
                    elif self.victory_animation:
                        cell_index = self.victory_wave_cells.index((row, col)) if (row, col) in self.victory_wave_cells else -1
                        if cell_index >= 0 and cell_index < self.victory_current_index:
                            color = COLOR_SOLVED
                        else:
                            color = COLOR_SOLVING
                    else:
                        color = COLOR_SOLVING
                    
                    # Render text
                    text = self.font_large.render(str(num), False, color)
                    text_rect = text.get_rect(
                        center=(col * CELL_SIZE + CELL_SIZE // 2,
                               row * CELL_SIZE + CELL_SIZE // 2)
                    )
                    self.screen.blit(text, text_rect)
    
    def draw_ui(self):
        """Draw UI text"""
        if self.victory_animation and self.victory_current_index >= len(self.victory_wave_cells):
            # Victory text blinking
            if self.victory_text_blink % 20 < 15:
                victory_msg = "SMART COMPLETE!" if self.mode == "smart" else "SOLVED!"
                text = self.font_victory.render(victory_msg, False, COLOR_VICTORY)
                text_rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
                
                # Glow effect
                for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
                    glow_rect = text_rect.copy()
                    glow_rect.x += offset[0]
                    glow_rect.y += offset[1]
                    self.screen.blit(text, glow_rect)
                
                self.screen.blit(text, text_rect)
            
            # Stats
            stats_text = f"Backtracks: {self.backtracks} | Steps: {self.steps}"
            text = self.font_small.render(stats_text, False, COLOR_TEXT)
            self.screen.blit(text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE // 2 + 60))
            
            # Instructions
            text = self.font_small.render("M: Toggle Mode | R: Reset", False, COLOR_TEXT)
            self.screen.blit(text, (WINDOW_SIZE // 2 - 110, WINDOW_SIZE - 25))
        elif self.paused:
            text = self.font_small.render("PAUSED - SPACE: resume | M: mode | R: reset", False, COLOR_TEXT)
            self.screen.blit(text, (10, WINDOW_SIZE - 25))
        else:
            info = f"Solving... Backtracks: {self.backtracks} | SPACE: pause | M: mode | R: reset"
            text = self.font_small.render(info, False, COLOR_TEXT)
            self.screen.blit(text, (10, WINDOW_SIZE - 25))
    
    def update_solver(self):
        """Advance the solver by multiple steps per frame"""
        if not self.solving or self.paused:
            return
        
        # Use different step counts based on mode
        steps = STEPS_PER_FRAME_SMART if self.mode == "smart" else STEPS_PER_FRAME_NAIVE
        
        for _ in range(steps):
            try:
                action, row, col, num = next(self.solve_gen)
                self.steps += 1
                
                if action == 'place':
                    self.current_board[row][col] = num
                    if self.mode == "smart":
                        self.audio.play_forward_smart()
                    else:
                        self.audio.play_forward_naive()
                        
                elif action == 'remove':
                    self.current_board[row][col] = 0
                    self.cell_flash[row][col] = 4
                    self.cell_bg_flash[row][col] = 5
                    self.backtracks += 1
                    self.audio.play_backward()
                    
            except StopIteration:
                self.solving = False
                self.solved = self.solver.solved
                if self.solved:
                    self.prepare_victory_animation()
                break
    
    def update_victory_animation(self):
        """Update the victory wave animation"""
        if not self.victory_animation:
            return
        
        self.victory_frame_counter += 1
        if self.victory_frame_counter >= 2:
            self.victory_frame_counter = 0
            if self.victory_current_index < len(self.victory_wave_cells):
                self.audio.play_victory_note(self.victory_current_index)
                self.victory_current_index += 1
        
        self.victory_text_blink += 1
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.victory_animation:
                        self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_m:
                    self.toggle_mode()
        
        return True
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update
            self.update_solver()
            self.update_victory_animation()
            
            # Draw
            self.screen.fill(COLOR_BG)
            self.draw_grid()
            self.draw_numbers()
            self.draw_mode_indicator()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    visualizer = SudokuVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
