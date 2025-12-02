from collections import Counter, deque

import numpy as np


class BoardPerception:
    def __init__(self, buffer_size: int = 10):
        """
        Initializes the BoardPerception object.

        Args:
            buffer_size: The number of recent board states to store.
        """
        self.buffer_size = buffer_size
        # Queue to hold 8x8 matrices of the last N frames
        self.history = deque(maxlen=buffer_size)

    def add_frame_observation(self, observation_8x8: np.ndarray):
        """
        Adds a new 8x8 board observation to the history buffer.

        Args:
            observation_8x8: A 2D NumPy array representing the board state.
        """
        self.history.append(observation_8x8)

    def get_stable_board(self) -> tuple[np.ndarray | None, bool]:
        """
        Analyzes the history of board observations to determine a stable board state.

        Returns:
            A tuple containing the stable board grid (or None) and a boolean indicating stability.
        """
        if len(self.history) < self.buffer_size:
            return None, False

        # Stack history to make it 3D: (buffer_size, 8, 8)
        stack = np.array(self.history)

        stable_grid = np.empty((8, 8), dtype=object)
        is_totally_stable = True

        for r in range(8):
            for c in range(8):
                # Get all detections for this specific square across history
                pixel_history = stack[:, r, c]

                # Find the most common element
                counts = Counter(pixel_history)
                most_common, freq = counts.most_common(1)[0]

                # Check confidence (e.g., must be present in 90% of frames)
                if freq < (self.buffer_size * 0.9):
                    is_totally_stable = False

                stable_grid[r][c] = most_common

        return stable_grid, is_totally_stable
