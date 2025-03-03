import numpy as np
import matplotlib.pyplot as plt
from numpy import array


class GridVisualizer:
    def __init__(self, grid, start_pos=(0, 0)):
        """
        Initializes the GridVisualizer class with a grid.

        Args:
        grid (np.ndarray): A 2D numpy array representing the grid.
        """
        self.grid = grid
        self.paths = []
        self.fig, self.ax = plt.subplots()
        self.start_pos = start_pos

    def add_movement_path(self, movements, color='red'):
        """
        Adds a movement path to the list of paths to be visualized.

        Args:
        movements (list): A list of np.array objects representing the movements.
        color (str): Color of the arrow for this path.
        """
        self.paths.append((movements, color))

    def visualize(self):
        """
        Visualizes the grid with arrows representing the player's movements and displays the cost of each cell.
        """
        n_rows, n_cols = self.grid.shape
        # Display the grid with numbers
        for i in range(n_rows):
            for j in range(n_cols):
                self.ax.text(j, i, str(self.grid[i, j]), ha='center', va='center', color='blue', fontsize=9)

        # Draw arrows for each path
        for path, color in self.paths:
            start_pos = np.array(self.start_pos)
            current_cost = 0
            path_len = len(path)
            for i, move in enumerate(path):
                new_pos = start_pos + move[::-1]  # reverse to fit the row/col structure of plt
                new_pos[0] = min(max(new_pos[0], 0), self.grid.shape[1]-1)
                new_pos[1] = min(max(new_pos[1], 0), self.grid.shape[0]-1)
                end_cell = np.array([int(new_pos[1]), int(new_pos[0])])
                mid_pos = (start_pos * (i + 1) + new_pos * (2*path_len - i)) / (2*path_len)
                current_cost += self.grid[end_cell[0], end_cell[1]]

                # Draw an arrow from the start to the new position
                self.ax.annotate('', xy=mid_pos + (new_pos - mid_pos) * 0.9,
                                 xytext=mid_pos + (start_pos - mid_pos) * 0.9,
                                 arrowprops=dict(arrowstyle="->", lw=1.5, color=color, shrinkA=5, shrinkB=5))

                self.ax.text(mid_pos[0], mid_pos[1] - 0.1, f'{i}', color=color, fontsize=6,
                             ha='center')

                # Update the start position
                start_pos = new_pos

        # Set grid lines and limits
        self.ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
        self.ax.grid(True, which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.set_xlim([0, n_cols])
        self.ax.set_ylim([n_rows, 0])
        self.ax.set_aspect('equal')

        # Hide axes
        self.ax.axis('off')

        plt.show()

def main():
    # Example usage
    grid = np.array([[1, 1, 10, -1],
                         [1, 0, 1, 10],
                         [1, 0, 0, 100],
                         [0, 0, 0, 1]])
    adversary_noise = [array([1, 0]), array([-1,  0]), array([-1,  0]), array([-1,  0])]
    reg_actions = [array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0])]
    look_ahead_actions = [array([1, 1]), array([1, 0]), array([0, 1]), array([0, 1])]
    movements = [adversary_noise[i] + look_ahead_actions[i] for i in range(len(adversary_noise))]
    visualizer = GridVisualizer(grid)
    visualizer.add_movement_path(movements)
    visualizer.visualize()

if __name__ == "__main__":
    main()
