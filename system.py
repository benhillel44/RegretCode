from abc import ABC, abstractmethod

import numpy as np


class System(ABC):
    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def get_valid_noises(self):
        pass

    @abstractmethod
    def get_valid_states(self):
        pass

    @abstractmethod
    def _validate_state(self, x):
        pass

    @abstractmethod
    def f(self, x, u, w):
        pass

    @abstractmethod
    def g(self, x, u, w) -> float:
        pass

    @abstractmethod
    def getId(self):
        pass

    def J(self, x, u_ls, w_ls) -> float:
        l = min(len(u_ls), len(w_ls))
        total_cost = 0
        x_t = x
        for i in range(l):
            w = w_ls[i]
            u = u_ls[i]
            total_cost += self.g(x_t, u, w)
            x_t = self.f(x_t, u, w)
        return total_cost


import hashlib

class GridSystemHasher:
    @staticmethod
    def generate_unique_id(grid_system):
        """Generates a unique deterministic ID for a GridSystem instance."""
        # Convert numpy arrays to tuples for hash stability
        valid_actions = sorted(map(tuple, grid_system.VALID_ACTIONS))
        valid_noises = sorted(map(tuple, grid_system.VALID_NOISES))
        cost_mat = tuple(map(tuple, grid_system.COST_MAT))

        # Create a hashable representation
        key_data = (tuple(valid_actions), tuple(valid_noises), cost_mat)
        key_string = str(key_data).encode()

        # Generate a compact hash (SHA256 truncated for brevity)
        hash_digest = hashlib.sha256(key_string).hexdigest()[:16]  # Truncated to 16 characters
        return hash_digest

class GridSystem(System):
    VALID_ACTIONS = [np.array((1, 0)), np.array((0, 1)),
                     np.array((-1, 0)), np.array((0, -1)),
                     np.array((1, 1)), np.array((1, -1)),
                     np.array((-1, 1)), np.array((-1, -1))]
    VALID_NOISES = [np.array((2, 0)), np.array((0, 2)),
                    np.array((-2, 0)), np.array((0, -2))]
    COST_MAT = np.array([[1, 1, 10, -1, 10],
                         [1, 0, 100, 10, 0],
                         [10, 1, 1, 100, 1],
                         [10, 1, 0, 1, 1],
                         [10, 1, 1, 10, 1]])
    RANDSEED = 0

    def __init__(self, walls=None):
        self.grid_width = GridSystem.COST_MAT.shape[0]
        self.grid_length = GridSystem.COST_MAT.shape[1]
        self.walls = [] if (walls is None) else walls
        self.x_0 = np.array((0, 0))
        self.f_mem = {}

    def set_random_grid(self, dimensions=(10, 10)):
        np.random.seed(self.RANDSEED)
        GridSystem.COST_MAT = np.random.randint(1, 100, dimensions)

    def set_version2(self):
        np.random.seed(self.RANDSEED)
        GridSystem.COST_MAT = np.random.randint(1, 100, (10, 10))
        GridSystem.VALID_ACTIONS = [np.array((1, 0)), np.array((-1, 0)), np.array((0, 1)), np.array((0, -1))]
        GridSystem.VALID_NOISES = [np.array((1, 1)), np.array((1, -1)), np.array((-1, 1)), np.array((-1, -1))]

    def getId(self):
        return GridSystemHasher().generate_unique_id(self)

    def get_valid_actions(self):
        return GridSystem.VALID_ACTIONS

    def get_valid_noises(self):
        return GridSystem.VALID_NOISES

    def get_valid_states(self):
        valid_states = [np.array((x, y)) for x in range(self.grid_width) for y in range(self.grid_length)
                        if ((x, y) not in self.walls)]
        return valid_states

    def _serialize_state(self, state):
        return bytes(state.data)

    def _serialize_ls(self, w_ls):
        return tuple(bytes(w.data) for w in w_ls)

    def _validate_state(self, x: np.ndarray):
        """
        :param x: the position to check
        :return: True if x is in the grid And not colliding with any walls, else False
        """
        new_x = max(min(x[0], self.grid_width-1), 0)
        new_y = max(min(x[1], self.grid_length-1), 0)
        return np.array([new_x, new_y])

    def f(self, x: np.ndarray, u: np.ndarray, w: np.ndarray):
        return self._validate_state(x + u + w)

    def g(self, x: np.ndarray, u: np.ndarray, w: np.ndarray):
        next_x = self.f(x, u, w)
        return self.COST_MAT[next_x[0], next_x[1]]

