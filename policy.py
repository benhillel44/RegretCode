import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from filelock import FileLock
from system import System
from constants import *


class Policy(ABC):
    def __init__(self, system: System, load_mem=True):
        self.sys = system
        self.f = system.f
        self.actions = self.sys.get_valid_actions()
        self.noises = self.sys.get_valid_noises()
        self.states = self.sys.get_valid_states()
        self.load_memory = load_mem
        if load_mem:
            self.memo = self.load_mem()
        else:
            self.memo = {}

    @abstractmethod
    def step(self, **information):
        pass

    @abstractmethod
    def GetClassID(self):
        pass

    def save_mem(self):
        """
        save the memoization dictionary to a file with the system id, k and N as the file name
        :return:
        """
        filename = f"{self.GetClassID()}_{self.sys.getId()}.pkl"
        filepath = PATH_TO_MEMOIZATION_FILES + filename
        lockfile = filepath + ".lock"
        with FileLock(lockfile):
            with open(filepath, 'wb') as f:
                pickle.dump(self.memo, f)

    def load_mem(self):
        """
        load the memoization dictionary from a file with the system id, k and N as the file name
        :return:
        """
        filename = f"{self.GetClassID()}_{self.sys.getId()}.pkl"
        # check if the file exists
        try:
            with open(PATH_TO_MEMOIZATION_FILES + filename, 'rb') as f:
                self.memo = pickle.load(f)
                if self.memo is None:
                    return {}
                return self.memo
        except FileNotFoundError:
            print("memoization file not found")
            return {}
        except EOFError:
            print(f"memoization file - {filename} is empty")
            return {}

    def __del__(self):
        if self.load_memory:
            self.save_mem()


class LookAheadPolicy(Policy):
    def __init__(self, system: System, look_ahead_depth=1, load_mem=True):
        self.look_ahead_depth = look_ahead_depth
        super().__init__(system, load_mem)

    def GetClassID(self):
        return f"LH"

    def _serialize_state(self, state):
        """ Serialize the numpy array state for memoization key. """
        return tuple(state.tobytes())

    def _serialize_w_ls(self, w_ls):
        """ Serialize list of numpy arrays for memoization key. """
        return tuple(w.tobytes() for w in w_ls)

    def step(self, x_t, w_ls):
        """
        :param x_t: current statr
        :param w_ls: list of future noises
        :return: [a_t: best action for time t, with k look ahead], total cost
        """
        actions = []
        total_cost = 0
        while len(w_ls) > 0:
            partial_actions, _ = self._step(x_t, w_ls[:self.look_ahead_depth])
            actions.append(partial_actions[0])
            total_cost += self.sys.g(x_t, actions[-1], w_ls[0])
            x_t = self.f(x_t, actions[-1], w_ls[0])
            w_ls = w_ls[1:]
        return actions, total_cost

    def _step(self, x_t, w_ls):
        if not w_ls:
            return [], 0

        # Create a memoization key from the state and disturbance list
        key = (self._serialize_state(x_t), self._serialize_w_ls(w_ls))
        if key in self.memo:
            return self.memo[key]

        first_w = w_ls[0]
        best_cost = float('inf')
        best_actions = []

        for u in self.actions:
            next_x = self.f(x_t, u, first_w)
            next_actions, price = self._step(next_x, w_ls[1:])
            price += self.sys.g(x_t, u, first_w)

            if price < best_cost:
                best_cost = price
                best_actions = [u] + next_actions

        result = (best_actions, best_cost)
        self.memo[key] = result
        return result

    def save_mem(self):
        """
        save the memoization dictionary to a file with the system id, k and N as the file name
        :return:
        """
        filename = f"{self.GetClassID()}_{self.sys.getId()}.pkl"
        # save the updated memoization
        filepath = PATH_TO_MEMOIZATION_FILES + filename
        lockfile = filepath + ".lock"
        with FileLock(lockfile):
            # check if the file updated since this instance was created
            if os.path.exists(PATH_TO_MEMOIZATION_FILES + filename):
                with open(PATH_TO_MEMOIZATION_FILES + filename, 'rb') as f:
                    old_memo = pickle.load(f)
                    self.memo.update(old_memo)
            with open(filepath, 'wb') as f:
                pickle.dump(self.memo, f)


class RegretPolicy(Policy):
    def __init__(self, system: System, star_policy=None, load_mem=True):
        super().__init__(system, load_mem)
        if star_policy is None:
            self.star_policy = LookAheadPolicy(system, look_ahead_depth=1, load_mem=load_mem)
        else:
            self.star_policy = star_policy

    def GetClassID(self):
        return "RP"

    def step(self, x_t, x_star_t, N=4):
        key = (tuple(x_t), tuple(x_star_t), N)
        if key in self.memo:
            return self.memo[key]

        if N == 0:
            return [], [], 0

        min_V = float('inf')
        best_worst_all_w = None
        best_worst_all_actions = None

        for u in self.actions:
            max_V = -float('inf')
            worst_all_w = None
            worst_all_actions = None

            for w in self.noises:
                u_star_ls, g_star = self.star_policy.step(x_star_t, [w])
                u_star = u_star_ls[0]
                g = self.sys.g(x_t, u, w)
                g_star = self.sys.g(x_star_t, u_star, w)
                next_x = self.f(x_t, u, w)
                next_x_star = self.f(x_star_t, u_star, w)
                V = g - g_star
                all_actions, all_noises, last_V = self.step(next_x, next_x_star, N - 1)

                V += last_V
                if V > max_V:
                    max_V = V
                    worst_all_w = [w] + all_noises
                    worst_all_actions = [u] + all_actions

            if max_V < min_V:
                min_V = max_V
                best_worst_all_w = deepcopy(worst_all_w)
                best_worst_all_actions = deepcopy(worst_all_actions)

        result = (best_worst_all_actions, best_worst_all_w, min_V)
        self.memo[key] = result
        return result


def _max_over_set(_set, ret_func, val_func):
    max_val = -float('inf')
    max_ret = None
    for elem in _set:
        val = val_func(elem)
        if val > max_val:
            max_val = val
            max_ret = ret_func(elem)
    return max_ret, max_val


def _min_over_set(_set, ret_func, val_func):
    min_val = float('inf')
    min_ret = None
    for elem in _set:
        val = val_func(elem)
        print(val)
        if val < min_val:
            min_val = val
            min_ret = ret_func(elem)
    return min_ret,


def _extremum_over_set(_set, is_min, _func):
    min_val = float('inf') if is_min else -float('inf')
    min_ret = None
    for elem in _set:
        ret, val = _func(elem)
        if is_min:
            if val < min_val:
                min_val = val
                min_ret = ret
        else:
            if val > min_val:
                min_val = val
                min_ret = ret
    if min_ret is None or min_val is None:
        print("ret is None")
    return min_ret, min_val


class KRegretPolicy(Policy):
    def __init__(self, system: System, N, look_ahead_depth=3, load_mem=True):
        self.k = look_ahead_depth
        super().__init__(system, load_mem)
        self.N = N
        self.star_policy = LookAheadPolicy(system, look_ahead_depth=look_ahead_depth, load_mem=load_mem)
        self._V_memo = {}

    def GetClassID(self):
        return f"KRP_{self.k}"

    def _serialize_w_ls(self, w_ls):
        """ Serialize list of numpy arrays for memoization key. """
        return tuple(bytes(w.data) for w in w_ls)

    def step(self, x_N_t, x_star, w_ls, t):
        """
        :param x_N_t:
        :param x_star:
        :param w_ls:
        :param t:
        :return: ((u_ls, w_ls), V)
            u_ls: list of actions
            w_ls: list of the noises that the adversary will play
            V: the value of the game
        """
        if t > self.N:
            raise ValueError("t must be less than N")

        key = (tuple(x_N_t), tuple(x_star), self._serialize_w_ls(w_ls), t)
        if key in self.memo:
            return self.memo[key]

        if t == 0:
            if len(w_ls) == 0:
                self.memo[key] = ([], []), 0
                return ([], []), 0
            u_star_ls, cost = self.star_policy.step(x_star, w_ls)
            V_0 = - cost
            self.memo[key] = ([], []), V_0
            return ([], []), V_0

        elif (t <= self.N - self.k):
            # min over u ( max over w ( min over u_star (
            #   g(x_N_t, u, w) - g(x_star, u_star, w) + V(x_N_t, x_star, w_ls + [w], t-1)
            # ) ) )

            # min over u ( max over w ( min over u_star ( ... ) ) )
            # g(x_N_t, u, w) - g(x_star, u_star, w) + V(x_N_t, x_star, w_ls + [w], t-1)
            def _temp_f(u, w, u_star):
                temp_w_ls = w_ls + [w]
                cost_x_N = self.sys.g(x_N_t, u, w)
                cost_x_star = self.sys.g(x_star, u_star, temp_w_ls[0])  # Compute once

                _val, _V = self.step(self.sys.f(x_N_t, u, w), self.sys.f(x_star, u_star, temp_w_ls[0]), temp_w_ls[1:],
                                     t - 1)
                _V += cost_x_N - cost_x_star  # Use precomputed values
                return _val, _V

            # min over u ( max over w ( ... ) )
            # w_ls + [w]
            def _temp_f_w(u, w):
                _val, _temp_V = _extremum_over_set(self.actions, False, lambda u_star: _temp_f(u, w, u_star))
                _temp_u_ls, _temp_w_ls = _val
                _temp_w_ls = [w] + _temp_w_ls
                return (_temp_u_ls, _temp_w_ls), _temp_V

            # min over u ( ... )
            # [u] + u_ls
            def _temp_f_u(u):
                _val, _temp_V = _extremum_over_set(self.noises, False, lambda w: _temp_f_w(u, w))
                _temp_u_ls, _temp_w_ls = _val
                _temp_u_ls = [u] + _temp_u_ls
                return (_temp_u_ls, _temp_w_ls), _temp_V

            _u_w_ls, V = _extremum_over_set(self.actions, True, _temp_f_u)
            self.memo[key] = _u_w_ls, V
            return _u_w_ls, V

        else:  # t >= self.N - self.k - 1
            # min over u ( max over w (
            #   g(x_N_t, u, w) + step(x_N_t, x_star, w_ls + [w], t-1)
            # ) )

            # min over u ( max over w ( ... ) )
            # w_ls + [w]
            def _temp_f2(u, w):
                _val, _V = self.step(self.sys.f(x_N_t, u, w), x_star, w_ls + [w], t - 1)
                _V += self.sys.g(x_N_t, u, w)
                _temp_u_ls, _temp_w_ls = _val
                _temp_w_ls = [w] + _temp_w_ls
                return (_temp_u_ls, _temp_w_ls), _V

            # min over u ( ... )
            # u_ls + [u]
            def _temp_f_u(u):
                _val, _V = _extremum_over_set(self.noises, False, lambda w: _temp_f2(u, w))
                _temp_u_ls, _temp_w_ls = _val
                _temp_u_ls = [u] + _temp_u_ls
                return (_temp_u_ls, _temp_w_ls), _V

            _u_w_ls, V = _extremum_over_set(self.actions, True, _temp_f_u)
            self.memo[key] = _u_w_ls, V
            return _u_w_ls, V

    def __V(self, x_N_t, x_star, w_ls, t):
        key = (tuple(x_N_t), tuple(x_star), self._serialize_w_ls(w_ls), t)
        if key in self._V_memo:
            return self._V_memo[key]

        if t == 0:
            if len(w_ls) == 0:
                return [], [], 0

            val_func = lambda _u_star: ((self.__V(x_N_t, self.sys.f(x_star, _u_star, w_ls[0]), w_ls[1:self.k], t))[-1]
                                        - self.sys.g(x_star, _u_star, w_ls[0]))
            ret_func = lambda _u_star: (self.__V(x_N_t, self.sys.f(x_star, _u_star, w_ls[0]), w_ls[1:self.k], t))[:-1]
            _, max_V = _max_over_set(self.actions, ret_func, val_func)
            # print(f"t = {t}, max_V = {max_V}")
            self._V_memo[key] = [], [], max_V
            return [], [], max_V

        if t < self.N - self.k:
            min_V_relative_u = float('inf')
            min_u_ls_relative_u = []
            min_w_ls_relative_u = []
            for u in self.actions:
                max_V_relative_w = -float('inf')
                max_u_ls_relative_w = []
                max_w_ls_relative_w = []
                for w in self.noises:
                    min_V_relative_u_star = float('inf')
                    min_u_ls_relative_u_star = []
                    min_w_ls_relative_u_star = []
                    for u_star in self.actions:
                        _u_ls, _w_ls, V = (self.__V(self.sys.f(x_N_t, u, w),
                                                    self.sys.f(x_star, u_star, w_ls[-1]), w_ls[1:] + [w], t - 1))
                        V += self.sys.g(x_N_t, u, w) - self.sys.g(x_star, u_star, w_ls[-1])
                        if V < min_V_relative_u_star:
                            min_V_relative_u_star = V
                            min_u_ls_relative_u_star = _u_ls
                            min_w_ls_relative_u_star = _w_ls
                    if min_V_relative_u_star > max_V_relative_w:
                        max_V_relative_w = min_V_relative_u_star
                        max_u_ls_relative_w = min_u_ls_relative_u_star
                        max_w_ls_relative_w = [w] + min_w_ls_relative_u_star
                if max_V_relative_w < min_V_relative_u:
                    min_V_relative_u = max_V_relative_w
                    min_u_ls_relative_u = [u] + max_u_ls_relative_w
                    min_w_ls_relative_u = max_w_ls_relative_w

            self._V_memo[key] = min_u_ls_relative_u, min_w_ls_relative_u, min_V_relative_u
            return min_u_ls_relative_u, min_w_ls_relative_u, min_V_relative_u
        else:  # t >= self.N - self.k - 1
            min_V_relative_u = float('inf')
            min_u_ls_relative_u = []
            min_w_ls_relative_u = []
            for u in self.actions:
                max_V_relative_w = -float('inf')
                max_u_ls_relative_w = []
                max_w_ls_relative_w = []
                for w in self.noises:
                    u_ls, w_ls, V = self.__V(self.sys.f(x_N_t, u, w), x_star, w_ls + [w], t - 1)
                    V += self.sys.g(x_N_t, u, w)
                    if V > max_V_relative_w:
                        max_V_relative_w = V
                        max_u_ls_relative_w = u_ls
                        max_w_ls_relative_w = [w] + w_ls
                if max_V_relative_w < min_V_relative_u:
                    min_V_relative_u = max_V_relative_w
                    min_u_ls_relative_u = [u] + max_u_ls_relative_w
                    min_w_ls_relative_u = max_w_ls_relative_w
            self._V_memo[key] = min_u_ls_relative_u, min_w_ls_relative_u, min_V_relative_u
            return min_u_ls_relative_u, min_w_ls_relative_u, min_V_relative_u