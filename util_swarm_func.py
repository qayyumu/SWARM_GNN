import time
import functools
import multiprocessing
import numpy as np

def run_simulation(simulation, args, instances, processes=1, batch=100, silent=False):
    pool = multiprocessing.Pool(processes=processes)
    timeseries_data_all = []
    edge_data_all = []
    time_data_all = []

    remaining_instances = instances

    prev_time = time.time()
    while remaining_instances > 0:
        n = min(remaining_instances, batch)
        func = functools.partial(simulation, args)
        data_pool = pool.map(func, range(n))

        timeseries_pool, edge_pool, time_pool = zip(*data_pool)

        remaining_instances -= n
        if not silent:
            print('Simulation {}/{}... {:.1f}s'.format(instances - remaining_instances,
                                                       instances, time.time()-prev_time))
        prev_time = time.time()

        timeseries_data_all.extend(timeseries_pool)
        edge_data_all.extend(edge_pool)
        time_data_all.extend(time_pool)

    return timeseries_data_all, edge_data_all, time_data_all


class Entity:
    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None):
        self._ndim = ndim if ndim else 3

        # Max speed the boid can achieve.
        self.max_speed = float(max_speed) if max_speed else None
        self.max_acceleration = float(max_acceleration) if max_acceleration else None

        self.reset(position, velocity, acceleration)

    def reset(self, position, velocity=None, acceleration=None):
        """Initialize agent's spactial state."""
        self._position = np.zeros(self._ndim)
        self._velocity = np.zeros(self._ndim)
        self._acceleration = np.zeros(self._ndim)

        self._position[:] = position[:]

        if velocity is not None:
            self.velocity = velocity

        if acceleration is not None:
            self.acceleration = acceleration

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position[:] = position[:]

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity[:] = velocity[:]
        self._regularize_v()

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration):
        self._acceleration[:] = acceleration[:]
        self._regularize_a()

    def _regularize_v(self):
        if self.max_speed and self.speed > self.max_speed:
            self._velocity *= self.max_speed / self.speed
            return True
        return False

    def _regularize_a(self):
        if self.max_acceleration and np.linalg.norm(self.acceleration) > self.max_acceleration:
            self._acceleration *= self.max_acceleration / np.linalg.norm(self.acceleration)
            return True
        return False

    def move(self, dt):
        dt = float(dt)
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt  # + 0.5 * self.acceleration * dt * dt


class Goal:
    def __init__(self, position, velocity=None, priority=1, ndim=3):
        self._ndim = ndim if ndim else 3

        self._position = np.zeros(self._ndim)
        self.position = position

        self._velocity = np.zeros(self._ndim)
        if velocity is not None:
            self.velocity = velocity

        self.priority = priority

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position[:] = position[:]

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity[:] = velocity[:]

    def move(self, dt):
        self.position += self.velocity * dt
