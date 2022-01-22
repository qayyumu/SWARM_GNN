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



class Obstacle(Entity):
    def __init__(self, position, velocity=None, ndim=None):
        """Base class `Obstacle`."""
        super().__init__(position, velocity, ndim=ndim)

        self.size = 0

    def distance(self, r):
        raise NotImplementedError()

    def direction(self, r):
        """Direction of position `r` relative to obstacle surface"""
        raise NotImplementedError()


class Wall(Obstacle):
    def __init__(self, direction, position, velocity=None, ndim=None):
        """
        A plane in space that repels free agents.

        Parameters:
            position: the position of a point the wall passes.
            direction: the normal direction of the plane wall.
        """
        super().__init__(position, velocity, ndim)

        self._direction = np.array(direction, dtype=float)
        if self._direction.shape != (self._ndim,):
            raise ValueError('direction must be of shape ({},)'.format(self._ndim))
        self._direction /= np.linalg.norm(self._direction)  # Normalization

    def distance(self, r):
        return np.dot(r - self.position, self.direction(r))

    def direction(self, r):
        return self._direction


class Sphere(Obstacle):
    def __init__(self, size, position, velocity=None, ndim=None):
        """
        A sphere in ndim space.
        """
        super().__init__(position, velocity, ndim)
        self.size = size

    def distance(self, r):
        d = np.linalg.norm(r - self.position) - self.size
        if d < 0.1:
            d = 0.1
        return d

    def direction(self, r):
        """Direction of position `r` relative to obstacle surface"""
        d = r - self.position
        return d / np.linalg.norm(d)


class Rectangle(Obstacle):
    def __init__(self, sides, position, orientation, velocity=None, ndim=None):
        """
        A generalized rectangle in ndim space.
        """
        super().__init__(position, velocity, ndim)

        if len(sides) != self.ndim:
            raise ValueError('number of side lengths does not match ndim')

        self._orientation = np.array(orientation, dtype=float)
        if self._orientation.shape != (self._ndim,):
            raise ValueError('direction must be of shape ({},)'.format(self._ndim))
        self._orientation /= np.linalg.norm(self._orientation)  # Normalization

class Agent(Entity):
    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None,
                 size=None, vision=None):
        """
        Create a boid with essential attributes.
        `ndim`: dimension of the space it resides in.
        `vision`: the visual range.
        `anticipation`: range of anticipation for its own motion.
        `comfort`: distance the agent wants to keep from other objects.
        `max_speed`: max speed the agent can achieve.
        `max_acceleratoin`: max acceleration the agent can achieve.
        """
        super().__init__(position, velocity, acceleration, ndim, max_speed, max_acceleration)

        self.size = float(size) if size else 0.
        self.vision = float(vision) if vision else np.inf

        # Goal.
        self.goal = None

        # Perceived neighbors and obstacles.
        self.neighbors = []
        self.obstacles = []

    def distance(self, other):
        """Distance from the other objects."""
        if isinstance(other, Agent):
            return np.linalg.norm(self.position - other.position)
        # If other is not agent, let other tell the distance.
        try:
            return other.distance(self.position)
        except AttributeError:
            raise ValueError(f'cannot determine distance with {type(other)}')

    def set_goal(self, goal):
        if (goal is not None) and not isinstance(goal, Goal):
            raise ValueError("'goal' must be an instance of 'Goal' or None")
        self.goal = goal

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision

    def observe(self, environment):
        """Observe the population and take note of neighbors."""
        self.neighbors = [other for other in environment.population
                          if self.can_see(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = [obstacle for obstacle in environment.obstacles
                          if self.can_see(obstacle)]

    def decide(self):
        raise NotImplementedError()



class Boid(Agent):
    """Boid agent"""
    config = {
        "cohesion": 0.2,
        "separation": 2,
        "alignment": 0.2,
        "obstacle_avoidance": 2,
        "goal_steering": 0.5,
        "neighbor_interaction_mode": "avg"
    }

    def _cohesion(self):
        """Boids try to fly towards the center of neighbors."""
        if not self.neighbors:
            return np.zeros(self._ndim)

        center = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            center += neighbor.position
        center /= len(self.neighbors)

        return center - self.position

    def _seperation(self):
        """Boids try to keep a small distance away from other objects."""
        repel = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            distance = self.distance(neighbor)
            if distance < self.size:
                # Divergence protection.
                if distance < 0.01:
                    distance = 0.01

                repel += (self.position - neighbor.position) / \
                    distance / distance
                # No averaging taken place.
                # When two neighbors are in the same position, a stronger urge
                # to move away is assumed, despite that distancing itself from
                # one neighbor automatically eludes the other.

        # Whether the avg or sum of the influence from neighbors is used.
        if self.config["neighbor_interaction_mode"] == 'avg' and self.neighbors:
            repel /= len(self.neighbors)
        return repel

    def _alignment(self):
        """Boids try to match velocity with neighboring boids."""
        # If no neighbors, no change.
        if not self.neighbors:
            return np.zeros(self._ndim)

        avg_velocity = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity /= len(self.neighbors)

        return avg_velocity - self.velocity

    def _obstacle_avoidance(self):
        """Boids try to avoid obstacles."""
        # NOTE: Assume there is always enough space between obstacles
        # Find the nearest obstacle in the front.
        min_distance = np.inf
        closest = -1
        for i, obstacle in enumerate(self.obstacles):
            distance = obstacle.distance(self.position)
            if (np.dot(-obstacle.direction(self.position), self.velocity) > 0  # In the front
                    and distance < min_distance):
                closest, min_distance = i, distance

        # No obstacles in front.
        if closest < 0:
            return np.zeros(self.ndim)

        obstacle = self.obstacles[closest]
        # normal distance of obstacle to velocity, note that min_distance is obstacle's distance
        obstacle_direction = -obstacle.direction(self.position)
        v_direction = self.velocity / self.speed
        sin_theta = np.linalg.norm(
            np.cross(v_direction, obstacle_direction))
        normal_distance = (min_distance + obstacle.size) * \
            sin_theta - obstacle.size
        # Decide if self is on course of collision.
        if normal_distance < self.size:
            # normal direction away from obstacle
            cos_theta = np.sqrt(1 - sin_theta * sin_theta)
            turn_direction = v_direction * cos_theta - obstacle_direction
            turn_direction = turn_direction / np.linalg.norm(turn_direction)
            # Stronger the obstrution, stronger the turn.
            return turn_direction * (self.size - normal_distance)**2 / max(min_distance, self.size)

        # Return 0 if obstacle does not obstruct.
        return np.zeros(self.ndim)

    def _goal_seeking(self):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not self.goal:
            return self.velocity / self.speed

        # The urge to chase the goal is stronger when farther.
        offset = self.goal.position - self.position
        distance = np.linalg.norm(offset)
        target_speed = self.max_speed * min(1, distance / 20)
        target_velocity = target_speed * offset / distance
        return target_velocity - self.velocity

    def decide(self):
        """Make decision for acceleration."""
        self.acceleration = (self.config['cohesion'] * self._cohesion() +
                             self.config['separation'] * self._seperation() +
                             self.config['alignment'] * self._alignment() +
                             self.config['obstacle_avoidance'] * self._obstacle_avoidance() +
                             self.config['goal_steering'] * self._goal_seeking())

    @classmethod
    def set_model(cls, config):
        # Throw out unmatched keys.
        config = {k: v for k, v in config.items() if k in cls.config}
        cls.config.update(config)


class Environment2D:
    """Environment that contains the population of boids, goals and obstacles."""

    def __init__(self, boundary):
        self.population = []
        self.goals = []

        xmin, xmax, ymin, ymax = boundary
        self.boundaries = [Wall((1, 0), (xmin, 0), ndim=2),
                           Wall((-1, 0), (xmax, 0), ndim=2),
                           Wall((0, 1), (0, ymin), ndim=2),
                           Wall((0, -1), (0, ymax), ndim=2)]

        self._obstacles = []

    @property
    def obstacles(self):
        return self.boundaries + self._obstacles

    def add_agent(self, agent):
        if not isinstance(agent, Agent):
            raise ValueError('agent must be an instance of Agent')

        if agent.ndim != 2:
            raise ValueError('position space of agent must be 2D')
        self.population.append(agent)

    def add_goal(self, goal):
        if not isinstance(goal, Goal):
            raise ValueError('goal must be an instance of Goal')
        if goal.ndim != 2:
            raise ValueError('position space of goal must be 2D')
        self.goals.append(goal)

    def add_obstacle(self, obstacle):
        if not isinstance(obstacle, Obstacle):
            raise ValueError('obstacle must be an instance of Obstacle')
        if obstacle.ndim != 2:
            raise ValueError('position space of obstacle must be 2D')
        self._obstacles.append(obstacle)

    def _move_agents(self, dt):
        for agent in self.population:
            agent.observe(self)
            agent.decide()

        # Hold off moving agents until all have made decision.
        # This ensures synchronous update.
        for agent in self.population:
            agent.move(dt)

    def _move_goals(self, dt):
        for goal in self.goals:
            goal.move(dt)

    def _move_obstacles(self, dt):
        for obstacle in self.obstacles:
            obstacle.move(dt)

    def update(self, dt):
        """
        Update the state of environment for one time step dt, during which the
        boids move.
        """
        self._move_agents(dt)
        self._move_goals(dt)
        self._move_obstacles(dt)
