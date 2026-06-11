"""
Simulates the microcanonical ensemble conditions (fixed energy, volume and number of particles) considering their collision with boundaries and each other.

Parameters:
test = {'delta_time': 0.01, -> time step of the simulation
        'time': 10, -> total simulated time
        'framerate': 40, -> framerate of the animation
        'boundary': [[0, 5], [0, 1], [0, 2.5]], -> container's walls in x, y and z axis, respectively, with minimum and maximum values
        'number_of_particles': 20, -> number of particles simulated
        'position_initialization': [0, 2], -> minimum and maximum randomly generated positions (if randomized_position is true) or
                                              the set with the x, y and z components of the position of each particle (if randomized_position is false)
        'randomized_position': True, 
        'velocity_initialization': [1, 2], -> minimum and maximum randomly generated velocities (if randomized_velocity is true) or
                                              the set with the x, y and z components of the velocity of each particle (if randomized_velocity is false)
        'randomized_velocity': True, 
        'mass_initialization': [1, 2], -> minimum and maximum randomly generated masses (if randomized_mass is true) or
                                              the set with the mass of each particle (if randomized_mass is false)
        'randomized_mass': True, 
        'radius_initialization': [0.05, 0.1], -> minimum and maximum randomly generated radii (if randomized_radius is true) or
                                              the set with the radius of each particle (if randomized_radius is false)
        'randomized_radius': True}

Returns:
- An animation of the particles with the respective total kinetic energy graph (see microcanonical_ensemble.gif).
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from numbers import Real
from math import floor

class Particle():
    
    def __init__(self, *, position: np.ndarray, velocity: np.ndarray, mass: Real, radius: Real) -> None:
        
        self.position, self.velocity, self.mass, self.radius = np.asarray(position), np.asarray(velocity), float(mass), float(radius)

        self.check_datatype()
        self.check_dimensions()
        self.check_negatives()

    def check_datatype(self) -> None:
        if not np.issubdtype(self.position.dtype, np.number):
            raise TypeError('Position components must be numeric.')
        if not np.issubdtype(self.velocity.dtype, np.number):
            raise TypeError('Velocity components must be numeric.')
    
    def check_dimensions(self) -> None:
        if self.position.shape != (3,):
            raise ValueError('Position needs to be a 3-component numpy array.')
        if self.velocity.shape != (3,):
            raise ValueError('Velocity needs to be a 3-component numpy array.')
    
    def check_negatives(self) -> None:
        if self.mass <= 0.0:
            raise ValueError('Mass needs to be positive.')
        if self.radius <= 0.0:
            raise ValueError('Radius needs to be positive.')


class System():
    
    def __init__(self, *, boundary: list, number_of_particles: int, position_set: list, velocity_set: list, mass_set: list, radius_set: list) -> None:

        self.boundary, self.number_of_particles = boundary, number_of_particles
        
        self.check_datatype()
        self.check_dimensions()

        self.particles = [Particle(position=position_set[particle], 
                                   velocity=velocity_set[particle], 
                                   mass=mass_set[particle], 
                                   radius=radius_set[particle]) 
                                   for particle in range(number_of_particles)]

    def movement(self, *, position_set: np.ndarray, velocity_set: np.ndarray, delta_time: float) -> None:
        position_set += velocity_set * delta_time
        for i, particle in enumerate(self.particles):
            particle.position = position_set[i]
            particle.velocity = velocity_set[i]

    def bounds(self, *, position_set: np.ndarray, velocity_set: np.ndarray, radius_set: np.ndarray) -> None:
        x = position_set[:, 0]
        y = position_set[:, 1]
        z = position_set[:, 2]
        x_bounded = (((x - radius_set) < self.boundary[0][0]) | ((x + radius_set) > self.boundary[0][1]))
        y_bounded = (((y - radius_set) < self.boundary[1][0]) | ((y + radius_set) > self.boundary[1][1]))
        z_bounded = (((z - radius_set) < self.boundary[2][0]) | ((z + radius_set) > self.boundary[2][1]))
        position_set[:, 0] = np.clip(position_set[:, 0], self.boundary[0][0] + radius_set, self.boundary[0][1] - radius_set)
        position_set[:, 1] = np.clip(position_set[:, 1], self.boundary[1][0] + radius_set, self.boundary[1][1] - radius_set)
        position_set[:, 2] = np.clip(position_set[:, 2], self.boundary[2][0] + radius_set, self.boundary[2][1] - radius_set)
        velocity_set[:, 0] = np.where(x_bounded, -velocity_set[:, 0], velocity_set[:, 0])
        velocity_set[:, 1] = np.where(y_bounded, -velocity_set[:, 1], velocity_set[:, 1])
        velocity_set[:, 2] = np.where(z_bounded, -velocity_set[:, 2], velocity_set[:, 2])
        return position_set, velocity_set

    def collision(self, *, position_set: np.ndarray, velocity_set: np.ndarray, mass_set: np.ndarray, radius_set: np.ndarray) -> None:
        for particle in range(self.number_of_particles):
            for second_particle in range(particle + 1, self.number_of_particles):
                distance = np.linalg.norm(position_set[particle] - position_set[second_particle])
                if ((distance > 1e-12) and (distance <= (radius_set[particle] + radius_set[second_particle]))):
                        collision_axis = (position_set[particle] - position_set[second_particle])/distance
                        collision_v1 = (velocity_set[particle].dot(collision_axis)) * collision_axis
                        collision_v2 = (velocity_set[second_particle].dot(collision_axis)) * collision_axis
                        m1 = mass_set[particle]
                        m2 = mass_set[second_particle]
                        velocity_set[particle] += (m1-m2)/(m1+m2) * collision_v1 + 2*m2/(m1+m2) * collision_v2 - collision_v1
                        velocity_set[second_particle] += (m2-m1)/(m1+m2) * collision_v2 + 2*m1/(m1+m2) * collision_v1 - collision_v2

                        overlap = max((radius_set[particle] + radius_set[second_particle]) - distance, 0.0)
                        position_set[particle] += collision_axis * overlap/2 + 1e-6
                        position_set[second_particle] -= collision_axis * overlap/2 + 1e-6
        return position_set, velocity_set

    def check_datatype(self) -> None:
        if not isinstance(self.boundary, list):
            raise TypeError('Boundary must be a 3x2 matrix.')
        if not isinstance(self.number_of_particles, int):
            raise TypeError('Number of particles must be an integer.')
        flatted_boundary = np.array(self.boundary).flatten()
        for item in flatted_boundary:
            if not isinstance(item, Real):
                raise TypeError('Boundary components must be numeric.')

    def check_dimensions(self) -> None:
        if len(self.boundary) != 3:
            raise ValueError('Boundary must be a 3x2 matrix.')
        for boundary in self.boundary:
            if len(boundary) != 2:
                raise ValueError('Boundary must be a 3x2 matrix.')


class Simulation():
    
    def __init__(self, *, delta_time: float, time: float, framerate: float, boundary: list, number_of_particles: int, 
                 position_initialization: list, randomized_position: bool = False, 
                 velocity_initialization: list, randomized_velocity: bool = False, 
                 mass_initialization: list, randomized_mass: bool = False, 
                 radius_initialization: list, randomized_radius: bool = False) -> None:

        self.delta_time, self.time, self.framerate = delta_time, time, framerate

        if randomized_velocity:
            self.check_dimensions(velocity_initialization)
            random_repeatable = np.random.rand(number_of_particles, 3)
            velocity_set = random_repeatable*(velocity_initialization[1]-velocity_initialization[0]) + velocity_initialization[0]
        else:
            velocity_set = velocity_initialization

        if randomized_mass:
            self.check_dimensions(mass_initialization)
            random_repeatable = np.random.rand(number_of_particles)
            mass_set = random_repeatable*(mass_initialization[1]-mass_initialization[0]) + mass_initialization[0]
        else:
            mass_set = mass_initialization

        if randomized_radius:
            self.check_dimensions(radius_initialization)
            random_repeatable = np.random.rand(number_of_particles)
            radius_set = random_repeatable*(radius_initialization[1]-radius_initialization[0]) + radius_initialization[0]
        else:
            radius_set = radius_initialization

        if randomized_position:
            self.check_dimensions(position_initialization)
            random_unique = self.random_unique(amount=number_of_particles, scale=position_initialization, size=radius_set)
            position_set = random_unique*(position_initialization[1]-position_initialization[0]) + position_initialization[0]
        else:
            position_set = position_initialization

        self.system = System(boundary=boundary, number_of_particles=number_of_particles, position_set=position_set, velocity_set=velocity_set, mass_set=mass_set, radius_set=radius_set)

    def increment(self, *, position_set: np.ndarray, velocity_set: np.ndarray, mass_set: np.ndarray, radius_set: np.ndarray) -> None:
        self.system.movement(position_set=position_set, velocity_set=velocity_set, delta_time=self.delta_time)
        position_set, velocity_set = self.system.bounds(position_set=position_set, velocity_set=velocity_set, radius_set=radius_set)
        position_set, velocity_set = self.system.collision(position_set=position_set, velocity_set=velocity_set, mass_set=mass_set, radius_set=radius_set)

    def simulate(self) -> np.ndarray:
        time = 0
        position_data = []
        velocity_data = []
        time_data = []
        
        mass_set = np.asarray([particle.mass for particle in self.system.particles])
        radius_set = np.asarray([particle.radius for particle in self.system.particles])
        
        while time < self.time:
            position_set = np.array([particle.position for particle in self.system.particles])
            velocity_set = np.array([particle.velocity for particle in self.system.particles])
            
            position_data.append(position_set.copy())
            velocity_data.append(velocity_set.copy())
            time_data.append(time)

            self.increment(position_set=position_set, velocity_set=velocity_set, mass_set=mass_set, radius_set=radius_set)
            time += self.delta_time

        data = {'position': position_data,
                'velocity': velocity_data,
                'mass': mass_set,
                'radius': radius_set,
                'time': time_data}
        return data

    def animate(self, data: np.ndarray) -> None:
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(1, 3, figure=fig)

        ax1 = fig.add_subplot(gs[:, :2], projection='3d')
        ax1.set_xlim(self.system.boundary[0][0], self.system.boundary[0][1])
        ax1.set_ylim(self.system.boundary[1][0], self.system.boundary[1][1])
        ax1.set_zlim(self.system.boundary[2][0], self.system.boundary[2][1])
        scat1 = ax1.scatter([], [], [], color='red')

        ax2 = fig.add_subplot(gs[0, 2])
        time = data.get('time')
        kinetic_energy = [0.5 * np.sum(data.get('mass') * norm(data.get('velocity')[step], axis=1)**2) for step in range(len(data.get('velocity')))]
        line, = ax2.plot([], [], color='red')

        plt.tight_layout()

        def update(frame):
            position_set = data.get('position')[frame]
            x = position_set[:, 0]
            y = position_set[:, 1]
            z = position_set[:, 2]
            scat1._offsets3d = (x, y, z)

            line.set_data(time[:frame], kinetic_energy[:frame])
            ax2.relim()
            ax2.autoscale_view()

        self.anim = FuncAnimation(fig=fig, func=update, frames=floor(self.time/self.delta_time), interval=1/self.framerate, repeat=False)
        plt.show()

    @staticmethod
    def random_unique(*, amount, scale, size):
        vectors = np.empty((0, 3))
        while len(vectors) < amount:
            vector = np.random.rand(3)
            vector = vector*(scale[1]-scale[0]) + scale[0]
            distances = norm(vectors - vector, axis=1)
            minimal_distance = size[:len(vectors)] + size[len(vectors)]
            if (distances > minimal_distance).all():
                vectors = np.vstack((vectors, vector))
                clock = 0
            else:
                clock += 1
            if clock > 1000:
                raise RuntimeError('1000 vectors were successively tried without success. Please reconsider the input parameters.')
        return vectors

    @staticmethod
    def check_dimensions(initialization_vector):
        if len(initialization_vector) != 2:
            raise ValueError('For randomized simulations the initialization vector must contain only two components.')


test = {'delta_time': 0.01,
        'time': 10,
        'framerate': 40,
        'boundary': [[0, 5], [0, 1], [0, 2.5]],
        'number_of_particles': 20, 
        'position_initialization': [0, 2],
        'randomized_position': True, 
        'velocity_initialization': [1, 2],
        'randomized_velocity': True, 
        'mass_initialization': [1, 2],
        'randomized_mass': True, 
        'radius_initialization': [0.05, 0.1],
        'randomized_radius': True}

hydrogen_gas = {'delta_time': 1.0e-5,
        'time': 1.0e-2,
        'framerate': 40,
        'boundary': [[0, 1], [0, 1], [0, 1]],
        'number_of_particles': 400, 
        'position_initialization': [0, 0.5],
        'randomized_position': True, 
        'velocity_initialization': [2.4e3, 3.0e3],
        'randomized_velocity': True, 
        'mass_initialization': [2.0e-27, 2.0e-27],
        'randomized_mass': True, 
        'radius_initialization': [1.2e-10, 1.2e-10],
        'randomized_radius': True}

if __name__ == '__main__': 
    simulation = Simulation(**test)
    data = simulation.simulate()
    simulation.animate(data)
