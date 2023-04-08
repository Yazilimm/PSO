#PSO MİN





import numpy as np
import random as rnd
import matplotlib.pyplot as plt


def create_particle(position_min, position_max):
    return [[rnd.uniform(position_min, position_max)] for i in range(9)]


def fitness_function(value):
    return np.sin(value) + np.sin((10 / 3) * value)


def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
    num_particle = len(particle)
    new_velocity = np.array([0 for i in range(num_particle)])
    r1 = rnd.uniform(0, max)
    r2 = rnd.uniform(0, max)
    w = rnd.uniform(w_min, max)
    c1 = c
    c2 = c

    for i in range(num_particle):
        new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])
        return new_velocity


def update_position(particle, velocity):
    new_particle = particle + velocity
    return new_particle


def pso(num_particle=9, position_min=-10, position_max=10, iterasyon=100):
    particles = create_particle(position_min, position_max)

    pbest_position = particles

    pbest_fitness = [fitness_function(p[0]) for p in particles]

    gbest_index = np.argmax(pbest_fitness)

    gbest_position = pbest_position[gbest_index]

    velocity = [[0] for i in range(num_particle)]



    for i in range(0, iterasyon):

        for n in range(num_particle):
            velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)

            particles[n] = update_position(particles[n], velocity[n])

        pbest_fitness = [fitness_function(p[0]) for p in particles]

        gbest_index = np.argmin(pbest_fitness)

        gbest_position = pbest_position[gbest_index]

        def grafik():
            iterasyon = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            plt.xlabel('Parcacik')
            plt.ylabel('Fitness')
            plt.title('Fitness-Iterasyon Grafigi')
            plt.plot(iterasyon, pbest_fitness)
            plt.show()

        grafik()

        print('Global Best Position: ', gbest_position)
        print('Best Fitness Value: ', min(pbest_fitness))

    iterasyon += 1



if __name__ == '__main__':
    pso()

