""" Written by Rafay Aamir and Ghaith Chamaa for VIBOT M1: Probabilistic Robotics (Summer 2025)
"""

import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            # Sample initial particles from a Gaussian around the initial mean and covariance
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles # Initialize weights to be uniform

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action (control input)
        z: landmark observation (bearing)
        marker_id: landmark ID
        """
        new_particles = np.zeros_like(self.particles)
        for i in range(self.num_particles):
            noisy_u = env.sample_noisy_action(u, self.alphas)
            new_particles[i, :] = env.forward(self.particles[i, :].reshape((-1, 1)), noisy_u).ravel()
            new_particles[i, 2] = minimized_angle(new_particles[i, 2])

        self.particles = new_particles

        for i in range(self.num_particles):
            predicted_z = env.observe(self.particles[i, :].reshape((-1, 1)), marker_id)
            innovation = z - predicted_z
            innovation[0, 0] = minimized_angle(innovation[0, 0])

            likelihood = env.likelihood(innovation, self.beta)

            self.weights[i] *= likelihood

        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            print("Warning: All particle weights are zero. Resetting to uniform weights.")
            self.weights = np.ones(self.num_particles) / self.num_particles

        self.particles, self.weights = self.resample(self.particles, self.weights)

        self.weights = np.ones(self.num_particles) / self.num_particles

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights (normalized)
        """
        n = self.num_particles
        new_particles = np.zeros_like(particles)
        new_weights = np.ones(n) / n

        r = np.random.uniform(0, 1.0 / n)

        c = weights[0]
        i = 0

        for m in range(n):
            U = r + m * (1.0 / n)

            while U > c:
                i += 1
                c += weights[i]

            new_particles[m, :] = particles[i, :]

        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean_x = particles[:, 0].mean()
        mean_y = particles[:, 1].mean()
        mean_theta = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )
        mean = np.array([mean_x, mean_y, mean_theta]).reshape((-1, 1))

        zero_mean_particles = particles - mean.ravel()
        for i in range(zero_mean_particles.shape[0]):
            zero_mean_particles[i, 2] = minimized_angle(zero_mean_particles[i, 2])

        cov = np.dot(zero_mean_particles.T, zero_mean_particles) / self.num_particles

        return mean, cov

