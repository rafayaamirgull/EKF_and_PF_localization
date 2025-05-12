""" Written by Rafay Aamir and Ghaith Chamaa for VIBOT M1: Probabilistic Robotics (Summer 2025)
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action (control input)
        z: landmark observation (bearing)
        marker_id: landmark ID
        """

        mu_bar = env.forward(self.mu, u)

        G = env.G(self.mu, u)
        V = env.V(self.mu, u)

        M = env.noise_from_motion(u, self.alphas)

        sigma_bar = G @ self.sigma @ G.T + V@M@V.T

        z_bar = env.observe(mu_bar, marker_id)

        H = env.H(mu_bar, marker_id)

        Q = self.beta

        S = H @ sigma_bar @ H.T + Q

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Singular matrix in Kalman Filter update. Using identity for S_inv.")
            S_inv = np.eye(S.shape[0])

        K = sigma_bar @ H.T @ S_inv

        innovation = z - z_bar
        innovation[0, 0] = minimized_angle(innovation[0, 0])
        
        self.mu = mu_bar + K*innovation[0][0]

        I = np.eye(self.sigma.shape[0])
        self.sigma = (I - K @ H) @ sigma_bar

        self.sigma = (self.sigma + self.sigma.T) / 2

        return self.mu, self.sigma

