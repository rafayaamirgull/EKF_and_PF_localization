"""Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
Modified by Rafay Aamir and Ghaith Chamaa for VIBOT M1: Probabilistic Robotics (Summer 2025)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time  # Import time for unique seeds

from utils import minimized_angle, plot_field, plot_robot, plot_path
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter


def localize(env, policy, filt, x0, num_steps, plot=False):
    # Collect data from an entire rollout
    states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real = (
        env.rollout(x0, policy, num_steps)
    )  # Generate all data (actual observations with and without noise) using ground-truth (action wihtout noise)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        fig = env.get_figure()
        plt.show(block=False)  # Use non-blocking show for animation

    for i in range(num_steps):
        x_real = states_real[i + 1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i + 1, :] = mean.ravel()

        if plot:
            fig.clear()
            plot_field(env, marker_id)
            plot_robot(env, x_real, z_real)
            plot_path(
                env, states_noisefree[: i + 2, :], "g", 0.5
            )  # Es lo deseado pero NO es el ground-truth
            plot_path(
                env, states_real[: i + 2, :], "b"
            )  # Es la pose real del robot (ground-truth)
            if filt is not None:
                plot_path(env, states_filter[: i + 2, :2], "r")
            fig.canvas.flush_events()
            plt.pause(0.01)  # Pause to allow plot to update

        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            # print('Badly conditioned cov (setting to identity):', cond_number)
            # print(cov)
            cov = np.eye(3)
        # Calculate Mahalanobis error, handling potential division by zero or negative values in inverse
        try:
            mahalanobis_errors[i] = (
                errors[i : i + 1, :].dot(np.linalg.inv(cov)).dot(errors[i : i + 1, :].T)
            )
        except np.linalg.LinAlgError:
            # If covariance is singular or not positive definite, Mahalanobis error is undefined.
            # Assign a large value or skip, depending on desired behavior.
            # Assigning NaN allows filtering them out later.
            mahalanobis_errors[i] = np.nan

    # Filter out NaN Mahalanobis errors before calculating mean
    valid_mahalanobis_errors = mahalanobis_errors[~np.isnan(mahalanobis_errors)]

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = (
        np.mean(valid_mahalanobis_errors)
        if valid_mahalanobis_errors.size > 0
        else np.nan
    )
    anees = (
        mean_mahalanobis_error / 3 if not np.isnan(mean_mahalanobis_error) else np.nan
    )

    if filt is not None:
        print("-" * 80)
        print("Mean position error:", mean_position_error)
        print("Mean Mahalanobis error:", mean_mahalanobis_error)
        print("ANEES:", anees)

    if plot:
        plt.show(block=True)  # Keep plot window open at the end

    # Return position_errors and mahalanobis_errors arrays
    return position_errors, mahalanobis_errors


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filter_type",
        choices=("none", "ekf", "pf"),
        help="filter to use for localization",
    )
    parser.add_argument("--plot", action="store_true", help="turn on plotting")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--num-steps", type=int, default=200, help="timesteps to simulate"
    )

    # Noise scaling factors
    parser.add_argument(
        "--data-factor",
        type=float,
        default=1,
        help="scaling factor for motion and observation noise (data)",
    )
    parser.add_argument(
        "--filter-factor",
        type=float,
        default=1,
        help="scaling factor for motion and observation noise (filter)",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=100,
        help="number of particles (particle filter only)",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    # Define the range of filter factors for the EKF experiment (Exercise 3c)
    ekf_filter_factors = np.array([1 / 64, 1 / 16, 1 / 4, 4, 16, 64])
    num_trials = 10  # Number of trials per filter factor for experiments

    # Lists to store average results for experiments
    avg_mean_pos_errors = []
    avg_anees_values = []

    if args.filter_type == "ekf" and not args.plot:
        # Run EKF experiment (Exercise 3c) if filter_type is ekf and plot is False
        print(
            f"Running EKF experiment with varying filter factors ({ekf_filter_factors}) over {num_trials} trials each."
        )
        data_factor = 1  # Data is generated with default noise

        for filter_factor in ekf_filter_factors:
            print(f"\nRunning trials for filter_factor = {filter_factor}")
            trial_mean_pos_errors = []
            trial_anees_values = []

            for trial in range(num_trials):
                # Use a unique seed for each trial based on current time and trial number
                seed = (
                    int(time.time()) + trial if args.seed is None else args.seed + trial
                )
                np.random.seed(seed)
                print(f"  Trial {trial + 1}/{num_trials} with seed {seed}")

                # Initialize environment and filter for the current trial
                alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
                beta = np.diag([np.deg2rad(5) ** 2])

                env = Field(data_factor * alphas, data_factor * beta)
                policy = policies.OpenLoopRectanglePolicy()

                initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
                initial_cov = np.diag([10, 10, 1])

                filt = ExtendedKalmanFilter(
                    initial_mean,
                    initial_cov,
                    filter_factor * alphas,  # Varying filter noise
                    filter_factor * beta,
                )

                # Run localization and get results (mean error and ANEES for experiment)
                mean_pos_error, anees = localize(
                    env, policy, filt, initial_mean, args.num_steps, plot=False
                )  # Ensure plot is False for experiment speed

                trial_mean_pos_errors.append(mean_pos_error)
                trial_anees_values.append(anees)

            # Calculate average results for the current filter factor
            avg_mean_pos_errors.append(np.mean(trial_mean_pos_errors))
            # Filter out NaN ANEES values before calculating mean
            valid_anees = np.array(trial_anees_values)[~np.isnan(trial_anees_values)]
            avg_anees_values.append(
                np.mean(valid_anees) if valid_anees.size > 0 else np.nan
            )

        # Plot the results of the EKF experiment
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.plot(ekf_filter_factors, avg_mean_pos_errors, marker="o")
        plt.xscale("log")  # Use log scale for the filter factors
        plt.xlabel("Filter Noise Factor (log scale)")
        plt.ylabel("Average Mean Position Error")
        plt.title("EKF: Mean Position Error vs. Filter Noise Factor (Experiment 3c)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # Filter out NaN ANEES values for plotting
        valid_filter_factors = ekf_filter_factors[~np.isnan(avg_anees_values)]
        valid_avg_anees = np.array(avg_anees_values)[~np.isnan(avg_anees_values)]
        plt.plot(valid_filter_factors, valid_avg_anees, marker="o", color="orange")
        plt.xscale("log")  # Use log scale for the filter factors
        plt.xlabel("Filter Noise Factor (log scale)")
        plt.ylabel("Average ANEES")
        plt.title("EKF: ANEES vs. Filter Noise Factor (Experiment 3c)")
        plt.grid(True)

        plt.tight_layout()
        plt.show(block=True)  # Keep plot window open

    else:
        # Original behavior for single run or other filter types
        print("Data factor:", args.data_factor)
        print("Filter factor:", args.filter_factor)

        if args.seed is not None:
            np.random.seed(args.seed)

        alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
        beta = np.diag([np.deg2rad(5) ** 2])

        env = Field(args.data_factor * alphas, args.data_factor * beta)
        policy = policies.OpenLoopRectanglePolicy()

        initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
        initial_cov = np.diag([10, 10, 1])

        filt = None
        if args.filter_type == "ekf":
            filt = ExtendedKalmanFilter(
                initial_mean,
                initial_cov,
                args.filter_factor * alphas,
                args.filter_factor * beta,
            )
        elif args.filter_type == "pf":
            filt = ParticleFilter(
                initial_mean,
                initial_cov,
                args.num_particles,
                args.filter_factor * alphas,
                args.filter_factor * beta,
            )

        # Run single localization simulation and get detailed errors
        pos_errors_over_time, mahalanobis_errors_over_time = localize(
            env, policy, filt, initial_mean, args.num_steps, args.plot
        )

        # Plot position_errors and mahalanobis_errors over time for a single run
        if filt is not None:  # Only plot errors over time if a filter was used
            time_steps = np.arange(args.num_steps)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(time_steps, pos_errors_over_time)
            plt.xlabel("Time Step")
            plt.ylabel("Position Error")
            plt.title(f"{args.filter_type.upper()}: Position Error Over Time")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            # Filter out NaN Mahalanobis errors for plotting
            valid_time_steps = time_steps[~np.isnan(mahalanobis_errors_over_time)]
            valid_mahalanobis_errors = mahalanobis_errors_over_time[
                ~np.isnan(mahalanobis_errors_over_time)
            ]
            plt.plot(valid_time_steps, valid_mahalanobis_errors, color="orange")
            plt.xlabel("Time Step")
            plt.ylabel("Mahalanobis Error")
            plt.title(f"{args.filter_type.upper()}: Mahalanobis Error Over Time")
            plt.grid(True)

            plt.tight_layout()
            plt.show(block=True)
