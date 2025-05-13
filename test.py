""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
    Modified for batch experiments and median plotting.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Assuming utils.py, soccer_field.py, policies.py, ekf.py, pf.py are in the same directory or accessible
from utils import minimized_angle, plot_field, plot_robot, plot_path
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter


def localize(env, policy, filt, x0, num_steps, plot=False):
    # Collect data from an entire rollout
    states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real = \
            env.rollout(x0, policy, num_steps)
    
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    component_errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        # Suppress a warning about too many figures if plotting is on for many trials
        # This is more relevant if plot=True was used in a loop, but good practice.
        if plt.get_fignums():
            fig = plt.figure(plt.get_fignums()[0]) # Try to reuse existing figure
        else:
            fig = env.get_figure()


    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        
        states_filter[i+1, :] = mean.ravel()

        if plot:
            fig.clear()
            plot_field(env, marker_id)
            plot_robot(env, x_real, z_real)
            plot_path(env, states_noisefree[:i+2, :], 'g', 0.5)
            plot_path(env, states_real[:i+2, :], 'b')
            if filt is not None:
                plot_path(env, states_filter[:i+2, :2], 'r')
            fig.canvas.flush_events()
            # plt.pause(0.01) 

        current_error = (mean - x_real).ravel()
        current_error[2] = minimized_angle(current_error[2])
        component_errors[i, :] = current_error
        position_errors[i] = np.linalg.norm(current_error[:2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            # print(f'Warning: Covariance matrix badly conditioned (cond={cond_number:.2e}) at step {i}. Using identity.')
            effective_cov = np.eye(3)
        else:
            effective_cov = cov
        
        try:
            inv_cov = np.linalg.inv(effective_cov)
            mahalanobis_errors[i] = component_errors[i:i+1, :].dot(inv_cov).dot(component_errors[i:i+1, :].T)
        except np.linalg.LinAlgError:
            # print(f"Warning: Could not invert covariance matrix at step {i}. Setting Mahalanobis error to NaN.")
            mahalanobis_errors[i] = np.nan

    mean_position_error = np.nanmean(position_errors)
    mean_mahalanobis_error = np.nanmean(mahalanobis_errors)
    anees = mean_mahalanobis_error / 3.0 # Assuming 3 DOF (x, y, theta)

    # Suppress print statements during batch runs unless specifically debugging
    # if filt is not None and not plot: # only print summary if not live plotting
    #     print(f'    Run Summary: Pos Err: {mean_position_error:.3f}, Mah Err: {mean_mahalanobis_error:.3f}, ANEES: {anees:.3f}')


    if plot:
        plt.show(block=True)

    localization_results = {
        'timestamps': np.arange(num_steps),
        'states_real': states_real,
        'states_filter': states_filter,
        'states_noisefree': states_noisefree,
        'component_errors': component_errors,
        'position_errors': position_errors,
        'mahalanobis_errors': mahalanobis_errors,
        'mean_position_error': mean_position_error,
        'mean_mahalanobis_error': mean_mahalanobis_error,
        'anees': anees
    }
    
    return localization_results


def run_experiment_for_r(r_value, base_alphas, base_beta, policy_instance, 
                         filter_type_str, num_steps, num_particles_pf, 
                         initial_mean_val, initial_cov_val, num_trials, 
                         base_seed=None, trial_idx_offset=0):
    """
    Runs num_trials for a given r_value applying r_value to data and filter noise.
    Returns arrays of mean errors and ANEES from these trials.
    """
    trial_position_errors = []
    trial_mahalanobis_errors = []
    trial_anees_values = []

    print(f"\n--- Running {num_trials} trials for r = {r_value:.4f} (Filter: {filter_type_str}) ---")

    for trial_idx in range(num_trials):
        if base_seed is not None:
            current_seed = base_seed + trial_idx_offset + trial_idx
            np.random.seed(current_seed)
        else:
            # If no base_seed, ensure np.random is re-seeded for some level of trial-to-trial variation
            # This relies on np.random.seed() pulling from system entropy or similar.
            np.random.seed() 

        # Environment noise scaled by r_value (acting as data_factor)
        env_alphas = r_value * base_alphas
        env_beta = r_value * base_beta
        env = Field(env_alphas, env_beta)
        
        # Filter noise model scaled by r_value (acting as filter_factor)
        filt_alphas = r_value * base_alphas
        filt_beta = r_value * base_beta
        
        filt = None
        if filter_type_str == 'ekf':
            filt = ExtendedKalmanFilter(
                initial_mean_val, initial_cov_val,
                filt_alphas, filt_beta
            )
        elif filter_type_str == 'pf':
            filt = ParticleFilter(
                initial_mean_val, initial_cov_val, num_particles_pf,
                filt_alphas, filt_beta
            )
        elif filter_type_str == 'none':
            filt = None

        # Run localization (plot=False for batch runs)
        results = localize(env, policy_instance, filt, initial_mean_val, num_steps, plot=False)

        trial_position_errors.append(results['mean_position_error'])
        if filter_type_str != 'none':
            trial_mahalanobis_errors.append(results['mean_mahalanobis_error'])
            trial_anees_values.append(results['anees'])
        else:
            trial_mahalanobis_errors.append(0.0) # For 'none', error is 0
            trial_anees_values.append(0.0)       # For 'none', ANEES is 0

        if (trial_idx + 1) % (num_trials // min(num_trials, 5)) == 0 or trial_idx == num_trials -1 : # Print progress
             print(f"  Trial {trial_idx+1}/{num_trials} done. Last Pos Err: {results['mean_position_error']:.3f}")


    return np.array(trial_position_errors), np.array(trial_mahalanobis_errors), np.array(trial_anees_values)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filter_type', choices=('none', 'ekf', 'pf'),
        help='filter to use for localization')
    parser.add_argument(
        '--live-plot', action='store_true', # Renamed from --plot to avoid confusion
        help='turn on live plotting for a single run (not batch experiment)')
    parser.add_argument(
        '--seed', type=int,
        help='random seed for reproducible experiments')
    parser.add_argument(
        '--num-steps', type=int, default=200,
        help='timesteps to simulate per trial')
    parser.add_argument(
        '--num-trials', type=int, default=10,
        help='number of trials per r_value for batch experiment')
    # These factors are NOT used in the r-value experiment directly, 
    # as r_value itself becomes the factor.
    # They are kept for compatibility if running a single default sim.
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='(single run) scaling factor for motion and obs noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='(single run) scaling factor for motion and obs noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles (particle filter only)')
    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()

    if args.seed is not None:
        print(f"Using base random seed: {args.seed}")
        np.random.seed(args.seed) # Seed the main RNG once, sub-seeds handled per trial

    # Base noise parameters (unscaled)
    base_alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2]) # Motion noise
    base_beta = np.diag([np.deg2rad(5)**2])                   # Observation noise (bearing)

    # Initial conditions
    initial_mean = np.array([180, 50, 0]).reshape((-1, 1))      # cm, cm, rad
    initial_cov = np.diag([10**2, 10**2, np.deg2rad(10)**2]) # (cm^2, cm^2, rad^2)

    # Policy
    policy = policies.OpenLoopRectanglePolicy() # Assuming one policy for all experiments

    # --- Batch Experiment Setup ---
    # r values as specified, with 1.0 added as a reference
    r_values = np.array(sorted(list(set([1/64, 1/16, 1/4, 1.0, 4, 16, 64]))))
    num_trials_per_r = args.num_trials

    median_position_errors_vs_r = []
    median_mahalanobis_errors_vs_r = []
    median_anees_vs_r = []
    
    print(f"Starting batch experiment for filter: {args.filter_type.upper()}")
    print(f"r values to test: {r_values}")
    print(f"Number of trials per r: {num_trials_per_r}")
    print(f"Number of simulation steps per trial: {args.num_steps}")
    if args.filter_type == 'pf':
        print(f"Number of particles (PF): {args.num_particles}")

    for r_idx, r_val in enumerate(r_values):
        trial_idx_offset_for_r = 0
        if args.seed is not None:
            # Ensure distinct seed sequences for each r_value block if a base seed is used
            trial_idx_offset_for_r = r_idx * num_trials_per_r 
            
        # Run trials for the current r_val
        pos_errs, mah_errs, anees_vals = run_experiment_for_r(
            r_value=r_val,
            base_alphas=base_alphas,
            base_beta=base_beta,
            policy_instance=policy,
            filter_type_str=args.filter_type,
            num_steps=args.num_steps,
            num_particles_pf=args.num_particles,
            initial_mean_val=initial_mean,
            initial_cov_val=initial_cov,
            num_trials=num_trials_per_r,
            base_seed=args.seed, # Pass the main seed if provided
            trial_idx_offset=trial_idx_offset_for_r
        )

        # Calculate and store medians
        median_position_errors_vs_r.append(np.median(pos_errs))
        if args.filter_type != 'none':
            # Filter out NaNs before taking median, in case any trial had an issue
            valid_mah_errs = mah_errs[~np.isnan(mah_errs)]
            median_mahalanobis_errors_vs_r.append(np.median(valid_mah_errs) if len(valid_mah_errs) > 0 else np.nan)
            
            valid_anees_vals = anees_vals[~np.isnan(anees_vals)]
            median_anees_vs_r.append(np.median(valid_anees_vals) if len(valid_anees_vals) > 0 else np.nan)
        else:
            median_mahalanobis_errors_vs_r.append(np.median(mah_errs)) # Will be 0.0 for 'none'
            median_anees_vs_r.append(np.median(anees_vals))            # Will be 0.0 for 'none'

    # --- Plotting Median Results ---
    fig_summary, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig_summary.suptitle(f"Median Error Metrics vs. Factor r (Filter: {args.filter_type.upper()}, {num_trials_per_r} trials/r)", fontsize=14)

    # Plot Median Position Error
    axs[0].plot(r_values, median_position_errors_vs_r, marker='o', linestyle='-')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylabel("Median of Mean Pos. Error")
    axs[0].set_title("Position Error")
    axs[0].grid(True, which="both", ls="--")

    # Plot Median Mahalanobis Error
    if args.filter_type != 'none':
        axs[1].plot(r_values, median_mahalanobis_errors_vs_r, marker='o', linestyle='-')
    else: # For 'none', error is 0
        axs[1].plot(r_values, np.zeros_like(r_values), marker='o', linestyle='-')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log') # Typically log, but if all zeros for 'none', may need adjustment or linear.
    axs[1].set_ylabel("Median of Mean Mahalanobis Err.")
    axs[1].set_title("Mahalanobis Error")
    axs[1].grid(True, which="both", ls="--")

    # Plot Median ANEES
    if args.filter_type != 'none':
        axs[2].plot(r_values, median_anees_vs_r, marker='o', linestyle='-')
        axs[2].axhline(1.0, color='k', linestyle=':', linewidth=0.8, label='Ideal ANEES (for 3 DOF)')
        axs[2].legend()
    else: # For 'none', ANEES is 0
        axs[2].plot(r_values, np.zeros_like(r_values), marker='o', linestyle='-')
    axs[2].set_xscale('log')
    # ANEES y-scale can be linear if values are centered around 1, or log if they vary widely.
    # Let's try linear first for ANEES, as ideal is 1.0
    min_anees = np.nanmin(median_anees_vs_r) if args.filter_type != 'none' and len(median_anees_vs_r)>0 and not all(np.isnan(m) for m in median_anees_vs_r) else 0
    max_anees = np.nanmax(median_anees_vs_r) if args.filter_type != 'none' and len(median_anees_vs_r)>0 and not all(np.isnan(m) for m in median_anees_vs_r) else 1
    if max_anees / (min_anees + 1e-9) > 10 and min_anees > 0 : # If ANEES varies by more than an order of magnitude
         axs[2].set_yscale('log')
    axs[2].set_ylabel("Median ANEES")
    axs[2].set_title("ANEES (Average Normalized Estimation Error Squared)")
    axs[2].grid(True, which="both", ls="--")
    
    axs[2].set_xlabel("Factor r (scales data noise and filter model noise)")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

    print("\n--- Experiment Summary ---")
    print(f"Filter: {args.filter_type.upper()}")
    print("r values:", r_values)
    print("Median Position Errors:", [f"{x:.3e}" for x in median_position_errors_vs_r])
    if args.filter_type != 'none':
        print("Median Mahalanobis Errors:", [f"{x:.3e}" for x in median_mahalanobis_errors_vs_r])
        print("Median ANEES values:", [f"{x:.3f}" for x in median_anees_vs_r])
    
    # If user specified --live-plot, run one instance with live plotting
    # This is separate from the batch experiment.
    if args.live_plot:
        print("\n--- Running a single instance with live plotting ---")
        print(f"Using data_factor: {args.data_factor}, filter_factor: {args.filter_factor}")
        env = Field(args.data_factor * base_alphas, args.data_factor * base_beta)
        filt_single = None
        if args.filter_type == 'ekf':
            filt_single = ExtendedKalmanFilter(initial_mean, initial_cov, 
                                               args.filter_factor * base_alphas, args.filter_factor * base_beta)
        elif args.filter_type == 'pf':
            filt_single = ParticleFilter(initial_mean, initial_cov, args.num_particles,
                                         args.filter_factor * base_alphas, args.filter_factor * base_beta)
        
        localize(env, policy, filt_single, initial_mean, args.num_steps, plot=True)