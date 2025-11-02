#!/usr/bin/env python3
"""
Comprehensive plotter script for visualizing training statistics from RL durotaxis runs.

This script provides visualization capabilities for both spawn parameter evolution and reward
component analysis from training runs. It reads JSON statistics files and generates professional
plots with error bands, statistical summaries, and multiple visualization modes.

FEATURES:
    • Spawn Parameter Evolution: gamma, alpha, noise, theta parameters with ±1σ error bands
    • Reward Components Analysis: graph, spawn, delete, edge, total node, and total rewards
    • Combined Normalized Views: overlay plots for parameter comparison
    • Interactive Display: matplotlib interactive viewing
    • High-Resolution Output: publication-ready PNG files (300 DPI)
    • Statistical Summaries: mean, std, min, max displayed on each subplot

USAGE EXAMPLES:
    # Basic spawn parameter plotting
    python plotter.py --input training_results/run0002
    
    # Add combined normalized overlay plot
    python plotter.py --input training_results/run0002 --combined
    
    # Add reward components evolution plot
    python plotter.py --input training_results/run0002 --rewards
    
    # Add loss evolution plot
    python plotter.py --input training_results/run0002 --loss
    
    # Generate all available plots
    python plotter.py --input training_results/run0002 --combined --rewards --loss
    
    # Interactive viewing (opens matplotlib windows)
    python plotter.py --input training_results/run0002 --show --combined --rewards --loss
    
    # Custom output directory
    python plotter.py --input training_results/run0002 --output ./analysis_plots --combined --rewards --loss
    
    # Direct JSON file input
    python plotter.py --input training_results/run0002/spawn_parameters_stats.json --rewards

COMMAND LINE ARGUMENTS:
    --input, -i         Path to input data. Can be:
                        • A run directory (e.g., training_results/run0002)
                        • A spawn_parameters_stats.json file directly
                        • If omitted, defaults to training_results/run0002/
                        
    --output, -o        Output directory for generated PNG plots.
                        • Defaults to same directory as input JSON files
                        • Directory is created automatically if it doesn't exist
                        
    --show              Show plots interactively using matplotlib.
                        • Opens plot windows for interactive exploration
                        • Useful for zooming, panning, and detailed inspection
                        • Warning: May require X11 forwarding in SSH environments
                        
    --combined          Generate combined normalized parameter plot.
                        • Overlays all spawn parameters (gamma, alpha, noise, theta)
                        • Normalizes to [0,1] scale for direct comparison
                        • Includes proportionally scaled error bands
                        • Useful for identifying parameter relationships and trends
                        
    --rewards           Generate reward components evolution plot.
                        • Requires reward_components_stats.json in the same directory
                        • Plots all 6 reward components: graph, spawn, delete, edge, total_node, total
                        • 3x2 subplot grid with ±1σ error bands
                        • Zero reference lines for positive/negative reward identification
                        
    --loss              Generate loss evolution plot.
                        • Requires loss_metrics.json in the same directory
                        • Plots both raw loss and smoothed loss over training episodes
                        • Episodes are batch endpoints (e.g., 10, 20, 30 for batch size 10)
                        • Automatic log scaling for large loss ranges
                        • Statistical summary with improvement percentage

OUTPUT FILES:
    spawn_parameters_evolution_[run].png    Individual parameter plots (always generated)
    spawn_parameters_combined_[run].png     Normalized overlay plot (--combined flag)
    reward_components_[run].png             Reward evolution plot (--rewards flag)
    loss_evolution_[run].png                Loss evolution plot (--loss flag)

INPUT FILE FORMATS:
    spawn_parameters_stats.json:
        [{"episode": 0, "parameters": {"gamma": {"mean": 6.95, "std": 1.62, ...}, ...}}, ...]
    
    reward_components_stats.json:
        [{"episode": 0, "reward_components": {"total_reward": {"mean": -12.1, "std": 14.3, ...}, ...}}, ...]
    
    loss_metrics.json:
        [{"episode": 10, "loss": 16560.71, "smoothed_loss": 16560.71}, ...]

"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def load_spawn_stats(filepath):
    """Load spawn parameter statistics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_parameter_means(data):
    """Extract parameter means and standard deviations for each episode from the loaded data."""
    episodes = []
    gamma_means = []
    alpha_means = []
    noise_means = []
    theta_means = []
    gamma_stds = []
    alpha_stds = []
    noise_stds = []
    theta_stds = []
    
    for episode_data in data:
        episodes.append(episode_data['episode'])
        
        # Extract mean and std values for each parameter (handle None/null values)
        params = episode_data['parameters']
        gamma_means.append(params['gamma']['mean'] if params['gamma']['mean'] is not None else 0.0)
        alpha_means.append(params['alpha']['mean'] if params['alpha']['mean'] is not None else 0.0)
        noise_means.append(params['noise']['mean'] if params['noise']['mean'] is not None else 0.0)
        theta_means.append(params['theta']['mean'] if params['theta']['mean'] is not None else 0.0)
        
        gamma_stds.append(params['gamma']['std'] if params['gamma']['std'] is not None else 0.0)
        alpha_stds.append(params['alpha']['std'] if params['alpha']['std'] is not None else 0.0)
        noise_stds.append(params['noise']['std'] if params['noise']['std'] is not None else 0.0)
        theta_stds.append(params['theta']['std'] if params['theta']['std'] is not None else 0.0)
    
    return (episodes, gamma_means, alpha_means, noise_means, theta_means,
            gamma_stds, alpha_stds, noise_stds, theta_stds)


def create_spawn_parameter_plots(episodes, gamma_means, alpha_means, noise_means, theta_means,
                                gamma_stds, alpha_stds, noise_stds, theta_stds, 
                                title_suffix="", save_path=None):
    """Create and display/save plots for spawn parameter means with standard deviation bands."""
    
    # Convert to numpy arrays for easier math
    episodes = np.array(episodes)
    gamma_means = np.array(gamma_means)
    alpha_means = np.array(alpha_means)
    noise_means = np.array(noise_means)
    theta_means = np.array(theta_means)
    gamma_stds = np.array(gamma_stds)
    alpha_stds = np.array(alpha_stds)
    noise_stds = np.array(noise_stds)
    theta_stds = np.array(theta_stds)
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Spawn Parameter Evolution Across Episodes{title_suffix}', fontsize=16, fontweight='bold')
    
    # Colors for each parameter
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot Gamma with std fill
    axes[0, 0].plot(episodes, gamma_means, 'o-', color=colors[0], linewidth=2, markersize=5, alpha=0.8, label='Mean')
    axes[0, 0].fill_between(episodes, gamma_means - gamma_stds, gamma_means + gamma_stds, 
                           color=colors[0], alpha=0.2, label='±1 std')
    axes[0, 0].set_title('Gamma Mean ± Std per Episode', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Gamma Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    
    # Plot Alpha with std fill
    axes[0, 1].plot(episodes, alpha_means, 'o-', color=colors[1], linewidth=2, markersize=5, alpha=0.8, label='Mean')
    axes[0, 1].fill_between(episodes, alpha_means - alpha_stds, alpha_means + alpha_stds,
                           color=colors[1], alpha=0.2, label='±1 std')
    axes[0, 1].set_title('Alpha Mean ± Std per Episode', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Alpha Value')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].legend(fontsize=9)
    
    # Plot Noise with std fill
    axes[1, 0].plot(episodes, noise_means, 'o-', color=colors[2], linewidth=2, markersize=5, alpha=0.8, label='Mean')
    axes[1, 0].fill_between(episodes, noise_means - noise_stds, noise_means + noise_stds,
                           color=colors[2], alpha=0.2, label='±1 std')
    axes[1, 0].set_title('Noise Mean ± Std per Episode', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Noise Value')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].legend(fontsize=9)
    
    # Plot Theta with std fill
    axes[1, 1].plot(episodes, theta_means, 'o-', color=colors[3], linewidth=2, markersize=5, alpha=0.8, label='Mean')
    axes[1, 1].fill_between(episodes, theta_means - theta_stds, theta_means + theta_stds,
                           color=colors[3], alpha=0.2, label='±1 std')
    axes[1, 1].set_title('Theta Mean ± Std per Episode', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Theta Value')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].legend(fontsize=9)
    
    # Add statistics text boxes
    for i, (param_name, values, ax) in enumerate([
        ('Gamma', gamma_means, axes[0, 0]),
        ('Alpha', alpha_means, axes[0, 1]),
        ('Noise', noise_means, axes[1, 0]),
        ('Theta', theta_means, axes[1, 1])
    ]):
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def create_combined_plot(episodes, gamma_means, alpha_means, noise_means, theta_means,
                        gamma_stds, alpha_stds, noise_stds, theta_stds,
                        title_suffix="", save_path=None):
    """Create a single plot with all parameters normalized and overlaid with error bands."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to numpy arrays
    episodes = np.array(episodes)
    gamma_means = np.array(gamma_means)
    alpha_means = np.array(alpha_means)
    noise_means = np.array(noise_means)
    theta_means = np.array(theta_means)
    gamma_stds = np.array(gamma_stds)
    alpha_stds = np.array(alpha_stds)
    noise_stds = np.array(noise_stds)
    theta_stds = np.array(theta_stds)
    
    # Normalize all parameters to [0, 1] for comparison
    def normalize(values):
        values = np.array(values)
        return (values - np.min(values)) / (np.max(values) - np.min(values)) if np.max(values) != np.min(values) else values
    
    def normalize_std(values, stds):
        """Normalize standard deviations according to the same scale as the means."""
        values = np.array(values)
        stds = np.array(stds)
        value_range = np.max(values) - np.min(values)
        if value_range != 0:
            return stds / value_range
        else:
            return np.zeros_like(stds)
    
    gamma_norm = normalize(gamma_means)
    gamma_std_norm = normalize_std(gamma_means, gamma_stds)
    alpha_norm = normalize(alpha_means)
    alpha_std_norm = normalize_std(alpha_means, alpha_stds)
    noise_norm = normalize(noise_means)
    noise_std_norm = normalize_std(noise_means, noise_stds)
    theta_norm = normalize(theta_means)
    theta_std_norm = normalize_std(theta_means, theta_stds)
    
    # Plot all parameters with error bands
    ax.plot(episodes, gamma_norm, 'o-', label='Gamma (normalized)', linewidth=2, markersize=4, alpha=0.8)
    ax.fill_between(episodes, gamma_norm - gamma_std_norm, gamma_norm + gamma_std_norm, alpha=0.2)
    
    ax.plot(episodes, alpha_norm, 's-', label='Alpha (normalized)', linewidth=2, markersize=4, alpha=0.8)
    ax.fill_between(episodes, alpha_norm - alpha_std_norm, alpha_norm + alpha_std_norm, alpha=0.2)
    
    ax.plot(episodes, noise_norm, '^-', label='Noise (normalized)', linewidth=2, markersize=4, alpha=0.8)
    ax.fill_between(episodes, noise_norm - noise_std_norm, noise_norm + noise_std_norm, alpha=0.2)
    
    ax.plot(episodes, theta_norm, 'd-', label='Theta (normalized)', linewidth=2, markersize=4, alpha=0.8)
    ax.fill_between(episodes, theta_norm - theta_std_norm, theta_norm + theta_std_norm, alpha=0.2)
    
    ax.set_title(f'Normalized Spawn Parameters Evolution{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Normalized Parameter Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    
    return fig


def load_reward_stats(filepath):
    """Load reward component statistics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_loss_metrics(filepath):
    """Load loss metrics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_loss_metrics(data):
    """Extract loss metrics for each episode from the loaded data."""
    episodes = []
    losses = []
    smoothed_losses = []
    
    for entry in data:
        episodes.append(entry['episode'])
        losses.append(entry['loss'])
        smoothed_losses.append(entry['smoothed_loss'])
    
    return episodes, losses, smoothed_losses


def create_loss_plot(episodes, losses, smoothed_losses, title_suffix="", save_path=None):
    """Create and display/save loss evolution plot."""
    
    # Convert to numpy arrays
    episodes = np.array(episodes)
    losses = np.array(losses)
    smoothed_losses = np.array(smoothed_losses)
    
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for loss components
    loss_color = '#d62728'
    smoothed_color = '#1f77b4'
    
    # Plot raw loss
    ax.plot(episodes, losses, 'o-', color=loss_color, linewidth=2, markersize=5, alpha=0.7, label='Raw Loss')
    
    # Plot smoothed loss
    ax.plot(episodes, smoothed_losses, 's-', color=smoothed_color, linewidth=3, markersize=6, alpha=0.9, label='Smoothed Loss')
    
    ax.set_title(f'Training Loss Evolution{title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode (Batch End)', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Use log scale if loss values span multiple orders of magnitude
    if np.max(losses) / np.min(losses) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Loss Value (log scale)', fontsize=12)
    
    # Add statistics text box
    stats_text = (
        f'Raw Loss Statistics:\n'
        f'  Mean: {np.mean(losses):.2e}\n'
        f'  Std: {np.std(losses):.2e}\n'
        f'  Min: {np.min(losses):.2e}\n'
        f'  Max: {np.max(losses):.2e}\n'
        f'  Final: {losses[-1]:.2e}\n\n'
        f'Smoothed Loss Statistics:\n'
        f'  Mean: {np.mean(smoothed_losses):.2e}\n'
        f'  Final: {smoothed_losses[-1]:.2e}\n'
        f'  Improvement: {((smoothed_losses[0] - smoothed_losses[-1]) / smoothed_losses[0] * 100):.1f}%'
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")
    
    return fig


def extract_reward_components(data):
    """Extract reward component means and standard deviations for each episode.
    
    DELETE RATIO ARCHITECTURE: Now tracks 5 components:
    - graph_reward (alias for centroid movement)
    - spawn_reward
    - delete_reward  
    - distance_signal
    - total_reward
    """
    episodes = []
    
    # Reward components (DELETE RATIO ARCHITECTURE)
    graph_reward_means = []
    spawn_reward_means = []
    delete_reward_means = []
    distance_signal_means = []
    total_reward_means = []
    
    # Standard deviations
    graph_reward_stds = []
    spawn_reward_stds = []
    delete_reward_stds = []
    distance_signal_stds = []
    total_reward_stds = []
    
    for episode_data in data:
        episodes.append(episode_data['episode'])
        
        # Extract mean values for each reward component (handle None/null values)
        rewards = episode_data['reward_components']
        graph_reward_means.append(rewards.get('graph_reward', {}).get('mean', 0.0) or 0.0)
        spawn_reward_means.append(rewards.get('spawn_reward', {}).get('mean', 0.0) or 0.0)
        delete_reward_means.append(rewards.get('delete_reward', {}).get('mean', 0.0) or 0.0)
        distance_signal_means.append(rewards.get('distance_signal', {}).get('mean', 0.0) or 0.0)
        total_reward_means.append(rewards.get('total_reward', {}).get('mean', 0.0) or 0.0)
        
        # Extract std values for each reward component (handle None/null values)
        graph_reward_stds.append(rewards.get('graph_reward', {}).get('std', 0.0) or 0.0)
        spawn_reward_stds.append(rewards.get('spawn_reward', {}).get('std', 0.0) or 0.0)
        delete_reward_stds.append(rewards.get('delete_reward', {}).get('std', 0.0) or 0.0)
        distance_signal_stds.append(rewards.get('distance_signal', {}).get('std', 0.0) or 0.0)
        total_reward_stds.append(rewards.get('total_reward', {}).get('std', 0.0) or 0.0)
    
    return (episodes, 
            graph_reward_means, spawn_reward_means, delete_reward_means, 
            distance_signal_means, total_reward_means,
            graph_reward_stds, spawn_reward_stds, delete_reward_stds,
            distance_signal_stds, total_reward_stds)


def create_reward_components_plot(episodes, graph_reward_means, spawn_reward_means, delete_reward_means,
                                 distance_signal_means, total_reward_means,
                                 graph_reward_stds, spawn_reward_stds, delete_reward_stds,
                                 distance_signal_stds, total_reward_stds,
                                 title_suffix="", save_path=None):
    """Create and display/save plots for reward components with standard deviation bands.
    
    DELETE RATIO ARCHITECTURE: Plots 5 reward components in a 3x2 grid:
    - graph_reward (centroid movement)
    - spawn_reward
    - delete_reward
    - distance_signal
    - total_reward
    """
    
    # Convert to numpy arrays for easier math
    episodes = np.array(episodes)
    
    # Means (DELETE RATIO ARCHITECTURE)
    graph_reward_means = np.array(graph_reward_means)
    spawn_reward_means = np.array(spawn_reward_means)
    delete_reward_means = np.array(delete_reward_means)
    distance_signal_means = np.array(distance_signal_means)
    total_reward_means = np.array(total_reward_means)
    
    # Standard deviations
    graph_reward_stds = np.array(graph_reward_stds)
    spawn_reward_stds = np.array(spawn_reward_stds)
    delete_reward_stds = np.array(delete_reward_stds)
    distance_signal_stds = np.array(distance_signal_stds)
    total_reward_stds = np.array(total_reward_stds)
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Reward Components Evolution - Delete Ratio Architecture{title_suffix}', fontsize=16, fontweight='bold')
    
    # Colors for each reward component (DELETE RATIO ARCHITECTURE: 5 components)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    
    # Plot Graph Reward (centroid movement)
    axes[0, 0].plot(episodes, graph_reward_means, 'o-', color=colors[0], linewidth=2, markersize=4, alpha=0.8, label='Mean')
    axes[0, 0].fill_between(episodes, graph_reward_means - graph_reward_stds, graph_reward_means + graph_reward_stds, 
                           color=colors[0], alpha=0.2, label='±1 std')
    axes[0, 0].set_title('Graph Reward (Centroid Movement) per Episode', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Graph Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot Spawn Reward
    axes[0, 1].plot(episodes, spawn_reward_means, 'o-', color=colors[1], linewidth=2, markersize=4, alpha=0.8, label='Mean')
    axes[0, 1].fill_between(episodes, spawn_reward_means - spawn_reward_stds, spawn_reward_means + spawn_reward_stds,
                           color=colors[1], alpha=0.2, label='±1 std')
    axes[0, 1].set_title('Spawn Reward per Episode', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Spawn Reward')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot Delete Reward
    axes[1, 0].plot(episodes, delete_reward_means, 'o-', color=colors[2], linewidth=2, markersize=4, alpha=0.8, label='Mean')
    axes[1, 0].fill_between(episodes, delete_reward_means - delete_reward_stds, delete_reward_means + delete_reward_stds,
                           color=colors[2], alpha=0.2, label='±1 std')
    axes[1, 0].set_title('Delete Reward per Episode', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Delete Reward')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot Distance Signal
    axes[1, 1].plot(episodes, distance_signal_means, 'o-', color=colors[3], linewidth=2, markersize=4, alpha=0.8, label='Mean')
    axes[1, 1].fill_between(episodes, distance_signal_means - distance_signal_stds, distance_signal_means + distance_signal_stds,
                           color=colors[3], alpha=0.2, label='±1 std')
    axes[1, 1].set_title('Distance Signal per Episode', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Distance Signal')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot Total Reward
    axes[2, 0].plot(episodes, total_reward_means, 'o-', color=colors[4], linewidth=2, markersize=4, alpha=0.8, label='Mean')
    axes[2, 0].fill_between(episodes, total_reward_means - total_reward_stds, total_reward_means + total_reward_stds,
                           color=colors[4], alpha=0.2, label='±1 std')
    axes[2, 0].set_title('Total Reward per Episode', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Total Reward')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Hide unused subplot (we have 5 components, not 6)
    axes[2, 1].axis('off')
    
    # Add statistics text boxes
    reward_data = [
        ('Graph', graph_reward_means, axes[0, 0]),
        ('Spawn', spawn_reward_means, axes[0, 1]),
        ('Delete', delete_reward_means, axes[1, 0]),
        ('Distance', distance_signal_means, axes[1, 1]),
        ('Total', total_reward_means, axes[2, 0])
    ]
    
    for param_name, values, ax in reward_data:
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reward components plot saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot spawn parameter statistics from training runs')
    parser.add_argument('--input', '-i', type=str, 
                       help='Path to spawn_parameters_stats.json file or run directory')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output directory for plots (default: same as input)')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots interactively')
    parser.add_argument('--combined', action='store_true',
                       help='Also create a combined normalized plot')
    parser.add_argument('--rewards', action='store_true',
                       help='Also create reward components plot from reward_components_stats.json')
    parser.add_argument('--loss', action='store_true',
                       help='Also create loss evolution plot from loss_metrics.json')
    
    args = parser.parse_args()
    
    # Determine input file path
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            # Look for spawn_parameters_stats.json in the directory
            json_file = input_path / 'spawn_parameters_stats.json'
            reward_json_file = input_path / 'reward_components_stats.json'
            loss_json_file = input_path / 'loss_metrics.json'
        else:
            json_file = input_path
            # Try to find reward and loss files in same directory
            reward_json_file = input_path.parent / 'reward_components_stats.json'
            loss_json_file = input_path.parent / 'loss_metrics.json'
    else:
        # Default to run0002 for demonstration
        json_file = Path('training_results/run0002/spawn_parameters_stats.json')
        reward_json_file = Path('training_results/run0002/reward_components_stats.json')
        loss_json_file = Path('training_results/run0002/loss_metrics.json')
    
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        return
    
    # Load and process data
    print(f"Loading spawn parameter data from: {json_file}")
    data = load_spawn_stats(json_file)
    (episodes, gamma_means, alpha_means, noise_means, theta_means,
     gamma_stds, alpha_stds, noise_stds, theta_stds) = extract_parameter_means(data)
    
    print(f"Loaded data for {len(episodes)} episodes (Episodes {min(episodes)} to {max(episodes)})")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = json_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine title suffix from run directory
    run_name = json_file.parent.name
    title_suffix = f" - {run_name}" if run_name.startswith('run') else ""
    
    # Create main parameter plots
    save_path = output_dir / f'spawn_parameters_evolution{f"_{run_name}" if run_name.startswith("run") else ""}.png'
    fig1 = create_spawn_parameter_plots(episodes, gamma_means, alpha_means, noise_means, theta_means,
                                        gamma_stds, alpha_stds, noise_stds, theta_stds,
                                        title_suffix, save_path)
    
    # Create combined plot if requested
    if args.combined:
        combined_save_path = output_dir / f'spawn_parameters_combined{f"_{run_name}" if run_name.startswith("run") else ""}.png'
        fig2 = create_combined_plot(episodes, gamma_means, alpha_means, noise_means, theta_means,
                                   gamma_stds, alpha_stds, noise_stds, theta_stds,
                                   title_suffix, combined_save_path)
    
    # Create reward components plot if requested
    if args.rewards:
        if reward_json_file.exists():
            print(f"Loading reward component data from: {reward_json_file}")
            reward_data = load_reward_stats(reward_json_file)
            (reward_episodes, graph_reward_means, spawn_reward_means, delete_reward_means,
             distance_signal_means, total_reward_means,
             graph_reward_stds, spawn_reward_stds, delete_reward_stds,
             distance_signal_stds, total_reward_stds) = extract_reward_components(reward_data)
            
            print(f"Loaded reward data for {len(reward_episodes)} episodes")
            reward_save_path = output_dir / f'reward_components{f"_{run_name}" if run_name.startswith("run") else ""}.png'
            fig3 = create_reward_components_plot(reward_episodes, graph_reward_means, spawn_reward_means, delete_reward_means,
                                               distance_signal_means, total_reward_means,
                                               graph_reward_stds, spawn_reward_stds, delete_reward_stds,
                                               distance_signal_stds, total_reward_stds,
                                               title_suffix, reward_save_path)
        else:
            print(f"Warning: Reward components file not found: {reward_json_file}")
            print("Skipping reward components plot. Use --rewards only when reward_components_stats.json is available.")
    
    # Create loss evolution plot if requested
    if args.loss:
        if loss_json_file.exists():
            print(f"Loading loss metrics data from: {loss_json_file}")
            loss_data = load_loss_metrics(loss_json_file)
            loss_episodes, losses, smoothed_losses = extract_loss_metrics(loss_data)
            
            print(f"Loaded loss data for {len(loss_episodes)} batch endpoints (Episodes {min(loss_episodes)} to {max(loss_episodes)})")
            loss_save_path = output_dir / f'loss_evolution{f"_{run_name}" if run_name.startswith("run") else ""}.png'
            fig4 = create_loss_plot(loss_episodes, losses, smoothed_losses, title_suffix, loss_save_path)
        else:
            print(f"Warning: Loss metrics file not found: {loss_json_file}")
            print("Skipping loss evolution plot. Use --loss only when loss_metrics.json is available.")
    
    # Show plots if requested
    if args.show:
        plt.show()
    
    print("Plotting completed!")


if __name__ == "__main__":
    main()