#!/usr/bin/env python3
"""
Parse grid search logs and generate plots.

Usage:
    python scripts/parse_and_plot_grid_search.py --log-file <log_file> --output-dir <output_dir>

Examples:
    python scripts/parse_and_plot_grid_search.py \
        --log-file results/grid_search3/grid_search.log \
        --output-dir results/plots
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast


# Apply the custom style with extended color/linestyle cycles
mpl.style.use("/home/dev10x/open-truco2/scripts/style.mplstyle")


def parse_log_file(log_file):
    """Parse the grid search log file and extract relevant data."""
    data = []
    experiment = None

    with open(log_file, "r") as f:
        for line in f:
            # Detect new experiment. Logs no longer include epsilon_decay; parse only hidden_layers and anticipatory.
            if "Testing configuration:" in line:
                # Accept patterns like: hidden_layers=(16, 16), anticipatory=0.1
                match = re.search(r"hidden_layers=\((\d+\s*,\s*\d+)\)\s*,\s*anticipatory=(\d+(?:\.\d+)?)", line)
                if match:
                    hidden_layers = match.group(1)
                    # Skip (32, 32) configurations
                    if "32" in hidden_layers:
                        experiment = None
                        continue
                    experiment = {
                        "hidden_layers": hidden_layers,
                        "anticipatory": float(match.group(2)),
                        "episodes": [],
                        "losses": [],
                        "win_rates": {},
                    }
                    data.append(experiment)

            # Extract episode losses
            elif "Episode" in line and "Losses:" in line:
                match = re.search(r"Episode (\d+)/\d+ \| Losses: \[(.+)\]", line)
                if match and experiment:
                    episode = int(match.group(1))
                    losses = parse_loss_string(match.group(2))  # Use the new function to parse losses
                    experiment["episodes"].append(episode)
                    experiment["losses"].append(losses)

            # Extract win rates
            elif "Win rate" in line and experiment:
                match = re.search(r"Win rate (.+): (\d+\.\d+)%", line)
                if match:
                    baseline = match.group(1)
                    win_rate = float(match.group(2))
                    if baseline not in experiment["win_rates"]:
                        experiment["win_rates"][baseline] = []
                    experiment["win_rates"][baseline].append((experiment["episodes"][-1], win_rate))

    return data


def parse_loss_string(loss_string):
    """Safely parse the loss string and extract numerical values."""
    try:
        # Replace 'Array' with a placeholder to safely evaluate the string
        loss_string = loss_string.replace("Array", "")
        loss_string = loss_string.replace("dtype=float32", "")
        return ast.literal_eval(loss_string)
    except Exception as e:
        print(f"Error parsing loss string: {loss_string}")
        return []


def plot_loss_curves(data, output_dir):
    """Plot loss curves for each experiment."""
    max_episode = 10000
    for experiment in data:
        plt.figure()
        for i, agent_losses in enumerate(zip(*experiment["losses"])):
            # Flatten and filter out None values
            agent_losses = [loss[0] if isinstance(loss, tuple) and loss[0] is not None else None for loss in agent_losses]
            # Filter episodes to match valid losses and cut at max_episode
            valid_episodes = [experiment["episodes"][j] for j, loss in enumerate(agent_losses) if loss is not None and experiment["episodes"][j] <= max_episode]
            agent_losses = [loss for j, loss in enumerate(agent_losses) if loss is not None and experiment["episodes"][j] <= max_episode]

            if agent_losses:  # Only plot if there are valid losses
                plt.plot(valid_episodes, agent_losses, label=f"Agent {i}")
        plt.xlabel("Episodios")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"loss_curve_{experiment['hidden_layers']}_{experiment['anticipatory']}.pdf"))
        plt.close()


def plot_win_rate_curves(data, output_dir):
    """Plot win rate curves against each baseline for each experiment."""
    max_episode = 10000
    for experiment in data:
        plt.figure()
        for baseline, win_rates in experiment["win_rates"].items():
            episodes, rates = zip(*win_rates)
            # Cut at max_episode
            filtered = [(e, r) for e, r in zip(episodes, rates) if e <= max_episode]
            if filtered:
                episodes, rates = zip(*filtered)
                plt.plot(episodes, rates, label=baseline)
        # Draw horizontal red dotted line at 50%
        plt.axhline(y=50, color='red', linestyle=':', linewidth=1)
        plt.xlabel("Episodios")
        plt.ylabel("Tasa de victorias (%)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"win_rate_curve_{experiment['hidden_layers']}_{experiment['anticipatory']}.pdf"))
        plt.close()


def plot_grouped_loss_curves(data, output_dir):
    """Plot grouped loss curves for all experiments, averaging losses across agents."""
    max_episode = 10000
    plt.figure()
    for experiment in data:
        # Collect losses from all agents and average them
        all_agent_losses = []
        for i, agent_losses in enumerate(zip(*experiment["losses"])):
            # Extract numeric values from nested tuples: ((value,), None) -> value
            processed_losses = []
            for loss in agent_losses:
                if isinstance(loss, tuple) and loss[0] is not None:
                    val = loss[0]
                    # Handle nested tuple like (3.14,)
                    while isinstance(val, tuple):
                        val = val[0]
                    processed_losses.append(float(val))
                else:
                    processed_losses.append(None)
            all_agent_losses.append(processed_losses)
        
        # Average losses across agents for each episode
        avg_losses = []
        valid_episodes = []
        for j, episode in enumerate(experiment["episodes"]):
            if episode > max_episode:
                continue
            losses_at_j = [agent_losses[j] for agent_losses in all_agent_losses if j < len(agent_losses) and agent_losses[j] is not None]
            if losses_at_j:
                avg_losses.append(sum(losses_at_j) / len(losses_at_j))
                valid_episodes.append(episode)
        
        if avg_losses:
            label = f"({experiment['hidden_layers']}), a={experiment['anticipatory']}"
            plt.plot(valid_episodes, avg_losses, label=label)
    plt.xlabel("Episodios")
    plt.ylabel("Loss")
    plt.legend(loc='upper right', fontsize=7, ncol=2)
    plt.savefig(os.path.join(output_dir, "grouped_loss_curves.pdf"))
    plt.close()


def plot_grouped_win_rate_curves(data, output_dir):
    """Plot grouped win rate curves for all experiments."""
    # Create two grouped plots: one for baselines containing 'random', another for baselines containing 'ismcts'
    groups = {
        "random": [],
        "ismcts": [],
    }

    # Collect series per group: list of (episodes, rates, label)
    max_episode = 10000
    for experiment in data:
        short_label = f"({experiment['hidden_layers']}), a={experiment['anticipatory']}"
        for baseline, win_rates in experiment["win_rates"].items():
            episodes, rates = zip(*win_rates)
            # Cut at max_episode
            filtered = [(e, r) for e, r in zip(episodes, rates) if e <= max_episode]
            if not filtered:
                continue
            episodes, rates = zip(*filtered)
            if "random" in baseline:
                groups["random"].append((episodes, rates, short_label))
            elif "ismcts" in baseline:
                groups["ismcts"].append((episodes, rates, short_label))

    # Plot each group separately
    for group_name, series_list in groups.items():
        if not series_list:
            continue
        plt.figure()
        for episodes, rates, label in series_list:
            plt.plot(episodes, rates, label=label)
        # Draw horizontal red dotted line at 50%
        plt.axhline(y=50, color='red', linestyle=':', linewidth=1, label='EV')
        plt.xlabel("Episodios")
        plt.ylabel("Tasa de victorias (%)")
        plt.ylim(0, 100)
        plt.legend(loc='upper left', fontsize=7, ncol=2)
        out_name = f"grouped_win_rate_curves_vs_{group_name}.pdf"
        plt.savefig(os.path.join(output_dir, out_name))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse grid search logs and generate plots.")
    parser.add_argument("--log-file", type=str, required=True, help="Path to the grid search log file.")
    parser.add_argument("--output-dir", type=str, help="Directory to save the plots. Defaults to the log file's directory.")
    args = parser.parse_args()

    # Ensure output directory exists
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse log file
    data = parse_log_file(args.log_file)

    # Generate plots
    plot_loss_curves(data, args.output_dir)
    plot_win_rate_curves(data, args.output_dir)
    # Generate grouped plots
    plot_grouped_loss_curves(data, args.output_dir)
    plot_grouped_win_rate_curves(data, args.output_dir)


if __name__ == "__main__":
    main()