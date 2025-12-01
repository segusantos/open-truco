#!/usr/bin/env python3
"""
Parse pkl checkpoints and generate plots.

Usage:
    python scripts/plot_from_checkpoints.py --checkpoints <pkl1> <pkl2> ... --output-dir <output_dir>

Examples:
    python scripts/plot_from_checkpoints.py \
        --checkpoints checkpoints/nfsp/final_60.pkl checkpoints/nfsp/final_128.pkl \
        --output-dir results/plots
"""

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl


# Apply the custom style with extended color/linestyle cycles
mpl.style.use("/home/dev10x/open-truco2/scripts/style.mplstyle")


def parse_checkpoint(checkpoint_path):
    """Parse a pkl checkpoint and extract relevant data."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    
    config = checkpoint['config']
    metrics_history = checkpoint['metrics_history']
    
    # Extract hidden layers size for label
    hidden_layers = config.get('hidden_layers_sizes', 'unknown')
    anticipatory = config.get('anticipatory_param', 'unknown')
    
    # Use filename (without extension) as label
    filename = os.path.basename(checkpoint_path)
    label = os.path.splitext(filename)[0].replace('_', ' ')
    
    # Build experiment data structure compatible with plotting functions
    experiment = {
        "hidden_layers": str(hidden_layers),
        "anticipatory": anticipatory,
        "label": label,
        "episodes": [],
        "losses": [],
        "win_rates": {
            "win_rate_vs_random": [],
            "win_rate_vs_ismcts_100": [],
        },
        "checkpoint_path": checkpoint_path,
    }
    
    for entry in metrics_history:
        episode = entry['episode']
        experiment["episodes"].append(episode)
        experiment["losses"].append(entry['losses'])
        
        # Win rates are stored as decimals (0.0-1.0), convert to percentage
        if 'win_rate_vs_random' in entry:
            experiment["win_rates"]["win_rate_vs_random"].append(
                (episode, entry['win_rate_vs_random'] * 100)
            )
        if 'win_rate_vs_ismcts_100' in entry:
            experiment["win_rates"]["win_rate_vs_ismcts_100"].append(
                (episode, entry['win_rate_vs_ismcts_100'] * 100)
            )
    
    return experiment


def parse_checkpoints(checkpoint_paths):
    """Parse multiple checkpoints and return list of experiment data."""
    data = []
    for path in checkpoint_paths:
        try:
            experiment = parse_checkpoint(path)
            data.append(experiment)
            print(f"Parsed: {path} -> {experiment['hidden_layers']}, a={experiment['anticipatory']}")
        except Exception as e:
            print(f"Error parsing {path}: {e}")
    return data


def plot_grouped_loss_curves(data, output_dir, max_episode=None):
    """Plot grouped loss curves for all experiments, averaging losses across agents."""
    plt.figure()
    for experiment in data:
        # Collect losses from all agents and average them
        all_agent_losses = []
        for i, agent_losses in enumerate(zip(*experiment["losses"])):
            # Extract numeric values from nested tuples: (Array(value), None) -> value
            processed_losses = []
            for loss in agent_losses:
                if isinstance(loss, tuple) and loss[0] is not None:
                    val = loss[0]
                    # Handle nested tuple like (3.14,) or Array objects
                    while isinstance(val, tuple):
                        val = val[0]
                    # Handle jax Array - convert to float
                    processed_losses.append(float(val))
                else:
                    processed_losses.append(None)
            all_agent_losses.append(processed_losses)
        
        # Average losses across agents for each episode
        avg_losses = []
        valid_episodes = []
        for j, episode in enumerate(experiment["episodes"]):
            if max_episode and episode > max_episode:
                continue
            losses_at_j = [agent_losses[j] for agent_losses in all_agent_losses if j < len(agent_losses) and agent_losses[j] is not None]
            if losses_at_j:
                avg_losses.append(sum(losses_at_j) / len(losses_at_j))
                valid_episodes.append(episode)
        
        if avg_losses:
            plt.plot(valid_episodes, avg_losses, label=experiment['label'])
    plt.xlabel("Episodios")
    plt.ylabel("Loss")
    plt.legend(loc='upper right', fontsize=7, ncol=2)
    plt.savefig(os.path.join(output_dir, "grouped_loss_curves.pdf"))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'grouped_loss_curves.pdf')}")


def plot_grouped_win_rate_curves(data, output_dir, max_episode=None):
    """Plot grouped win rate curves for all experiments."""
    # Create two grouped plots: one for random, another for ismcts
    groups = {
        "random": [],
        "ismcts": [],
    }

    # Collect series per group: list of (episodes, rates, label)
    for experiment in data:
        short_label = experiment['label']
        for baseline, win_rates in experiment["win_rates"].items():
            if not win_rates:
                continue
            episodes, rates = zip(*win_rates)
            # Cut at max_episode if specified
            if max_episode:
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
        print(f"Saved: {os.path.join(output_dir, out_name)}")


def main():
    parser = argparse.ArgumentParser(description="Parse pkl checkpoints and generate plots.")
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True, 
                        help="Paths to the pkl checkpoint files.")
    parser.add_argument("--output-dir", type=str, default="results/plots",
                        help="Directory to save the plots.")
    parser.add_argument("--max-episode", type=int, default=None,
                        help="Maximum episode to plot (cuts off data after this).")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse checkpoints
    data = parse_checkpoints(args.checkpoints)
    
    if not data:
        print("No valid checkpoints found!")
        return

    # Generate grouped plots
    plot_grouped_loss_curves(data, args.output_dir, args.max_episode)
    plot_grouped_win_rate_curves(data, args.output_dir, args.max_episode)
    
    print(f"\nGenerated plots in: {args.output_dir}")


if __name__ == "__main__":
    main()
