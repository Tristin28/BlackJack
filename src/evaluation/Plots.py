import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from agent.MonteCarloAgent import MonteCarloAgent
from agent.QLearningAgent import QLearningAgent
from agent.DoubleQLearningAgent import DoubleQLearningAgent
from agent.SarsaAgent import SarsaAgent
from evaluation.Evaluation import run_episodes

def ensure_output_dir(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

def safe_name(text):
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

def format_state_action_label(state, action):
    player_sum, dealer_card, usable_ace = state
    ace_text = "Soft" if usable_ace else "Hard"
    return f"({player_sum},{dealer_card},{ace_text})-{action}"

def extract_history_series(history):
    episodes = [item[0] for item in history]
    wins = [item[1] for item in history]
    losses = [item[2] for item in history]
    draws = [item[3] for item in history]
    return episodes, wins, losses, draws

def plot_win_loss_draw_history(history, algorithm_name, config_name, output_dir="plots"):
    ensure_output_dir(output_dir)

    episodes, wins, losses, draws = extract_history_series(history)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, wins, label="Wins")
    plt.plot(episodes, losses, label="Losses")
    plt.plot(episodes, draws, label="Draws")
    plt.xlabel("Episodes")
    plt.ylabel("Count in previous 1000 episodes")
    plt.title(f"{algorithm_name} - {config_name}: Wins, Losses and Draws")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_{safe_name(config_name)}_history.png"))
    plt.close()

def plot_combined_histories(histories_by_config, algorithm_name, output_dir):
    ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (config_name, history) in zip(axes, histories_by_config.items()):
        episodes, wins, losses, draws = extract_history_series(history)

        ax.plot(episodes, wins, label="Wins")
        ax.plot(episodes, losses, label="Losses")
        ax.plot(episodes, draws, label="Draws")
        ax.set_title(config_name)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Count in previous 1000 episodes")
        ax.legend()

    fig.suptitle(f"{algorithm_name}: History Comparison Across Configurations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_combined_histories.png"))
    plt.close()

def get_sorted_pairs_and_counts(count_table):
    pairs_and_counts = []

    for state, actions in count_table.items():
        for action, count in actions.items():
            if count > 0:
                label = format_state_action_label(state, action)
                pairs_and_counts.append((label, count))

    pairs_and_counts.sort(key=lambda x: x[1], reverse=True)
    return pairs_and_counts

def plot_state_action_counts(count_table, algorithm_name, config_name, output_dir="plots"):
    ensure_output_dir(output_dir)

    pairs_and_counts = get_sorted_pairs_and_counts(count_table)

    if not pairs_and_counts:
        print(f"No visited state-action pairs to plot for {algorithm_name} - {config_name}")
        return

    labels = [item[0] for item in pairs_and_counts]
    counts = [item[1] for item in pairs_and_counts]

    figure_height = max(8, len(labels) * 0.18)

    plt.figure(figsize=(14, figure_height))
    plt.barh(labels, counts)
    plt.gca().invert_yaxis()
    plt.xlabel("Count")
    plt.ylabel("State-Action Pair")
    plt.title(f"{algorithm_name} - {config_name}: State-Action Pair Counts (Full)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_{safe_name(config_name)}_state_action_counts_full.png"))
    plt.close()

def plot_state_action_counts_top_n(count_table, algorithm_name, config_name, top_n, output_dir):
    ensure_output_dir(output_dir)

    pairs_and_counts = get_sorted_pairs_and_counts(count_table)

    if not pairs_and_counts:
        print(f"No visited state-action pairs to plot for {algorithm_name} - {config_name}")
        return

    top_pairs = pairs_and_counts[:top_n]

    labels = [item[0] for item in top_pairs]
    counts = [item[1] for item in top_pairs]

    figure_height = max(6, len(labels) * 0.35)

    plt.figure(figsize=(12, figure_height))
    plt.barh(labels, counts)
    plt.gca().invert_yaxis()
    plt.xlabel("Count")
    plt.ylabel("Top State-Action Pairs")
    plt.title(f"{algorithm_name} - {config_name}: Top {top_n} State-Action Pair Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_{safe_name(config_name)}_top_{top_n}_state_action_counts.png"))
    plt.close()

def plot_unique_pairs_across_configs(unique_counts_by_config, algorithm_name, output_dir="plots"):
    ensure_output_dir(output_dir)

    config_names = list(unique_counts_by_config.keys())
    counts = list(unique_counts_by_config.values())

    plt.figure(figsize=(10, 6))
    plt.bar(config_names, counts)
    plt.xlabel("Configuration")
    plt.ylabel("Number of Unique State-Action Pairs")
    plt.title(f"{algorithm_name}: Unique State-Action Pairs Across Configurations")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_unique_pairs_across_configs.png"))
    plt.close()

def evaluate_algorithm(algorithm_name, agent_factory, configs, num_episodes=100000, output_dir="plots"):
    unique_counts_by_config = {}
    histories_by_config = {}

    additional_dir = os.path.join(output_dir, "additional")
    ensure_output_dir(additional_dir)

    for display_name, epsilon_config, exploring_starts in configs:
        print(f"Running {algorithm_name} - {display_name}")

        agent = agent_factory()
        history, q_table, count_table, visited_pairs, num_visited_pairs = run_episodes(
            agent,
            epsilon_config,
            num_episodes=num_episodes,
            exploring_starts=exploring_starts
        )

        unique_counts_by_config[display_name] = num_visited_pairs
        histories_by_config[display_name] = history

        # Required plots
        plot_win_loss_draw_history(history, algorithm_name, display_name, output_dir)
        plot_state_action_counts(count_table, algorithm_name, display_name, output_dir)
        plot_unique_pairs_across_configs(unique_counts_by_config, algorithm_name, output_dir)

        # Additional plots
        plot_state_action_counts_top_n(
            count_table,
            algorithm_name,
            display_name,
            top_n=30,
            output_dir=additional_dir
        )

    # Additional summary plot
    plot_combined_histories(histories_by_config, algorithm_name, additional_dir)

if __name__ == "__main__":
    NUM_EPISODES = 100000
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")

    mc_configs = [
        ("Exploring Starts (1/k)", "1_over_k", True),
        ("No ES (1/k)", "1_over_k", False),
        ("No ES (exp/1000)", "exp_1000", False),
        ("No ES (exp/10000)", "exp_10000", False),
    ]

    td_configs = [
        ("0.1", "fixed_0.1", False),
        ("1/k", "1_over_k", False),
        ("exp/1000", "exp_1000", False),
        ("exp/10000", "exp_10000", False),
    ]

    evaluate_algorithm(
        "MonteCarlo",
        lambda: MonteCarloAgent({}, {}),
        mc_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    evaluate_algorithm(
        "SARSA",
        lambda: SarsaAgent({}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    evaluate_algorithm(
        "QLearning",
        lambda: QLearningAgent({}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    evaluate_algorithm(
        "DoubleQLearning",
        lambda: DoubleQLearningAgent({}, {}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )