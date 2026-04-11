import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from agent.MonteCarloAgent import MonteCarloAgent
from agent.QLearningAgent import QLearningAgent
from agent.DoubleQLearningAgent import DoubleQLearningAgent
from agent.SarsaAgent import SarsaAgent
from evaluation.Evaluation import run_episodes, get_optimal_policy, build_strategy_table

# --- Utility functions for plotting and saving results ---
def ensure_output_dir(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

def build_output_dirs(base_output_dir="plots"):
    """
    Creates and returns a dictionary of well-structured output directories.
    Required outputs are separated from additional outputs.
    """
    paths = {
        "base": base_output_dir,
        "required": os.path.join(base_output_dir, "required"),
        "required_line_charts": os.path.join(base_output_dir, "required", "line_charts"),
        "required_bar_charts": os.path.join(base_output_dir, "required", "bar_charts"),
        "required_strategy_tables": os.path.join(base_output_dir, "required", "strategy_tables"),
        "required_exports": os.path.join(base_output_dir, "required", "exports"),
        "additional": os.path.join(base_output_dir, "additional"),
        "additional_combined_histories": os.path.join(base_output_dir, "additional", "combined_histories"),
        "additional_top_state_action_counts": os.path.join(base_output_dir, "additional", "top_state_action_counts")
    }

    for path in paths.values():
        ensure_output_dir(path)

    return paths

def safe_name(text):
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

def format_state_action_label(state, action):
    player_sum, dealer_card, usable_ace = state
    ace_text = "Soft" if usable_ace else "Hard"
    return f"({player_sum},{dealer_card},{ace_text})-{action}"

# --- Evaluation Data Extraction Helpers ---
def extract_history_series(history):
    episodes = [item[0] for item in history]
    wins = [item[1] for item in history]
    losses = [item[2] for item in history]
    draws = [item[3] for item in history]
    return episodes, wins, losses, draws

def get_sorted_pairs_and_counts(count_table):
    pairs_and_counts = []

    for state, actions in count_table.items():
        for action, count in actions.items():
            if count > 0:
                label = format_state_action_label(state, action)
                pairs_and_counts.append((label, count))

    pairs_and_counts.sort(key=lambda x: x[1], reverse=True)
    return pairs_and_counts

def compute_last_10000_summary(history):
    last_10 = history[-10:]

    mean_wins = sum(item[1] for item in last_10) / len(last_10)
    mean_losses = sum(item[2] for item in last_10) / len(last_10)
    mean_draws = sum(item[3] for item in last_10) / len(last_10)

    denominator = mean_losses + mean_wins
    if denominator == 0:
        dealer_advantage = 0.0
    else:
        dealer_advantage = (mean_losses - mean_wins) / denominator

    return {
        "mean_wins": mean_wins,
        "mean_losses": mean_losses,
        "mean_draws": mean_draws,
        "dealer_advantage": dealer_advantage
    }

# --- Required Plots ---
def plot_win_loss_draw_history(history, algorithm_name, config_name, output_dir="plots"):
    """
    One line chart per algorithm configuration showing wins, losses and draws over episodes
    """
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

def plot_state_action_counts(count_table, algorithm_name, config_name, output_dir="plots"):
    """
    A bar chart sorted by highest count first, showing how often
    each unique state-action pair was executed.
    """
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

def plot_unique_pairs_across_configs(unique_counts_by_config, algorithm_name, output_dir="plots"):
    """
    A 4-bar chart for each algorithm showing the number of unique
    state-action pairs explored across its four configurations.
    """
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

def plot_dealer_advantage_across_configs(summary_by_config, algorithm_name, output_dir="plots"):
    """
    Dealer advantage for the four configurations of one algorithm.
    """
    ensure_output_dir(output_dir)

    config_names = list(summary_by_config.keys())
    values = [summary_by_config[name]["dealer_advantage"] for name in config_names]

    plt.figure(figsize=(10, 6))
    plt.bar(config_names, values)
    plt.xlabel("Configuration")
    plt.ylabel("Dealer Advantage")
    plt.title(f"{algorithm_name}: Dealer Advantage Across Configurations")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_dealer_advantage.png"))
    plt.close()

def plot_global_dealer_advantage(all_summaries, output_dir="plots"):
    """
    One chart comparing dealer advantage across all algorithm
    configurations, to determine which minimises dealer advantage most.
    """
    ensure_output_dir(output_dir)

    labels = []
    values = []

    for algorithm_name, summary_by_config in all_summaries.items():
        for config_name, stats in summary_by_config.items():
            labels.append(f"{algorithm_name}\n{config_name}")
            values.append(stats["dealer_advantage"])

    plt.figure(figsize=(14, 7))
    plt.bar(labels, values)
    plt.xlabel("Algorithm / Configuration")
    plt.ylabel("Dealer Advantage")
    plt.title("Dealer Advantage Across All Algorithm Configurations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_algorithms_dealer_advantage.png"))
    plt.close()

# ---Data Exports ---
def save_strategy_table_csv(table, algorithm_name, config_name, usable_ace, output_dir="plots"):
    """
    Strategy table exported as CSV. Two tables must exist for each configuration:
    one with a usable ace and one without.
    """
    ensure_output_dir(output_dir)

    ace_label = "usable_ace" if usable_ace else "no_usable_ace"
    filename = f"{safe_name(algorithm_name)}_{safe_name(config_name)}_{ace_label}_strategy_table.csv"
    filepath = os.path.join(output_dir, filename)

    dealer_columns = list(range(2, 12))   # 11 represents Ace
    player_rows = list(range(20, 11, -1)) # 20 down to 12

    with open(filepath, "w", encoding="utf-8") as f:
        headers = ["Player Sum"] + [("A" if col == 11 else str(col)) for col in dealer_columns]
        f.write(",".join(headers) + "\n")

        for row_label, row_values in zip(player_rows, table):
            row = [str(row_label)] + [str(v) for v in row_values]
            f.write(",".join(row) + "\n")

def plot_strategy_table(table, algorithm_name, config_name, usable_ace, output_dir="plots"):
    """
    Strategy table as an image for readability in the report.
    """
    ensure_output_dir(output_dir)

    ace_label = "Usable Ace" if usable_ace else "No Usable Ace"
    filename = f"{safe_name(algorithm_name)}_{safe_name(config_name)}_{safe_name(ace_label)}_strategy_table.png"

    dealer_columns = list(range(2, 12))   # 11 represents Ace
    player_rows = list(range(20, 11, -1)) # 20 down to 12

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    cell_text = []
    for row_label, row_values in zip(player_rows, table):
        cell_text.append([str(row_label)] + [str(v) for v in row_values])

    col_labels = ["Player Sum"] + [("A" if col == 11 else str(col)) for col in dealer_columns]

    plot_table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    plot_table.auto_set_font_size(False)
    plot_table.set_fontsize(10)
    plot_table.scale(1.1, 1.4)

    ax.set_title(f"{algorithm_name} - {config_name} - {ace_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def save_q_values_csv(q_table, algorithm_name, config_name, output_dir="plots"):
    """
    Estimated Q-values for each unique state-action pair.
    """
    ensure_output_dir(output_dir)

    filepath = os.path.join(
        output_dir,
        f"{safe_name(algorithm_name)}_{safe_name(config_name)}_q_values.csv"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Player Sum,Dealer Card,Usable Ace,Action,Q Value\n")
        for state, actions in q_table.items():
            player_sum, dealer_card, usable_ace = state
            for action, q_value in actions.items():
                f.write(f"{player_sum},{dealer_card},{usable_ace},{action},{q_value}\n")

def save_summary_csv(summary_by_config, algorithm_name, output_dir="plots"):
    """
    Mean wins, losses, draws over the last 10,000 episodes, plus dealer advantage.
    """
    ensure_output_dir(output_dir)

    filepath = os.path.join(output_dir, f"{safe_name(algorithm_name)}_last_10000_summary.csv")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Configuration,Mean Wins,Mean Losses,Mean Draws,Dealer Advantage\n")
        for config_name, stats in summary_by_config.items():
            f.write(
                f"{config_name},"
                f"{stats['mean_wins']:.4f},"
                f"{stats['mean_losses']:.4f},"
                f"{stats['mean_draws']:.4f},"
                f"{stats['dealer_advantage']:.6f}\n"
            )

# --- Additional Plots ---
def plot_combined_histories(histories_by_config, algorithm_name, output_dir):
    """
    Combines the four history plots of one algorithm into a single figure.
    """
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

def plot_state_action_counts_top_n(count_table, algorithm_name, config_name, top_n, output_dir):
    """
    A top-N readable version of the full state-action count chart.
    """
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
    
def plot_all_algorithms_unique_pairs_grid(all_unique_counts, output_dir="plots"):
    """
    all_unique_counts example:
    {
        "MonteCarlo": {"Exploring Starts (1/k)": 358, "No ES (1/k)": 343, ...},
        "SARSA": {"0.1": 359, "1/k": 341, ...},
        "QLearning": {"0.1": 358, "1/k": 344, ...},
        "DoubleQLearning": {"0.1": 359, "1/k": 342, ...}
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (algorithm_name, unique_counts_by_config) in zip(axes, all_unique_counts.items()):
        config_names = list(unique_counts_by_config.keys())
        counts = list(unique_counts_by_config.values())

        ax.bar(config_names, counts)
        ax.set_title(f"{algorithm_name}: Unique State-Action Pairs Across Configurations")
        ax.set_ylabel("Number of Unique State-Action Pairs")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_algorithms_unique_pairs_grid.png"))
    plt.close()

def evaluate_algorithm(algorithm_name, agent_factory, configs, num_episodes=100000, output_dir="plots"):
    unique_counts_by_config = {}
    histories_by_config = {}
    summary_by_config = {}

    output_paths = build_output_dirs(output_dir)

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
        summary_by_config[display_name] = compute_last_10000_summary(history)

        # Required plots
        plot_win_loss_draw_history(history, algorithm_name, display_name, output_paths["required_line_charts"])
        plot_state_action_counts(count_table, algorithm_name, display_name, output_paths["required_bar_charts"])

        # Required exports
        save_q_values_csv(q_table, algorithm_name, display_name, output_paths["required_exports"])

        optimal_policy = get_optimal_policy(q_table)

        table_no_ace = build_strategy_table(optimal_policy, usable_ace=False)
        table_with_ace = build_strategy_table(optimal_policy, usable_ace=True)

        save_strategy_table_csv(table_no_ace, algorithm_name, display_name, False, output_paths["required_strategy_tables"])
        save_strategy_table_csv(table_with_ace, algorithm_name, display_name, True, output_paths["required_strategy_tables"])

        plot_strategy_table(table_no_ace, algorithm_name, display_name, False, output_paths["required_strategy_tables"])
        plot_strategy_table(table_with_ace, algorithm_name, display_name, True, output_paths["required_strategy_tables"])

        # Additional readable plots
        plot_state_action_counts_top_n(
            count_table,
            algorithm_name,
            display_name,
            top_n=30,
            output_dir=output_paths["additional_top_state_action_counts"]
        )
    
    # Required plots after all 4 configurations are complete
    plot_unique_pairs_across_configs(unique_counts_by_config, algorithm_name, output_paths["required_bar_charts"])
    plot_dealer_advantage_across_configs(summary_by_config, algorithm_name, output_paths["required_bar_charts"])

    # Required summary export
    save_summary_csv(summary_by_config, algorithm_name, output_paths["required_exports"])

    # Additional summary plot
    plot_combined_histories(histories_by_config, algorithm_name, output_paths["additional_combined_histories"])

    return summary_by_config, unique_counts_by_config

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

    all_summaries = {}
    all_unique_counts = {}

    all_summaries["MonteCarlo"], all_unique_counts["MonteCarlo"] = evaluate_algorithm(
        "MonteCarlo",
        lambda: MonteCarloAgent({}, {}),
        mc_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    all_summaries["SARSA"], all_unique_counts["SARSA"] = evaluate_algorithm(
        "SARSA",
        lambda: SarsaAgent({}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    all_summaries["QLearning"], all_unique_counts["QLearning"] = evaluate_algorithm(
        "QLearning",
        lambda: QLearningAgent({}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    all_summaries["DoubleQLearning"], all_unique_counts["DoubleQLearning"] = evaluate_algorithm(
        "DoubleQLearning",
        lambda: DoubleQLearningAgent({}, {}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    final_output_paths = build_output_dirs(OUTPUT_DIR)
    plot_all_algorithms_unique_pairs_grid(all_unique_counts, final_output_paths["additional"])
    plot_global_dealer_advantage(all_summaries, final_output_paths["required_bar_charts"])