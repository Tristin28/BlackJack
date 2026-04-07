import os
import matplotlib.pyplot as plt

from agent import DoubleQLearningAgent, MonteCarloAgent, QLearningAgent, SarsaAgent
from evaluation.Evaluation import run_episodes

def ensure_output_dir(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

def safe_name(text):
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

def format_state_action_label(state, action):
    player_sum, dealer_card, usable_ace = state
    ace_text = "Soft" if usable_ace else "Hard"
    return f"({player_sum},{dealer_card},{ace_text})-{action}"

def plot_win_loss_draw_history(history, algorithm_name, config_name, output_dir="plots"):
    ensure_output_dir(output_dir)

    episodes = [item[0] for item in history]
    wins = [item[1] for item in history]
    losses = [item[2] for item in history]
    draws = [item[3] for item in history]

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
    ensure_output_dir(output_dir)

    pairs_and_counts = []

    for state, actions in count_table.items():
        for action, count in actions.items():
            if count > 0:
                label = format_state_action_label(state, action)
                pairs_and_counts.append((label, count))

    pairs_and_counts.sort(key=lambda x: x[1], reverse=True)

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
    plt.title(f"{algorithm_name} - {config_name}: State-Action Pair Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name(algorithm_name)}_{safe_name(config_name)}_state_action_counts.png"))
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

        plot_win_loss_draw_history(history, algorithm_name, display_name, output_dir)
        plot_state_action_counts(count_table, algorithm_name, display_name, output_dir)

    plot_unique_pairs_across_configs(unique_counts_by_config, algorithm_name, output_dir)

if __name__ == "__main__":
    NUM_EPISODES = 100000
    OUTPUT_DIR = "plots"

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
        lambda: DoubleQLearningAgent({}, {}),
        td_configs,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR
    )

    print(f"All plots saved in: {OUTPUT_DIR}")