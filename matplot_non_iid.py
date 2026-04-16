import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Experiment settings
output_folder = "experiments"
dataset = "fashion_mnist"
num_clients = "100C"
epochs = "40ep"
start_round_val = 10 # Numerical value for the vertical line
attack_percentage = 30
start_round_str = f"{start_round_val}start"
dirichlet_ratios = [0.9, 0.7, 0.5, 0.3]

# Control variables for the vertical line
show_vertical_line = True
vertical_line_position = start_round_val 

scalar_tags = {
    "Accuracy (%)": "test/accuracy", 
}

# Generate experiment paths based on: new_dirichlet_0.7_30percent_100C_40ep_10start_0_fashion_mnist_mudhog
experiments = [
    f"new_dirichlet_{ratio}_{attack_percentage}percent_{num_clients}_{epochs}_{start_round_str}_0_{dataset}_mudhog"
    for ratio in dirichlet_ratios
]
experiments_titles = [f"Dirichlet Alpha: {ratio}" for ratio in dirichlet_ratios]

# Output folder
results_folder = f'./08-01-2026_results/accuracy_plots_{dataset}'
os.makedirs(results_folder, exist_ok=True)

# Markers for comparison
markers = ["o", "s", "D", "^"]

# Plotting Loop
for scalar_name, scalar_tag in scalar_tags.items():
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    for i, (experiment_name, title) in enumerate(zip(experiments, experiments_titles)):
        log_dir = f'./logs/{output_folder}/{experiment_name}'
        if not os.path.exists(log_dir):
            print(f"Skipping: {log_dir}")
            continue

        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
        ea.Reload()

        if scalar_tag in ea.Tags()["scalars"]:
            events = ea.Scalars(scalar_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, marker=markers[i % len(markers)], linestyle="-", label=title)
        else:
            print(f"Warning: {scalar_tag} not found in {experiment_name}")

    # Add the vertical line to represent the starting attack point
    if show_vertical_line:
        plt.axvline(
            x=vertical_line_position,
            color="grey",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label="Attack Start" if i == 0 else "" # Optional label
        )

    plt.xlabel("Rounds", fontsize=28)
    plt.ylabel(scalar_name, fontsize=28)
    plt.grid(True, linestyle="--", alpha=0.6, linewidth=2)
    plt.legend(loc='upper right', fontsize=16)

    # Save the plot
    filename_prefix = f"{dataset}_dirichlet_vline{vertical_line_position}_{attack_percentage}"
    plot_filename = os.path.join(results_folder, f"{filename_prefix}_accuracy_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
