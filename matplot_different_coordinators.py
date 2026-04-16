import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Experiment settings
output_folder = "experiments"
dataset = "fashion_mnist"
num_clients = "100C"
epochs = "40ep"
start_round_val = 10
start_round_str = f"{start_round_val}start"

# Parameters for the 3 separate figures
coordinators = [1, 2, 3]
attack_percentages = [10, 20, 30]

# Control variables for the vertical line
show_vertical_line = True
vertical_line_position = start_round_val 

scalar_tags = {"Accuracy (%)": "test/accuracy"}
markers = ["o", "s", "D"] # Only 3 markers needed per figure now

# Outer loop to create 3 separate figures
for coord in coordinators:
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Inner loop to plot the 3 attack percentages for the current coordinator count
    for i, percent in enumerate(attack_percentages):
        experiment_name = f"multi_coordinator_{coord}_coordinator_{percent}percent_{num_clients}_{epochs}_{start_round_str}_0_{dataset}_mudhog"
        title = f"{percent}% Attack"
        
        log_dir = f'./logs/{output_folder}/{experiment_name}'
        if not os.path.exists(log_dir):
            print(f"Skipping: {log_dir}")
            continue

        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
        ea.Reload()

        if "test/accuracy" in ea.Tags()["scalars"]:
            events = ea.Scalars("test/accuracy")
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, marker=markers[i], linestyle="-", label=title)

    # Vertical line and plot styling
    if show_vertical_line:
        plt.axvline(x=vertical_line_position, color="grey", linestyle="--", linewidth=2.5, alpha=0.9)

    # plt.title(f"Performance with {coord} Orchestrator(s)", fontsize=22)
    plt.xlabel("Rounds", fontsize=24)
    plt.ylabel("Accuracy (%)", fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)
    plt.legend(loc='upper right', fontsize=15)

    current_bottom, current_top = plt.ylim()
    plt.ylim(top=100)

    # Save each figure separately
    results_folder = f'./08-01-2026_results/multi_coordinator_plots_{dataset}'
    os.makedirs(results_folder, exist_ok=True)
    
    plot_filename = os.path.join(results_folder, f"{dataset}_{coord}coord_accuracy_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {plot_filename}")
    plt.show()
    plt.close()
