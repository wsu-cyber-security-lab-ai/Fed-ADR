import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Set this to the iteration where the vertical line should appear
vertical_line_position = 30

# List of experiments
output_folder = "experiments"
dataset = "mnist"
# dataset = "fashion_mnist"
tag = "Gassuian_and_label_flipping"
# tag = "Gassuian_alone"

experiments = [
    f"{tag}_Exp_40C_40ep_{vertical_line_position}start_0_{dataset}_mudhog",
    f"{tag}_Exp_40C_40ep_{vertical_line_position}start_1_{dataset}_mudhog",
    f"{tag}_Exp_40C_40ep_{vertical_line_position}start_2_{dataset}_mudhog",
    f"{tag}_Exp_40C_40ep_{vertical_line_position}start_3_{dataset}_mudhog",
]

experiments_titles = [
    "15%",
    "25%",
    "35%",
    "45%",
]

scalar_tags = {
    "Loss": "test/loss",
    "Accuracy": "test/accuracy",
    "Label Flip Undetected": "test/label_malicious_marked_as_bengin",
    "Multi-Label Flip Undetected": "test/multi_label_malicious_marked_as_bengin",
    "Benign Falsely Detected": "test/num_of_bengin_marked_as_malicious",
    "Malicious Undetected": "test/num_of_malicious_marked_as_bengin",
    "Malicious Detected": "test/num_of_malicoius_marked_as_malicious",
}

# Control variables
show_vertical_line = True  # Set to True to show the vertical line, False to hide it

# Create results folder if it doesn't exist
results_tag = f'{tag}_40C_{vertical_line_position}ep_{dataset}'
results_folder = f'./experiments_results/{results_tag}'
os.makedirs(results_folder, exist_ok=True)

# Different markers for different experiments
markers = ["o", "s", "D", "^", "v", "<", ">", "p", "X"]

# Iterate over each scalar type
for scalar_name, scalar_tag in scalar_tags.items():
    plt.figure(figsize=(8, 6))

    # Iterate over experiments and plot their data
    for i, (experiment_name, title) in enumerate(zip(experiments, experiments_titles)):
        log_dir = f'./logs/{output_folder}/{experiment_name}'
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
        ea.Reload()  # Load the data

        if scalar_tag in ea.Tags()["scalars"]:
            events = ea.Scalars(scalar_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            plt.plot(steps, values, marker=markers[i % len(markers)], linestyle="-", label=title)
        else:
            print(f"Warning: Scalar tag {scalar_tag} not found in {experiment_name}")

    # Add a vertical line if enabled
    if show_vertical_line:
        plt.axvline(
            x=vertical_line_position,
            color="grey",             # Change to red for better visibility
            linestyle="--",
            linewidth=2.5,           # Thicker line
            alpha=0.9                # More opaque
        )

    # Axis labels with bold font and 24pt size
    plt.xlabel("Iteration Number", fontsize=20, fontweight='bold')
    plt.ylabel(scalar_name.replace("_", " ").title(), fontsize=20, fontweight='bold')

    plt.grid(True, linestyle="--", alpha=0.6)

    # Fixed legend location to top left
    plt.legend(loc='upper left', fontsize=10)

    # Construct filename with dataset, tag, and vertical line position
    filename_prefix = f"{dataset}_{tag}_vline{vertical_line_position}"
    plot_filename = os.path.join(results_folder, f"{filename_prefix}_{scalar_name}_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
