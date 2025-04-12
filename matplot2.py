import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Control variables
vertical_line_position = 0
output_folder = "experiments"
# dataset = "mnist"
dataset = "fashion_mnist"
tag = "Gassuian_only_different_aggregation_methods_comparison"
scalar_name = "accuracy"
scalar_tag = "test/accuracy"
show_vertical_line = True

# List of methods (suffixes in folder names)
methods = [
    "mudhog", 
    "krum", 
    "trimmed"
]

# Updated error ratios from 5% to 45%
error_ratios = ["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%"]
start_indices = list(range(len(error_ratios)))  # [0, 1, 2, ..., 8]

# Create results folder
results_tag = f'{tag}_40C_{vertical_line_position}ep_{dataset}'
results_folder = f'./experiments_results/{results_tag}'
os.makedirs(results_folder, exist_ok=True)

# Plot setup
plt.figure(figsize=(8, 6))

# Iterate over each method
for method in methods:
    final_accuracies = []
    for idx in start_indices:
        experiment_name = f"{tag}_Exp_40C_40ep_{vertical_line_position}start_{idx}_{dataset}_{method}"
        log_dir = f'./logs/{output_folder}/{experiment_name}'
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
        ea.Reload()

        if scalar_tag in ea.Tags()["scalars"]:
            events = ea.Scalars(scalar_tag)
            values = [e.value for e in events]
            final_accuracies.append(values[-1])  # Final accuracy
        else:
            print(f"Warning: Scalar tag {scalar_tag} not found in {experiment_name}")
            final_accuracies.append(None)

    plt.plot(
        error_ratios,
        final_accuracies,
        marker="o",
        linestyle="-",
        label=method.upper()
    )

# Axis setup
plt.xlabel("Malicious Clients Ratio", fontsize=20, fontweight='bold')
plt.ylabel("Accuracy", fontsize=20, fontweight='bold')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", fontsize=12)

# Save plot
filename_prefix = f"{dataset}_{tag}_vline{vertical_line_position}"
plot_filename = os.path.join(results_folder, f"{filename_prefix}_accuracy_vs_ratio_across_methods.png")
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
plt.close()
