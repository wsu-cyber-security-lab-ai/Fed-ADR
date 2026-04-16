import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Experiment settings
output_base_folder = "experiments"
dataset = "fashion_mnist"
epochs = "40ep"
start_round_val = 10 
start_round_str = f"{start_round_val}start"

# Mapping for the client count ID in the filename
# 0 -> 100 Clients
# 1 -> 200 Clients
client_configs = {
    "100C": "0",
    "200C": "1"
}

# Aggregation methods to compare
aggregation_methods = ["fedavg", "foolsgold", "krum", "trimmed"]

# Control variables for the vertical line
show_vertical_line = True
vertical_line_position = start_round_val

scalar_tags = {
    "Accuracy (%)": "test/accuracy",
}

# Output folder base
results_base_folder = f'./08-01-2026_results/accuracy_plots_{dataset}'
os.makedirs(results_base_folder, exist_ok=True)

# Markers for comparison
markers = ["o", "^", "v", "p", "*", "X"]

# Loop through each client configuration
for client_label, client_id in client_configs.items():
    print(f"Processing plots for {client_label} (ID: {client_id})...")
    
    # Generate experiment paths based on the specific naming convention:
    # different_aggregation_methods_100C_40ep_10start_{client_id}_{dataset}_{agg}
    experiments = [
        f"different_aggregation_methods_100C_40ep_{start_round_str}_{client_id}_{dataset}_{agg}"
        for agg in aggregation_methods
    ]
    
    # Titles for the legend
    experiments_titles = [agg.replace("_", " ").title() for agg in aggregation_methods]

    # Create a subfolder for this client count
    current_results_folder = os.path.join(results_base_folder, client_label)
    os.makedirs(current_results_folder, exist_ok=True)

    # Plotting Loop
    for scalar_name, scalar_tag in scalar_tags.items():
        plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        plot_has_data = False

        for i, (experiment_name, title) in enumerate(zip(experiments, experiments_titles)):
            log_dir = f'./logs/{output_base_folder}/{experiment_name}'
            if not os.path.exists(log_dir):
                print(f"Skipping (not found): {log_dir}")
                continue

            try:
                ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
                ea.Reload()

                if scalar_tag in ea.Tags()["scalars"]:
                    events = ea.Scalars(scalar_tag)
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    
                    plt.plot(
                        steps, 
                        values, 
                        marker=markers[i % len(markers)], 
                        linestyle="-", 
                        linewidth=2.5,
                        markersize=9,
                        label=title
                    )
                    plot_has_data = True
                else:
                    print(f"Warning: {scalar_tag} not found in {experiment_name}")
            except Exception as e:
                print(f"Error reading {log_dir}: {e}")

        if not plot_has_data:
            print(f"No data found for {client_label}. Skipping plot.")
            plt.close()
            continue

        # Add the vertical line to represent the starting attack point
        if show_vertical_line:
            plt.axvline(
                x=vertical_line_position,
                color="grey",
                linestyle="--",
                linewidth=3,
                alpha=0.8,
                # label="Attack Start"
            )

        # Set Y-axis limit to 100
        plt.ylim(0, 100)

        plt.xlabel("Rounds", fontsize=28)
        plt.ylabel(scalar_name, fontsize=28)
        plt.grid(True, linestyle="--", alpha=0.6, linewidth=2)
        
        # Legend placement
        plt.legend(loc='upper right', fontsize=18, ncol=2)

        # Save the plot
        filename_prefix = f"{dataset}_{client_label}_agg_comparison"
        plot_filename = os.path.join(current_results_folder, f"{filename_prefix}_accuracy_plot.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {plot_filename}")
        plt.close()
