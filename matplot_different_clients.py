import os
import matplotlib.pyplot as plt

# Data from LaTeX Table
malicious_ratios = [10, 20, 30, 40, 50]
client_counts = [100, 200, 300]
datasets = ["MNIST", "Fashion-MNIST"]

# data[clients][dataset][defense]
data = {
    100: {
        "MNIST":   {"mudhog": [9.82, 10.09, 9.80, 10.32, 9.80], "trimmed": [9.80, 9.80, 9.80, 9.80, 9.80]},
        "Fashion-MNIST": {"mudhog": [36.07, 10.00, 10.00, 10.00, 10.00], "trimmed": [10.00, 10.00, 10.00, 10.00, 10.00]}
    },
    200: {
        "MNIST":   {"mudhog": [9.82, 9.80, 9.80, 9.80, 9.80], "trimmed": [9.80, 9.80, 9.80, 9.80, 9.80]},
        "Fashion-MNIST": {"mudhog": [10.00, 10.00, 10.00, 10.00, 10.00], "trimmed": [10.00, 10.00, 10.00, 10.00, 10.00]}
    },
    300: {
        "MNIST":   {"mudhog": [9.80, 9.80, 9.58, 10.10, 9.80], "trimmed": [9.80, 9.80, 9.80, 9.80, 9.80]},
        "Fashion-MNIST": {"mudhog": [10.00, 10.00, 9.99, 10.00, 10.00], "trimmed": [10.00, 10.00, 10.00, 10.00, 10.00]}
    }
}

os.makedirs('./final_plots', exist_ok=True)

# Loop to create and save each figure separately
for ds in datasets:
    for clients in client_counts:
        plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        
        # MUD-HoG (Blue Circle)
        plt.plot(malicious_ratios, data[clients][ds]["mudhog"], 
                marker='o', linestyle='-', linewidth=3, markersize=12, label='MUD-HoG')
        
        # Trimmed Mean (Orange Square)
        plt.plot(malicious_ratios, data[clients][ds]["trimmed"], 
                marker='s', linestyle='--', linewidth=3, markersize=12, label='Trimmed Mean')
        
        # Aesthetics matching your paper's style
        plt.grid(True, linestyle="--", alpha=0.6, linewidth=2)
        plt.xlabel("Malicious Ratio (%)", fontsize=28, labelpad=15)
        plt.ylabel("Accuracy (%)", fontsize=28, labelpad=15)
        plt.ylim(0, 100)
        plt.xticks(malicious_ratios)

        # Legend included in every plot
        plt.legend(loc='upper right', fontsize=18)

        # No upper title as requested

        # Save separately
        filename = f"{ds}_{clients}C_comparison.png"
        plt.savefig(f'./final_plots/{filename}', dpi=300, bbox_inches="tight")
        plt.close()
