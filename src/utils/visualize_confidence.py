import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

CURRENT_THRESHOLD = 0.3

def visualize_confidence(csv_file, output_plot):
    df = pd.read_csv(csv_file)

    # plot
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of CLIP Confidence Scores (Unseen Test Data)')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    
    # threshold line
    plt.axvline(x=CURRENT_THRESHOLD, color='r', linestyle='--', label='Current Threshold')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # stats
    print("\nConfidence Statistics:")
    print(df['confidence'].describe())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Confidence Distribution')
    parser.add_argument('--csv_file', type=str, default='visualize_confidence.csv', help='Path to confidence CSV')
    parser.add_argument('--output_plot', type=str, default='confidence_distribution.png', help='Path to save plot')
    
    args = parser.parse_args()
    visualize_confidence(args.csv_file, args.output_plot)
