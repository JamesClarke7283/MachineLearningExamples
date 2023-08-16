import pandas as pd
import os

# Constants
DATASETS_DIR = "./datasets/raw/"
TARGET_SAMPLE_SIZE = 1_000_000  # or whatever size you desire

# Manually define the list of dataset files
dataset_files = ['imdb_reviews.csv', 'yelp_reviews.csv']  # add other filenames as needed

def load_and_sample_data(file_path, sample_size, random_state=42):
    """Load and sample data from the given file path."""
    data = pd.read_csv(file_path)
    return data.sample(min(len(data), sample_size), random_state=random_state)

# Distribute the target sample size among datasets
remaining_size = TARGET_SAMPLE_SIZE
samples_per_dataset = remaining_size // len(dataset_files)

# Load and sample each dataset
all_samples = []
for dataset_file in dataset_files:
    current_sample_size = min(remaining_size, samples_per_dataset)
    sampled_data = load_and_sample_data(os.path.join(DATASETS_DIR, dataset_file), current_sample_size)
    all_samples.append(sampled_data)
    remaining_size -= len(sampled_data)

# Concatenate all the sampled data
combined_reviews = pd.concat(all_samples, axis=0).reset_index(drop=True)

# Save to a new CSV
output_path = os.path.join(DATASETS_DIR, "combined_reviews.csv")
combined_reviews.to_csv(output_path, index=False)
print(f"Saved combined dataset to {output_path}")
