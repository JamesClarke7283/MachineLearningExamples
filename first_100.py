import pandas as pd

# Load the first 100 entries from the yelp_reviews.csv dataset
data_path = './datasets/raw/yelp_reviews.csv'
data = pd.read_csv(data_path)

# Shuffle the data
shuffled_data = data.sample(frac=1, random_state=42)

# Get the first 100 entries from the shuffled data
first_100_data = shuffled_data.iloc[:1000]

# Save to a new CSV file called first_100_yelp.csv
output_path = './datasets/raw/first_1000_yelp.csv'
first_100_data.to_csv(output_path, index=False)
