import pandas as pd
from model import compute_similarity

# Load dataset
df = pd.read_csv("F:/NEURON_PROJECT/Dataset/DataNeuron_Text_Similarity.csv")  # replace with the actual file name

# Check columns (example names: 'text1', 'text2')
print(df.head())

# Compute similarity for each row
df['similarity_score'] = df.apply(
    lambda row: compute_similarity(row['text1'], row['text2']),
    axis=1
)

# Save results
df.to_csv("similarity_results.csv", index=False)

print("Saved similarity results to similarity_results.csv")
