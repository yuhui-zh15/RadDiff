import pandas as pd
from pathlib import Path
import json

print("Creating small dataset for quick training...")

# Load full dataset
train_df = pd.read_csv('large_race_dataset/vit_train_mixed.csv')
val_df = pd.read_csv('large_race_dataset/vit_val_mixed.csv')

print(f"Original training set: {len(train_df):,}")
print(f"Original validation set: {len(val_df):,}")

# Sample 5000 per race from training
small_train = []
for label in [0, 1, 2]:
    race_data = train_df[train_df['label'] == label]
    n_sample = min(5000, len(race_data))
    sampled = race_data.sample(n=n_sample, random_state=42)
    small_train.append(sampled)
    print(f"  Train label {label}: sampled {n_sample:,}")

small_train_df = pd.concat(small_train).sample(frac=1, random_state=42)

# Sample 1000 per race from validation
small_val = []
for label in [0, 1, 2]:
    race_data = val_df[val_df['label'] == label]
    n_sample = min(1000, len(race_data))
    sampled = race_data.sample(n=n_sample, random_state=42)
    small_val.append(sampled)
    print(f"  Val label {label}: sampled {n_sample:,}")

small_val_df = pd.concat(small_val).sample(frac=1, random_state=42)

# Create directory
Path('small_race_dataset').mkdir(exist_ok=True)

# Save
small_train_df.to_csv('small_race_dataset/vit_train_mixed.csv', index=False)
small_val_df.to_csv('small_race_dataset/vit_val_mixed.csv', index=False)

# Copy and update metadata
with open('large_race_dataset/metadata_mixed.json', 'r') as f:
    metadata = json.load(f)

metadata['train_size'] = len(small_train_df)
metadata['val_size'] = len(small_val_df)
metadata['note'] = 'Small subset for quick training'

with open('small_race_dataset/metadata_mixed.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Small dataset created!")
print(f"  Training: {len(small_train_df):,} images")
print(f"  Validation: {len(small_val_df):,} images")
print(f"  Saved to: small_race_dataset/")
