import sys; sys.path.insert(0, '.')
from data.glucose_dataset import GlucoseDataset
print('Loading dataset...')
dataset = GlucoseDataset(data_dir='data', seq_len=24, context_dim=5)
print(f'Dataset len: {len(dataset)}')
batch = dataset[0]
print(f'control_sequence shape: {batch["control_sequence"].shape}')
print(f'initial_glucose: {batch["initial_glucose"]}')
print(f'context shape: {batch["context"].shape}')
print(f'target shape: {batch["target"].shape}')
