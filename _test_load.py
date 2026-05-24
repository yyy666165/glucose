import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.neural_ode_glucose import NeuralODEGlucosePredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
state_dict = checkpoint['model_state_dict']

hidden_dim = state_dict['context_encoder.0.weight'].shape[0]
context_dim = state_dict['context_encoder.0.weight'].shape[1]
control_dim = state_dict['ode_func.dynamics_net.0.weight'].shape[1] - 1 - hidden_dim

print(f'inferred: hidden_dim={hidden_dim}, context_dim={context_dim}, control_dim={control_dim}')

model = NeuralODEGlucosePredictor(
    context_dim=context_dim,
    hidden_dim=hidden_dim,
    control_dim=control_dim,
).to(device)
model.load_state_dict(state_dict)
model.eval()
print('Model loaded successfully!')

# Quick test prediction
import numpy as np
initial_glucose = torch.tensor([[150.0]], dtype=torch.float32, device=device)
control = torch.zeros(1, 8, control_dim, dtype=torch.float32, device=device)
context = torch.zeros(1, context_dim, dtype=torch.float32, device=device)
with torch.no_grad():
    pred = model(initial_glucose, control, context)
print(f'Test prediction shape: {pred.shape}, values: {pred[0].cpu().numpy()}')
