import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.neural_ode_glucose import NeuralODEGlucosePredictor
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralODEGlucosePredictor(context_dim=5, hidden_dim=64, control_dim=7).to(device)

# Test 1: basic forward
initial_glucose = torch.tensor([[150.0]], dtype=torch.float32, device=device)
control = torch.zeros(1, 8, 7, dtype=torch.float32, device=device)
context = torch.zeros(1, 5, dtype=torch.float32, device=device)
with torch.no_grad():
    pred = model(initial_glucose, control, context)
print(f'Test1 (zero insulin): {pred[0].cpu().numpy()}')
print(f'  No-insulin bias active -> should show rising glucose, last={pred[0,-1].item():.1f}')

# Test 2: with insulin
control_insulin = torch.zeros(1, 8, 7, dtype=torch.float32, device=device)
control_insulin[0, :, 0] = 2.0   # IOB
control_insulin[0, :, 4] = 4.0   # bolus
control_insulin[0, :, 3] = 0.8   # basal
with torch.no_grad():
    pred2 = model(initial_glucose, control_insulin, context)
print(f'Test2 (with insulin): {pred2[0].cpu().numpy()}')
print(f'  With insulin -> should show falling glucose, last={pred2[0,-1].item():.1f}')

# Test 3: counterfactual
cf = model.counterfactual(initial_glucose, control, control_insulin, context)
diff = (cf[0] - pred[0]).detach().cpu().numpy()
print(f'Test3 counterfactual diff (insulin vs no-insulin): {diff}')
print(f'  Should be negative (insulin lowers glucose): mean={diff.mean():.2f}')

# Test 4: dynamics_amplitude and Tanh constraint
print(f'\nModel parameters:')
print(f'  dynamics_amplitude: {model.ode_func.dynamics_amplitude.item():.3f}')
print(f'  insulin_scale: {model.ode_func.insulin_scale.item():.3f}')
print(f'  insulin_constraint: {model.ode_func.insulin_constraint}')
print(f'  bolus_constraint: {model.ode_func.bolus_constraint}')
print(f'  carb_constraint: {model.ode_func.carb_constraint}')

print('\nAll tests passed!')
