import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
print('=== checkpoint metadata ===')
print('context_dim:', ckpt.get('context_dim'))
print('control_dim:', ckpt.get('control_dim'))
print('hidden_dim:', ckpt.get('hidden_dim'))
print()
print('=== actual weight shapes from state_dict ===')
for k, v in sd.items():
    print(f'{k}: {v.shape}')
print()
print('=== inferred dims ===')
context_encoder_w = sd['context_encoder.0.weight']
print(f'context_dim = {context_encoder_w.shape[1]}')
print(f'hidden_dim = {context_encoder_w.shape[0]}')
dynamics_w = sd['ode_func.dynamics_net.0.weight']
control_dim = dynamics_w.shape[1] - 1 - context_encoder_w.shape[0]
print(f'control_dim = {dynamics_w.shape[1]} - 1 - {context_encoder_w.shape[0]} = {control_dim}')
