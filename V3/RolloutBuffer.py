import torch

class RolloutBuffer:
    def __init__(self, num_steps, state_dim=28, action_dim=6, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        
        # Pré-allocation des tenseurs
        self.states = torch.zeros((num_steps, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, action_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.values = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.dones = torch.zeros(num_steps, dtype=torch.float32).to(device)
        
        self.step = 0

    def add(self, state, action, logprob, reward, value, done):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1

    def reset(self):
        self.step = 0

def collect_trajectories(env, policy, buffer, current_state, device="cpu"):
    hidden = getattr(policy, "current_hidden", None)
    for _ in range(buffer.num_steps):
        state_tensor = torch.as_tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action_tensor, logprob, _, value, hidden = policy.get_action_and_value(
                state_tensor, hidden=hidden, return_hidden=True
            )
            
        action_numpy = action_tensor.detach().cpu().numpy().flatten()
        next_state, reward, done = env.step(action_numpy)
        
        buffer.add(
            state_tensor.squeeze(0),
            action_tensor.squeeze(0),
            logprob.squeeze(),
            reward,
            value.squeeze(),
            done
        )
        
        current_state = next_state
        
        if done:
            current_state = env.reset()
            hidden = None  # Réinitialiser la mémoire récurrente à la fin de l'épisode
            
    policy.current_hidden = hidden
    
    with torch.no_grad():
        final_state_tensor = torch.as_tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, _, next_value, _ = policy.get_action_and_value(
            final_state_tensor, hidden=hidden, return_hidden=True
        )
        
    return current_state, next_value.squeeze()
