import torch

class RolloutBuffer:
    def __init__(self, num_steps, state_dim, action_dim, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        
        # Pré-allocation des tenseurs pour éviter de ralentir la boucle
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
    for _ in range(buffer.num_steps):
        # Conversion de l'array Numpy en tenseur PyTorch
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
        
        #On désactive le calcul de gradient
        with torch.no_grad():
            action_tensor, logprob, _, value = policy.get_action_and_value(state_tensor)
            
        # Conversion du tenseur action vers un array NumPy pour le moteur physique
        action_numpy = action_tensor.cpu().numpy().flatten()
        next_state, reward, done = env.step(action_numpy)
        
        # Stockage dans le buffer
        buffer.add(
            state_tensor.squeeze(0),
            action_tensor.squeeze(0),
            logprob.squeeze(),
            torch.tensor(reward, dtype=torch.float32),
            value.squeeze(),
            torch.tensor(done, dtype=torch.float32)
        )
        
        current_state = next_state
        
        if done:
            current_state = env.reset()
            
    with torch.no_grad():
        final_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
        _, _, _, next_value = policy.get_action_and_value(final_state_tensor)
        
    return current_state, next_value.squeeze()