import torch


class RolloutBuffer:
    def __init__(self, num_steps, state_dim=34, action_dim=8, device="cpu"):
        self.num_steps = num_steps
        self.device = device

        # Pré-allocation des tenseurs
        self.states = torch.zeros((num_steps, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(num_steps, dtype=torch.float32, device=device)

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
    total_height = 0.0
    total_vx = 0.0
    falls_count = 0
    timeouts_count = 0

    for _ in range(buffer.num_steps):
        state_tensor = torch.as_tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, logprob, _, value, hidden = policy.get_action_and_value(
                state_tensor, hidden=hidden, return_hidden=True
            )

        action_numpy = action_tensor.detach().cpu().numpy().flatten()
        next_state, reward, done = env.step(action_numpy)

        if env.torso is not None:
            total_height += float(env.y_sol - env.torso.position.y)
            total_vx += float(env.torso.velocity.x)

        buffer.add(
            state_tensor.squeeze(0),
            action_tensor.squeeze(0),
            logprob.squeeze(),
            reward,
            value.squeeze(),
            done
        )

        current_state = next_state

        if getattr(env, "just_fallen", False):
            falls_count += 1

        if done:
            if getattr(env, "last_fallen", False) and not getattr(env, "just_fallen", False):
                falls_count += 1
            elif not getattr(env, "last_fallen", False):
                timeouts_count += 1
            current_state = env.reset()
            hidden = None  # Réinitialiser la mémoire récurrente à la fin de l'épisode

    policy.current_hidden = hidden

    with torch.no_grad():
        final_state_tensor = torch.as_tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, _, next_value, _ = policy.get_action_and_value(
            final_state_tensor, hidden=hidden, return_hidden=True
        )

    traj_info = {
        "mean_height": total_height / max(1, buffer.num_steps),
        "mean_vx": total_vx / max(1, buffer.num_steps),
        "falls": falls_count,
        "timeouts": timeouts_count,
    }

    return current_state, next_value.squeeze(), traj_info

