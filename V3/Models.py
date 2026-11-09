import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Architecture Actor-Critic Récurrente (LSTM) pour le contrôle de locomotion,
    basée sur l'article Visual CPG-RL (Bellegarda, Shafiee, Ijspeert).
    Une couche LSTM (mémoire temporelle) traite la dynamique séquentielle,
    suivie de têtes denses pour la politique (Actor) et la fonction de valeur (Critic).
    """
    def __init__(self, state_dim=28, action_dim=6, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Couche récurrente LSTM (mémoire temporelle)
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)

        # Tête Critic V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Tête Actor : moyenne des paramètres de modulation du CPG
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward_lstm(self, state, hidden=None):
        # Format d'entrée : (batch_size, seq_len, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
        lstm_out, next_hidden = self.lstm(state, hidden)
        return lstm_out, next_hidden

    def get_action_and_value(self, state, action=None, hidden=None, deterministic=False, return_hidden=False):
        lstm_out, next_hidden = self.forward_lstm(state, hidden)
        feat = lstm_out[:, -1, :]

        action_mean = self.actor_mean(feat)
        value = self.critic(feat)

        clamped_log_std = torch.clamp(self.actor_log_std, min=-1.6, max=0.5)
        action_std = torch.exp(clamped_log_std.expand_as(action_mean))
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean if deterministic else probs.sample()

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)

        if return_hidden:
            return action, log_prob, entropy, value, next_hidden
        return action, log_prob, entropy, value

    def evaluate_sequence(self, states, actions, seq_len=64, initial_h=None):
        """Évaluation séquentielle vectorisée pour les mises à jour PPO."""
        total_steps = states.shape[0]
        rem = total_steps % seq_len
        if rem != 0:
            pad_len = seq_len - rem
            states_padded = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)
            actions_padded = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)
            total_padded = states_padded.shape[0]
        else:
            states_padded = states
            actions_padded = actions
            total_padded = total_steps

        num_seqs = total_padded // seq_len
        states_seq = states_padded.view(num_seqs, seq_len, self.state_dim)
        lstm_out, _ = self.lstm(states_seq, initial_h)
        feat = lstm_out.reshape(total_padded, self.hidden_dim)[:total_steps]

        action_mean = self.actor_mean(feat)
        values = self.critic(feat).squeeze(-1)

        clamped_log_std = torch.clamp(self.actor_log_std, min=-1.6, max=0.5)
        action_std = torch.exp(clamped_log_std.expand_as(action_mean))
        probs = Normal(action_mean, action_std)

        log_probs = probs.log_prob(actions).sum(-1)
        entropy = probs.entropy().sum(-1)
        return log_probs, entropy, values
