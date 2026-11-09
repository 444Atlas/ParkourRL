import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Evalue l'état V(s)
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        # Calcule la moyenne de la distribution d'actions
        self.actor_mean = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim))
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        
    def get_action_and_value(self, state, action=None, deterministic=False):
        # Calcul de la valeur V(s)
        value = self.critic(state)
        
        # Paramétrage de la distribution Normale avec plancher d'exploration (Feature 4.1)
        action_mean = self.actor_mean(state)
        clamped_log_std = torch.clamp(self.actor_log_std, min=-1.6, max=0.5)
        action_log_std = clamped_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        probs = Normal(action_mean, action_std)
        
        # Action déterministe (évaluation) ou échantillonnage (entraînement)
        if action is None:
            action = action_mean if deterministic else probs.sample()

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        
        return action, log_prob, entropy, value