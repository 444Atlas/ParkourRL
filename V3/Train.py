import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Env import Env, Normalizer, get_available_models
from Models import ActorCritic
import RolloutBuffer

import os
import sys
import json
import time
import argparse


def load_warmstart(policy, source_model_path, target_action_dim=6):
    """
    Transfère les poids d'un modèle pré-entraîné (ex: ppo_standing.pth)
    pour initialiser l'apprentissage.
    """
    if not os.path.exists(source_model_path):
        print(f"[Warm-start] Fichier '{source_model_path}' introuvable. Entraînement à partir de zéro.")
        return False

    loaded = torch.load(source_model_path, map_location="cpu")
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded

    curr_dict = policy.state_dict()
    transferred = []

    for name, param in curr_dict.items():
        if name in state_dict:
            src = state_dict[name]
            if src.shape == param.shape:
                param.copy_(src)
                transferred.append(name)
            elif "weight" in name and param.dim() == 2 and src.dim() == 2:
                r = min(param.shape[0], src.shape[0])
                c = min(param.shape[1], src.shape[1])
                param[:r, :c].copy_(src[:r, :c])
                transferred.append(f"{name} ({r}x{c})")
            elif "bias" in name and param.dim() == 1 and src.dim() == 1:
                r = min(param.shape[0], src.shape[0])
                param[:r].copy_(src[:r])
                transferred.append(f"{name} ({r})")
            elif name == "actor_log_std" and src.shape[1] <= param.shape[1]:
                param[:, :src.shape[1]].copy_(src)
                transferred.append(f"{name} (partiel)")

    policy.load_state_dict(curr_dict)
    print(f"[Warm-start] Succès ! Poids transférés depuis '{source_model_path}' ({len(transferred)} tenseurs).")
    return True


def resolve_models_dir(models_dir=None):
    if models_dir is not None and os.path.isdir(models_dir):
        return models_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(script_dir, "models")
    if os.path.isdir(cand):
        return cand
    return models_dir or "models"


def prompt_model_choice(available_models=None, save_dir=None):
    """Propose un menu interactif dans le terminal pour choisir la tâche à entraîner."""
    if not sys.stdin.isatty():
        return "walking", "walking"

    models_dir = resolve_models_dir(save_dir)
    walk_path = os.path.join(models_dir, "ppo_walking.pth")
    stand_path = os.path.join(models_dir, "ppo_standing.pth")

    entries = [
        {
            "key": "walking",
            "label": "WALKING (Marche Bipède)",
            "task": "walking",
            "path": walk_path,
            "status": "existant" if os.path.exists(walk_path) else "nouveau",
        },
        {
            "key": "standing",
            "label": "STANDING (Équilibre Debout)",
            "task": "standing",
            "path": stand_path,
            "status": "existant" if os.path.exists(stand_path) else "nouveau",
        },
    ]

    print("\n" + "=" * 70)
    print("      SÉLECTION DE LA TÂCHE À ENTRAÎNER (ROBOT BIPÈDE)")
    print("=" * 70)
    for idx, entry in enumerate(entries, 1):
        status_tag = f"[{entry['status']}]"
        print(f"  [{idx}] {entry['label']:<28} ({entry['path']}) {status_tag:<11}")
    print("=" * 70)

    try:
        choice = input(f"Choisissez le numéro [1-{len(entries)}] (défaut: 1) : ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            sel = entries[int(choice) - 1]
            return sel["task"], sel["key"]
        elif choice:
            for entry in entries:
                if choice.lower() in (entry["key"].lower(), entry["label"].lower(), entry["task"].lower()):
                    return entry["task"], entry["key"]
    except Exception:
        pass

    return entries[0]["task"], entries[0]["key"]


def train(task="walking",total_timesteps=500000,num_steps=2048,lr=3e-4,warmstart="auto",save_dir=None,model_name=None,device="cpu"):
    save_dir = resolve_models_dir(save_dir)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dummy_env = Env(render_mode=False, task=task)
    state_dim = dummy_env.state_dim
    action_dim = dummy_env.action_dim
    dummy_env.close()
    normalizer = Normalizer(state_dim)

    # Nom du modèle cible et des fichiers
    if model_name:
        out_name = model_name
    else:
        out_name = f"{task}"
    model_filename = out_name if out_name.endswith(".pth") else (f"{out_name}.pth" if out_name.startswith("ppo_") else f"ppo_{out_name}.pth")
    meta_filename = model_filename.replace(".pth", ".json")

    num_updates = max(1, total_timesteps // num_steps)

    print("=" * 70)
    print("      LANCEMENT DE L'ENTRAÎNEMENT PPO (ROBOT BIPÈDE V3 - CPG)")
    print("=" * 70)
    print(f"Modèle cible        : {out_name.upper()} ({model_filename})")
    print(f"Morphologie robot   : Robot Bipède articulé (Tronc + 2 Bras + 2 Cuisses + 2 Tibias + Pieds)")
    print(f"Tâche               : {task.upper()} ({'Marche - reward_walking' if task == 'walking' else 'Équilibre - reward_standing'})")
    print(f"Dimension d'état    : {state_dim} (Normalisation en ligne Welford O(1))")
    print(f"Dimension d'action  : {action_dim} (2 hanches + 2 genoux + 2 bras)")
    print(f"Volume timesteps    : {total_timesteps:,} ({num_updates} itérations de {num_steps} pas)")
    print(f"Learning Rate       : {lr}")
    print(f"Appareil (Device)   : {device}")
    print(f"Dossier de sortie   : {save_dir}/")
    print("=" * 70)
    
    # Environnement et politique
    env = Env(render_mode=False, task=task, normalizer=normalizer, update_normalizer=True)
    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)

    # Warm-start depuis le modèle d'équilibre
    warmstart_path = None
    if warmstart and warmstart.lower() != "none":
        if warmstart != "auto" and os.path.exists(warmstart):
            warmstart_path = warmstart
        elif warmstart == "auto" and task == "walking":
            candidate = os.path.join(save_dir, "ppo_standing.pth")
            if os.path.exists(candidate):
                warmstart_path = candidate

    if warmstart_path:
        load_warmstart(policy, warmstart_path, action_dim)
        warm_meta = warmstart_path.replace(".pth", ".json")
        if os.path.exists(warm_meta):
            try:
                with open(warm_meta, "r") as f:
                    meta_data = json.load(f)
                if meta_data.get("normalizer") is not None:
                    old_mean = meta_data["normalizer"].get("mean", [])
                    old_var = meta_data["normalizer"].get("var", [])
                    n_old = min(len(old_mean), state_dim)
                    normalizer.mean[:n_old] = np.array(old_mean[:n_old], dtype=np.float32)
                    normalizer.var[:n_old] = np.array(old_var[:n_old], dtype=np.float32)
                    normalizer.count = float(meta_data["normalizer"].get("count", 1.0))
                    print(f"[Warm-start] Normalisateur Welford chargé depuis '{warm_meta}'.")
            except Exception as e:
                print(f"[Warm-start] Avertissement chargement normalisateur : {e}")
    else:
        if task == "standing":
            policy.actor_log_std.data.fill_(-0.7)
            print("[Initialisation] actor_log_std fixé à -0.7 (sigma ≈ 0.50) pour un équilibrage fin.")
        else:
            print("[Warm-start] Aucun point de départ spécifié. Entraînement depuis zéro.")

    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer.RolloutBuffer(num_steps, state_dim, action_dim, device)
    
    current_state = env.reset()
    start_time = time.time()
    save_path = os.path.join(save_dir, model_filename)
    meta_path = os.path.join(save_dir, meta_filename)

    print("-" * 75)
    print(f"{'Update':<12}{'Progression':<14}{'Reward':<12}{'Vx (px/s)':<12}{'Hauteur':<10}{'Chutes':<8}{'PGLoss':<10}")
    print("-" * 75)

    vx_idx = 4
    hauteur_idx = 0

    for update in range(num_updates):
        buffer.reset()
        
        current_state, next_value = RolloutBuffer.collect_trajectories(env, policy, buffer, current_state, device)
        advantages, returns = compute_gae(buffer, next_value)
        pg_loss, v_loss, ent_loss = ppo_update(policy, optimizer, buffer, advantages, returns)
        
        mean_reward = buffer.rewards.mean().item()
        mean_vx = buffer.states[:, vx_idx].mean().item() * 100.0
        mean_height = buffer.states[:, hauteur_idx].mean().item() * 150.0
        current_steps = (update + 1) * num_steps
        pct = (current_steps / total_timesteps) * 100.0
        dones_count = int(buffer.dones.sum().item())

        if (update + 1) % 5 == 0 or update == 0 or update == num_updates - 1:
            print(f"{update+1:<4}/{num_updates:<6} {pct:5.1f}% ({current_steps:>8,})   {mean_reward:+8.3f}    {mean_vx:+7.2f}    {mean_height:6.1f}px   {dones_count:<7} {pg_loss:+8.4f}")

    elapsed = time.time() - start_time
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), save_path)

    metadata = {
        "task": task,
        "model_name": out_name,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "normalizer": normalizer.state_dict(),
        "total_timesteps": total_timesteps,
        "warmstart": warmstart_path,
        "training_time_seconds": round(elapsed, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("-" * 70)
    print(f"\nEntraînement terminé avec succès en {elapsed:.1f}s ({total_timesteps / elapsed:.0f} steps/s) !")
    print(f"Modèle sauvegardé sous : '{save_path}'")
    print(f"Métadonnées sauvegardées sous : '{meta_path}'")


def ppo_update(policy, optimizer, buffer, advantages, returns, clip_coef=0.2, ent_coef=0.05, vf_coef=0.5, epochs=4, seq_len=64):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    last_pg = 0.0
    last_v = 0.0
    last_ent = 0.0

    for _ in range(epochs):
        if hasattr(policy, "evaluate_sequence"):
            newlogprob, entropy, newvalue = policy.evaluate_sequence(buffer.states, buffer.actions, seq_len=seq_len)
        else:
            _, newlogprob, entropy, newvalue = policy.get_action_and_value(buffer.states, buffer.actions)
        
        logratio = newlogprob - buffer.logprobs
        ratio = logratio.exp()
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        v_loss = F.mse_loss(newvalue.squeeze(), returns)
        entropy_loss = entropy.mean()

        loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        last_pg = pg_loss.item()
        last_v = v_loss.item()
        last_ent = entropy_loss.item()

    return last_pg, last_v, last_ent


def compute_gae(buffer, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(buffer.rewards)
    lastgaelam = 0.0
    
    for t in reversed(range(buffer.num_steps)):
        if t == buffer.num_steps - 1:
            nextnonterminal = 1.0 - buffer.dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - buffer.dones[t]
            nextvalues = buffer.values[t + 1]
            
        delta = buffer.rewards[t] + gamma * nextvalues * nextnonterminal - buffer.values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
    returns = advantages + buffer.values
    return advantages, returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement PPO pour Robot Bipède (V2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["walking", "standing"],
        default=None,
        help="Tâche à entraîner : 'walking' (marche) ou 'standing' (équilibre debout)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nom personnalisé du modèle cible."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Nombre total de pas d'entraînement."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate pour l'optimiseur Adam."
    )
    parser.add_argument(
        "--warmstart",
        type=str,
        default="auto",
        help="Modèle pré-entraîné pour warm-start ('auto' ou chemin .pth)."
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Désactive le warm-start et force un entraînement depuis zéro."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Dossier où stocker le modèle et ses métadonnées (défaut: models)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Appareil PyTorch (défaut: cpu, ~3.5x plus rapide pour Pymunk pas-à-pas monothread)."
    )
    args = parser.parse_args()

    models_dir = resolve_models_dir(args.save_dir)
    available_models = get_available_models(models_dir)
    selected_task = args.task
    selected_model = args.model

    if selected_task is None and selected_model is None:
        selected_task, selected_model = prompt_model_choice(available_models, save_dir=models_dir)
    elif selected_model is not None and selected_task is None:
        selected_task = "walking" if "walk" in selected_model.lower() else "standing"
    elif selected_task is not None and selected_model is None:
        selected_model = f"ppo_{selected_task}"

    warmstart_arg = "none" if args.no_warmstart else args.warmstart

    train(
        task=selected_task,
        total_timesteps=args.timesteps,
        lr=args.lr,
        warmstart=warmstart_arg,
        save_dir=args.save_dir,
        model_name=selected_model,
        device=args.device
    )

