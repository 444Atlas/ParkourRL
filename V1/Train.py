import torch
import torch.optim as optim
import torch.nn.functional as F
from Env import Env, Normalizer, get_available_models
from Models import ActorCritic
import RolloutBuffer

import os
import sys
import json
import time


def load_warmstart(policy, source_model_path, target_action_dim):
    """
    Transfère les poids d'un modèle pré-entraîné (ex: ppo_standing.pth)
    pour initialiser l'apprentissage de la marche (Feature 3.1).
    Gère la compatibilité 1D -> 2D et formes identiques automatiquement.
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
            elif name == "actor_mean.4.weight" and src.shape[0] == 1 and param.shape[0] == 2:
                # jambe droite = +w, jambe gauche = -w pour reproduire l'équilibre
                param[0].copy_(src[0])
                param[1].copy_(-src[0])
                transferred.append(f"{name} (1D->2D symétrie)")
            elif name == "actor_mean.4.bias" and src.shape[0] == 1 and param.shape[0] == 2:
                param[0].copy_(src[0])
                param[1].copy_(-src[0])
                transferred.append(f"{name} (1D->2D symétrie)")
            elif name == "actor_log_std" and src.shape[1] == 1 and param.shape[1] == 2:
                param[0, 0].copy_(src[0, 0])
                param[0, 1].copy_(src[0, 0])
                transferred.append(f"{name} (1D->2D expand)")

    policy.load_state_dict(curr_dict)
    print(f"[Warm-start] Succès ! Poids transférés depuis '{source_model_path}' ({len(transferred)} tenseurs).")
    return True


def prompt_model_choice(available_models=None):
    """Propose un menu interactif dans le terminal pour choisir le modèle et le robot à entraîner."""
    if not sys.stdin.isatty():
        return "walking", "5_membres", "walking_5_membres"

    entries = [
        {
            "key": "walking_5_membres",
            "label": "WALKING (5 MEMBRES)",
            "task": "walking",
            "robot_type": "5_membres",
            "path": "models/ppo_walking_5_membres.pth",
            "status": "existant" if os.path.exists("models/ppo_walking_5_membres.pth") else "nouveau",
        },
        {
            "key": "standing_5_membres",
            "label": "STANDING (5 MEMBRES)",
            "task": "standing",
            "robot_type": "5_membres",
            "path": "models/ppo_standing_5_membres.pth",
            "status": "existant" if os.path.exists("models/ppo_standing_5_membres.pth") else "nouveau",
        },
        {
            "key": "walking_3_membres",
            "label": "WALKING (3 MEMBRES)",
            "task": "walking",
            "robot_type": "3_membres",
            "path": "models/ppo_walking_3_membres.pth",
            "status": "existant" if os.path.exists("models/ppo_walking_3_membres.pth") else "nouveau",
        },
        {
            "key": "standing_3_membres",
            "label": "STANDING (3 MEMBRES)",
            "task": "standing",
            "robot_type": "3_membres",
            "path": "models/ppo_standing_3_membres.pth",
            "status": "existant" if os.path.exists("models/ppo_standing_3_membres.pth") else "nouveau",
        },
        {
            "key": "walking_2_membres",
            "label": "WALKING (2 MEMBRES)",
            "task": "walking",
            "robot_type": "2_membres",
            "path": "models/ppo_walking_2_membres.pth",
            "status": "existant" if os.path.exists("models/ppo_walking_2_membres.pth") else "nouveau",
        },
        {
            "key": "standing_2_membres",
            "label": "STANDING (2 MEMBRES)",
            "task": "standing",
            "robot_type": "2_membres",
            "path": "models/ppo_standing_2_membres.pth",
            "status": "existant" if os.path.exists("models/ppo_standing_2_membres.pth") else "nouveau",
        },
    ]

    print("\n" + "=" * 70)
    print("      SÉLECTION INTERACTIVE DU MODÈLE ET DU ROBOT À ENTRAÎNER")
    print("=" * 70)
    for idx, entry in enumerate(entries, 1):
        status_tag = f"[{entry['status']}]"
        print(f"  [{idx}] {entry['label']:<24} ({entry['path']}) {status_tag:<11}")
    print("=" * 70)

    try:
        choice = input(f"Choisissez le numéro [1-{len(entries)}] (défaut: 1) : ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            sel = entries[int(choice) - 1]
            return sel["task"], sel["robot_type"], sel["key"]
        elif choice:
            for entry in entries:
                if choice.lower() in (entry["key"].lower(), entry["label"].lower(), entry["task"].lower()):
                    return entry["task"], entry["robot_type"], entry["key"]
    except Exception:
        pass

    return entries[0]["task"], entries[0]["robot_type"], entries[0]["key"]


def train(
    task="walking",
    robot_type="5_membres",
    action_dim=None,
    total_timesteps=500000,
    num_steps=2048,
    lr=3e-4,
    warmstart="auto",
    save_dir="models",
    model_name=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "5" in str(robot_type):
        robot_type = "5_membres"
    elif "2" in str(robot_type):
        robot_type = "2_membres"
    else:
        robot_type = "3_membres"

    if robot_type == "5_membres":
        state_dim = 20
        action_dim = 4 if action_dim is None else action_dim
        normalizer = Normalizer(state_dim)
    else:
        state_dim = 9
        action_dim = 2 if action_dim is None else action_dim
        normalizer = None
    
    # Résolution du nom de modèle cible et des noms de fichiers
    if model_name:
        out_name = model_name
    else:
        out_name = f"{task}_{robot_type}"
    model_filename = out_name if out_name.endswith(".pth") else (f"{out_name}.pth" if out_name.startswith("ppo_") else f"ppo_{out_name}.pth")
    meta_filename = model_filename.replace(".pth", ".json")

    num_updates = max(1, total_timesteps // num_steps)

    print("=" * 70)
    print("      LANCEMENT DE L'ENTRAÎNEMENT PPO AVANCÉ")
    print("=" * 70)
    print(f"Modèle cible        : {out_name.upper()} ({model_filename})")
    print(f"Morphologie robot   : {robot_type.upper()}")
    print(f"Tâche               : {task.upper()} ({'Marche - reward_walking' if task == 'walking' else 'Équilibre - reward_standing'})")
    print(f"Dimension d'état    : {state_dim} ({'Normalisation en ligne Welford' if normalizer is not None else 'Statique'})")
    print(f"Dimension d'action  : {action_dim} ({'4 joints (2 hanches + 2 genoux)' if action_dim == 4 else '2 jambes'})")
    print(f"Volume timesteps    : {total_timesteps:,} ({num_updates} itérations de {num_steps} pas)")
    print(f"Learning Rate       : {lr}")
    print(f"Appareil (Device)   : {device}")
    print(f"Dossier de sortie   : {save_dir}/")
    print("=" * 70)
    
    # Instanciations
    env = Env(render_mode=False, task=task, robot_type=robot_type, normalizer=normalizer, update_normalizer=(normalizer is not None))
    policy = ActorCritic(state_dim, action_dim).to(device)

    # Feature 3.1 : Warm-start depuis le modèle d'équilibre (standing) correspondant
    warmstart_path = None
    if warmstart and warmstart.lower() != "none":
        if warmstart != "auto" and os.path.exists(warmstart):
            warmstart_path = warmstart
        elif warmstart == "auto" and task == "walking":
            candidate = os.path.join(save_dir, f"ppo_standing_{robot_type}.pth")
            if os.path.exists(candidate):
                warmstart_path = candidate
            elif os.path.exists(os.path.join(save_dir, "ppo_standing.pth")) and robot_type != "5_membres":
                warmstart_path = os.path.join(save_dir, "ppo_standing.pth")
            elif os.path.exists("ppo_robot_model.pth") and robot_type != "5_membres":
                warmstart_path = "ppo_robot_model.pth"

    if warmstart_path:
        load_warmstart(policy, warmstart_path, action_dim)
        if normalizer is not None:
            warm_meta = warmstart_path.replace(".pth", ".json")
            if os.path.exists(warm_meta):
                try:
                    with open(warm_meta, "r") as f:
                        meta_data = json.load(f)
                    if meta_data.get("normalizer") is not None:
                        normalizer.load_state_dict(meta_data["normalizer"])
                        print(f"[Warm-start] Normalisateur d'état Welford chargé depuis '{warm_meta}'.")
                except Exception as e:
                    print(f"[Warm-start] Avertissement : impossible de charger le normalisateur : {e}")
    else:
        if task == "standing":
            # Exploration initiale modérée pour l'équilibre (sigma ≈ 0.50 au lieu de 1.0)
            policy.actor_log_std.data.fill_(-0.7)
            print("[Initialisation] actor_log_std fixé à -0.7 (sigma ≈ 0.50) pour un équilibrage fin.")
        else:
            print("[Warm-start] Aucun point de départ spécifié ou trouvé. Entraînement depuis zéro.")

    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer.RolloutBuffer(num_steps, state_dim, action_dim, device)
    
    current_state = env.reset()
    start_time = time.time()
    save_path = os.path.join(save_dir, model_filename)
    meta_path = os.path.join(save_dir, meta_filename)
    save_interval = 100000

    print("-" * 75)
    print(f"{'Update':<12}{'Progression':<14}{'Reward':<12}{'Vx (px/s)':<12}{'Hauteur':<10}{'Chutes':<8}{'PGLoss':<10}")
    print("-" * 75)

    vx_idx = 4 if robot_type == "5_membres" else 7

    # Boucle RL
    for update in range(num_updates):
        buffer.reset()
        
        # Collecte
        current_state, next_value = RolloutBuffer.collect_trajectories(env, policy, buffer, current_state, device)
        
        # Avantages GAE
        advantages, returns = compute_gae(buffer, next_value)
        
        # Optimisation PPO
        pg_loss, v_loss, ent_loss = ppo_update(policy, optimizer, buffer, advantages, returns)
        
        # Métriques du lot
        mean_reward = buffer.rewards.mean().item()
        mean_vx = buffer.states[:, vx_idx].mean().item() * 100.0
        mean_height = buffer.states[:, 0].mean().item() * 150.0
        current_steps = (update + 1) * num_steps
        pct = (current_steps / total_timesteps) * 100.0
        dones_count = int(buffer.dones.sum().item())

        if (update + 1) % 5 == 0 or update == 0 or update == num_updates - 1:
            print(f"{update + 1}/{num_updates:<6} {pct:5.1f}% ({current_steps:7,d})  {mean_reward:+8.3f}   {mean_vx:+8.2f}    {mean_height:6.1f}px   {dones_count:<6d}  {pg_loss:+.4f}")

        # Sauvegarde périodique de sécurité
        if current_steps % save_interval < num_steps and update < num_updates - 1:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(policy.state_dict(), save_path)

    elapsed = time.time() - start_time

    # Sauvegarde finale du modèle et des métadonnées
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), save_path)

    metadata = {
        "task": task,
        "robot_type": robot_type,
        "model_name": out_name,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
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

    if task == "standing" and robot_type == "2_membres":
        torch.save(policy.state_dict(), "ppo_robot_model.pth")


def ppo_update(policy, optimizer, buffer, advantages, returns, clip_coef=0.2, ent_coef=0.05, vf_coef=0.5, epochs=4):
    # Normalisation des avantages pour stabiliser l'apprentissage
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    last_pg = 0.0
    last_v = 0.0
    last_ent = 0.0

    for _ in range(epochs):
        _, newlogprob, entropy, newvalue = policy.get_action_and_value(buffer.states, buffer.actions)
        
        logratio = newlogprob - buffer.logprobs
        ratio = logratio.exp()
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        v_loss = F.mse_loss(newvalue.squeeze(), returns)
        entropy_loss = entropy.mean()

        loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
        
        # Descente de gradient
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        last_pg = pg_loss.item()
        last_v = v_loss.item()
        last_ent = entropy_loss.item()

    return last_pg, last_v, last_ent


def compute_gae(buffer, next_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(buffer.rewards).to(buffer.device)
    lastgaelam = 0
    
    for t in reversed(range(buffer.num_steps)):
        if t == buffer.num_steps - 1:
            nextnonterminal = 1.0 
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - buffer.dones[t+1]
            nextvalues = buffer.values[t+1]
            
        delta = buffer.rewards[t] + gamma * nextvalues * nextnonterminal - buffer.values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        
    returns = advantages + buffer.values
    return advantages, returns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement PPO du robot (marche ou standing).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nom ou identifiant du modèle à entraîner (ex: 'walking', 'standing'). Si omis, sélection interactive."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["walking", "standing"],
        default=None,
        help="Tâche à entraîner : 'walking' (avancer vers la droite) ou 'standing' (tenir debout)."
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        choices=[1, 2, 4],
        default=None,
        help="Dimension d'action (4 pour 5_membres, 2 pour 2 ou 3 membres, 1 pour couple couplé)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Nombre total de pas d'entraînement (défaut: 500 000)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate pour l'optimiseur Adam (défaut: 3e-4)."
    )
    parser.add_argument(
        "--warmstart",
        type=str,
        default="auto",
        help="Chemin du modèle pré-entraîné pour warm-start ('auto' détecte ppo_standing_{robot}.pth si task=walking, ou 'none')."
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Désactive le warm-start et force un entraînement depuis zéro."
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=["2", "3", "5", "2_membres", "3_membres", "5_membres"],
        default=None,
        help="Morphologie du robot : '5_membres' (genoux articulés), '3_membres' (tronc vertical), ou '2_membres'."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Dossier où stocker proprement le modèle et ses métadonnées (défaut: 'models')."
    )
    args = parser.parse_args()

    available_models = get_available_models()

    selected_task = args.task
    selected_model = args.model
    selected_robot = args.robot

    # Sélection interactive si ni task ni model n'ont été passés en ligne de commande
    if selected_task is None and selected_model is None:
        selected_task, sel_robot, selected_model = prompt_model_choice(available_models)
        if selected_robot is None:
            selected_robot = sel_robot
    elif selected_model is not None:
        if selected_task is None:
            selected_task = "walking" if "walk" in selected_model.lower() else "standing"
        if selected_robot is None:
            if "5" in selected_model.lower():
                selected_robot = "5_membres"
            elif "2" in selected_model.lower():
                selected_robot = "2_membres"
            else:
                selected_robot = "3_membres"
    else:
        if selected_robot is None:
            selected_robot = "5_membres"
        r_suffix = "5_membres" if "5" in str(selected_robot) else ("2_membres" if "2" in str(selected_robot) else "3_membres")
        selected_model = f"{selected_task}_{r_suffix}"

    if selected_robot is None:
        selected_robot = "5_membres"
    if "5" in str(selected_robot):
        selected_robot = "5_membres"
    elif "2" in str(selected_robot):
        selected_robot = "2_membres"
    else:
        selected_robot = "3_membres"

    warmstart_arg = "none" if args.no_warmstart else args.warmstart

    train(
        task=selected_task,
        robot_type=selected_robot,
        action_dim=args.action_dim,
        total_timesteps=args.timesteps,
        lr=args.lr,
        warmstart=warmstart_arg,
        save_dir=args.save_dir,
        model_name=selected_model,
    )