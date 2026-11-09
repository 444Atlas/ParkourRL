import torch
import math
import os
import sys
import pygame
import argparse
from Env import Env, get_available_models, load_policy_auto


def prompt_model_choice(available_models):
    """Propose un menu interactif dans le terminal pour choisir le modèle à évaluer."""
    if not available_models:
        return None

    keys = list(available_models.keys())
    if len(keys) == 1:
        return keys[0]

    if not sys.stdin.isatty():
        return keys[0]

    print("\n" + "=" * 70)
    print("      SÉLECTION DU MODÈLE À ÉVALUER (ROBOT BIPÈDE V2)")
    print("=" * 70)
    for idx, key in enumerate(keys, 1):
        path = available_models[key]
        task = "Marche (Walking)" if "walk" in key.lower() else "Équilibre (Standing)"
        print(f"  [{idx}] {key.upper():<28} ({path}) - Tâche: {task}")
    print("=" * 70)

    try:
        choice = input(f"Choisissez le numéro [1-{len(keys)}] (défaut: 1) : ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        elif choice.lower() in [k.lower() for k in keys]:
            for k in keys:
                if k.lower() == choice.lower():
                    return k
    except Exception:
        pass

    return keys[0]


def evaluate(model_name=None, num_episodes=None, models_dir=None):
    device = torch.device("cpu")
    available = get_available_models(models_dir)
    model_keys = list(available.keys())

    if not available:
        print("Erreur : Aucun modèle .pth trouvé.")
        print("Pour entraîner un modèle, lancez : python Train.py --task walking")
        return

    # Résolution du modèle initial
    current_model_key = None
    if model_name:
        if os.path.exists(model_name):
            current_model_key = os.path.splitext(os.path.basename(model_name))[0]
            available[current_model_key] = model_name
            if current_model_key not in model_keys:
                model_keys.append(current_model_key)
        else:
            for k in model_keys:
                if model_name.lower() in k.lower():
                    current_model_key = k
                    break

    if current_model_key is None:
        current_model_key = prompt_model_choice(available)

    task = "walking" if "walk" in current_model_key.lower() else "standing"
    env = Env(render_mode=True, task=task)

    resolved_path = available[current_model_key]
    try:
        policy, _ = load_policy_auto(resolved_path, state_dim=env.state_dim, action_dim=env.action_dim, device=device)
        if getattr(policy, "normalizer", None):
            env.normalizer = policy.normalizer
    except Exception as e:
        print(f"Erreur lors du chargement de '{resolved_path}' : {e}")
        return

    print("\n" + "=" * 65)
    print("      ÉVALUATION INTERACTIVE - ROBOT BIPÈDE V2")
    print("=" * 65)
    print(f"Modèle actif       : {current_model_key.upper()} ({resolved_path})")
    print(f"Tâche              : {task.upper()} ({'reward_walking' if task == 'walking' else 'reward_standing'})")
    print(f"Dimension d'action : {env.action_dim} articulations (2 hanches + 2 genoux + 2 bras)")
    print(f"Dimension d'état   : {env.state_dim} variables")
    print(f"Normalisation état : {'Normalizer Welford actif' if getattr(policy, 'normalizer', None) else 'Brut'}")
    print(f"Vitesse initiale   : {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS - temps réel)")
    print("\nContrôles interactifs dans la fenêtre :")
    print("  - P ou Tab         : Changer de modèle à la volée (Standing / Walking)")
    print("  - Espace           : Pause / Reprendre")
    print("  - R                : Réinitialiser l'épisode (position verticale standard)")
    print("  - U                : Appliquer une grande force sur la tête (perturbation dynamique gauche/droite ±45°)")
    print("  - D                : Basculer mode Déterministe / Stochastique")
    print("  - V ou +/-         : Ajuster la vitesse d'affichage")
    print("  - Échap / Croix    : Quitter")
    print("=" * 65 + "\n")

    episode = 1
    total_reward = 0.0
    reward = 0.0
    current_state = env.reset(randomize=False)
    paused = False
    action_numpy = [0.0] * env.action_dim
    deterministic = (task == "standing")

    try:
        with torch.no_grad():
            while num_episodes is None or episode <= num_episodes:
                robot_x = env.torso.position.x
                vx = env.torso.velocity.x
                head_deg = math.degrees(env.torso.angle)
                speed_mult = env.render_fps / 60.0

                if paused:
                    overlay = [
                        f"ÉVALUATION [{current_model_key.upper()}] | ROBOT BIPÈDE - [PAUSE]",
                        f"Vitesse: {speed_mult:.2f}x ({env.render_fps} FPS)",
                        f"Épisode: {episode} | Step: {env.current_step}/{env.max_steps} | X: {robot_x:+.0f}px",
                        f"Reward cumulé: {total_reward:.2f}",
                        "Touches: [Espace] Reprendre | [P/Tab] Modèle | [R] Reset | [U] Perturber tête",
                    ]
                    events = env.render(overlay_info=overlay)
                else:
                    state_in = policy.normalizer.normalize(current_state) if getattr(policy, "normalizer", None) else current_state
                    state_tensor = torch.FloatTensor(state_in).unsqueeze(0).to(device)
                    action_tensor, _, _, _ = policy.get_action_and_value(state_tensor, deterministic=deterministic)
                    act_vals = action_tensor.cpu().numpy().flatten()
                    if len(act_vals) < env.action_dim:
                        padded = np.zeros(env.action_dim, dtype=np.float32)
                        padded[:len(act_vals)] = act_vals
                        act_vals = padded
                    action_numpy = act_vals[:env.action_dim]
                    
                    act_str = f"H_D:{math.degrees(env.target_angles[0]):+.0f}° G_D:{math.degrees(env.target_angles[1]):+.0f}° B_D:{math.degrees(env.target_angles[4]):+.0f}° | H_G:{math.degrees(env.target_angles[2]):+.0f}° G_G:{math.degrees(env.target_angles[3]):+.0f}° B_G:{math.degrees(env.target_angles[5]):+.0f}°"
                    head_status = "Horizontal" if abs(head_deg) < 15 else ("Incliné" if abs(head_deg) <= 90 else "À l'envers")
                    overlay = [
                        f"MODÈLE: {current_model_key.upper()} [{'DÉT' if deterministic else 'STOCH'}] | ROBOT BIPÈDE",
                        f"Vitesse: {speed_mult:.2f}x ({env.render_fps} FPS) [V ou +/-] | [D] Mode Dét/Stoch",
                        f"Épisode: {episode} | Step: {env.current_step}/{env.max_steps} | X: {robot_x:+.0f}px | Vx: {vx:+.1f}px/s",
                        f"Tête: {head_deg:+.1f}° ({head_status}) | Angles PD: {act_str}",
                        f"Reward pas: {reward:+.2f} | Cumul: {total_reward:+.2f}",
                        "Touches: [P/Tab] Modèle | [Espace] Pause | [R] Reset | [U] Perturber tête",
                    ]
                    if getattr(env, "perturbation_visual_timer", 0) > 0 and getattr(env, "last_head_perturbation", None):
                        p_info = env.last_head_perturbation
                        overlay.insert(4, f">> PERTURBATION TÊTE: {p_info['side'].upper()} ({p_info['angle_deg']:+.1f}°, {p_info['magnitude']:.0f} N·s) <<")
                    events = env.render(overlay_info=overlay)

                    obs, reward, done = env.step(action_numpy)
                    current_state = obs
                    total_reward += reward

                    if done:
                        print(f"Fin Épisode {episode:<3} | Total Reward: {total_reward:+8.2f} | Distance X: {robot_x:+.1f}px | Steps: {env.current_step}")
                        current_state = env.reset(randomize=(task == "standing"))
                        episode += 1
                        total_reward = 0.0
                        reward = 0.0

                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                            print("Simulation en PAUSE" if paused else "Simulation REPRISE")

                        elif event.key == pygame.K_r:
                            current_state = env.reset(randomize=False)
                            total_reward = 0.0
                            reward = 0.0
                            print("Reset standard (vertical)")

                        elif event.key == pygame.K_u:
                            info = env.apply_head_perturbation()
                            if info:
                                print(f"[Perturbation Tête] Coup appliqué vers la {info['side'].upper()} ({info['angle_deg']:+.1f}°, {info['magnitude']:.0f} N·s)")

                        elif event.key == pygame.K_d:
                            deterministic = not deterministic
                            print(f"Mode politique -> {'DÉTERMINISTE' if deterministic else 'STOCHASTIQUE'}")

                        elif event.key in (pygame.K_TAB, pygame.K_p) and len(available) > 1:
                            keys_list = list(available.keys())
                            cur_idx = keys_list.index(current_model_key) if current_model_key in keys_list else 0
                            current_model_key = keys_list[(cur_idx + 1) % len(keys_list)]
                            resolved_path = available[current_model_key]
                            task = "walking" if "walk" in current_model_key.lower() else "standing"
                            deterministic = (task == "standing")
                            try:
                                policy, _ = load_policy_auto(resolved_path, state_dim=env.state_dim, action_dim=env.action_dim, device=device)
                                env.task = task
                                env.normalizer = getattr(policy, "normalizer", None)
                                current_state = env.reset(randomize=False)
                                total_reward = 0.0
                                reward = 0.0
                                print(f"Bascule modèle -> {current_model_key.upper()} ({resolved_path})")
                            except Exception as e:
                                print(f"Erreur changement modèle : {e}")

                        elif event.key in (pygame.K_v, pygame.K_KP_PLUS, pygame.K_PLUS, pygame.K_EQUALS):
                            fps_levels = [15, 30, 60, 120, 240]
                            cur_idx = fps_levels.index(env.render_fps) if env.render_fps in fps_levels else 2
                            env.render_fps = fps_levels[(cur_idx + 1) % len(fps_levels)]
                            print(f"Vitesse d'affichage : {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS)")

                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            fps_levels = [15, 30, 60, 120, 240]
                            cur_idx = fps_levels.index(env.render_fps) if env.render_fps in fps_levels else 2
                            env.render_fps = fps_levels[(cur_idx - 1) % len(fps_levels)]
                            print(f"Vitesse d'affichage : {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS)")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation interactive du robot bipède (V2)")
    parser.add_argument("--model", type=str, default=None, help="Chemin du modèle .pth à évaluer.")
    parser.add_argument("--episodes", type=int, default=None, help="Nombre d'épisodes à jouer (défaut: infini).")
    parser.add_argument("--models_dir", type=str, default=None, help="Dossier contenant les modèles (défaut: models).")
    args = parser.parse_args()

    evaluate(model_name=args.model, num_episodes=args.episodes, models_dir=args.models_dir)

