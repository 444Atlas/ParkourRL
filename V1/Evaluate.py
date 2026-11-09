import torch
import math
import os
import sys
import pygame
import argparse
from Env import Env, get_available_models, load_policy_auto, detect_robot_type


def prompt_model_choice(available_models):
    """Propose un menu interactif dans le terminal pour choisir le modèle à évaluer."""
    if not available_models:
        return None

    keys = list(available_models.keys())
    if len(keys) == 1:
        return keys[0]

    # Si stdin n'est pas un terminal interactif, on prend le premier
    if not sys.stdin.isatty():
        return keys[0]

    print("\n" + "=" * 70)
    print("      SÉLECTION INTERACTIVE DU MODÈLE À ÉVALUER")
    print("=" * 70)
    for idx, key in enumerate(keys, 1):
        path = available_models[key]
        task = "walking" if "walk" in key.lower() else "standing"
        robot = detect_robot_type(key)
        print(f"  [{idx}] {key.upper():<25} ({path}) - Tâche: {task:<8} - Robot: {robot}")
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


def evaluate(model_name=None, num_episodes=None):
    device = torch.device("cpu")
    available = get_available_models()
    model_keys = list(available.keys())

    if not available:
        print("Erreur : Aucun modèle .pth trouvé dans 'models/' ou à la racine.")
        print("Pour entraîner un modèle, lancez : python Train.py --task walking")
        print("Pour observer l'environnement brut, lancez : python Env.py")
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

    resolved_path = available[current_model_key]
    try:
        policy, action_dim = load_policy_auto(resolved_path, device=device)
    except Exception as e:
        print(f"Erreur lors du chargement de '{resolved_path}' : {e}")
        return

    task = "walking" if "walk" in current_model_key.lower() else "standing"
    robot_type = detect_robot_type(current_model_key)
    env = Env(render_mode=True, task=task, robot_type=robot_type, normalizer=getattr(policy, "normalizer", None), update_normalizer=False)

    print("\n" + "=" * 65)
    print("      ÉVALUATION INTERACTIVE DU MODÈLE")
    print("=" * 65)
    print(f"Modèle actif       : {current_model_key.upper()} ({resolved_path})")
    print(f"Morphologie robot  : {env.robot_type.upper()}")
    print(f"Tâche              : {task.upper()} ({'reward_walking' if task == 'walking' else 'reward_standing'})")
    print(f"Dimension d'action : {action_dim} ({'4 joints' if action_dim == 4 else '2 jambes'})")
    print(f"Normalisation état : {'Normalizer Welford actif' if getattr(policy, 'normalizer', None) else 'Statique'}")
    print(f"Vitesse initiale   : {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS - temps réel)")
    print("\nContrôles interactifs dans la fenêtre :")
    print("  - P ou Tab         : Changer de modèle à la volée")
    print("  - B                : Basculer morphologie (2 -> 3 -> 5 MEMBRES)")
    print("  - Espace           : Pause / Reprendre")
    print("  - R                : Réinitialiser l'épisode (position verticale standard)")
    print("  - U                : Réinitialiser avec perturbation aléatoire (test d'adaptabilité)")
    print("  - D                : Basculer mode Déterministe / Stochastique")
    print("  - V ou +/-         : Ajuster la vitesse d'affichage")
    print("  - Échap / Croix    : Quitter")
    print("=" * 65 + "\n")

    episode = 1
    total_reward = 0.0
    reward = 0.0
    current_state = env.reset(randomize=False)
    paused = False
    action_numpy = [0.0] * action_dim
    # Standing utilise le mode déterministe (maintien parfait à 0°), Walking utilise le mode stochastique (oscillations réelles entraînées)
    deterministic = (task == "standing")

    try:
        with torch.no_grad():
            while num_episodes is None or episode <= num_episodes:
                if env.robot_type in ("3_membres", "5_membres") and hasattr(env, "torso") and env.torso is not None:
                    robot_x = env.torso.position.x
                    vx = env.torso.velocity.x
                    head_deg = math.degrees(env.torso.angle)
                elif env.bodies:
                    robot_x = (env.bodies[0].position.x + env.bodies[1].position.x) / 2.0 if len(env.bodies) > 1 else env.bodies[0].position.x
                    vx = (env.bodies[0].velocity.x + env.bodies[1].velocity.x) / 2.0 if len(env.bodies) > 1 else env.bodies[0].velocity.x
                    head_deg = math.degrees((env.bodies[0].angle + env.bodies[1].angle) / 2.0) if len(env.bodies) > 1 else 0.0
                else:
                    robot_x = 0.0
                    vx = 0.0
                    head_deg = 0.0

                speed_mult = env.render_fps / 60.0

                # En pause
                if paused:
                    overlay = [
                        f"ÉVALUATION [{current_model_key.upper()}] | ROBOT: {env.robot_type.upper()} - [PAUSE]",
                        f"Vitesse: {speed_mult:.2f}x ({env.render_fps} FPS)",
                        f"Épisode: {episode} | Step: {env.current_step}/{env.max_steps} | X: {robot_x:+.0f}px",
                        f"Reward cumulé: {total_reward:.2f}",
                        "Touches: [Espace] Reprendre | [P/Tab] Modèle | [B] Robot | [R] Reset | [U] Perturber",
                    ]
                    events = env.render(overlay_info=overlay)
                else:
                    # Calcul de l'action par le modèle (stochastique pour la marche, déterministe pour standing)
                    state_in = policy.normalizer.normalize(current_state) if getattr(policy, "normalizer", None) else current_state
                    state_tensor = torch.FloatTensor(state_in).unsqueeze(0).to(device)
                    action_tensor, _, _, _ = policy.get_action_and_value(state_tensor, deterministic=deterministic)
                    action_numpy = action_tensor.cpu().numpy().flatten()
                    
                    if len(env.target_angles) >= 4:
                        act_str = f"H_D:{math.degrees(env.target_angles[0]):+.0f}° G_D:{math.degrees(env.target_angles[1]):+.0f}° | H_G:{math.degrees(env.target_angles[2]):+.0f}° G_G:{math.degrees(env.target_angles[3]):+.0f}°"
                    elif len(env.target_angles) >= 2:
                        act_str = f"D (Rouge): {math.degrees(env.target_angles[0]):+.0f}° | G (Bleue): {math.degrees(env.target_angles[1]):+.0f}°"
                    else:
                        act_str = f"Cible: {math.degrees(env.target_angles[0]):+.0f}°"

                    head_status = "Horizontal" if abs(head_deg) < 15 else ("Incliné" if abs(head_deg) <= 90 else "À l'envers")
                    mode_label = "DÉTERMINISTE" if deterministic else "STOCHASTIQUE (oscillations entraînées)"

                    overlay = [
                        f"ÉVALUATION [{current_model_key.upper()}] | ROBOT: {env.robot_type.upper()}  [{mode_label}]",
                        f"Vitesse: {speed_mult:.2f}x ({env.render_fps} FPS) [V ou +/-] | Mode: [Touche D]",
                        f"Épisode: {episode} | Step: {env.current_step}/{env.max_steps} | X: {robot_x:+.0f}px | Vx: {vx:+.1f}px/s",
                        f"Tête: {head_deg:+.1f}° ({head_status}) | Angles PD: {act_str}",
                        f"Reward pas: {reward:+.2f} | Total: {total_reward:.2f}",
                        "Touches: [P/Tab] Modèle | [B] Robot | [D] Mode | [Esp] Pause | [R] Reset | [U] Perturber",
                    ]
                    env.set_overlay(overlay)

                    # env.step avance et affiche chaque sous-pas avec micro-déplacements fluides
                    current_state, reward, done = env.step(action_numpy)
                    total_reward += reward
                    events = env.get_events()

                    if done:
                        print(f"Épisode {episode} terminé | Modèle: {current_model_key.upper()} [{env.robot_type}] | Reward total: {total_reward:.2f} | X: {robot_x:.1f}px")
                        episode += 1
                        total_reward = 0.0
                        reward = 0.0
                        current_state = env.reset(randomize=False)

                # Gestion des événements clavier
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                            print("Simulation en PAUSE" if paused else "Simulation REPRISE")

                        elif event.key == pygame.K_b:
                            if env.robot_type == "2_membres":
                                next_robot = "3_membres"
                            elif env.robot_type == "3_membres":
                                next_robot = "5_membres"
                            else:
                                next_robot = "2_membres"
                            env.robot_type = next_robot
                            env.action_dim = 4 if next_robot == "5_membres" else 2
                            env.state_dim = 20 if next_robot == "5_membres" else 9
                            current_state = env.reset(randomize=False)
                            total_reward = 0.0
                            reward = 0.0
                            episode += 1
                            print(f"Bascule robot -> Morphologie : {env.robot_type.upper()}")

                        elif event.key == pygame.K_d:
                            deterministic = not deterministic
                            mode_name = "DÉTERMINISTE (sans bruit)" if deterministic else "STOCHASTIQUE (avec oscillations entraînées)"
                            print(f"Mode d'action changé -> {mode_name}")

                        elif event.key == pygame.K_r:
                            current_state = env.reset(randomize=False)
                            print(f"Reset standard vertical [{env.robot_type}] | Épisode {episode} | Reward: {total_reward:.2f} | X: {robot_x:.1f}px")
                            episode += 1
                            total_reward = 0.0
                            reward = 0.0

                        elif event.key == pygame.K_u:
                            current_state = env.reset(randomize=True)
                            print(f"Reset perturbé aléatoire [{env.robot_type}] | Épisode {episode}")
                            episode += 1
                            total_reward = 0.0
                            reward = 0.0

                        elif event.key in (pygame.K_p, pygame.K_TAB) and len(model_keys) > 1:
                            # Changement de modèle à chaud
                            curr_idx = model_keys.index(current_model_key) if current_model_key in model_keys else 0
                            current_model_key = model_keys[(curr_idx + 1) % len(model_keys)]
                            resolved_path = available[current_model_key]
                            try:
                                policy, action_dim = load_policy_auto(resolved_path, device=device)
                                task = "walking" if "walk" in current_model_key.lower() else "standing"
                                env.robot_type = detect_robot_type(current_model_key)
                                env.action_dim = 4 if env.robot_type == "5_membres" else 2
                                env.state_dim = 20 if env.robot_type == "5_membres" else 9
                                env.normalizer = getattr(policy, "normalizer", None)
                                env.update_normalizer = False
                                env.task = task
                                deterministic = (task == "standing")
                                current_state = env.reset()
                                total_reward = 0.0
                                reward = 0.0
                                episode += 1
                                print(f"Changement de modèle -> [{current_model_key.upper()}] (robot={env.robot_type}, tâche={task}, dim={action_dim})")
                            except Exception as e:
                                print(f"Erreur chargement modèle : {e}")
                                env.robot_type = detect_robot_type(current_model_key)
                                env.task = task
                                deterministic = (task == "standing")
                                current_state = env.reset()
                                total_reward = 0.0
                                reward = 0.0
                                episode += 1
                                print(f"Changement de modèle -> [{current_model_key.upper()}] (robot={env.robot_type}, tâche={task}, dim={action_dim})")
                            except Exception as e:
                                print(f"Erreur chargement modèle : {e}")

                        elif event.key == pygame.K_v:
                            # Préréglages de vitesse
                            presets = [20, 40, 60, 90]
                            curr_fps = env.render_fps
                            next_fps = presets[0]
                            for p in presets:
                                if p > curr_fps:
                                    next_fps = p
                                    break
                            env.render_fps = next_fps
                            print(f"Vitesse d'affichage : {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS)")

                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                            env.render_fps = min(120, env.render_fps + 5)
                            print(f"Accélération -> {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS)")

                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            env.render_fps = max(10, env.render_fps - 5)
                            print(f"Ralentissement -> {env.render_fps / 60.0:.2f}x ({env.render_fps} FPS)")

    except SystemExit:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation interactive de modèles PPO pour le robot.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modèle à évaluer (ex: 'walking', 'standing'). Si non spécifié, un menu interactif est proposé."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Nombre maximal d'épisodes à évaluer (défaut: infini jusqu'à fermeture)."
    )
    args = parser.parse_args()
    evaluate(model_name=args.model, num_episodes=args.episodes)