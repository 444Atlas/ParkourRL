import math
import os
import sys
import pygame
import torch
import numpy as np
import argparse
from Env import Env, get_available_models, load_policy_auto


def run_evaluation(model_name=None, models_dir=None):
    """
    Fonction standardisée de visualisation et d'évaluation interactive du Robot Quadrupède V4.
    
    Spécifications :
      - Mode initial : STANDING (équilibre debout à 4 pattes)
      - Touche 'P' : Basculer entre mode standing et mode walking
      - Touche 'M' : Basculer entre sol plat et sol imparfait (accidenté)
      - Touche 'Flèche Droite' : Appliquer une perturbation à l'extrémité droite du corps (+X)
      - Touche 'Flèche Gauche' : Appliquer une perturbation à l'extrémité gauche du corps (-X)
      - Touche 'R' ou 'Espace' : Réinitialiser manuellement le robot
      - Temps de simulation infini sans reset automatique
      - Fréquence temps réel fluide (60 FPS)
    """
    device = torch.device("cpu")
    available = get_available_models(models_dir)

    task = "standing"
    current_model_key = None

    # Recherche du modèle standing initial
    if model_name:
        for k in available:
            if model_name.lower() in k.lower():
                current_model_key = k
                break
    if current_model_key is None:
        for k in available:
            if "standing" in k.lower() or "stand" in k.lower():
                current_model_key = k
                break
        if current_model_key is None and available:
            current_model_key = list(available.keys())[0]

    # Environnement quadrupède V4 en temps infini (max_steps=None)
    env = Env(render_mode=True, task=task, max_steps=None)
    obs = env.reset(randomize=False)

    policy = None
    loaded_policies = {}
    if current_model_key and current_model_key in available:
        try:
            policy, _ = load_policy_auto(
                available[current_model_key],
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device
            )
            loaded_policies[current_model_key] = policy
            print(f"[Modèle Chargé] '{current_model_key}' activé pour la tâche {task.upper()}.")
        except Exception as e:
            print(f"[Info] Impossible de charger le modèle '{current_model_key}' : {e}")
            policy = None

    print("\n" + "=" * 75)
    print("      VISUALISATION DU ROBOT QUADRUPÈDE V4 (Go1 Visual CPG-RL)")
    print("=" * 75)
    print(f"Mode initial         : {task.upper()} (Châssis horizontal + 4 pattes sans pieds ni tête)")
    print(f"Modèles détectés     : {list(available.keys()) if available else 'Aucun (mode passif nominal)'}")
    print("\nContrôles interactifs :")
    print("  [P]             : Basculer entre mode STANDING et WALKING")
    print("  [M]             : Basculer entre sol PLAT et sol IMPARFAIT (accidenté)")
    print("  [O]             : Téléporter le robot 250 px devant la prochaine boîte d'obstacle")
    print("  [Flèche Droite] : Perturber l'extrémité droite du corps (avant du châssis)")
    print("  [Flèche Gauche] : Perturber l'extrémité gauche du corps (arrière du châssis)")
    print("  [R / Espace]    : Réinitialiser manuellement le robot au centre")
    print("  [Échap / Croix] : Quitter la simulation")
    print("  * La durée de simulation est infinie en temps réel standard (60 FPS).")
    print("  * Aucun reset automatique : réinitialisation uniquement sur commande manuelle [R/Espace].")
    print("=" * 75 + "\n")

    episode = 1
    episode_reward = 0.0
    lstm_hidden = None

    try:
        while True:
            # Synchronisation automatique de la politique si l'environnement a changé de mode (ex: détection d'obstacle par vision ou fin de saut)
            if env.task != task:
                prev_task = task
                task = env.task
                target_key = None
                for k in available:
                    if task in k.lower():
                        target_key = k
                        break
                if target_key:
                    current_model_key = target_key
                    if target_key in loaded_policies:
                        policy = loaded_policies[target_key]
                    else:
                        try:
                            policy, _ = load_policy_auto(
                                available[target_key],
                                state_dim=env.state_dim,
                                action_dim=env.action_dim,
                                device=device
                            )
                            loaded_policies[target_key] = policy
                        except Exception as e:
                            print(f"[Avertissement] Erreur chargement modèle : {e}")
                            policy = None
                else:
                    policy = None
                    current_model_key = None
                lstm_hidden = None
                if task == "jumping":
                    print(f"\n[Détection Obstacle] Les 3 rayons < 150px -> Passage automatique en mode JUMPING (Politique '{current_model_key}')")
                else:
                    print(f"\n[Fin de Saut] Obstacle franchi avec succès -> Reprise automatique du mode {task.upper()} (Politique '{current_model_key}')")

            # Calcul de l'action
            if policy is not None:
                with torch.no_grad():
                    state_in = policy.normalizer.normalize(obs) if getattr(policy, "normalizer", None) else obs
                    state_tensor = torch.as_tensor(state_in, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor, _, _, _, lstm_hidden = policy.get_action_and_value(
                        state_tensor, hidden=lstm_hidden, deterministic=True, return_hidden=True
                    )
                    action = action_tensor.detach().cpu().numpy().flatten()
            else:
                # Mode nominal CPG sans modèle entraîné
                action = np.zeros(env.action_dim, dtype=np.float32)

            chassis_x = env.torso.position.x
            vx = env.torso.velocity.x
            chassis_deg = math.degrees(env.torso.angle)
            hauteur = max(0.0, env.get_terrain_height(chassis_x) - env.torso.position.y) if env.torso else (env.y_sol - env.torso.position.y)

            if task == "jumping":
                mode_label = f"MODE: JUMPING [{getattr(env, 'jump_phase', 'CROUCH')}]"
            else:
                mode_label = f"MODE: {task.upper()}"
            if current_model_key and policy is not None:
                mode_label += f" [Modèle: {current_model_key.upper()}]"
            else:
                mode_label += " [Générateur 4-Phases]" if task == "jumping" else " [CPG Nominal]"

            terrain_label = "IMPARFAIT (Accidenté)" if getattr(env, "terrain_mode", "flat") == "rough" else "PLAT"

            # Prochain obstacle en amont
            future_obs = [bx for _, _, bx in getattr(env, "obstacle_boxes", []) if bx > chassis_x - 50.0]
            if future_obs:
                next_bx = min(future_obs)
                dist_obs = next_bx - chassis_x
                obs_status = f"Boîte: {next_bx:.0f}px ({dist_obs:+.0f}px)"
            else:
                obs_status = "Boîte: Aucune"

            # État des rayons de vision
            if getattr(env, "vision_rays", None) and len(env.vision_rays) == 3:
                r0 = env.vision_rays[0]["dist"]
                r_up = env.vision_rays[1]["dist"]
                r_dn = env.vision_rays[2]["dist"]
                vis_tag = "Visibles (3 en contact)" if getattr(env, "rays_visible", False) else "Invisibles"
                ray_info = f"Rayons [{vis_tag}]: [0°: {r0:.0f}px | -20°: {r_up:.0f}px | +20°: {r_dn:.0f}px]"
            else:
                ray_info = "Rayons: Inactifs"

            det_alert = " >> OBSTACLE DÉTECTÉ (<150px) : JUMPING <<" if getattr(env, "obstacle_detected", False) else ""

            overlay = [
                f"{mode_label} | SOL: {terrain_label} | QUADRUPÈDE V4",
                f"Épisode: {episode} | Step: {env.current_step} (Infini) | X: {chassis_x:+.0f}px | Vx: {vx:+.1f}px/s | {obs_status}",
                f"Ligne de vue : {ray_info}{det_alert}",
                f"Hauteur Châssis: {hauteur:.1f}px | Assiette (Tilt): {chassis_deg:+.1f}°",
                f"CPG Phase: {math.degrees(env.cpg.theta[0]):.0f}° | Trot: FR+HL vs FL+HR",
                "Touches : [P] Stand/Walk | [J] Saut (Jump) | [M] Sol | [O] Devant Boîte | [->]/[<-] Pousse | [R] Reset",
            ]

            if env.perturbation_timer > 0 and env.last_perturbation:
                p_side = env.last_perturbation["side"].upper()
                p_mag = env.last_perturbation["magnitude"]
                overlay.insert(3, f">> PERTURBATION CORPS : EXTRÉMITÉ {p_side} ({p_mag:.0f} N·s) <<")

            env.set_overlay(overlay)
            events = env.render()

            # Gestion des entrées utilisateur standardisées
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        # Bascule entre Standing et Walking
                        task = "walking" if task == "standing" else "standing"
                        env.set_task(task)
                        lstm_hidden = None
                        print(f"\n[Bascule Mode] -> {task.upper()}")

                        # Recherche et chargement du modèle correspondant
                        target_key = None
                        for k in available:
                            if task in k.lower():
                                target_key = k
                                break
                        if target_key:
                            current_model_key = target_key
                            if target_key in loaded_policies:
                                policy = loaded_policies[target_key]
                            else:
                                try:
                                    policy, _ = load_policy_auto(
                                        available[target_key],
                                        state_dim=env.state_dim,
                                        action_dim=env.action_dim,
                                        device=device
                                    )
                                    loaded_policies[target_key] = policy
                                    print(f"[Modèle Chargé] '{target_key}' activé pour {task.upper()}.")
                                except Exception as e:
                                    print(f"[Avertissement] Erreur chargement modèle : {e}")
                                    policy = None
                        else:
                            policy = None
                            current_model_key = None
                            print(f"[Info] Aucun modèle spécifique pour '{task}'. Contrôle CPG nominal.")

                    elif event.key == pygame.K_j:
                        # Déclenchement manuel du saut (préserve le mode d'origine pour la reprise post-saut)
                        env.post_jump_task = task
                        task = "jumping"
                        upcoming = [bx for bx in getattr(env, "obstacle_positions", []) if bx > (chassis_x - 30.0)]
                        if upcoming:
                            env.cleared_obstacles.add(min(upcoming))
                        env.set_task("jumping")
                        lstm_hidden = None
                        target_key = None
                        for k in available:
                            if "jump" in k.lower():
                                target_key = k
                                break
                        if target_key:
                            current_model_key = target_key
                            if target_key in loaded_policies:
                                policy = loaded_policies[target_key]
                            else:
                                try:
                                    policy, _ = load_policy_auto(
                                        available[target_key],
                                        state_dim=env.state_dim,
                                        action_dim=env.action_dim,
                                        device=device
                                    )
                                    loaded_policies[target_key] = policy
                                except Exception:
                                    policy = None
                        else:
                            policy = None
                            current_model_key = None
                        print(f"\n[Commande Manuelle] Déclenchement du saut (JUMPING) !")

                    elif event.key == pygame.K_m:
                        # Bascule entre sol plat et sol imparfait (accidenté)
                        new_terrain = env.toggle_terrain()
                        label = "SOL IMPARFAIT (Accidenté / Bumps & Relief)" if new_terrain == "rough" else "SOL PLAT (Nominal)"
                        print(f"\n[Terrain] -> {label}")

                    elif event.key == pygame.K_o:
                        # Téléporter le robot à 250 px en amont de la prochaine boîte
                        future_obs = [bx for _, _, bx in getattr(env, "obstacle_boxes", []) if bx > chassis_x + 120.0]
                        target_obs = min(future_obs) if future_obs else 2000.0
                        teleport_x = target_obs - 250.0
                        env.teleport_to(teleport_x)
                        env.cleared_obstacles.discard(target_obs)
                        env.jump_cooldown = 0
                        lstm_hidden = None
                        print(f"\n[Téléportation] Robot positionné à X={teleport_x:.0f}px (250 px devant la boîte à {target_obs:.0f}px)")

                    elif event.key == pygame.K_RIGHT:
                        # Perturbation à l'extrémité droite du corps (avant du châssis, 2000 N·s)
                        p = env.apply_body_perturbation(side="right")
                        mag = p["magnitude"] if p else 2000.0
                        print(f"[Perturbation] Impulsion appliquée à l'avant DROIT du châssis (+{mag:.0f} N·s vers la droite).")

                    elif event.key == pygame.K_LEFT:
                        # Perturbation à l'extrémité gauche du corps (arrière du châssis, 2000 N·s)
                        p = env.apply_body_perturbation(side="left")
                        mag = p["magnitude"] if p else 2000.0
                        print(f"[Perturbation] Impulsion appliquée à l'arrière GAUCHE du châssis (-{mag:.0f} N·s vers la gauche).")

                    elif event.key in (pygame.K_SPACE, pygame.K_r):
                        obs = env.reset(randomize=False)
                        task = "standing"
                        target_key = None
                        for k in available:
                            if "standing" in k.lower():
                                target_key = k
                                break
                        policy = loaded_policies.get(target_key)
                        current_model_key = target_key
                        lstm_hidden = None
                        print(f"Reset standard | Épisode {episode} terminé | Reward: {episode_reward:.2f}")
                        episode += 1
                        episode_reward = 0.0

            obs, reward, done = env.step(action)
            episode_reward += reward

            # Simulation continue sans aucun reset automatique (reset manuel uniquement via [R] ou [Espace])
            if done:
                lstm_hidden = None

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation interactive du Robot Quadrupède V4")
    parser.add_argument("--model", type=str, default=None, help="Modèle spécifique à charger.")
    parser.add_argument("--models_dir", type=str, default=None, help="Dossier des modèles (défaut: models).")
    args = parser.parse_args()

    run_evaluation(model_name=args.model, models_dir=args.models_dir)

