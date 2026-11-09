import numpy as np
import math
import pygame
import pymunk
import pymunk.pygame_util


class Normalizer:
    """Normalisateur d'état en ligne O(1) basé sur l'algorithme de Welford / Chan et al.
    Calcule la moyenne et la variance cumulées de manière numériquement stable."""
    def __init__(self, size, eps=1e-8, clip=10.0):
        self.size = size
        self.eps = eps
        self.clip = clip
        self.count = 1e-4
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = new_mean.astype(np.float32)
        self.var = np.maximum(new_var, 1e-4).astype(np.float32)
        self.count = tot_count

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var + self.eps)
        normalized = (x - self.mean) / std
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized

    def state_dict(self):
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
            "size": int(self.size),
            "clip": float(self.clip),
        }

    def load_state_dict(self, d):
        self.mean = np.array(d["mean"], dtype=np.float32)
        self.var = np.array(d["var"], dtype=np.float32)
        self.count = float(d["count"])
        self.size = int(d.get("size", len(self.mean)))
        self.clip = float(d.get("clip", 10.0))


#Environnement PyGame avec moteur physique Pymunk et contrôleur PD
class Env:
    def __init__(self, max_steps=500, render_mode=False, task="standing", random_reset=True, robot_type="3_membres", normalizer=None, update_normalizer=False):
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.task = task
        self.random_reset = random_reset
        if "5" in str(robot_type):
            self.robot_type = "5_membres"
        elif "2" in str(robot_type):
            self.robot_type = "2_membres"
        else:
            self.robot_type = "3_membres"
        
        self.normalizer = normalizer
        self.update_normalizer = update_normalizer

        self.SCREEN_LENGHT = 600
        self.SCREEN_HEIGHT = 400
        self.y_sol = 350
        self.max_torque = 200000.0

        # Contrôleur PD en position
        self.kp = 80000.0
        self.kd = 6000.0
        self.kp_knee = 90000.0
        self.kd_knee = 6000.0

        self.action_dim = 4 if self.robot_type == "5_membres" else 2
        self.state_dim = 20 if self.robot_type == "5_membres" else 9

        self.target_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

        # Vitesse cible inatteignable pour favoriser la démarche la plus rapide possible (Feature 3.2)
        self.target_vx = 200.0

        self.bodies = []
        self.joints = []
        self.prev_joint_x = 300.0
        self.camera_x = 0.0

        # Initialisation de la physique
        self.space = pymunk.Space()
        self.space.iterations = 35  # Résolution rigide
        self.space.gravity = (0, 981)

        # Création du sol statique avec coefficient de friction Box2D (2.5 comme dans le repo ARS)
        static_body = self.space.static_body
        floor = pymunk.Segment(static_body, (-1000000, self.y_sol), (1000000, self.y_sol), 5)
        floor.friction = 2.5
        self.floor = floor
        self.space.add(floor)

        self.render_fps = 60  # À 60 FPS avec 1 sous-pas par frame = 1.0x temps réel
        self.substep_render = True
        self.overlay_info = None
        self.pending_events = []

        if self.render_mode:
            pygame.init()
            pygame.display.set_caption("Simulation Robot - Environnement")
            self.screen = pygame.display.set_mode((self.SCREEN_LENGHT, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.font = None

    # Visualisation avec défilement de caméra suivant le robot
    def render(self, overlay_info=None):
        if not self.render_mode:
            return []
            
        if overlay_info is not None:
            self.overlay_info = overlay_info

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                exit()
            self.pending_events.append(event)

        # Calcul de la position de la caméra centrée sur le robot
        if hasattr(self, "torso") and self.torso is not None:
            robot_x = self.torso.position.x
            self.camera_x = robot_x - (self.SCREEN_LENGHT / 2.0)
        elif self.bodies:
            robot_x = self.bodies[0].position.x
            self.camera_x = robot_x - (self.SCREEN_LENGHT / 2.0)
        else:
            self.camera_x = 0.0

        # Décalage de la vue Pymunk pour centrer la caméra
        self.draw_options.transform = pymunk.Transform.translation(-self.camera_x, 0)
                
        self.screen.fill((255, 255, 255))
        
        # Dessin des repères sur le sol pour visualiser le défilement
        self._draw_ground_markers()

        # Dessin du robot et des éléments physiques
        self.space.debug_draw(self.draw_options)

        # Dessin de la tête au niveau du joint (rond avec 2 yeux et bouche)
        self._draw_head()

        if self.overlay_info:
            self._draw_overlay(self.overlay_info)

        pygame.display.flip()
        self.clock.tick(self.render_fps)

        evs = list(self.pending_events)
        if not getattr(self, "_in_substep", False):
            self.pending_events.clear()
        return evs

    # Définit les lignes de texte de l'affichage tête haute (HUD)
    def set_overlay(self, info_lines):
        self.overlay_info = info_lines

    # Récupère tous les événements clavier accumulés pendant les sous-pas
    def get_events(self):
        evs = list(self.pending_events)
        self.pending_events.clear()
        return evs

    # Dessine des repères gradués sur le sol infini
    def _draw_ground_markers(self):
        try:
            if self.font is None:
                if not pygame.font.get_init():
                    pygame.font.init()
                self.font = pygame.font.Font(None, 20)

            step_dist = 100
            start_m = int((self.camera_x - 100) // step_dist) * step_dist
            end_m = int((self.camera_x + self.SCREEN_LENGHT + 100) // step_dist) * step_dist

            for mx in range(start_m, end_m + 1, step_dist):
                sx = int(mx - self.camera_x)
                # Petit repère vertical sur la ligne de sol
                pygame.draw.line(self.screen, (200, 200, 200), (sx, self.y_sol - 8), (sx, self.y_sol + 8), 2)
                # Label de distance en pixels
                txt = self.font.render(f"{mx}px", True, (160, 160, 160))
                self.screen.blit(txt, (sx - 16, self.y_sol + 12))
        except Exception:
            pass

    # Affiche un panneau semi-transparent avec des informations
    def _draw_overlay(self, info_lines):
        try:
            if self.font is None:
                if not pygame.font.get_init():
                    pygame.font.init()
                self.font = pygame.font.Font(None, 22)

            padding = 6
            line_height = 20
            box_width = 460
            box_height = len(info_lines) * line_height + padding * 2

            overlay_surf = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            overlay_surf.fill((240, 240, 240, 210))
            self.screen.blit(overlay_surf, (8, 8))

            for i, line in enumerate(info_lines):
                text_surf = self.font.render(line, True, (20, 20, 20))
                self.screen.blit(text_surf, (14, 10 + i * line_height))
        except Exception:
            pass

    # Dessine la tête (au sommet du tronc pour 3 et 5 membres, au niveau du joint pivot pour 2 membres)
    def _draw_head(self):
        try:
            if not self.bodies:
                return

            if self.robot_type in ("3_membres", "5_membres") and hasattr(self, "torso") and self.torso is not None:
                neck = self.torso.local_to_world((0, -35))
                screen_x = int(neck.x - self.camera_x)
                screen_y = int(neck.y)
                angle = self.torso.angle
            else:
                screen_x = int(self.bodies[0].position.x - self.camera_x)
                screen_y = int(self.bodies[0].position.y)
                angle = (self.bodies[0].angle + self.bodies[1].angle) / 2.0 if len(self.bodies) > 1 else self.bodies[0].angle

            # Création de la surface pour la tête
            head_radius = 16
            head_size = (head_radius + 4) * 2
            surf = pygame.Surface((head_size, head_size), pygame.SRCALPHA)
            center = head_size // 2

            # 1. Tête : rond blanc avec contour noir
            pygame.draw.circle(surf, (255, 255, 255), (center, center), head_radius)
            pygame.draw.circle(surf, (0, 0, 0), (center, center), head_radius, 2)

            # 2. Deux yeux : ronds noirs tournés vers la droite
            pygame.draw.circle(surf, (0, 0, 0), (center - 3, center - 4), 3)
            pygame.draw.circle(surf, (0, 0, 0), (center + 6, center - 4), 3)

            # 3. Bouche souriante noire
            mouth_rect = pygame.Rect(center - 5, center + 1, 10, 7)
            pygame.draw.arc(surf, (0, 0, 0), mouth_rect, math.pi, 2 * math.pi, 2)

            # Rotation de la tête selon l'inclinaison physique réelle du tronc
            rotated = pygame.transform.rotate(surf, -math.degrees(angle))
            new_rect = rotated.get_rect(center=(screen_x, screen_y))
            self.screen.blit(rotated, new_rect)

            # Dessin de marqueurs pour les genoux si robot à 5 membres
            if self.robot_type == "5_membres" and hasattr(self, "thigh0") and hasattr(self, "thigh1"):
                k0 = self.thigh0.local_to_world((0, 28))
                pygame.draw.circle(self.screen, (220, 40, 40), (int(k0.x - self.camera_x), int(k0.y)), 4)
                k1 = self.thigh1.local_to_world((0, 28))
                pygame.draw.circle(self.screen, (30, 90, 220), (int(k1.x - self.camera_x), int(k1.y)), 4)
        except Exception:
            pass

    # Ferme la fenêtre Pygame proprement
    def close(self):
        if self.render_mode and pygame.get_init():
            pygame.quit() 

    # Supprime le robot précédent et initialise le nouveau robot selon self.robot_type
    def reset(self, randomize=None):
        self.current_step = 0
        self.target_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

        if self.bodies:
            shapes = [s for b in self.bodies for s in b.shapes]
            self.space.remove(*self.joints, *shapes, *self.bodies)
            
        self.bodies.clear()
        self.joints.clear()
        self.torso = None
        self.thigh0 = None
        self.thigh1 = None
        self.shin0 = None
        self.shin1 = None

        do_random = (self.random_reset and self.task == "standing") if randomize is None else bool(randomize)

        if self.robot_type == "5_membres":
            # 5 MEMBRES : Tronc + 2 Cuisses (28px) + 2 Tibias/Mollets (28px) avec genoux articulés
            px = 300.0
            if do_random:
                # Perturbations réalistes modérées pour permettre l'apprentissage de l'équilibre
                max_angle_deg = 5.0
                theta_torso = float(np.random.uniform(-math.radians(3.0), math.radians(3.0)))
                theta_hip0 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
                theta_hip1 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
                theta_knee0 = float(np.random.uniform(0.0, math.radians(5.0)))
                theta_knee1 = float(np.random.uniform(0.0, math.radians(5.0)))

                th0_angle = theta_torso + theta_hip0
                th1_angle = theta_torso + theta_hip1
                sh0_angle = th0_angle + theta_knee0
                sh1_angle = th1_angle + theta_knee1

                reach0 = 28.0 * math.cos(th0_angle) + 28.0 * math.cos(sh0_angle)
                reach1 = 28.0 * math.cos(th1_angle) + 28.0 * math.cos(sh1_angle)
                max_reach = max(reach0, reach1)
                # Position y précise : le pied le plus bas effleure le sol sans pénétration ni chute dans le vide
                py = float(self.y_sol - max_reach - 4.5)

                # 1. Tronc
                self.torso = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.torso.position = (px, py)
                self.torso.angle = theta_torso
                torso_seg = pymunk.Segment(self.torso, (0, 0), (0, -35), 6)
                torso_seg.mass = 2.0
                torso_seg.color = (65, 70, 80, 255)
                torso_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.torso, torso_seg)

                # 2. Cuisses (droite = rouge, gauche = bleu)
                self.thigh0 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.thigh0.position = (px, py)
                self.thigh0.angle = th0_angle
                th0_seg = pymunk.Segment(self.thigh0, (0, 0), (0, 28), 5)
                th0_seg.mass = 0.8
                th0_seg.friction = 1.0
                th0_seg.color = (235, 75, 75, 255)
                th0_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.thigh0, th0_seg)

                self.thigh1 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.thigh1.position = (px, py)
                self.thigh1.angle = th1_angle
                th1_seg = pymunk.Segment(self.thigh1, (0, 0), (0, 28), 5)
                th1_seg.mass = 0.8
                th1_seg.friction = 1.0
                th1_seg.color = (45, 125, 240, 255)
                th1_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.thigh1, th1_seg)

                # 3. Tibias reliés aux genoux
                k0 = self.thigh0.local_to_world((0, 28))
                self.shin0 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.shin0.position = k0
                self.shin0.angle = sh0_angle
                sh0_seg = pymunk.Segment(self.shin0, (0, 0), (0, 28), 4)
                sh0_seg.mass = 0.6
                sh0_seg.friction = 1.0
                sh0_seg.color = (255, 130, 110, 255)
                sh0_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.shin0, sh0_seg)

                k1 = self.thigh1.local_to_world((0, 28))
                self.shin1 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.shin1.position = k1
                self.shin1.angle = sh1_angle
                sh1_seg = pymunk.Segment(self.shin1, (0, 0), (0, 28), 4)
                sh1_seg.mass = 0.6
                sh1_seg.friction = 1.0
                sh1_seg.color = (110, 185, 255, 255)
                sh1_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.shin1, sh1_seg)

                # 4. Pieds solidaires à profil berceau / toe-spring anti-trébuchement
                foot0_base = pymunk.Segment(self.shin0, (-5, 28), (4, 28), 4)
                foot0_base.mass = 0.10
                foot0_base.friction = 1.0
                foot0_base.color = (255, 130, 110, 255)
                foot0_base.filter = pymunk.ShapeFilter(group=1)

                foot0_toe = pymunk.Segment(self.shin0, (4, 28), (10, 25), 3.5)
                foot0_toe.mass = 0.05
                foot0_toe.friction = 1.0
                foot0_toe.color = (255, 130, 110, 255)
                foot0_toe.filter = pymunk.ShapeFilter(group=1)
                self.space.add(foot0_base, foot0_toe)

                foot1_base = pymunk.Segment(self.shin1, (-5, 28), (4, 28), 4)
                foot1_base.mass = 0.10
                foot1_base.friction = 1.0
                foot1_base.color = (110, 185, 255, 255)
                foot1_base.filter = pymunk.ShapeFilter(group=1)

                foot1_toe = pymunk.Segment(self.shin1, (4, 28), (10, 25), 3.5)
                foot1_toe.mass = 0.05
                foot1_toe.friction = 1.0
                foot1_toe.color = (110, 185, 255, 255)
                foot1_toe.filter = pymunk.ShapeFilter(group=1)
                self.space.add(foot1_base, foot1_toe)

                # Hanches
                pj_h0 = pymunk.PivotJoint(self.torso, self.thigh0, (0, 0), (0, 0))
                rj_h0 = pymunk.RotaryLimitJoint(self.torso, self.thigh0, math.radians(-90), math.radians(90))
                pj_h1 = pymunk.PivotJoint(self.torso, self.thigh1, (0, 0), (0, 0))
                rj_h1 = pymunk.RotaryLimitJoint(self.torso, self.thigh1, math.radians(-90), math.radians(90))

                # Genoux (flexion arrière uniquement [-5°, +105°])
                pj_k0 = pymunk.PivotJoint(self.thigh0, self.shin0, (0, 28), (0, 0))
                rj_k0 = pymunk.RotaryLimitJoint(self.thigh0, self.shin0, math.radians(-5), math.radians(105))
                pj_k1 = pymunk.PivotJoint(self.thigh1, self.shin1, (0, 28), (0, 0))
                rj_k1 = pymunk.RotaryLimitJoint(self.thigh1, self.shin1, math.radians(-5), math.radians(105))

                self.space.add(pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1)
                self.joints.extend([pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1])
                self.bodies = [self.thigh0, self.shin0, self.thigh1, self.shin1, self.torso]

                init_vx = float(np.random.uniform(-1.0, 1.0))
                for b in self.bodies:
                    b.velocity = (init_vx, 0.0)
                    self.space.reindex_shapes_for_body(b)
            else:
                py = float(self.y_sol - 56.0 - 4.5)
                # 1. Tronc
                self.torso = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.torso.position = (px, py)
                torso_seg = pymunk.Segment(self.torso, (0, 0), (0, -35), 6)
                torso_seg.mass = 2.0
                torso_seg.color = (65, 70, 80, 255)
                torso_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.torso, torso_seg)

                # 2. Cuisses
                self.thigh0 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.thigh0.position = (px, py)
                th0_seg = pymunk.Segment(self.thigh0, (0, 0), (0, 28), 5)
                th0_seg.mass = 0.8
                th0_seg.friction = 1.0
                th0_seg.color = (235, 75, 75, 255)
                th0_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.thigh0, th0_seg)

                self.thigh1 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.thigh1.position = (px, py)
                th1_seg = pymunk.Segment(self.thigh1, (0, 0), (0, 28), 5)
                th1_seg.mass = 0.8
                th1_seg.friction = 1.0
                th1_seg.color = (45, 125, 240, 255)
                th1_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.thigh1, th1_seg)

                # 3. Tibias
                self.shin0 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.shin0.position = (px, py + 28)
                sh0_seg = pymunk.Segment(self.shin0, (0, 0), (0, 28), 4)
                sh0_seg.mass = 0.6
                sh0_seg.friction = 1.0
                sh0_seg.color = (255, 130, 110, 255)
                sh0_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.shin0, sh0_seg)

                self.shin1 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.shin1.position = (px, py + 28)
                sh1_seg = pymunk.Segment(self.shin1, (0, 0), (0, 28), 4)
                sh1_seg.mass = 0.6
                sh1_seg.friction = 1.0
                sh1_seg.color = (110, 185, 255, 255)
                sh1_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.shin1, sh1_seg)

                # 4. Pieds solidaires à profil berceau / toe-spring anti-trébuchement
                foot0_base = pymunk.Segment(self.shin0, (-5, 28), (4, 28), 4)
                foot0_base.mass = 0.10
                foot0_base.friction = 1.0
                foot0_base.color = (255, 130, 110, 255)
                foot0_base.filter = pymunk.ShapeFilter(group=1)

                foot0_toe = pymunk.Segment(self.shin0, (4, 28), (10, 25), 3.5)
                foot0_toe.mass = 0.05
                foot0_toe.friction = 1.0
                foot0_toe.color = (255, 130, 110, 255)
                foot0_toe.filter = pymunk.ShapeFilter(group=1)
                self.space.add(foot0_base, foot0_toe)

                foot1_base = pymunk.Segment(self.shin1, (-5, 28), (4, 28), 4)
                foot1_base.mass = 0.10
                foot1_base.friction = 1.0
                foot1_base.color = (110, 185, 255, 255)
                foot1_base.filter = pymunk.ShapeFilter(group=1)

                foot1_toe = pymunk.Segment(self.shin1, (4, 28), (10, 25), 3.5)
                foot1_toe.mass = 0.05
                foot1_toe.friction = 1.0
                foot1_toe.color = (110, 185, 255, 255)
                foot1_toe.filter = pymunk.ShapeFilter(group=1)
                self.space.add(foot1_base, foot1_toe)

                # Hanches
                pj_h0 = pymunk.PivotJoint(self.torso, self.thigh0, (0, 0), (0, 0))
                rj_h0 = pymunk.RotaryLimitJoint(self.torso, self.thigh0, math.radians(-90), math.radians(90))
                pj_h1 = pymunk.PivotJoint(self.torso, self.thigh1, (0, 0), (0, 0))
                rj_h1 = pymunk.RotaryLimitJoint(self.torso, self.thigh1, math.radians(-90), math.radians(90))

                # Genoux
                pj_k0 = pymunk.PivotJoint(self.thigh0, self.shin0, (0, 28), (0, 0))
                rj_k0 = pymunk.RotaryLimitJoint(self.thigh0, self.shin0, math.radians(-5), math.radians(105))
                pj_k1 = pymunk.PivotJoint(self.thigh1, self.shin1, (0, 28), (0, 0))
                rj_k1 = pymunk.RotaryLimitJoint(self.thigh1, self.shin1, math.radians(-5), math.radians(105))

                self.space.add(pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1)
                self.joints.extend([pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1])
                self.bodies = [self.thigh0, self.shin0, self.thigh1, self.shin1, self.torso]

            self.prev_joint_x = self.torso.position.x
        elif self.robot_type == "3_membres":
            if do_random:
                max_angle_deg = 25.0
                theta1 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
                theta2 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
                theta_torso = float(np.random.uniform(-math.radians(10.0), math.radians(10.0)))

                py = float(self.y_sol - 50.0 * max(math.cos(theta1), math.cos(theta2)) - 5.0)
                px = 300.0

                # 1. Tronc physique visible (segment de (0, 0) hanches jusqu'à (0, -35) cou)
                self.torso = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.torso.position = (px, py)
                self.torso.angle = theta_torso
                torso_seg = pymunk.Segment(self.torso, (0, 0), (0, -35), 6)
                torso_seg.mass = 2.0
                torso_seg.color = (65, 70, 80, 255)
                torso_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.torso, torso_seg)

                # 2. Deux jambes : droite (rouge corail), gauche (bleu royal)
                self.create_segment(px, py, px, py + 50, color=(235, 75, 75, 255))
                self.create_segment(px, py, px, py + 50, color=(45, 125, 240, 255))
                self.bodies[0].angle = theta1
                self.bodies[1].angle = theta2
                self.bodies.append(self.torso)

                # 3. Articulations de hanche indépendantes reliant chaque jambe au tronc
                pj0 = pymunk.PivotJoint(self.torso, self.bodies[0], (0, 0), (0, 0))
                rj0 = pymunk.RotaryLimitJoint(self.torso, self.bodies[0], math.radians(-90), math.radians(90))
                pj1 = pymunk.PivotJoint(self.torso, self.bodies[1], (0, 0), (0, 0))
                rj1 = pymunk.RotaryLimitJoint(self.torso, self.bodies[1], math.radians(-90), math.radians(90))
                self.space.add(pj0, rj0, pj1, rj1)
                self.joints.extend([pj0, rj0, pj1, rj1])

                for b in self.bodies:
                    self.space.reindex_shapes_for_body(b)

                init_vx = float(np.random.uniform(-20.0, 20.0))
                for b in self.bodies:
                    b.velocity = (init_vx, 0.0)
                self.bodies[0].angular_velocity = float(np.random.uniform(-0.5, 0.5))
                self.bodies[1].angular_velocity = float(np.random.uniform(-0.5, 0.5))
                self.torso.angular_velocity = float(np.random.uniform(-0.2, 0.2))
            else:
                px, py = 300.0, float(self.y_sol - 50.0 - 5.0)
                # 1. Tronc physique visible
                self.torso = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
                self.torso.position = (px, py)
                torso_seg = pymunk.Segment(self.torso, (0, 0), (0, -35), 6)
                torso_seg.mass = 2.0
                torso_seg.color = (65, 70, 80, 255)
                torso_seg.filter = pymunk.ShapeFilter(group=1)
                self.space.add(self.torso, torso_seg)

                # 2. Deux jambes
                self.create_segment(px, py, px, py + 50, color=(235, 75, 75, 255))
                self.create_segment(px, py, px, py + 50, color=(45, 125, 240, 255))
                self.bodies.append(self.torso)

                # 3. Articulations de hanche indépendantes
                pj0 = pymunk.PivotJoint(self.torso, self.bodies[0], (0, 0), (0, 0))
                rj0 = pymunk.RotaryLimitJoint(self.torso, self.bodies[0], math.radians(-90), math.radians(90))
                pj1 = pymunk.PivotJoint(self.torso, self.bodies[1], (0, 0), (0, 0))
                rj1 = pymunk.RotaryLimitJoint(self.torso, self.bodies[1], math.radians(-90), math.radians(90))
                self.space.add(pj0, rj0, pj1, rj1)
                self.joints.extend([pj0, rj0, pj1, rj1])

                for b in self.bodies:
                    b.velocity = (0.0, 0.0)
                    b.angular_velocity = 0.0

            self.prev_joint_x = self.torso.position.x
        else:
            # 2_membres (2 jambes reliées directement à la hanche pivot)
            self.torso = None
            if do_random:
                max_angle_deg = 25.0
                theta1 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
                theta2 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
                py = float(self.y_sol - 50.0 * max(math.cos(theta1), math.cos(theta2)) - 5.0)
                px = 300.0

                self.create_segment(px, py, px, py + 50, color=(235, 75, 75, 255))
                self.create_segment(px, py, px, py + 50, color=(45, 125, 240, 255))
                self.create_joint(self.bodies[0], self.bodies[1], px, py, -180, 180)

                self.bodies[0].angle = theta1
                self.bodies[1].angle = theta2
                self.space.reindex_shapes_for_body(self.bodies[0])
                self.space.reindex_shapes_for_body(self.bodies[1])

                init_vx = float(np.random.uniform(-20.0, 20.0))
                self.bodies[0].velocity = (init_vx, 0.0)
                self.bodies[1].velocity = (init_vx, 0.0)
                self.bodies[0].angular_velocity = float(np.random.uniform(-0.5, 0.5))
                self.bodies[1].angular_velocity = float(np.random.uniform(-0.5, 0.5))
            else:
                py = float(self.y_sol - 50.0 - 5.0)
                self.create_segment(300, py, 300, py + 50, color=(235, 75, 75, 255))
                self.create_segment(300, py, 300, py + 50, color=(45, 125, 240, 255))
                self.create_joint(self.bodies[0], self.bodies[1], 300, py, -180, 180)
                self.bodies[0].angle = 0.0
                self.bodies[1].angle = 0.0
                self.bodies[0].velocity = (0.0, 0.0)
                self.bodies[1].velocity = (0.0, 0.0)
                self.bodies[0].angular_velocity = 0.0
                self.bodies[1].angular_velocity = 0.0

            self.prev_joint_x = self.bodies[0].position.x

        return self._get_obs()

    # Avance la simulation de 4 pas pour accélérer l'entraînement
    def step(self, action):
        self.current_step += 1
        self._apply_action(action)
        self._in_substep = True
        try:
            for _ in range(4):
                # Application continue du contrôleur PD à 60 Hz (Feature 1)
                self._update_pd_torques()
                self.space.step(1.0 / 60.0)
                self._clamp_legs()
                self._sync_joint()
                # Affichage fluide sous-pas par sous-pas pour diviser le déplacement par frame par 4
                if self.render_mode and getattr(self, "substep_render", True):
                    self.render()
        finally:
            self._in_substep = False

        obs = self._get_obs()

        if self.task == "walking":
            reward = self._reward_walking(action)
        else:
            reward = self._reward_standing(action)

        done = False
        # Arrêt précoce en cas de chute
        if self.max_steps is not None and self._is_fallen():
            done = True

        if self.max_steps is not None and self.current_step >= self.max_steps:
            done = True
        return obs, reward, done

    # Détecte si le robot est tombé (selon la tâche et la morphologie)
    def _is_fallen(self):
        if not self.bodies or len(self.bodies) < 2:
            return False
        if self.robot_type == "5_membres":
            if not hasattr(self, "torso") or self.torso is None:
                return False
            y_pivot = self.torso.position.y
            hauteur = self.y_sol - y_pivot
            head_angle = self.torso.angle
            if self.task == "standing":
                max_tilt = math.radians(65)
                return (hauteur < 18.0) or (abs(head_angle) > max_tilt)
            else:
                k0_clearance = self.y_sol - self.thigh0.local_to_world((0, 28)).y if hasattr(self, "thigh0") and self.thigh0 else 99.0
                k1_clearance = self.y_sol - self.thigh1.local_to_world((0, 28)).y if hasattr(self, "thigh1") and self.thigh1 else 99.0
                knee_crash = min(k0_clearance, k1_clearance) < 5.0
                return (hauteur < 22.0) or (math.cos(head_angle) <= 0.0) or knee_crash
        elif self.robot_type == "3_membres":
            if not hasattr(self, "torso") or self.torso is None:
                return False
            y_pivot = self.torso.position.y
            hauteur = self.y_sol - y_pivot
            head_angle = self.torso.angle
            if self.task == "standing":
                max_tilt = math.radians(65)
                return (hauteur < 28.0) or (abs(head_angle) > max_tilt)
            else:
                return (hauteur < 20.0) or (math.cos(head_angle) <= 0.0)
        else:
            y_pivot = self.bodies[0].position.y
            hauteur = self.y_sol - y_pivot
            b1, b2 = self.bodies[0], self.bodies[1]
            head_angle = (b1.angle + b2.angle) / 2.0
            if self.task == "standing":
                max_tilt = math.radians(65)
                return (hauteur < 28.0) or (abs(b1.angle) > max_tilt) or (abs(b2.angle) > max_tilt)
            else:
                return (hauteur < 20.0) or (math.cos(head_angle) <= 0.0)

    # Contrôleur PD haute fréquence appliqué à chaque sous-pas physique (Feature 1)
    def _update_pd_torques(self):
        if self.robot_type == "5_membres":
            # 4 joints : Hanches (torso <-> cuisses) et Genoux (cuisses <-> tibias)
            for i, (thigh, shin) in enumerate([(self.thigh0, self.shin0), (self.thigh1, self.shin1)]):
                # Hanche
                rel_th = thigh.angle - self.torso.angle
                rel_w_th = thigh.angular_velocity - self.torso.angular_velocity
                err_hip = self.target_angles[i * 2] - rel_th
                torque_hip = float(np.clip(self.kp * err_hip - self.kd * rel_w_th, -self.max_torque, self.max_torque))

                # Genou
                rel_sh = shin.angle - thigh.angle
                rel_w_sh = shin.angular_velocity - thigh.angular_velocity
                err_knee = self.target_angles[i * 2 + 1] - rel_sh
                torque_knee = float(np.clip(self.kp_knee * err_knee - self.kd_knee * rel_w_sh, -self.max_torque, self.max_torque))

                shin.torque = torque_knee
                thigh.torque = torque_hip - torque_knee  # Action-réaction au niveau du genou

            # Stabilisation posturale active du tronc (réflexes vestibulospinaux et spinaux)
            t_torso = -self.kp * self.torso.angle - self.kd * self.torso.angular_velocity
            self.torso.torque = float(np.clip(t_torso, -self.max_torque, self.max_torque))
        elif self.robot_type == "3_membres":
            # Contrôleur PD indépendant sur chaque jambe par rapport au tronc
            for i in range(2):
                leg = self.bodies[i]
                rel_angle = leg.angle - self.torso.angle
                rel_w = leg.angular_velocity - self.torso.angular_velocity
                error = self.target_angles[i] - rel_angle
                torque = self.kp * error - self.kd * rel_w
                leg.torque = float(np.clip(torque, -self.max_torque, self.max_torque))

            # Aucune force extérieure artificielle appliquée au torse
            self.torso.torque = 0.0
        else:
            for i in range(min(len(self.bodies), len(self.target_angles))):
                b = self.bodies[i]
                error = self.target_angles[i] - b.angle
                torque = self.kp * error - self.kd * b.angular_velocity
                b.torque = float(np.clip(torque, -self.max_torque, self.max_torque))

    # Maintient le joint parfaitement soudé pour empêcher toute séparation visuelle des corps
    def _sync_joint(self):
        if self.robot_type == "5_membres":
            if hasattr(self, "torso") and self.torso is not None and hasattr(self, "thigh0") and hasattr(self, "thigh1"):
                avg_hip = (self.torso.position * 2.0 + self.thigh0.position + self.thigh1.position) / 4.0
                self.torso.position = avg_hip
                self.thigh0.position = avg_hip
                self.thigh1.position = avg_hip
                if hasattr(self, "shin0") and hasattr(self, "shin1"):
                    self.shin0.position = self.thigh0.local_to_world((0, 28))
                    self.shin1.position = self.thigh1.local_to_world((0, 28))
        elif self.robot_type == "3_membres":
            if hasattr(self, "torso") and self.torso is not None and len(self.bodies) >= 2:
                avg_p = (self.torso.position * 2.0 + self.bodies[0].position + self.bodies[1].position) / 4.0
                self.torso.position = avg_p
                self.bodies[0].position = avg_p
                self.bodies[1].position = avg_p
        else:
            if len(self.bodies) >= 2:
                avg_p = (self.bodies[0].position + self.bodies[1].position) / 2.0
                self.bodies[0].position = avg_p
                self.bodies[1].position = avg_p

    # Limite strictement le mouvement des articulations
    def _clamp_legs(self):
        if self.robot_type == "5_membres":
            min_hip = math.radians(-90)
            max_hip = math.radians(90)
            min_knee = math.radians(-5)
            max_knee = math.radians(105)

            for thigh in (self.thigh0, self.thigh1):
                rel = thigh.angle - self.torso.angle
                if rel > max_hip:
                    thigh.angle = self.torso.angle + max_hip
                    if thigh.angular_velocity > self.torso.angular_velocity:
                        thigh.angular_velocity = self.torso.angular_velocity
                elif rel < min_hip:
                    thigh.angle = self.torso.angle + min_hip
                    if thigh.angular_velocity < self.torso.angular_velocity:
                        thigh.angular_velocity = self.torso.angular_velocity

            for thigh, shin in [(self.thigh0, self.shin0), (self.thigh1, self.shin1)]:
                rel_k = shin.angle - thigh.angle
                if rel_k > max_knee:
                    shin.angle = thigh.angle + max_knee
                    if shin.angular_velocity > thigh.angular_velocity:
                        shin.angular_velocity = thigh.angular_velocity
                elif rel_k < min_knee:
                    shin.angle = thigh.angle + min_knee
                    if shin.angular_velocity < thigh.angular_velocity:
                        shin.angular_velocity = thigh.angular_velocity
        elif self.robot_type == "3_membres":
            min_rad = math.radians(-90)
            max_rad = math.radians(90)
            for i in (0, 1):
                b = self.bodies[i]
                rel_angle = b.angle - self.torso.angle
                if rel_angle > max_rad:
                    b.angle = self.torso.angle + max_rad
                    if b.angular_velocity > self.torso.angular_velocity:
                        b.angular_velocity = self.torso.angular_velocity
                elif rel_angle < min_rad:
                    b.angle = self.torso.angle + min_rad
                    if b.angular_velocity < self.torso.angular_velocity:
                        b.angular_velocity = self.torso.angular_velocity
        else:
            min_rad = math.radians(-90)
            max_rad = math.radians(90)
            for b in (self.bodies[0], self.bodies[1]):
                if b.angle > max_rad:
                    b.angle = max_rad
                    if b.angular_velocity > 0:
                        b.angular_velocity = 0.0
                elif b.angle < min_rad:
                    b.angle = min_rad
                    if b.angular_velocity < 0:
                        b.angular_velocity = 0.0

    # Récupère les observations adaptées à la morphologie
    def _get_obs(self):
        if self.robot_type == "5_membres":
            torso = self.torso
            hauteur = (self.y_sol - torso.position.y) / 150.0
            sin_torso = math.sin(torso.angle)
            cos_torso = math.cos(torso.angle)
            w_torso = torso.angular_velocity / 10.0
            vx = torso.velocity.x / 100.0
            vy = torso.velocity.y / 100.0

            # Jambe droite (thigh0, shin0)
            rel_h0 = self.thigh0.angle - torso.angle
            sin_h0, cos_h0 = math.sin(rel_h0), math.cos(rel_h0)
            w_h0 = (self.thigh0.angular_velocity - torso.angular_velocity) / 10.0

            rel_k0 = self.shin0.angle - self.thigh0.angle
            sin_k0, cos_k0 = math.sin(rel_k0), math.cos(rel_k0)
            w_k0 = (self.shin0.angular_velocity - self.thigh0.angular_velocity) / 10.0

            # Jambe gauche (thigh1, shin1)
            rel_h1 = self.thigh1.angle - torso.angle
            sin_h1, cos_h1 = math.sin(rel_h1), math.cos(rel_h1)
            w_h1 = (self.thigh1.angular_velocity - torso.angular_velocity) / 10.0

            rel_k1 = self.shin1.angle - self.thigh1.angle
            sin_k1, cos_k1 = math.sin(rel_k1), math.cos(rel_k1)
            w_k1 = (self.shin1.angular_velocity - self.thigh1.angular_velocity) / 10.0

            toe0_y = self.shin0.local_to_world((10, 25)).y
            mid0_y = self.shin0.local_to_world((4, 28)).y
            heel0_y = self.shin0.local_to_world((-5, 28)).y
            ankle0_y = self.shin0.local_to_world((0, 28)).y
            foot0_max_y = max(toe0_y, mid0_y, heel0_y, ankle0_y)

            toe1_y = self.shin1.local_to_world((10, 25)).y
            mid1_y = self.shin1.local_to_world((4, 28)).y
            heel1_y = self.shin1.local_to_world((-5, 28)).y
            ankle1_y = self.shin1.local_to_world((0, 28)).y
            foot1_max_y = max(toe1_y, mid1_y, heel1_y, ankle1_y)

            contact0 = 1.0 if foot0_max_y >= (self.y_sol - 10.0) else 0.0
            contact1 = 1.0 if foot1_max_y >= (self.y_sol - 10.0) else 0.0

            raw_obs = np.array([
                hauteur, sin_torso, cos_torso, w_torso, vx, vy,
                sin_h0, cos_h0, w_h0, sin_k0, cos_k0, w_k0,
                sin_h1, cos_h1, w_h1, sin_k1, cos_k1, w_k1,
                contact0, contact1
            ], dtype=np.float32)

            if self.normalizer is not None:
                if self.update_normalizer:
                    self.normalizer.update(raw_obs)
                return self.normalizer.normalize(raw_obs)
            return raw_obs

        elif self.robot_type == "3_membres":
            b1, b2 = self.bodies[0], self.bodies[1]
            torso = self.torso
            hauteur = (self.y_sol - torso.position.y) / 150.0  
            rel_1 = b1.angle - torso.angle
            rel_2 = b2.angle - torso.angle
            sin_1, cos_1 = math.sin(rel_1), math.cos(rel_1)
            sin_2 = math.sin(rel_2)
            w_torso = torso.angular_velocity / 10.0
            w1 = (b1.angular_velocity - torso.angular_velocity) / 10.0
            w2 = (b2.angular_velocity - torso.angular_velocity) / 10.0
            vx = torso.velocity.x / 100.0
            sin_torso = math.sin(torso.angle)
            raw_obs = np.array([hauteur, sin_1, cos_1, sin_2, w_torso, w1, w2, vx, sin_torso], dtype=np.float32)
            if self.normalizer is not None:
                if self.update_normalizer:
                    self.normalizer.update(raw_obs)
                return self.normalizer.normalize(raw_obs)
            return raw_obs
        else:
            b1, b2 = self.bodies[0], self.bodies[1]
            hauteur = (self.y_sol - b1.position.y) / 150.0  
            sin_1, cos_1 = math.sin(b1.angle), math.cos(b1.angle)
            sin_2, cos_2 = math.sin(b2.angle), math.cos(b2.angle)
            w1 = b1.angular_velocity / 10.0
            w2 = b2.angular_velocity / 10.0
            vx = b1.velocity.x / 100.0
            vy = b1.velocity.y / 100.0
            raw_obs = np.array([hauteur, sin_1, cos_1, sin_2, cos_2, w1, w2, vx, vy], dtype=np.float32)
            if self.normalizer is not None:
                if self.update_normalizer:
                    self.normalizer.update(raw_obs)
                return self.normalizer.normalize(raw_obs)
            return raw_obs

    # Définit les angles cibles à partir de l'action [-1, 1]
    def _apply_action(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        self.prev_action = self.current_action.copy()

        if self.robot_type == "5_membres":
            if len(act) < 4:
                padded = np.zeros(4, dtype=np.float32)
                padded[:len(act)] = act
                act = padded
            self.current_action = act[:4].copy()
            # Hanches : [-pi/2, +pi/2]
            target_hip0 = self.current_action[0] * (math.pi / 2.0)
            target_hip1 = self.current_action[2] * (math.pi / 2.0)
            # Genoux : flexion [0, 90°] (action positive = plié vers l'arrière, action <= 0 = tendu droit)
            target_knee0 = np.clip(self.current_action[1], 0.0, 1.0) * math.radians(90.0)
            target_knee1 = np.clip(self.current_action[3], 0.0, 1.0) * math.radians(90.0)
            self.target_angles = np.array([target_hip0, target_knee0, target_hip1, target_knee1], dtype=np.float32)
        else:
            if len(act) == 1:
                self.current_action = np.array([act[0], -act[0]], dtype=np.float32)
            else:
                self.current_action = act[:2].copy()
            self.target_angles = self.current_action * (math.pi / 2.0)

    # Récompense pour l'équilibre debout avec bonus de survie et pénalité de chute
    def _reward_standing(self, action):
        if self._is_fallen():
            return -2.0

        if self.robot_type == "5_membres":
            hauteur = max(0.0, self.y_sol - self.torso.position.y)
            head_angle = self.torso.angle
            alive_bonus = 1.0
            height_bonus = 0.5 * min(1.0, hauteur / 55.0)
            upright_bonus = 0.25 * (math.cos(self.thigh0.angle) + math.cos(self.thigh1.angle) + math.cos(self.shin0.angle) + math.cos(self.shin1.angle))
            cos_head = max(0.0, math.cos(head_angle))
            align = cos_head * math.exp(-(head_angle ** 2) / (2.0 * (math.radians(20) ** 2)))
            head_reward = 0.5 * (2.0 * align - 1.0)
            action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
            smoothness_penalty = 0.03 * action_diff
            effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))
            reward = (
                alive_bonus
                + height_bonus
                + upright_bonus
                + head_reward
                - smoothness_penalty
                - effort_penalty
            )
            return float(reward)

        elif self.robot_type == "3_membres":
            b1, b2 = self.bodies[0], self.bodies[1]
            hauteur = max(0.0, self.y_sol - self.torso.position.y)
            head_angle = self.torso.angle
            alive_bonus = 1.0
            height_bonus = 0.5 * min(1.0, hauteur / 50.0)
            upright_bonus = 0.5 * (math.cos(b1.angle) + math.cos(b2.angle))
            cos_head = max(0.0, math.cos(head_angle))
            align = cos_head * math.exp(-(head_angle ** 2) / (2.0 * (math.radians(20) ** 2)))
            head_reward = 0.5 * (2.0 * align - 1.0)
            action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
            smoothness_penalty = 0.05 * action_diff
            effort_penalty = 0.01 * float(np.sum(np.square(self.current_action)))
            reward = (
                alive_bonus
                + height_bonus
                + upright_bonus
                + head_reward
                - smoothness_penalty
                - effort_penalty
            )
            return float(reward)
        else:
            b1, b2 = self.bodies[0], self.bodies[1]
            hauteur = max(0.0, self.y_sol - b1.position.y)
            head_angle = (b1.angle + b2.angle) / 2.0
            alive_bonus = 1.0
            height_bonus = 0.5 * min(1.0, hauteur / 50.0)
            upright_bonus = 0.5 * (math.cos(b1.angle) + math.cos(b2.angle))
            cos_head = max(0.0, math.cos(head_angle))
            align = cos_head * math.exp(-(head_angle ** 2) / (2.0 * (math.radians(20) ** 2)))
            head_reward = 0.5 * (2.0 * align - 1.0)
            action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
            smoothness_penalty = 0.05 * action_diff
            effort_penalty = 0.01 * float(np.sum(np.square(self.current_action)))
            reward = (
                alive_bonus
                + height_bonus
                + upright_bonus
                + head_reward
                - smoothness_penalty
                - effort_penalty
            )
            return float(reward)

    # Récompense pour la marche : incitation continue à la vitesse vers l'avant pour faire émerger une démarche naturelle rapide
    def _reward_walking(self, action):
        if self.robot_type == "5_membres":
            vx = self.torso.velocity.x
            hauteur = max(0.0, self.y_sol - self.torso.position.y)
            head_angle = self.torso.angle
            current_joint_x = self.torso.position.x

            target_vx = 200.0
            max_speed_reward = 4.0
            if vx >= 0:
                # Progression linéaire de 0 à 4.0 strictement proportionnelle à la vitesse, plafonnée à 4.0 à 200 px/s
                speed_reward = min(max_speed_reward, (vx / target_vx) * max_speed_reward)
            else:
                # Pénalité proportionnelle si le robot recule
                speed_reward = (vx / target_vx) * 2.0

            # Bonus de maintien de la hauteur (1.0 quand le robot est debout à hauteur normale >= 48px)
            height_bonus = 1.0 * min(1.0, hauteur / 48.0)

            # Pénalité anti-glisse sur les genoux : dissuade le robot de fléchir excessivement ou de racler ses genoux au sol
            k0_dist = max(0.0, self.y_sol - self.thigh0.local_to_world((0, 28)).y)
            k1_dist = max(0.0, self.y_sol - self.thigh1.local_to_world((0, 28)).y)
            min_knee = min(k0_dist, k1_dist)
            knee_penalty = 1.5 if min_knee < 8.0 else 0.0

            cos_head = max(0.0, math.cos(head_angle))
            head_align = cos_head * math.exp(-(head_angle ** 2) / (2.0 * (math.radians(25) ** 2)))
            head_reward = 0.6 * (2.0 * head_align - 1.0)

            action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
            smoothness_penalty = 0.02 * action_diff
            effort_penalty = 0.004 * float(np.sum(np.square(self.current_action)))

            reward = (
                speed_reward
                + height_bonus
                + head_reward
                - knee_penalty
                - smoothness_penalty
                - effort_penalty
            )
            self.prev_joint_x = current_joint_x
            return float(reward)

        elif self.robot_type == "3_membres":
            vx = self.torso.velocity.x
            hauteur = max(0.0, self.y_sol - self.torso.position.y)
            head_angle = self.torso.angle
            current_joint_x = self.torso.position.x
            if vx >= 0:
                speed_reward = min(4.0, vx / 50.0)
            else:
                speed_reward = 0.5 * (vx / 50.0)
            height_bonus = 0.5 * min(1.0, hauteur / 40.0)
            cos_head = max(0.0, math.cos(head_angle))
            head_align = cos_head * math.exp(-(head_angle ** 2) / (2.0 * (math.radians(25) ** 2)))
            head_reward = 0.6 * (2.0 * head_align - 1.0)
            action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
            smoothness_penalty = 0.03 * action_diff
            effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))
            reward = (
                speed_reward
                + height_bonus
                + head_reward
                - smoothness_penalty
                - effort_penalty
            )
            self.prev_joint_x = current_joint_x
            return float(reward)
        else:
            vx = (self.bodies[0].velocity.x + self.bodies[1].velocity.x) / 2.0
            hauteur = max(0.0, self.y_sol - self.bodies[0].position.y)
            head_angle = (self.bodies[0].angle + self.bodies[1].angle) / 2.0
            current_joint_x = self.bodies[0].position.x
            if vx >= 0:
                speed_reward = min(4.0, vx / 50.0)
            else:
                speed_reward = 0.5 * (vx / 50.0)
            height_bonus = 0.5 * min(1.0, hauteur / 40.0)
            cos_head = max(0.0, math.cos(head_angle))
            head_align = cos_head * math.exp(-(head_angle ** 2) / (2.0 * (math.radians(25) ** 2)))
            head_reward = 0.6 * (2.0 * head_align - 1.0)
            action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
            smoothness_penalty = 0.03 * action_diff
            effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))
            reward = (
                speed_reward
                + height_bonus
                + head_reward
                - smoothness_penalty
                - effort_penalty
            )
            self.prev_joint_x = current_joint_x
            return float(reward)

    # Alias publics
    def reward_standing(self, action):
        return self._reward_standing(action)

    def reward_walking(self, action):
        return self._reward_walking(action)

    # Creation des jambes du robot (Friction = 1.0, effective 2.5 sur le sol à 2.5)
    def create_segment(self, x1, y1, x2, y2, thickness=5, mass=1.0, friction=1.0, color=None, add_foot=True):
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = (x1, y1)
        local_end = (x2 - x1, y2 - y1)
        shape = pymunk.Segment(body, (0, 0), local_end, thickness)
        shape.mass = mass
        shape.friction = friction
        shape.filter = pymunk.ShapeFilter(group=1)
        if color is not None:
            shape.color = color
    
        self.space.add(body, shape)

        # Pied solidaire à profil berceau / toe-spring (talon -6px, pointe relevée à +11px)
        if add_foot:
            foot_base = pymunk.Segment(body, (-6, local_end[1]), (5, local_end[1]), thickness)
            foot_base.mass = 0.14
            foot_base.friction = friction
            foot_base.filter = pymunk.ShapeFilter(group=1)
            if color is not None:
                foot_base.color = color

            foot_toe = pymunk.Segment(body, (5, local_end[1]), (11, local_end[1] - 3), thickness - 1)
            foot_toe.mass = 0.06
            foot_toe.friction = friction
            foot_toe.filter = pymunk.ShapeFilter(group=1)
            if color is not None:
                foot_toe.color = color

            self.space.add(foot_base, foot_toe)

        self.bodies.append(body)

    # Création du joint
    def create_joint(self, b1, b2, x, y, min_angle=-180, max_angle=180):
        # Utilisation des ancres locales (0, 0) pour que le joint soit exactement confondu sur les deux corps
        joint = pymunk.PivotJoint(b1, b2, (0, 0), (0, 0))
        limit = pymunk.RotaryLimitJoint(b1, b2, math.radians(min_angle), math.radians(max_angle))
        
        self.space.add(joint, limit)
        self.joints.extend([joint, limit])


def detect_robot_type(model_path_or_key):
    """Détecte si un modèle ou une clé correspond au robot 2_membres, 3_membres ou 5_membres."""
    if not model_path_or_key:
        return "3_membres"
    str_val = str(model_path_or_key).lower()
    if "5_membres" in str_val or "5membres" in str_val or "5_membre" in str_val:
        return "5_membres"
    if "2_membres" in str_val or "2membres" in str_val or "2_membre" in str_val:
        return "2_membres"
    if "3_membres" in str_val or "3membres" in str_val or "3_membre" in str_val:
        return "3_membres"

    # Vérification du fichier JSON de métadonnées
    import os, json
    candidates = []
    if os.path.exists(str_val):
        candidates.append(os.path.splitext(str_val)[0] + ".json")
    if os.path.isdir("models"):
        base = os.path.splitext(os.path.basename(str_val))[0]
        candidates.append(os.path.join("models", f"{base}.json"))
        candidates.append(os.path.join("models", f"ppo_{base}.json"))

    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r") as f:
                    meta = json.load(f)
                    if "robot_type" in meta:
                        r = str(meta["robot_type"])
                        if "5" in r:
                            return "5_membres"
                        if "2" in r:
                            return "2_membres"
                        return "3_membres"
            except Exception:
                pass

    if "5" in str_val:
        return "5_membres"
    if "2" in str_val:
        return "2_membres"
    return "3_membres"


def get_available_models():
    """Détecte et liste tous les modèles disponibles dans models/ et à la racine."""
    import os
    models = {}
    if os.path.isdir("models"):
        for f in sorted(os.listdir("models")):
            if f.endswith(".pth") and "checkpoint" not in f.lower():
                key = os.path.splitext(f)[0].replace("ppo_", "").replace("robot_", "")
                models[key] = os.path.join("models", f)
    if os.path.exists("ppo_robot_model.pth") and "standing" not in models and "standing_2_membres" not in models:
        models["standing_legacy"] = "ppo_robot_model.pth"
    return models


def get_standing_model_for_robot(robot_type, models):
    """Retourne la clé du modèle standing correspondant à la morphologie du robot."""
    if not models:
        return None
    r_str = str(robot_type)
    if "5" in r_str:
        target_substr = "5_membres"
    elif "2" in r_str:
        target_substr = "2_membres"
    else:
        target_substr = "3_membres"

    # 1. Recherche exacte : standing ET le suffixe du robot
    for k in models:
        if "standing" in k.lower() and target_substr in k.lower():
            return k
    # 2. Recherche par détection automatique du type de robot
    for k in models:
        if "standing" in k.lower() and detect_robot_type(k) == target_substr:
            return k
    # 3. Premier modèle standing trouvé
    for k in models:
        if "standing" in k.lower():
            return k
    return list(models.keys())[0] if models else None


def load_policy_auto(model_path, state_dim=None, device="cpu"):
    """Charge un modèle PyTorch et déduit automatiquement ses dimensions d'état et d'action, ainsi que son Normalizer."""
    import torch
    import os, json
    from Models import ActorCritic
    loaded = torch.load(model_path, map_location=device)
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    
    if state_dim is None:
        if "actor_mean.0.weight" in state_dict:
            state_dim = state_dict["actor_mean.0.weight"].shape[1]
        elif "critic.0.weight" in state_dict:
            state_dim = state_dict["critic.0.weight"].shape[1]
        else:
            state_dim = 9
            
    if "actor_log_std" in state_dict:
        action_dim = state_dict["actor_log_std"].shape[1]
    elif "actor_mean.4.weight" in state_dict:
        action_dim = state_dict["actor_mean.4.weight"].shape[0]
    else:
        action_dim = 2
        
    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Détection automatique du Normalizer dans le fichier .json correspondant
    policy.normalizer = None
    json_path = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
                if "normalizer" in meta and meta["normalizer"]:
                    norm = Normalizer(size=state_dim)
                    norm.load_state_dict(meta["normalizer"])
                    policy.normalizer = norm
        except Exception:
            pass

    return policy, action_dim


def run_env(mode="manual", model_name=None, max_steps=500, robot_type="3_membres"):
    """
    Lance l'environnement en mode observation pour visualiser la physique
    sans lancer d'entraînement.

    Modes disponibles :
      - 'manual'   : Contrôle au clavier indépendant assisté par le modèle standing du robot (temps infini).
      - 'random'   : Actions aléatoires.
      - 'zero'     : Couple nul (simulation passive avec gravité sur sol infini).
      - 'model'    : Utilise une politique entraînée choisie (standing, walking, etc.).
    """
    import os
    available_models = get_available_models()
    model_keys = list(available_models.keys())
    
    # Détermination de la morphologie initiale du robot
    if "5" in str(robot_type):
        selected_robot = "5_membres"
    elif "2" in str(robot_type):
        selected_robot = "2_membres"
    else:
        selected_robot = "3_membres"

    # Choix du modèle initial
    current_model_key = None
    if model_name:
        for k, p in available_models.items():
            if model_name.lower() in (k.lower(), p.lower()):
                current_model_key = k
                selected_robot = detect_robot_type(current_model_key)
                break

    # En mode manuel (ou sans modèle spécifié), charger impérativement le modèle standing associé au robot
    if current_model_key is None or mode == "manual":
        standing_key = get_standing_model_for_robot(selected_robot, available_models)
        if standing_key:
            current_model_key = standing_key

    # En mode manuel, le temps est infini (pas de réinitialisation automatique)
    init_max_steps = None if mode == "manual" else max_steps
    task = "walking" if (current_model_key and "walk" in current_model_key and mode != "manual") else "standing"
    env = Env(max_steps=init_max_steps, render_mode=True, task=task, robot_type=selected_robot)
    obs = env.reset()

    # Chargement de la politique
    policy = None
    policy_action_dim = env.action_dim
    if current_model_key and current_model_key in available_models:
        try:
            policy, policy_action_dim = load_policy_auto(available_models[current_model_key])
            if getattr(policy, "normalizer", None):
                env.normalizer = policy.normalizer
                env.update_normalizer = False
            print(f"[Modèle Chargé] '{current_model_key}' activé pour le robot {selected_robot.upper()}.")
        except Exception as e:
            print(f"Avertissement : Impossible de charger '{current_model_key}' ({e}).")
            policy = None

    available_modes = ["manual", "random", "zero"]
    if policy is not None:
        available_modes.append("model")

    if mode not in available_modes:
        print(f"Mode '{mode}' non disponible. Utilisation du mode 'manual'.")
        mode = "manual"

    print("=" * 70)
    print("      OBSERVATION DE L'ENVIRONNEMENT (SANS ENTRAÎNEMENT)")
    print("=" * 70)
    print(f"Mode actif           : {mode.upper()}")
    print(f"Morphologie robot    : {env.robot_type.upper()}")
    print(f"Modèles détectés     : {list(available_models.keys()) if available_models else 'Aucun'}")
    if current_model_key:
        print(f"Modèle sélectionné   : {current_model_key.upper()} ({available_models[current_model_key]})")
    print("\nContrôles interactifs :")
    if env.robot_type == "5_membres":
        print("  - Jambe Droite (Rouge) : Flèches G/D (Hanche) | Flèches H/B (Genou)")
        print("  - Jambe Gauche (Bleue) : Q/D (Hanche)         | Z/S (Genou)")
    else:
        print("  - Jambe Droite (Rouge) : Flèche Gauche (Arrière) / Flèche Droite (Avant)")
        print("  - Jambe Gauche (Bleue) : Q (Arrière) / D (Avant)")
    print("  - M                    : Changer de mode (Manuel / Aléatoire / Passif / Modèle)")
    print("  - B                    : Basculer morphologie Robot (2 -> 3 -> 5 MEMBRES)")
    print("  - P                    : Changer de modèle entraîné")
    print("  - R ou Espace          : Réinitialiser le robot (Reset)")
    print("  - Échap / Croix        : Quitter")
    print("  * En mode manuel, la durée est infinie et le sol défile librement.")
    print("=" * 70)

    episode = 1
    episode_reward = 0.0
    reward = 0.0
    deterministic = (task == "standing")

    try:
        while True:
            keys = pygame.key.get_pressed()

            # Détermination de l'action selon le mode
            if mode == "manual":
                # Action de base d'équilibre issue du modèle standing associé au robot
                if policy is not None:
                    import torch
                    with torch.no_grad():
                        state_in = policy.normalizer.normalize(obs) if getattr(policy, "normalizer", None) else obs
                        state_tensor = torch.FloatTensor(state_in).unsqueeze(0)
                        action_tensor, _, _, _ = policy.get_action_and_value(state_tensor, deterministic=True)
                        action = action_tensor.cpu().numpy().flatten().copy()
                else:
                    action = np.zeros(env.action_dim, dtype=np.float32)

                # Commandes manuelles prioritaires
                if env.robot_type == "5_membres":
                    if len(action) < 4:
                        padded = np.zeros(4, dtype=np.float32)
                        padded[:len(action)] = action
                        action = padded
                    # Jambe Droite : Flèches
                    if keys[pygame.K_LEFT]:
                        action[0] = 1.0   # Hanche D arrière (-X)
                    elif keys[pygame.K_RIGHT]:
                        action[0] = -1.0  # Hanche D avant (+X)

                    if keys[pygame.K_UP]:
                        action[1] = 1.0   # Genou D plié
                    elif keys[pygame.K_DOWN]:
                        action[1] = 0.0   # Genou D tendu

                    # Jambe Gauche : Q/D et Z/S
                    if keys[pygame.K_q] or keys[pygame.K_a]:
                        action[2] = 1.0   # Hanche G arrière
                    elif keys[pygame.K_d]:
                        action[2] = -1.0  # Hanche G avant

                    if keys[pygame.K_z] or keys[pygame.K_w]:
                        action[3] = 1.0   # Genou G plié
                    elif keys[pygame.K_s]:
                        action[3] = 0.0   # Genou G tendu
                else:
                    if len(action) < 2:
                        action = np.zeros(2, dtype=np.float32)
                    if keys[pygame.K_q] or keys[pygame.K_a]:
                        action[1] = 1.0  # Jambe G arrière (-X)
                    elif keys[pygame.K_d]:
                        action[1] = -1.0  # Jambe G avant (+X)

                    if keys[pygame.K_LEFT]:
                        action[0] = 1.0  # Jambe D arrière (-X)
                    elif keys[pygame.K_RIGHT]:
                        action[0] = -1.0  # Jambe D avant (+X)

            elif mode == "random":
                action = np.random.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)

            elif mode == "zero":
                action = np.zeros(env.action_dim, dtype=np.float32)

            elif mode == "model" and policy is not None:
                import torch
                with torch.no_grad():
                    state_in = policy.normalizer.normalize(obs) if getattr(policy, "normalizer", None) else obs
                    state_tensor = torch.FloatTensor(state_in).unsqueeze(0)
                    action_tensor, _, _, _ = policy.get_action_and_value(state_tensor, deterministic=deterministic)
                    action = action_tensor.cpu().numpy().flatten()
            else:
                action = np.zeros(env.action_dim, dtype=np.float32)

            # Position et vitesse du robot selon la morphologie
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

            # Affichage du HUD
            if mode == "model" and current_model_key:
                mode_label = f"MODE: MODÈLE [{current_model_key.upper()}] (P: changer)"
            elif mode == "manual":
                loaded_str = current_model_key.upper() if current_model_key else "AUCUN"
                mode_label = f"MODE: MANUEL (Équilibre: {loaded_str})"
            else:
                mode_label = f"MODE: {mode.upper()}"

            step_str = f"{env.current_step} (infini)" if env.max_steps is None else f"{env.current_step}/{env.max_steps}"
            if len(env.target_angles) >= 4:
                act_str = f"H_D:{math.degrees(env.target_angles[0]):+.0f}° G_D:{math.degrees(env.target_angles[1]):+.0f}° | H_G:{math.degrees(env.target_angles[2]):+.0f}° G_G:{math.degrees(env.target_angles[3]):+.0f}°"
            elif len(env.target_angles) >= 2:
                act_str = f"D (Rouge): {math.degrees(env.target_angles[0]):+.0f}° | G (Bleue): {math.degrees(env.target_angles[1]):+.0f}°"
            else:
                act_str = f"Cible: {math.degrees(env.target_angles[0]):+.0f}°"

            head_status = "Horizontal" if abs(head_deg) < 15 else ("Incliné" if abs(head_deg) <= 90 else "À l'envers")
            mode_act = f" [{'DÉT' if deterministic else 'STOCH'}]" if mode == "model" else ""
            overlay = [
                f"{mode_label}{mode_act} | ROBOT: {env.robot_type.upper()} [B: changer]",
                f"Vitesse affichage: {speed_mult:.2f}x ({env.render_fps} FPS) [V ou +/-]" + (" [T: Dét/Stoch]" if mode == "model" else ""),
                f"Épisode: {episode} | Step: {step_str} | X: {robot_x:+.0f}px | Vx: {vx:+.1f}px/s",
                f"Tête: {head_deg:+.1f}° ({head_status}) | Angles PD: {act_str}",
                f"Reward pas: {reward:+.2f} | Total: {episode_reward:+.2f}",
                "Touches: B: Robot (2/3/5) | M: Mode | V: Vitesse | R: Reset | U: Perturber",
            ]
            env.set_overlay(overlay)

            obs, reward, done = env.step(action)
            episode_reward += reward
            events = env.get_events()

            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_r):
                        obs = env.reset(randomize=False)
                        print(f"Reset standard vertical [{env.robot_type}] | Épisode {episode} | Reward final: {episode_reward:.2f} | X final: {robot_x:.1f}px")
                        episode += 1
                        episode_reward = 0.0
                        reward = 0.0

                    elif event.key == pygame.K_u:
                        obs = env.reset(randomize=True)
                        print(f"Reset avec perturbation aléatoire [{env.robot_type}] | Épisode {episode}")
                        episode += 1
                        episode_reward = 0.0
                        reward = 0.0

                    elif event.key == pygame.K_b:
                        # Basculer en direct : 2_membres -> 3_membres -> 5_membres -> 2_membres
                        if env.robot_type == "2_membres":
                            next_robot = "3_membres"
                        elif env.robot_type == "3_membres":
                            next_robot = "5_membres"
                        else:
                            next_robot = "2_membres"

                        env.robot_type = next_robot
                        env.action_dim = 4 if next_robot == "5_membres" else 2
                        env.state_dim = 20 if next_robot == "5_membres" else 9
                        standing_key = get_standing_model_for_robot(env.robot_type, available_models)
                        if standing_key and standing_key in available_models:
                            current_model_key = standing_key
                            try:
                                policy, policy_action_dim = load_policy_auto(available_models[current_model_key])
                                if getattr(policy, "normalizer", None):
                                    env.normalizer = policy.normalizer
                                    env.update_normalizer = False
                                else:
                                    env.normalizer = None
                            except Exception as e:
                                print(f"Avertissement : Erreur chargement modèle '{current_model_key}' : {e}")
                                policy = None
                        else:
                            policy = None
                            env.normalizer = None

                        obs = env.reset(randomize=False)
                        episode += 1
                        episode_reward = 0.0
                        reward = 0.0
                        print(f"Bascule robot -> Morphologie : {env.robot_type.upper()} | Modèle standing : {current_model_key.upper() if current_model_key else 'Aucun'}")

                    elif event.key == pygame.K_m:
                        idx = available_modes.index(mode)
                        mode = available_modes[(idx + 1) % len(available_modes)]
                        env.max_steps = None if mode == "manual" else max_steps
                        if mode == "manual":
                            standing_key = get_standing_model_for_robot(env.robot_type, available_models)
                            if standing_key and standing_key in available_models and current_model_key != standing_key:
                                current_model_key = standing_key
                                try:
                                    policy, policy_action_dim = load_policy_auto(available_models[current_model_key])
                                except Exception:
                                    pass
                        print(f"Changement de mode -> {mode.upper()} (max_steps={env.max_steps})")

                    elif event.key == pygame.K_t and mode == "model":
                        deterministic = not deterministic
                        mode_name = "DÉTERMINISTE" if deterministic else "STOCHASTIQUE"
                        print(f"Mode d'action modèle -> {mode_name}")

                    elif event.key == pygame.K_p and model_keys:
                        # Changer de modèle entraîné
                        curr_idx = model_keys.index(current_model_key) if current_model_key in model_keys else 0
                        current_model_key = model_keys[(curr_idx + 1) % len(model_keys)]
                        try:
                            policy, policy_action_dim = load_policy_auto(available_models[current_model_key])
                            env.robot_type = detect_robot_type(current_model_key)
                            env.action_dim = 4 if env.robot_type == "5_membres" else 2
                            env.state_dim = 20 if env.robot_type == "5_membres" else 9
                            if getattr(policy, "normalizer", None):
                                env.normalizer = policy.normalizer
                                env.update_normalizer = False
                            else:
                                env.normalizer = None
                            env.task = "walking" if "walk" in current_model_key else "standing"
                            deterministic = (env.task == "standing")
                            obs = env.reset()
                            episode += 1
                            episode_reward = 0.0
                            reward = 0.0
                            print(f"Changement de modèle -> [{current_model_key.upper()}] (robot={env.robot_type}, tâche={env.task}, dim={policy_action_dim})")
                        except Exception as e:
                            print(f"Erreur chargement modèle : {e}")

                    elif event.key == pygame.K_v:
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

            if done:
                print(f"Fin épisode {episode} | Reward: {episode_reward:.2f} | X: {robot_x:.1f}px")
                obs = env.reset()
                episode += 1
                episode_reward = 0.0
                reward = 0.0

    except SystemExit:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Observer l'environnement physique et tester les modèles sans entraînement.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "random", "zero", "model"],
        default="manual",
        help="Mode d'observation: 'manual' (touches Q/D et flèches, temps infini), 'random', 'zero', 'model'."
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=["2", "3", "5", "2_membres", "3_membres", "5_membres"],
        default="3_membres",
        help="Morphologie du robot : '5_membres' (genoux articulés), '3_membres' (tronc vertical), ou '2_membres'."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nom ou fichier du modèle à exécuter en mode model (ex: 'walking', 'standing' ou chemin .pth)."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Nombre maximal d'étapes avant réinitialisation automatique hors mode manuel (défaut: 500)."
    )
    args = parser.parse_args()
    run_env(mode=args.mode, model_name=args.model, max_steps=args.max_steps, robot_type=args.robot)