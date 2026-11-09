import numpy as np
import math
import pygame
import pymunk
import pymunk.pygame_util
import os
import json


class Normalizer:
    """Normalisateur d'état en ligne O(1) basé sur l'algorithme de Welford / Chan et al.
    Calcule la moyenne et la variance cumulées de manière numériquement stable."""
    def __init__(self, size=26, eps=1e-8, clip=10.0):
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


# Environnement PyGame / Pymunk pour Robot Bipède articulé (avec 2 bras actionnables)
class Env:
    def __init__(self, max_steps=500, render_mode=False, task="standing", random_reset=True, normalizer=None, update_normalizer=False):
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.task = task
        self.random_reset = random_reset
        
        self.normalizer = normalizer
        self.update_normalizer = update_normalizer

        self.SCREEN_LENGHT = 600
        self.SCREEN_HEIGHT = 400
        self.y_sol = 350
        self.max_torque = 200000.0

        # Dimensions robot
        self.leg_length = 56.0  # 28 (cuisse) + 28 (tibia)
        self.arm_length = 49.0  # 7/8 de la taille de la jambe (56 * 7 / 8 = 49)

        # Contrôleur PD en position
        self.kp = 80000.0
        self.kd = 6000.0
        self.kp_knee = 90000.0
        self.kd_knee = 6000.0
        # Contrôleur PD souple (compliant) pour les bras (amortisseur dynamique)
        self.kp_arm = 12000.0
        self.kd_arm = 1500.0
        self.max_arm_torque = 30000.0

        self.action_dim = 6
        self.state_dim = 26

        self.target_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

        self.target_vx = 200.0

        self.bodies = []
        self.joints = []
        self.torso = None
        self.thigh0 = None
        self.thigh1 = None
        self.shin0 = None
        self.shin1 = None
        self.arm0 = None
        self.arm1 = None
        self.prev_joint_x = 300.0
        self.camera_x = 0.0

        # Physique Pymunk
        self.space = pymunk.Space()
        self.space.iterations = 35
        self.space.gravity = (0, 981)

        # Sol statique infini avec coefficient de friction Box2D (2.5)
        static_body = self.space.static_body
        floor = pymunk.Segment(static_body, (-1000000, self.y_sol), (1000000, self.y_sol), 5)
        floor.friction = 2.5
        self.floor = floor
        self.space.add(floor)

        self.render_fps = 60
        self.substep_render = True
        self.overlay_info = None
        self.pending_events = []
        self.last_head_perturbation = None
        self.perturbation_visual_timer = 0

        if self.render_mode:
            pygame.init()
            pygame.display.set_caption("Simulation Robot Bipède - V2")
            self.screen = pygame.display.set_mode((self.SCREEN_LENGHT, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.font = None

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

        if self.torso is not None:
            robot_x = self.torso.position.x
            self.camera_x = robot_x - (self.SCREEN_LENGHT / 2.0)
        else:
            self.camera_x = 0.0

        self.draw_options.transform = pymunk.Transform.translation(-self.camera_x, 0)
        self.screen.fill((255, 255, 255))
        
        self._draw_ground_markers()
        self.space.debug_draw(self.draw_options)
        self._draw_head()
        self._draw_perturbation_indicator()

        if self.overlay_info:
            self._draw_overlay(self.overlay_info)

        pygame.display.flip()
        self.clock.tick(self.render_fps)

        evs = list(self.pending_events)
        if not getattr(self, "_in_substep", False):
            self.pending_events.clear()
        return evs

    def set_overlay(self, info_lines):
        self.overlay_info = info_lines

    def get_events(self):
        evs = list(self.pending_events)
        self.pending_events.clear()
        return evs

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
                pygame.draw.line(self.screen, (200, 200, 200), (sx, self.y_sol - 8), (sx, self.y_sol + 8), 2)
                txt = self.font.render(f"{mx}px", True, (160, 160, 160))
                self.screen.blit(txt, (sx - 16, self.y_sol + 12))
        except Exception:
            pass

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

    def _draw_head(self):
        try:
            if self.torso is None:
                return

            angle = self.torso.angle
            deg = -math.degrees(angle)

            # Cou reliant les épaules (haut du torse) à la base de la tête
            neck_base = self.torso.local_to_world((0, -35))
            neck_top = self.torso.local_to_world((0, -40))
            p_base = (int(neck_base.x - self.camera_x), int(neck_base.y))
            p_top = (int(neck_top.x - self.camera_x), int(neck_top.y))
            pygame.draw.line(self.screen, (230, 195, 165), p_base, p_top, 5)
            pygame.draw.line(self.screen, (60, 60, 60), p_base, p_top, 1)

            # Tête surélevée au-dessus des épaules pour ne pas superposer les bras vers le bas
            head_center = self.torso.local_to_world((0, -50))
            screen_x = int(head_center.x - self.camera_x)
            screen_y = int(head_center.y)

            head_radius = 14
            head_size = (head_radius + 4) * 2
            surf = pygame.Surface((head_size, head_size), pygame.SRCALPHA)
            center = head_size // 2

            pygame.draw.circle(surf, (255, 218, 185), (center, center), head_radius)
            pygame.draw.circle(surf, (60, 60, 60), (center, center), head_radius, 2)

            eye_radius = 2.5
            eye_y = center - 2
            pygame.draw.circle(surf, (30, 30, 30), (center + 5, eye_y), eye_radius)
            pygame.draw.circle(surf, (255, 255, 255), (center + 6, eye_y - 1), 1)

            pygame.draw.circle(surf, (30, 30, 30), (center - 5, eye_y), eye_radius)
            pygame.draw.circle(surf, (255, 255, 255), (center - 4, eye_y - 1), 1)

            mouth_rect = pygame.Rect(center - 4, center + 3, 9, 5)
            pygame.draw.arc(surf, (180, 50, 50), mouth_rect, 3.14, 0, 2)

            rotated_surf = pygame.transform.rotate(surf, deg)
            rect = rotated_surf.get_rect(center=(screen_x, screen_y))
            self.screen.blit(rotated_surf, rect)
        except Exception:
            pass

    def _draw_perturbation_indicator(self):
        try:
            if getattr(self, "perturbation_visual_timer", 0) <= 0:
                return
            if not hasattr(self, "torso") or self.torso is None:
                return
            if not hasattr(self, "last_head_perturbation") or self.last_head_perturbation is None:
                return

            head_center = self.torso.local_to_world((0, -50))
            hx = int(head_center.x - self.camera_x)
            hy = int(head_center.y)

            imp_x, imp_y = self.last_head_perturbation["impulse"]
            length = math.hypot(imp_x, imp_y)
            if length > 1e-3:
                dir_x = imp_x / length
                dir_y = imp_y / length
                arrow_len = 50.0
                start_x = int(hx - dir_x * arrow_len)
                start_y = int(hy - dir_y * arrow_len)

                # Trait de flèche de force (orange vif / rouge)
                pygame.draw.line(self.screen, (255, 60, 20), (start_x, start_y), (hx, hy), 4)

                # Tête de la flèche
                tip_angle = math.atan2(dir_y, dir_x)
                p1 = (
                    int(hx - 12 * math.cos(tip_angle - math.pi / 6)),
                    int(hy - 12 * math.sin(tip_angle - math.pi / 6)),
                )
                p2 = (
                    int(hx - 12 * math.cos(tip_angle + math.pi / 6)),
                    int(hy - 12 * math.sin(tip_angle + math.pi / 6)),
                )
                pygame.draw.polygon(self.screen, (255, 60, 20), [(hx, hy), p1, p2])

                # Onde d'impact autour de la tête
                pygame.draw.circle(self.screen, (255, 120, 0), (hx, hy), 18, 2)
        except Exception:
            pass

    def close(self):
        if self.render_mode:
            pygame.quit() 

    def reset(self, randomize=None):
        self.current_step = 0
        self.target_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_head_perturbation = None
        self.perturbation_visual_timer = 0

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
        self.arm0 = None
        self.arm1 = None

        do_random = (self.random_reset and self.task == "standing") if randomize is None else bool(randomize)
        px = 300.0

        if do_random:
            # Perturbations réalistes modérées pour permettre l'apprentissage de l'équilibre
            max_angle_deg = 5.0
            theta_torso = float(np.random.uniform(-math.radians(3.0), math.radians(3.0)))
            theta_hip0 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
            theta_hip1 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
            theta_knee0 = float(np.random.uniform(0.0, math.radians(5.0)))
            theta_knee1 = float(np.random.uniform(0.0, math.radians(5.0)))
            theta_arm0 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))
            theta_arm1 = float(np.random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg)))

            th0_angle = theta_torso + theta_hip0
            th1_angle = theta_torso + theta_hip1
            sh0_angle = th0_angle + theta_knee0
            sh1_angle = th1_angle + theta_knee1
            arm0_angle = theta_torso + theta_arm0
            arm1_angle = theta_torso + theta_arm1

            reach0 = 28.0 * math.cos(th0_angle) + 28.0 * math.cos(sh0_angle)
            reach1 = 28.0 * math.cos(th1_angle) + 28.0 * math.cos(sh1_angle)
            max_reach = max(reach0, reach1)
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

            # 5. Bras (rouge = arm0 droit, bleu = arm1 gauche) attachés au haut du torse (longueur = 7/8 jambe = 49 px, masse allégée = 0.20 kg)
            shoulder = self.torso.local_to_world((0, -35))
            self.arm0 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            self.arm0.position = shoulder
            self.arm0.angle = arm0_angle
            arm0_seg = pymunk.Segment(self.arm0, (0, 0), (0, 49), 3)
            arm0_seg.mass = 0.20
            arm0_seg.friction = 1.0
            arm0_seg.color = (235, 75, 75, 255)
            arm0_seg.filter = pymunk.ShapeFilter(group=1)
            self.space.add(self.arm0, arm0_seg)

            self.arm1 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            self.arm1.position = shoulder
            self.arm1.angle = arm1_angle
            arm1_seg = pymunk.Segment(self.arm1, (0, 0), (0, 49), 3)
            arm1_seg.mass = 0.20
            arm1_seg.friction = 1.0
            arm1_seg.color = (45, 125, 240, 255)
            arm1_seg.filter = pymunk.ShapeFilter(group=1)
            self.space.add(self.arm1, arm1_seg)

            # Épaules (balancement avant/arrière [-110°, +110°])
            pj_a0 = pymunk.PivotJoint(self.torso, self.arm0, (0, -35), (0, 0))
            rj_a0 = pymunk.RotaryLimitJoint(self.torso, self.arm0, math.radians(-110), math.radians(110))
            pj_a1 = pymunk.PivotJoint(self.torso, self.arm1, (0, -35), (0, 0))
            rj_a1 = pymunk.RotaryLimitJoint(self.torso, self.arm1, math.radians(-110), math.radians(110))

            self.space.add(pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1, pj_a0, rj_a0, pj_a1, rj_a1)
            self.joints.extend([pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1, pj_a0, rj_a0, pj_a1, rj_a1])
            self.bodies = [self.thigh0, self.shin0, self.thigh1, self.shin1, self.torso, self.arm0, self.arm1]

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

            # 5. Bras (rouge = arm0 droit, bleu = arm1 gauche) attachés au haut du torse (longueur = 7/8 jambe = 49 px, masse allégée = 0.20 kg)
            shoulder = self.torso.local_to_world((0, -35))
            self.arm0 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            self.arm0.position = shoulder
            arm0_seg = pymunk.Segment(self.arm0, (0, 0), (0, 49), 3)
            arm0_seg.mass = 0.20
            arm0_seg.friction = 1.0
            arm0_seg.color = (235, 75, 75, 255)
            arm0_seg.filter = pymunk.ShapeFilter(group=1)
            self.space.add(self.arm0, arm0_seg)

            self.arm1 = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            self.arm1.position = shoulder
            arm1_seg = pymunk.Segment(self.arm1, (0, 0), (0, 49), 3)
            arm1_seg.mass = 0.20
            arm1_seg.friction = 1.0
            arm1_seg.color = (45, 125, 240, 255)
            arm1_seg.filter = pymunk.ShapeFilter(group=1)
            self.space.add(self.arm1, arm1_seg)

            # Épaules
            pj_a0 = pymunk.PivotJoint(self.torso, self.arm0, (0, -35), (0, 0))
            rj_a0 = pymunk.RotaryLimitJoint(self.torso, self.arm0, math.radians(-110), math.radians(110))
            pj_a1 = pymunk.PivotJoint(self.torso, self.arm1, (0, -35), (0, 0))
            rj_a1 = pymunk.RotaryLimitJoint(self.torso, self.arm1, math.radians(-110), math.radians(110))

            self.space.add(pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1, pj_a0, rj_a0, pj_a1, rj_a1)
            self.joints.extend([pj_h0, rj_h0, pj_h1, rj_h1, pj_k0, rj_k0, pj_k1, rj_k1, pj_a0, rj_a0, pj_a1, rj_a1])
            self.bodies = [self.thigh0, self.shin0, self.thigh1, self.shin1, self.torso, self.arm0, self.arm1]

            for b in self.bodies:
                b.velocity = (0.0, 0.0)
                b.angular_velocity = 0.0

        self.prev_joint_x = self.torso.position.x
        return self._get_obs()

    def apply_head_perturbation(self, magnitude=None, side=None, angle_deg=None):
        """
        Applique une grande force/impulsion sur la tête (point local (0, -50) du torse)
        dans une direction aléatoire comprise entre -45° et +45° dans le repère trigonométrique,
        orientée soit vers la droite (side=+1) soit vers la gauche (side=-1).
        Calibrée (250-320 N·s) pour que le robot ait besoin d'un ou deux pas pour rétablir
        son équilibre sans chuter par terre.
        """
        if not hasattr(self, "torso") or self.torso is None:
            return None
        if magnitude is None:
            magnitude = float(np.random.uniform(250.0, 320.0))
        if side is None:
            side = 1.0 if np.random.rand() > 0.5 else -1.0
        if angle_deg is None:
            angle_deg = float(np.random.uniform(-45.0, 45.0))

        phi = math.radians(angle_deg)
        imp_x = side * magnitude * math.cos(phi)
        imp_y = -magnitude * math.sin(phi)  # Repère Pymunk (+y vers le bas, -y vers le haut)
        self.torso.apply_impulse_at_local_point((imp_x, imp_y), (0, -50))
        self.perturbation_visual_timer = 20

        info = {
            "side": "droite" if side > 0 else "gauche",
            "angle_deg": angle_deg,
            "magnitude": magnitude,
            "impulse": (imp_x, imp_y),
            "step": self.current_step,
        }
        self.last_head_perturbation = info
        return info

    def step(self, action):
        self.current_step += 1

        # Aux 3/4 de la simulation standing, application automatique d'une grande force sur la tête
        if self.task == "standing" and self.max_steps is not None:
            if self.current_step == int(self.max_steps * 0.75):
                self.apply_head_perturbation()

        if getattr(self, "perturbation_visual_timer", 0) > 0:
            self.perturbation_visual_timer -= 1

        self._apply_action(action)
        self._in_substep = True
        try:
            for _ in range(4):
                self._update_pd_torques()
                self.space.step(1.0 / 60.0)
                self._clamp_legs()
                self._sync_joint()
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
        if self.max_steps is not None and self._is_fallen():
            done = True

        if self.max_steps is not None and self.current_step >= self.max_steps:
            done = True
        return obs, reward, done

    def _is_fallen(self):
        if not hasattr(self, "torso") or self.torso is None:
            return False
        hauteur = self.y_sol - self.torso.position.y
        head_angle = self.torso.angle
        if self.task == "standing":
            max_tilt = math.radians(65)
            return (hauteur < 18.0) or (abs(head_angle) > max_tilt)
        else:
            k0_clearance = self.y_sol - self.thigh0.local_to_world((0, 28)).y if hasattr(self, "thigh0") and self.thigh0 else 99.0
            k1_clearance = self.y_sol - self.thigh1.local_to_world((0, 28)).y if hasattr(self, "thigh1") and self.thigh1 else 99.0
            knee_crash = min(k0_clearance, k1_clearance) < 5.0
            return (hauteur < 22.0) or (math.cos(head_angle) <= 0.0) or knee_crash

    def _update_pd_torques(self):
        # Hanches (torso <-> cuisses) et Genoux (cuisses <-> tibias)
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
            thigh.torque = torque_hip - torque_knee

        # Épaules (bras rouge = arm0 droit, bras bleu = arm1 gauche) : contrôle souple (compliant)
        if hasattr(self, "arm0") and self.arm0 is not None:
            rel_a0 = self.arm0.angle - self.torso.angle
            rel_w_a0 = self.arm0.angular_velocity - self.torso.angular_velocity
            err_a0 = self.target_angles[4] - rel_a0
            torque_a0 = float(np.clip(self.kp_arm * err_a0 - self.kd_arm * rel_w_a0, -self.max_arm_torque, self.max_arm_torque))
            self.arm0.torque = torque_a0

        if hasattr(self, "arm1") and self.arm1 is not None:
            rel_a1 = self.arm1.angle - self.torso.angle
            rel_w_a1 = self.arm1.angular_velocity - self.torso.angular_velocity
            err_a1 = self.target_angles[5] - rel_a1
            torque_a1 = float(np.clip(self.kp_arm * err_a1 - self.kd_arm * rel_w_a1, -self.max_arm_torque, self.max_arm_torque))
            self.arm1.torque = torque_a1

        # Stabilisation posturale active du tronc (réflexes vestibulospinaux et spinaux)
        t_torso = -self.kp * self.torso.angle - self.kd * self.torso.angular_velocity
        self.torso.torque = float(np.clip(t_torso, -self.max_torque, self.max_torque))

    def _sync_joint(self):
        if hasattr(self, "torso") and self.torso is not None and hasattr(self, "thigh0") and hasattr(self, "thigh1"):
            avg_hip = (self.torso.position * 2.0 + self.thigh0.position + self.thigh1.position) / 4.0
            self.torso.position = avg_hip
            self.thigh0.position = avg_hip
            self.thigh1.position = avg_hip
            if hasattr(self, "shin0") and hasattr(self, "shin1"):
                self.shin0.position = self.thigh0.local_to_world((0, 28))
                self.shin1.position = self.thigh1.local_to_world((0, 28))
            if hasattr(self, "arm0") and self.arm0 and hasattr(self, "arm1") and self.arm1:
                shoulder = self.torso.local_to_world((0, -35))
                self.arm0.position = shoulder
                self.arm1.position = shoulder

    def _clamp_legs(self):
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

        for arm in (self.arm0, self.arm1):
            if arm is None:
                continue
            rel_a = arm.angle - self.torso.angle
            max_arm = math.radians(110)
            min_arm = math.radians(-110)
            if rel_a > max_arm:
                arm.angle = self.torso.angle + max_arm
                if arm.angular_velocity > self.torso.angular_velocity:
                    arm.angular_velocity = self.torso.angular_velocity
            elif rel_a < min_arm:
                arm.angle = self.torso.angle + min_arm
                if arm.angular_velocity < self.torso.angular_velocity:
                    arm.angular_velocity = self.torso.angular_velocity

    def _get_obs(self):
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

        # Bras droit (arm0 - rouge)
        rel_a0 = self.arm0.angle - torso.angle if self.arm0 is not None else 0.0
        sin_a0, cos_a0 = math.sin(rel_a0), math.cos(rel_a0)
        w_a0 = ((self.arm0.angular_velocity - torso.angular_velocity) / 10.0) if self.arm0 is not None else 0.0

        # Bras gauche (arm1 - bleu)
        rel_a1 = self.arm1.angle - torso.angle if self.arm1 is not None else 0.0
        sin_a1, cos_a1 = math.sin(rel_a1), math.cos(rel_a1)
        w_a1 = ((self.arm1.angular_velocity - torso.angular_velocity) / 10.0) if self.arm1 is not None else 0.0

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
            sin_a0, cos_a0, w_a0, sin_a1, cos_a1, w_a1,
            contact0, contact1
        ], dtype=np.float32)

        if self.normalizer is not None:
            if self.update_normalizer:
                self.normalizer.update(raw_obs)
            return self.normalizer.normalize(raw_obs)
        return raw_obs

    def _apply_action(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        self.prev_action = self.current_action.copy()

        if len(act) < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[:len(act)] = act
            act = padded
        self.current_action = act[:self.action_dim].copy()

        # Hanches : [-pi/2, +pi/2]
        target_hip0 = self.current_action[0] * (math.pi / 2.0)
        target_hip1 = self.current_action[2] * (math.pi / 2.0)
        # Genoux : flexion [0, 90°] (action positive = plié vers l'arrière, action <= 0 = tendu droit)
        target_knee0 = np.clip(self.current_action[1], 0.0, 1.0) * math.radians(90.0)
        target_knee1 = np.clip(self.current_action[3], 0.0, 1.0) * math.radians(90.0)
        # Bras (épaules) : balancement avant/arrière [-pi/2, +pi/2]
        target_arm0 = self.current_action[4] * (math.pi / 2.0)
        target_arm1 = self.current_action[5] * (math.pi / 2.0)
        self.target_angles = np.array([
            target_hip0, target_knee0, target_hip1, target_knee1, target_arm0, target_arm1
        ], dtype=np.float32)

    def _reward_standing(self, action):
        if self._is_fallen():
            return -2.0

        hauteur = max(0.0, self.y_sol - self.torso.position.y)
        head_angle = self.torso.angle
        alive_bonus = 1.0
        height_bonus = min(1.0, hauteur / 55.0)
        upright_bonus = (math.cos(self.thigh0.angle) + math.cos(self.thigh1.angle) + math.cos(self.shin0.angle) + math.cos(self.shin1.angle))
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

    def _reward_walking(self, action):
        vx = self.torso.velocity.x
        hauteur = max(0.0, self.y_sol - self.torso.position.y)
        head_angle = self.torso.angle
        current_joint_x = self.torso.position.x

        target_vx = 200.0
        max_speed_reward = 4.0
        if vx >= 0:
            # Progression linéaire de 0 à 4.0 proportionnelle à la vitesse, plafonnée à 4.0 à 200 px/s
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

    def reward_standing(self, action):
        return self._reward_standing(action)

    def reward_walking(self, action):
        return self._reward_walking(action)


def get_available_models(models_dir=None):
    """Détecte et liste les modèles dans models/."""
    if models_dir is None or not os.path.isdir(models_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "models")
        if os.path.isdir(candidate):
            models_dir = candidate
        elif models_dir is None:
            models_dir = "models"

    models = {}
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pth") and "checkpoint" not in f.lower():
                key = os.path.splitext(f)[0].replace("ppo_", "").replace("robot_", "")
                models[key] = os.path.join(models_dir, f)
    return models


def load_policy_auto(model_path, state_dim=26, action_dim=6, device="cpu"):
    """Charge un modèle PyTorch pour le robot bipède et attache son Normalizer Welford s'il existe."""
    import torch
    from Models import ActorCritic

    loaded = torch.load(model_path, map_location=device)
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded

    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)

    loaded_state_dim = state_dict["actor_mean.0.weight"].shape[1]
    loaded_action_dim = state_dict["actor_mean.4.weight"].shape[0]

    if loaded_state_dim == state_dim and loaded_action_dim == action_dim:
        policy.load_state_dict(state_dict)
    else:
        # Transfert adaptable des poids
        curr_dict = policy.state_dict()
        for name, param in curr_dict.items():
            if name in state_dict:
                src = state_dict[name]
                if src.shape == param.shape:
                    param.copy_(src)
                elif "weight" in name and param.dim() == 2 and src.dim() == 2:
                    r = min(param.shape[0], src.shape[0])
                    c = min(param.shape[1], src.shape[1])
                    param[:r, :c].copy_(src[:r, :c])
                elif "bias" in name and param.dim() == 1 and src.dim() == 1:
                    r = min(param.shape[0], src.shape[0])
                    param[:r].copy_(src[:r])
                elif name == "actor_log_std" and src.shape[1] <= param.shape[1]:
                    param[:, :src.shape[1]].copy_(src)
        policy.load_state_dict(curr_dict)

    policy.eval()

    policy.normalizer = None
    json_path = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
                if "normalizer" in meta and meta["normalizer"]:
                    norm = Normalizer(size=state_dim)
                    old_mean = meta["normalizer"].get("mean", [])
                    old_var = meta["normalizer"].get("var", [])
                    n_old = min(len(old_mean), state_dim)
                    norm.mean[:n_old] = np.array(old_mean[:n_old], dtype=np.float32)
                    norm.var[:n_old] = np.array(old_var[:n_old], dtype=np.float32)
                    norm.count = float(meta["normalizer"].get("count", 1.0))
                    policy.normalizer = norm
        except Exception:
            pass

    return policy, action_dim


def run_env(mode="manual", model_name=None, max_steps=500):
    """
    Lance l'environnement en mode observation pour visualiser la physique du robot bipède.

    Modes disponibles :
      - 'manual'   : Contrôle au clavier indépendant assisté par le modèle standing (durée infinie).
      - 'random'   : Actions aléatoires.
      - 'zero'     : Couple nul (simulation passive).
      - 'model'    : Utilise une politique entraînée choisie (standing, walking).
    """
    available_models = get_available_models()
    
    current_model_key = None
    if model_name:
        for k, p in available_models.items():
            if model_name.lower() in (k.lower(), p.lower()):
                current_model_key = k
                break

    if current_model_key is None:
        for k in available_models:
            if "standing" in k.lower():
                current_model_key = k
                break
        if current_model_key is None and available_models:
            current_model_key = list(available_models.keys())[0]

    init_max_steps = None if mode == "manual" else max_steps
    task = "walking" if (current_model_key and "walk" in current_model_key and mode != "manual") else "standing"
    env = Env(max_steps=init_max_steps, render_mode=True, task=task)
    obs = env.reset()

    policy = None
    if current_model_key and current_model_key in available_models:
        try:
            policy, _ = load_policy_auto(available_models[current_model_key], state_dim=env.state_dim, action_dim=env.action_dim)
            if getattr(policy, "normalizer", None):
                env.normalizer = policy.normalizer
                env.update_normalizer = False
            print(f"[Modèle Chargé] '{current_model_key}' activé.")
        except Exception as e:
            print(f"Avertissement : Impossible de charger '{current_model_key}' : {e}")
            policy = None

    available_modes = ["manual", "random", "zero"]
    if policy is not None:
        available_modes.append("model")

    if mode not in available_modes:
        mode = "manual"

    print("=" * 70)
    print("      OBSERVATION DU ROBOT BIPÈDE (SANS ENTRAÎNEMENT)")
    print("=" * 70)
    print(f"Mode actif           : {mode.upper()}")
    print(f"Modèles détectés     : {list(available_models.keys()) if available_models else 'Aucun'}")
    if current_model_key:
        print(f"Modèle sélectionné   : {current_model_key.upper()} ({available_models[current_model_key]})")
    print("\nContrôles interactifs :")
    print("  - Jambe Droite (Rouge) : Flèches G/D (Hanche) | Flèches H/B (Genou)")
    print("  - Jambe Gauche (Bleue) : Q/D (Hanche)         | Z/S (Genou)")
    print("  - Bras Droit (Rouge)   : I/K (Épaule)")
    print("  - Bras Gauche (Bleu)   : E/R (Épaule)")
    print("  - M                    : Changer de mode (Manuel / Aléatoire / Passif / Modèle)")
    print("  - P                    : Changer de modèle entraîné (Standing / Walking)")
    print("  - R ou Espace          : Réinitialiser le robot (Reset standard)")
    print("  - U                    : Appliquer une grande force sur la tête (perturbation dynamique gauche/droite ±45°)")
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

            if mode == "manual":
                if policy is not None:
                    import torch
                    with torch.no_grad():
                        state_in = policy.normalizer.normalize(obs) if getattr(policy, "normalizer", None) else obs
                        state_tensor = torch.FloatTensor(state_in).unsqueeze(0)
                        action_tensor, _, _, _ = policy.get_action_and_value(state_tensor, deterministic=True)
                        act_vals = action_tensor.cpu().numpy().flatten().copy()
                        if len(act_vals) < env.action_dim:
                            padded = np.zeros(env.action_dim, dtype=np.float32)
                            padded[:len(act_vals)] = act_vals
                            act_vals = padded
                        action = act_vals[:env.action_dim]
                else:
                    action = np.zeros(env.action_dim, dtype=np.float32)

                # Commandes manuelles prioritaires
                if keys[pygame.K_LEFT]:
                    action[0] = 1.0
                elif keys[pygame.K_RIGHT]:
                    action[0] = -1.0

                if keys[pygame.K_UP]:
                    action[1] = 1.0
                elif keys[pygame.K_DOWN]:
                    action[1] = 0.0

                if keys[pygame.K_q] or keys[pygame.K_a]:
                    action[2] = 1.0
                elif keys[pygame.K_d]:
                    action[2] = -1.0

                if keys[pygame.K_z] or keys[pygame.K_w]:
                    action[3] = 1.0
                elif keys[pygame.K_s]:
                    action[3] = 0.0

                # Bras
                if keys[pygame.K_i]:
                    action[4] = 1.0
                elif keys[pygame.K_k]:
                    action[4] = -1.0

                if keys[pygame.K_e]:
                    action[5] = 1.0
                elif keys[pygame.K_r]:
                    action[5] = -1.0

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
                    act_vals = action_tensor.cpu().numpy().flatten()
                    if len(act_vals) < env.action_dim:
                        padded = np.zeros(env.action_dim, dtype=np.float32)
                        padded[:len(act_vals)] = act_vals
                        act_vals = padded
                    action = act_vals[:env.action_dim]
            else:
                action = np.zeros(env.action_dim, dtype=np.float32)

            robot_x = env.torso.position.x
            vx = env.torso.velocity.x
            head_deg = math.degrees(env.torso.angle)
            speed_mult = env.render_fps / 60.0

            if mode == "model" and current_model_key:
                mode_label = f"MODE: MODÈLE [{current_model_key.upper()}] (P: changer)"
            elif mode == "manual":
                loaded_str = current_model_key.upper() if current_model_key else "AUCUN"
                mode_label = f"MODE: MANUEL (Assistance équilibre: {loaded_str})"
            else:
                mode_label = f"MODE: {mode.upper()}"

            step_str = f"{env.current_step} (infini)" if env.max_steps is None else f"{env.current_step}/{env.max_steps}"
            act_str = f"H_D:{math.degrees(env.target_angles[0]):+.0f}° G_D:{math.degrees(env.target_angles[1]):+.0f}° B_D:{math.degrees(env.target_angles[4]):+.0f}° | H_G:{math.degrees(env.target_angles[2]):+.0f}° G_G:{math.degrees(env.target_angles[3]):+.0f}° B_G:{math.degrees(env.target_angles[5]):+.0f}°"
            head_status = "Horizontal" if abs(head_deg) < 15 else ("Incliné" if abs(head_deg) <= 90 else "À l'envers")
            mode_act = f" [{'DÉT' if deterministic else 'STOCH'}]" if mode == "model" else ""
            overlay = [
                f"{mode_label}{mode_act} | ROBOT BIPÈDE",
                f"Vitesse affichage: {speed_mult:.2f}x ({env.render_fps} FPS) [V ou +/-]" + (" [T: Dét/Stoch]" if mode == "model" else ""),
                f"Épisode: {episode} | Step: {step_str} | X: {robot_x:+.0f}px | Vx: {vx:+.1f}px/s",
                f"Tête: {head_deg:+.1f}° ({head_status}) | Angles PD: {act_str}",
                f"Reward pas: {reward:+.2f} | Total: {episode_reward:+.2f}",
                "Touches: M: Mode | P: Modèle | V: Vitesse | R: Reset | U: Perturber tête",
            ]
            if getattr(env, "perturbation_visual_timer", 0) > 0 and getattr(env, "last_head_perturbation", None):
                p_info = env.last_head_perturbation
                overlay.insert(4, f">> PERTURBATION TÊTE: {p_info['side'].upper()} ({p_info['angle_deg']:+.1f}°, {p_info['magnitude']:.0f} N·s) <<")
            env.set_overlay(overlay)

            obs, reward, done = env.step(action)
            episode_reward += reward
            events = env.get_events()

            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_r):
                        obs = env.reset(randomize=False)
                        print(f"Reset standard vertical | Épisode {episode} | Reward final: {episode_reward:.2f} | X: {robot_x:.1f}px")
                        episode += 1
                        episode_reward = 0.0
                        reward = 0.0

                    elif event.key == pygame.K_u:
                        info = env.apply_head_perturbation()
                        if info:
                            print(f"[Perturbation Tête] Coup appliqué vers la {info['side'].upper()} ({info['angle_deg']:+.1f}°, {info['magnitude']:.0f} N·s)")

                    elif event.key == pygame.K_m:
                        idx = available_modes.index(mode)
                        mode = available_modes[(idx + 1) % len(available_modes)]
                        env.max_steps = None if mode == "manual" else max_steps
                        obs = env.reset(randomize=False)
                        print(f"Bascule mode -> {mode.upper()}")

                    elif event.key == pygame.K_p and len(available_models) > 1:
                        keys_list = list(available_models.keys())
                        cur_idx = keys_list.index(current_model_key) if current_model_key in keys_list else 0
                        current_model_key = keys_list[(cur_idx + 1) % len(keys_list)]
                        task = "walking" if "walk" in current_model_key else "standing"
                        deterministic = (task == "standing")
                        env.task = task
                        try:
                            policy, _ = load_policy_auto(available_models[current_model_key], state_dim=env.state_dim, action_dim=env.action_dim)
                            if getattr(policy, "normalizer", None):
                                env.normalizer = policy.normalizer
                            print(f"Modèle actif -> '{current_model_key.upper()}' (Tâche: {task.upper()})")
                        except Exception as e:
                            print(f"Erreur changement modèle : {e}")

                    elif event.key == pygame.K_t:
                        deterministic = not deterministic
                        print(f"Mode politique -> {'DÉTERMINISTE' if deterministic else 'STOCHASTIQUE'}")

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

            if done:
                print(f"Fin épisode {episode} | Step: {env.current_step} | Reward: {episode_reward:.2f} | X: {robot_x:.1f}px")
                obs = env.reset(randomize=(task == "standing"))
                episode += 1
                episode_reward = 0.0
                reward = 0.0

    finally:
        env.close()


if __name__ == "__main__":
    import sys
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "manual"
    run_env(mode=mode_arg)

