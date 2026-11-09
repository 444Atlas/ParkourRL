import math
import os
import json
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame


class Normalizer:
    """Algorithme de Welford en ligne O(1) pour standardiser les états."""
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


def leg_ik(x, z, L1=28.0, L2=28.0, knee_backward=True):
    """
    Cinématique inverse analytique pour une patte planaire à 2 segments (cuisse L1, tibia L2).
    x : position horizontale dans le repère de la hanche (+ vers l'avant, - vers l'arrière)
    z : profondeur verticale vers le sol (+ vers le bas)
    knee_backward=True : genoux orientés vers l'arrière (< <, configuration Unitree Go1 / mammifère)
    knee_backward=False : genoux orientés vers l'avant (> >)
    Retourne (q_hip, q_knee) compatibles avec le repère angulaire de Pymunk.
    """
    D = math.sqrt(x * x + z * z)
    D = max(1.0, min(D, 0.98 * (L1 + L2)))

    # Loi des cosinus pour l'angle du genou
    cos_beta = (L1 * L1 + L2 * L2 - D * D) / (2.0 * L1 * L2)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)

    # Angle de la ligne hanche -> pied
    gamma = math.atan2(-x, z)
    cos_psi = (L1 * L1 + D * D - L2 * L2) / (2.0 * L1 * D)
    cos_psi = max(-1.0, min(1.0, cos_psi))
    psi = math.acos(cos_psi)

    if knee_backward:
        # Genou fléchi vers l'arrière (Go1 / mammifère) : qk <= 0, qh >= 0
        qk = -(math.pi - beta)
        qh = gamma + psi
    else:
        # Genou fléchi vers l'avant : qk >= 0, qh <= 0
        qk = math.pi - beta
        qh = gamma - psi

    return qh, qk


class HopfCPG:
    """
    Générateur de Motifs Centraux (CPG) de Hopf pour quadrupède (Trot Gait),
    conforme à l'article Visual CPG-RL (Bellegarda, Shafiee, Ijspeert).

    4 pattes :
      0: Front-Right (FR)
      1: Front-Left (FL)
      2: Hind-Right (HR)
      3: Hind-Left (HL)

    Allure de trot :
      Les paires diagonales oscillent en phase : (FR, HL) et (FL, HR).
      Les deux paires diagonales sont en opposition de phase (pi).
    """
    def __init__(self, f_base=1.5, dt=1.0 / 60.0, alpha=25.0, w_coupling=4.0):
        self.dt = float(dt)
        self.f_base = float(f_base)
        self.omega_base = 2.0 * math.pi * self.f_base
        self.alpha = float(alpha)
        self.w = float(w_coupling)

        # Paramètres géométriques de la trajectoire cartésienne des pieds (Eq. 4-6)
        self.base_dstep = 16.0    # demi-amplitude de pas (px) -> foulée nominale 32 px
        self.base_h = 47.0        # hauteur nominale d'extension des pattes en marche (px, 84% H_max conforme à Spot)
        self.base_gc = 16.0       # garde au sol symétrique sur les 4 pattes (avec profil Fast-Lift)
        self.base_gp = 1.5        # trajectoire de stance avec pénétration au niveau du sol (adhérence active)
        # Offsets adaptés aux hanches Spot (+27px avant / -27px arrière avec overhangs)
        self.base_front_x_offset = -6.0
        self.base_rear_x_offset = -20.0
        self.base_x_offset = -13.0

        self.r = np.ones(4, dtype=np.float32)
        self.mu = np.ones(4, dtype=np.float32)

        # Déphasage initial pour l'allure de trot :
        # Paires diagonales : (FR=0, HL=3) en phase 0, (FL=1, HR=2) en phase pi
        self.theta = np.array([0.0, math.pi, math.pi, 0.0], dtype=np.float32)
        self.omega = self.omega_base * np.ones(4, dtype=np.float32)

        # Matrice de déphasage nominal phi_ij (Trot gait de l'article)
        self.phi = np.zeros((4, 4), dtype=np.float32)
        # FR (0)
        self.phi[0, 1] = math.pi
        self.phi[0, 2] = math.pi
        self.phi[0, 3] = 0.0
        # FL (1)
        self.phi[1, 0] = -math.pi
        self.phi[1, 2] = 0.0
        self.phi[1, 3] = math.pi
        # HR (2)
        self.phi[2, 0] = -math.pi
        self.phi[2, 1] = 0.0
        self.phi[2, 3] = math.pi
        # HL (3)
        self.phi[3, 0] = 0.0
        self.phi[3, 1] = -math.pi
        self.phi[3, 2] = -math.pi

        self.dstep = self.base_dstep
        self.h = self.base_h
        self.gc = self.base_gc
        self.gp = self.base_gp
        self.pitch_offset = 0.0

    def reset(self, initial_phase=0.0):
        self.theta = np.array([
            initial_phase,
            (initial_phase + math.pi) % (2.0 * math.pi),
            (initial_phase + math.pi) % (2.0 * math.pi),
            initial_phase
        ], dtype=np.float32)
        self.r = self.mu.copy()

    def set_modulations(self, action):
        """Modulation descendante des paramètres du CPG par le réseau DRL (action in [-1, 1]^8)."""
        a = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        if len(a) < 8:
            padded = np.zeros(8, dtype=np.float32)
            padded[:len(a)] = a
            a = padded

        # 1. Fréquence globale (rythme CPG : ~1.1 Hz à 2.1 Hz)
        f = self.f_base * (1.0 + 0.3 * a[0])
        self.omega[:] = 2.0 * math.pi * f

        # 2. Longueur de pas (demi-amplitude dstep : 13 px à 19 px)
        self.dstep = float(np.clip(16.0 + 3.0 * a[1], 13.0, 19.0))

        # 3. Garde au sol symétrique en swing (gc : 12 px à 20 px)
        self.gc = float(np.clip(16.0 + 3.0 * a[2], 12.0, 20.0))

        # 4. Hauteur du corps (h : 45.0 px à 48.0 px, conforme à Spot 82% H_max)
        self.h = float(np.clip(46.5 + 1.5 * a[3], 45.0, 48.0))

        # 5. Décalage d'assiette longitudinale modéré (pitch ±2.0°)
        self.pitch_offset = math.radians(2.0) * a[4]

        # 6. Asymétrie bilatérale
        asym = 0.15 * a[5]
        self.mu[0] = max(0.5, 1.0 + asym)
        self.mu[1] = max(0.5, 1.0 - asym)
        self.mu[2] = max(0.5, 1.0 + asym)
        self.mu[3] = max(0.5, 1.0 - asym)

    def step(self):
        """Intégration numérique vectorisée avec couplage de Kuramoto."""
        substeps = 4
        dt_sub = self.dt / substeps
        for _ in range(substeps):
            if self.w > 0:
                diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis] - self.phi
                coupling = self.w * np.sum(np.sin(diff), axis=1)
                d_theta = self.omega + coupling
            else:
                d_theta = self.omega

            dr = self.alpha * (self.mu - self.r)
            self.r += dr * dt_sub
            self.theta = (self.theta + d_theta * dt_sub) % (2.0 * math.pi)

    def get_joint_targets(self, knee_backward=True):
        """
        Génère les trajectoires cartésiennes (x, z) de chaque patte (Équations 4-6 de l'article),
        puis applique la cinématique inverse pour produire les 8 angles articulaires :
        [qh_FR, qk_FR, qh_FL, qk_FL, qh_HR, qk_HR, qh_HL, qk_HL].
        """
        targets = []
        for i in range(4):
            th = self.theta[i]
            # Solution A : Découplage de l'offset des pieds (Avant -8px, Arrière -18px)
            if knee_backward:
                x_offset = getattr(self, "base_front_x_offset", -8.0) if i in (0, 1) else getattr(self, "base_rear_x_offset", -18.0)
            else:
                x_offset = 0.0

            # Longueur de pas pondérée par l'amplitude r_i
            x_foot = x_offset - self.dstep * self.r[i] * math.cos(th)
            s = math.sin(th)
            if s > 0:
                # Phase de swing (en l'air) avec profil Fast-Lift symétrique sur les 4 pattes
                s_eff = s ** 0.6
                z_foot = self.h - self.gc * s_eff
            else:
                # Phase de stance (au sol)
                z_foot = self.h - self.gp * s

            # Décalage de pitch : pattes avant vs pattes arrière
            if i in (0, 1):
                z_foot += 35.0 * math.tan(self.pitch_offset)
            else:
                z_foot -= 35.0 * math.tan(self.pitch_offset)

            qh, qk = leg_ik(x_foot, z_foot, knee_backward=knee_backward)
            targets.extend([qh, qk])

        return np.array(targets, dtype=np.float32)

    def get_phases(self):
        return self.theta.copy()

    def get_phase_features(self):
        feats = []
        for i in range(4):
            feats.extend([math.sin(self.theta[i]), math.cos(self.theta[i])])
        return feats


class Env:
    """
    Environnement de simulation physique pour Robot Quadrupède V4 (Pymunk + Pygame).
    Morphologie :
      - 1 tronc / châssis rectangulaire horizontal (pas de tête)
      - 4 pattes (8 segments : 4 cuisses + 4 tibias)
      - Pas de pieds (extrémités sphériques en caoutchouc à haute friction)
      - Contrôle PD à 1 kHz / Action à 60 Hz
    """
    def __init__(self, render_mode=False, task="standing", max_steps=None, normalizer=None, update_normalizer=True, knee_backward=True, enable_obstacles=True):
        self.render_mode = render_mode
        self.task = task
        self.max_steps = max_steps
        self.current_step = 0
        self.normalizer = normalizer
        self.update_normalizer = update_normalizer
        self.knee_backward = bool(knee_backward)
        self.enable_obstacles = bool(enable_obstacles)

        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 650
        self.y_sol = 500.0

        # Contrôleur PD pour les 8 articulations (4 hanches + 4 genoux) amorti de manière critique
        self.kp_hip = 150000.0
        self.kd_hip = 5000.0
        self.kp_knee = 180000.0
        self.kd_knee = 6000.0
        self.max_torque = 90000.0

        self.action_dim = 8
        # State dim : orientation (3) + vel (2) + height (1) + 8 joint angles + 8 joint vel + 4 contacts + 8 CPG phases = 34
        self.state_dim = 34

        # CPG de Hopf avec allure de trot quadrupède
        self.cpg = HopfCPG(f_base=1.5, dt=1.0 / 60.0)
        self.use_cpg = True

        self.target_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

        self.target_vx = 120.0

        # Pymunk Space
        self.space = pymunk.Space()
        self.space.iterations = 35
        self.space.gravity = (0, 981)

        # Sol statique
        static_body = self.space.static_body
        floor = pymunk.Segment(static_body, (-1000000, self.y_sol), (1000000, self.y_sol), 5)
        floor.friction = 2.5
        self.floor = floor
        self.space.add(floor)

        # Boîtes d'obstacles statiques carrées fixées au sol tous les 2000 px (à 2000, 4000, etc.)
        self.obstacle_size = 120.0
        self.obstacle_spacing = 2000.0
        self.obstacle_positions = list(range(2000, 42000, 2000))
        self.obstacle_boxes = []
        if self.enable_obstacles:
            self._create_obstacle_boxes()

        # Rayons de vision (ligne de vue depuis la tête : 0°, -20°, +20°)
        self.vision_rays = []
        self.obstacle_detected = False
        self.vision_threshold = 150.0

        # Mode jumping (saut d'obstacle à 4 phases : CROUCH, THRUST, FLIGHT, LANDING)
        self.jump_step = 0
        self.jump_phase = "NONE"
        self.cleared_obstacles = set()
        self.jump_cooldown = 0

        # Gestion dynamique du terrain : 'flat' (sol plat) ou 'rough' (sol imparfait / accidenté)
        self.terrain_mode = "flat"
        self.rough_segments = []
        self.rough_profile_fn = None

        self.bodies = []
        self.joints = []
        self.torso = None
        self.thighs = [None] * 4
        self.shins = [None] * 4
        self.shin_tips = [None] * 4

        self.camera_x = 0.0
        self.pending_events = []
        self.last_perturbation = None
        self.perturbation_timer = 0
        self.last_fallen = False
        self.just_fallen = False
        self.last_timeout = False
        self.num_standing_perturbations = 10
        self.perturbation_magnitude = 2000.0 # Force de perturbation doublée (2000 N·s)
        self.perturbation_steps = set()
        self.overlay_info = None

        if self.render_mode:
            pygame.init()
            pygame.display.set_caption("Simulation Robot Quadrupède V4 (Go1 CPG-RL)")
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.font = None

    def reset(self, randomize=False):
        self.current_step = 0
        self.target_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_perturbation = None
        self.perturbation_timer = 0
        self.last_fallen = False
        self.just_fallen = False
        self.last_timeout = False
        self.cleared_obstacles = set()
        self.jump_cooldown = 0
        self.cpg.reset(initial_phase=0.0)

        # Planification de 10 perturbations régulières réparties sur l'épisode d'entraînement standing
        if self.task == "standing" and self.max_steps is not None:
            num_perts = getattr(self, "num_standing_perturbations", 10)
            self.perturbation_steps = set(
                int(self.max_steps * (k / (num_perts + 1))) for k in range(1, num_perts + 1)
            )
        else:
            self.perturbation_steps = set()

        if self.bodies:
            shapes = [s for b in self.bodies for s in b.shapes]
            self.space.remove(*self.joints, *shapes, *self.bodies)

        self.bodies.clear()
        self.joints.clear()

        # Position initiale
        if self.task == "jumping":
            # Placé devant la première boîte d'obstacle (boîte à 2000 px, front à 1940 px)
            px = float(np.random.uniform(1815.0, 1830.0)) if randomize else 1825.0
            self.jump_step = 0
            self.jump_phase = "CROUCH"
        else:
            px = 400.0
            self.jump_step = 0
            self.jump_phase = "NONE"

        standing_D = 50.0
        standing_h = standing_D + 8.5  # 58.5 px pour posture haute redressée
        py = float(self.get_terrain_height(px) - standing_h)

        # 1. Tronc / Châssis (proportions Spot de référence : ratio jambe/corps ~0.81, overhang avant/arrière)
        self.torso = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.torso.position = (px, py)
        self.torso.angle = float(np.random.uniform(-0.02, 0.02)) if randomize else 0.0

        # Châssis de 71 px de long (-33.5 à +37.5, ratio jambe/corps = 56/71 = 0.79 ~ 0.81)
        self.chassis_rear_x = -33.5
        self.chassis_front_x = 37.5
        chassis_seg = pymunk.Segment(self.torso, (self.chassis_rear_x, 0), (self.chassis_front_x, 0), 7)
        chassis_seg.mass = 3.2 if self.knee_backward else 4.0
        chassis_seg.color = (55, 60, 70, 255)
        chassis_seg.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.torso, chassis_seg)

        # Solution B : Pack batterie arrière physique et visuel (masse 0.8 kg) + recul du centre de gravité
        if self.knee_backward:
            battery_seg = pymunk.Segment(self.torso, (-27, -7), (-7, -7), 4)
            battery_seg.mass = 0.8
            battery_seg.color = (30, 35, 45, 255)
            battery_seg.filter = pymunk.ShapeFilter(group=1)
            self.space.add(battery_seg)
            self.torso.center_of_gravity = (-6.0, 0.0)
        else:
            self.torso.center_of_gravity = (0.0, 0.0)

        # Positions d'attache des hanches Spot : inter-hanches 54 px (-27 arrière, +27 avant)
        # Overhang avant : 10.5 px (19.4% de inter-hanches)
        # Overhang arrière : 6.5 px (11.7% de inter-hanches)
        self.hip_attachments = [(27.0, 0.0), (27.0, 0.0), (-27.0, 0.0), (-27.0, 0.0)]
        hip_attachments = self.hip_attachments
        self.thighs = [None] * 4
        self.shins = [None] * 4
        self.shin_tips = [None] * 4

        # Paires droite en rouge-corail, gauche en bleu-cyan
        leg_colors = [
            (235, 75, 75, 255),   # 0: FR (avant droit)
            (45, 125, 240, 255),  # 1: FL (avant gauche)
            (235, 75, 75, 255),   # 2: HR (arrière droit)
            (45, 125, 240, 255)   # 3: HL (arrière gauche)
        ]
        shin_colors = [
            (255, 130, 110, 255),
            (110, 185, 255, 255),
            (255, 130, 110, 255),
            (110, 185, 255, 255)
        ]

        # Posture debout conforme à Spot : H = 46.3 px (82.7% de H_max = 56 px), qh = +38°, qk = -68°
        init_targets = []
        for i in range(4):
            xt_init = -3.2 if self.knee_backward else 0.0
            standing_D = 46.3 if self.knee_backward else 50.0
            qh_i, qk_i = leg_ik(xt_init, standing_D, knee_backward=self.knee_backward)
            init_targets.extend([qh_i, qk_i])
        self.target_angles = np.array(init_targets, dtype=np.float32)

        for i in range(4):
            qh_init = init_targets[i * 2]
            qk_init = init_targets[i * 2 + 1]
            hip_local = hip_attachments[i]
            hip_world = self.torso.local_to_world(hip_local)

            # Cuisses (thigh)
            thigh = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            thigh.position = hip_world
            thigh.angle = self.torso.angle + qh_init
            th_seg = pymunk.Segment(thigh, (0, 0), (0, 28), 4.5)
            th_seg.mass = 0.5
            th_seg.friction = 1.0
            th_seg.color = leg_colors[i]
            th_seg.filter = pymunk.ShapeFilter(group=1)
            self.space.add(thigh, th_seg)
            self.thighs[i] = thigh

            # Pivot Hanche
            pj_hip = pymunk.PivotJoint(self.torso, thigh, hip_local, (0, 0))
            rj_hip = pymunk.RotaryLimitJoint(self.torso, thigh, math.radians(-80), math.radians(80))
            self.space.add(pj_hip, rj_hip)
            self.joints.extend([pj_hip, rj_hip])

            # Tibias (shin)
            knee_world = thigh.local_to_world((0, 28))
            shin = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
            shin.position = knee_world
            shin.angle = thigh.angle + qk_init
            sh_seg = pymunk.Segment(shin, (0, 0), (0, 28), 3.5)
            sh_seg.mass = 0.35
            sh_seg.friction = 1.0
            sh_seg.color = shin_colors[i]
            sh_seg.filter = pymunk.ShapeFilter(group=1)

            # Embout de patte sphérique (pattes sans pieds)
            tip_circle = pymunk.Circle(shin, 3.5, (0, 28))
            tip_circle.mass = 0.05
            tip_circle.friction = 2.0
            tip_circle.color = (30, 30, 30, 255)
            tip_circle.filter = pymunk.ShapeFilter(group=1)
            self.space.add(shin, sh_seg, tip_circle)
            self.shins[i] = shin
            self.shin_tips[i] = tip_circle

            # Pivot Genou
            pj_knee = pymunk.PivotJoint(thigh, shin, (0, 28), (0, 0))
            if self.knee_backward:
                rj_knee = pymunk.RotaryLimitJoint(thigh, shin, math.radians(-130), math.radians(10))
            else:
                rj_knee = pymunk.RotaryLimitJoint(thigh, shin, math.radians(-10), math.radians(110))
            self.space.add(pj_knee, rj_knee)
            self.joints.extend([pj_knee, rj_knee])

        self.bodies = [self.torso] + self.thighs + self.shins

        for b in self.bodies:
            b.velocity = (0.0, 0.0)
            b.angular_velocity = 0.0
            self.space.reindex_shapes_for_body(b)

        # Calcul initial des rayons de vision
        self.check_vision_rays(threshold=self.vision_threshold)

        return self._get_obs()

    def _sync_joints(self):
        """Maintient la cohérence géométrique rigide des articulations en sous-pas."""
        if self.torso is None:
            return
        hip_attachments = getattr(self, "hip_attachments", [(27.0, 0.0), (27.0, 0.0), (-27.0, 0.0), (-27.0, 0.0)])
        for i in range(4):
            hip_world = self.torso.local_to_world(hip_attachments[i])
            self.thighs[i].position = hip_world
            self.shins[i].position = self.thighs[i].local_to_world((0, 28))

    def _update_pd_torques(self):
        """Contrôleur PD d'actionnement à chaque sous-pas physique avec réaction Newtonienne équilibrée."""
        total_reaction = 0.0
        for i in range(4):
            thigh = self.thighs[i]
            shin = self.shins[i]

            # Hanche
            rel_h = thigh.angle - self.torso.angle
            rel_w_h = thigh.angular_velocity - self.torso.angular_velocity
            err_h = self.target_angles[i * 2] - rel_h
            torque_h = float(np.clip(self.kp_hip * err_h - self.kd_hip * rel_w_h, -self.max_torque, self.max_torque))

            # Genou avec Solution C : contrôle anti-plongée (anti-dive) asymétrique en appui stance
            rel_k = shin.angle - thigh.angle
            rel_w_k = shin.angular_velocity - thigh.angular_velocity
            err_k = self.target_angles[i * 2 + 1] - rel_k
            cur_kp_knee = self.kp_knee
            is_swing = (self.task == "walking" and self.use_cpg and self.cpg is not None and math.sin(self.cpg.theta[i]) > 0.0)
            if self.task == "walking" and self.knee_backward and i in (0, 1) and self.torso.angle > 0 and not is_swing:
                dive_factor = 1.0 + 0.8 * min(1.0, math.degrees(self.torso.angle) / 10.0)
                cur_kp_knee = min(320000.0, self.kp_knee * dive_factor)
            torque_k = float(np.clip(cur_kp_knee * err_k - self.kd_knee * rel_w_k, -self.max_torque, self.max_torque))

            shin.torque = torque_k
            thigh.torque = torque_h - torque_k
            total_reaction += torque_h

        # Réflexe postural stabilisateur d'assiette du châssis :
        # En saut (vol et réception) : contrôle actif haute autorité d'assiette gyroscopique
        if self.task == "jumping" and getattr(self, "jump_phase", "") in ("FLIGHT", "LANDING"):
            pitch_stiffness = 300000.0
            pitch_damping = 22000.0
            self.torso.torque = float(np.clip(-pitch_stiffness * self.torso.angle - pitch_damping * self.torso.angular_velocity, -500000.0, 500000.0))
        else:
            # Raideur élevée pour redresser le châssis après perturbation sans que la réaction des hanches ne plaque le nez au sol
            pitch_stiffness = 80000.0 if self.knee_backward else 50000.0
            pitch_damping = 4500.0 if self.knee_backward else 3000.0
            pitch_restoring = -pitch_stiffness * self.torso.angle - pitch_damping * self.torso.angular_velocity
            t_torso = pitch_restoring - 0.25 * total_reaction
            self.torso.torque = float(np.clip(t_torso, -self.max_torque, self.max_torque))

    def apply_body_perturbation(self, side="right", magnitude=2000.0):
        """
        Applique une impulsion horizontale calibrée sur une extrémité du corps :
          - 'right' : à l'extrémité droite (avant du châssis en +70, vers la droite)
          - 'left'  : à l'extrémité gauche (arrière du châssis en -70, vers la gauche)
        """
        if self.torso is None:
            return None

        if side == "right":
            point = (getattr(self, "chassis_front_x", 37.5), 0.0)
            impulse = (float(magnitude), 0.0)
        else:
            point = (getattr(self, "chassis_rear_x", -33.5), 0.0)
            impulse = (-float(magnitude), 0.0)

        self.torso.apply_impulse_at_local_point(impulse, point)
        self.last_perturbation = {
            "side": side,
            "point": point,
            "magnitude": magnitude,
            "step": self.current_step
        }
        self.perturbation_timer = 25
        return self.last_perturbation

    def set_task(self, task):
        """Transition fluide et standardisée entre modes standing, walking et jumping."""
        if task == self.task:
            if task == "jumping":
                self.jump_step = 0
                self.jump_phase = "CROUCH"
            return
        if self.task != "jumping":
            self.task_before_jump = self.task
        self.task = task
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)
        if task == "walking" and self.use_cpg and self.cpg is not None:
            # Réinitialisation propre de la phase nominale (phase 0) pour un décollage franc vers l'avant
            self.cpg.reset(initial_phase=0.0)
        elif task == "jumping":
            self.jump_step = 0
            self.jump_phase = "CROUCH"

    def get_terrain_height(self, x):
        """Retourne l'ordonnée Y de la surface du sol au point d'abscisse X."""
        if getattr(self, "terrain_mode", "flat") == "rough" and getattr(self, "rough_profile_fn", None) is not None:
            return float(self.rough_profile_fn(x))
        return float(self.y_sol)

    def get_surface_height(self, x):
        """Retourne l'ordonnée Y de la surface physique supérieure (sol ou sommet d'obstacle)."""
        gy = self.get_terrain_height(x)
        if getattr(self, "enable_obstacles", False):
            half = getattr(self, "obstacle_size", 120.0) / 2.0
            for bx in getattr(self, "obstacle_positions", []):
                if (bx - half) <= x <= (bx + half):
                    top_y = gy - getattr(self, "obstacle_size", 120.0)
                    return float(top_y)
        return float(gy)

    def set_terrain(self, mode="flat"):
        """Configure le profil du sol : 'flat' (sol plat nominal) ou 'rough' (sol imparfait / accidenté)."""
        if mode not in ("flat", "rough"):
            mode = "flat"
        self.terrain_mode = mode

        # Nettoyage des segments accidentés existants
        if hasattr(self, "rough_segments") and self.rough_segments:
            self.space.remove(*self.rough_segments)
            self.rough_segments.clear()

        # Retrait du sol plat si présent
        if hasattr(self, "floor") and self.floor in self.space.shapes:
            self.space.remove(self.floor)

        if mode == "flat":
            self.space.add(self.floor)
            self.rough_profile_fn = None
        else:
            anchor_x = self.torso.position.x if self.torso else 400.0
            static_body = self.space.static_body

            def profile_fn(x, ax=anchor_x, y_base=self.y_sol):
                d = abs(x - ax)
                # Zone de transition douce sous le robot (±45px) pour éviter tout décrochage brutal à l'activation
                ramp = min(1.0, max(0.0, (d - 45.0) / 40.0))
                # Relief multi-fréquence : ondulations, bosses et petits ressauts
                w1 = 6.0 * math.sin((x - ax) / 35.0)
                w2 = 4.5 * math.sin((x - ax) / 13.0 + 1.3)
                w3 = 2.0 * math.sin((x - ax) / 5.0 + 0.7)
                step = 4.0 * math.tanh(math.sin((x - ax) / 90.0) * 3.0)
                raw = 0.5 * w1 + 0.45 * w2 + 0.25 * w3 + 0.5 * step
                return y_base + ramp * float(np.clip(raw, -8.5, 8.5))

            self.rough_profile_fn = profile_fn

            # Maillage de segments physiques Pymunk de x=-3000 à x=+35000 px
            step_dx = 15.0
            cur_x = -3000.0
            cur_y = profile_fn(cur_x)
            max_x = 35000.0
            while cur_x < max_x:
                next_x = cur_x + step_dx
                next_y = profile_fn(next_x)
                seg = pymunk.Segment(static_body, (cur_x, cur_y), (next_x, next_y), 3.0)
                seg.friction = 2.5
                seg.color = (130, 95, 60, 255)
                self.rough_segments.append(seg)
                cur_x, cur_y = next_x, next_y

            self.space.add(*self.rough_segments)

        # Mise à jour de l'élévation des boîtes d'obstacles selon le relief du sol
        if hasattr(self, "obstacle_boxes") and self.obstacle_boxes:
            for body, shape, bx in self.obstacle_boxes:
                gy = self.get_terrain_height(bx)
                body.position = (float(bx), float(gy - (self.obstacle_size / 2.0)))
                self.space.reindex_shapes_for_body(body)

        return self.terrain_mode

    def _create_obstacle_boxes(self):
        """
        Crée les boîtes carrées de 150x150 px fixées au sol tous les 2000 px (à 2000, 4000, etc.).
        Chaque boîte est un corps statique Pymunk toute grise sans texture ni inscription.
        """
        for bx in self.obstacle_positions:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            gy = self.get_terrain_height(bx)
            body.position = (float(bx), float(gy - (self.obstacle_size / 2.0)))

            box_shape = pymunk.Poly.create_box(body, (self.obstacle_size, self.obstacle_size))
            box_shape.friction = 2.5
            box_shape.elasticity = 0.0
            box_shape.color = (135, 135, 135, 255)

            self.space.add(body, box_shape)
            self.obstacle_boxes.append((body, box_shape, float(bx)))

    def teleport_to(self, target_x):
        """
        Téléporte le robot à une position horizontale cible (ex: devant une boîte d'obstacle).
        Préserve l'intégrité articulaire des 4 pattes et annule les vitesses parasites.
        """
        if self.torso is None:
            return
        dx = float(target_x - self.torso.position.x)
        gy_target = self.get_terrain_height(target_x)
        gy_cur = self.get_terrain_height(self.torso.position.x)
        dy = float(gy_target - gy_cur)

        for b in self.bodies:
            b.position = (b.position.x + dx, b.position.y + dy)
            b.velocity = (0.0, 0.0)
            b.angular_velocity = 0.0
            self.space.reindex_shapes_for_body(b)

        self.torso.angle = 0.0
        self._sync_joints()
        self.check_vision_rays(threshold=self.vision_threshold)

    def check_vision_rays(self, threshold=150.0):
        """
        Projette 3 rayons semi-transparents rouges de vision (ligne de vue depuis la tête) :
          - 0° selon x (axe horizontal vers l'avant)
          - -20° (orienté vers le haut)
          - +20° (orienté vers le bas)

        Règles :
          - Les rayons n'ignorent pas le sol : le sol (plat ou accidenté) et les boîtes sont des objets détectables.
          - Les 3 rayons sont visibles UNIQUEMENT quand tous les 3 touchent un objet.
          - Quand les 3 détectent un objet à moins de threshold (150 px), le robot passe en mode standing.
        """
        if self.torso is None:
            return False, []

        local_head = (getattr(self, "chassis_front_x", 37.5) + 7.0, 0.0)
        head_world = self.torso.local_to_world(local_head)

        max_range = 250.0
        ray_results = []
        # Stabilisation de la ligne de vue vers l'avant même en cas de tangage ou rebond
        torso_angle = float(np.clip(self.torso.angle, -math.radians(35.0), math.radians(35.0)))

        for deg in [0.0, -20.0, 20.0]:
            rad = torso_angle + math.radians(deg)
            dir_vec = pymunk.Vec2d(math.cos(rad), math.sin(rad))
            start_pt = head_world
            end_pt = start_pt + dir_vec * max_range

            # Ignorer uniquement le corps du robot (group 1) : le sol et les boîtes sont pris en compte
            q = self.space.segment_query_first(start_pt, end_pt, 1.0, pymunk.ShapeFilter(group=1))

            hit_shape = q.shape if q else None
            is_obstacle = False
            if q:
                hit_pt = q.point
                dist = float((hit_pt - start_pt).length)
                touched = True
                # L'objet touché est un obstacle s'il s'agit d'une boîte en amont non encore franchie
                if q.shape != getattr(self, "floor", None) and q.shape not in getattr(self, "rough_segments", []):
                    for _, box_s, bx in getattr(self, "obstacle_boxes", []):
                        if q.shape == box_s:
                            if bx > (self.torso.position.x - 20.0) and bx not in getattr(self, "cleared_obstacles", set()):
                                is_obstacle = True
                            break
                    else:
                        if hit_pt.x > self.torso.position.x:
                            is_obstacle = True
            else:
                hit_pt = end_pt
                dist = float(max_range)
                touched = False

            detected = bool(touched and dist < threshold and is_obstacle)
            ray_results.append({
                "deg": deg,
                "start": start_pt,
                "hit": hit_pt,
                "dist": dist,
                "touched": touched,
                "detected": detected,
                "is_obstacle": is_obstacle,
                "shape": hit_shape
            })

        # Les 3 rayons touchent-ils un objet ?
        all_touching = bool(len(ray_results) == 3 and all(r["touched"] for r in ray_results))
        # Les 3 rayons détectent-ils à moins de 150 px avec présence d'un obstacle ?
        has_obstacle = any(r["is_obstacle"] for r in ray_results)
        all_detected = bool(all_touching and all(r["detected"] for r in ray_results) and has_obstacle)

        self.vision_rays = ray_results
        self.rays_visible = all_touching
        self.obstacle_detected = all_detected
        return all_detected, ray_results

    def toggle_terrain(self):
        """Bascule entre sol plat et sol imparfait."""
        new_mode = "rough" if getattr(self, "terrain_mode", "flat") == "flat" else "flat"
        return self.set_terrain(new_mode)

    def _get_jumping_targets(self, action):
        """
        Générateur de trajectoire de saut d'obstacle à haute puissance et 4 phases physiques modulé par l'action RL :
          1. CROUCH  (steps 0-14)   : compression squat d'armement et ancrage au sol (H ~ 24 px)
          2. THRUST  (steps 15-26)  : décharge de puissance crête explosive (900k N·px) projetant le robot en avant et haut
          3. FLIGHT  (steps 27-74)  : rétraction tuck complète sous le châssis pour franchir la boîte (apogée > 300 px)
          4. LANDING (steps 75-114) : déploiement amorti et réception stable
          5. Terminé (steps >= 115) : transition automatique en mode STANDING
        """
        self.jump_step += 1
        act = np.asarray(action, dtype=np.float32).flatten()
        if len(act) < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[:len(act)] = act
            act = padded

        # 1. Phase CROUCH (steps 0-14) : armement, freinage d'assiette et compression
        if self.jump_step < 15:
            self.jump_phase = "CROUCH"
            self.kp_hip = 250000.0
            self.kp_knee = 300000.0
            self.max_torque = 250000.0
            crouch_targets = []
            for i in range(4):
                mod_x = act[i * 2] * 3.0
                mod_z = act[i * 2 + 1] * 2.5
                qh, qk = leg_ik(-2.0 + mod_x, 24.0 + mod_z, knee_backward=self.knee_backward)
                crouch_targets.extend([qh, qk])
            return np.array(crouch_targets, dtype=np.float32)

        # 2. Phase THRUST (steps 15-26) : impulsion crête explosive haute puissance avec propulsion vers l'avant
        elif self.jump_step < 27:
            self.jump_phase = "THRUST"
            # Décharge de puissance crête (Peak power burst : 950 000 N·px, Kp genoux 1 150 000)
            self.kp_hip = 950000.0
            self.kp_knee = 1150000.0
            self.max_torque = 950000.0
            thrust_targets = [
                math.radians(65.0) + act[0] * math.radians(5.0), math.radians(-16.0) + act[1] * math.radians(5.0),
                math.radians(65.0) + act[2] * math.radians(5.0), math.radians(-16.0) + act[3] * math.radians(5.0),
                math.radians(92.0) + act[4] * math.radians(5.0), math.radians(-16.0) + act[5] * math.radians(5.0),
                math.radians(92.0) + act[6] * math.radians(5.0), math.radians(-16.0) + act[7] * math.radians(5.0),
            ]
            return np.array(thrust_targets, dtype=np.float32)

        # 3. Phase FLIGHT / TUCK (steps 27-74) : rétraction compacte des 4 pattes sous le tronc
        elif self.jump_step < 75:
            self.jump_phase = "FLIGHT"
            self.kp_hip = 250000.0
            self.kp_knee = 300000.0
            self.max_torque = 250000.0
            # Repliage serré sous le tronc pour franchir les 120 px de la boîte avec marge (> 300 px de hauteur)
            tuck_targets = [
                math.radians(35.0) + act[0] * math.radians(5.0), math.radians(-120.0) + act[1] * math.radians(5.0),
                math.radians(35.0) + act[2] * math.radians(5.0), math.radians(-120.0) + act[3] * math.radians(5.0),
                math.radians(45.0) + act[4] * math.radians(5.0), math.radians(-120.0) + act[5] * math.radians(5.0),
                math.radians(45.0) + act[6] * math.radians(5.0), math.radians(-120.0) + act[7] * math.radians(5.0),
            ]
            return np.array(tuck_targets, dtype=np.float32)

        # 4. Phase LANDING (steps 75-114) : déploiement et amortissement progressif
        elif self.jump_step < 115:
            self.jump_phase = "LANDING"
            self.kp_hip = 220000.0
            self.kp_knee = 260000.0
            self.max_torque = 180000.0
            land_targets = [
                math.radians(38.0) + act[0] * math.radians(6.0), math.radians(-68.0) + act[1] * math.radians(6.0),
                math.radians(38.0) + act[2] * math.radians(6.0), math.radians(-68.0) + act[3] * math.radians(6.0),
                math.radians(38.0) + act[4] * math.radians(6.0), math.radians(-68.0) + act[5] * math.radians(6.0),
                math.radians(38.0) + act[6] * math.radians(6.0), math.radians(-68.0) + act[7] * math.radians(6.0),
            ]
            return np.array(land_targets, dtype=np.float32)

        # 5. Terminé (step >= 115) -> retour automatique au mode walking (ou tâche précédant le saut)
        else:
            self.jump_phase = "DONE"
            self.jump_cooldown = 90
            target_task = getattr(self, "post_jump_task", getattr(self, "task_before_jump", "walking"))
            self.set_task(target_task)
            return self.target_angles.copy()

    def _apply_action(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        if len(act) < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[:len(act)] = act
            act = padded

        # Filtrage passe-bas du 1er ordre (évite les secousses / tremblements haute fréquence)
        alpha = 0.35
        self.current_action = (1.0 - alpha) * self.prev_action + alpha * act[:self.action_dim]
        self.prev_action = self.current_action.copy()

        if self.task == "walking" and self.use_cpg and self.cpg is not None:
            # Modulation DRL des paramètres intrinsèques du CPG
            self.cpg.set_modulations(self.current_action)
            raw_targets = self.cpg.get_joint_targets(knee_backward=self.knee_backward)
        elif self.task == "jumping":
            # Mode jumping : trajectoire à 4 phases modulée par RL
            raw_targets = self._get_jumping_targets(self.current_action)
        else:
            # Mode standing : posture d'équilibre redressée conforme à Spot (qh=+38°, qk=-68°, H=46.3 px)
            vx = self.torso.velocity.x if self.torso else 0.0
            brake_offset = float(np.clip(-0.0035 * vx, -math.radians(16.0), math.radians(16.0)))
            targets = []
            for i in range(4):
                xt = -3.2 if self.knee_backward else 0.0
                standing_D = 46.3 if self.knee_backward else 50.0
                qh_base, qk_base = leg_ik(xt, standing_D, knee_backward=self.knee_backward)
                th_adj = self.current_action[i * 2] * math.radians(24.0) + brake_offset
                k_adj = self.current_action[i * 2 + 1] * math.radians(26.0)
                targets.extend([qh_base + th_adj, qk_base + k_adj])
            raw_targets = np.array(targets, dtype=np.float32)

        # Limiteur de vitesse angulaire articulaire (slew-rate limiter) :
        # Libère la dynamique explosive (50°/pas) lors de la phase THRUST pour une détente maximale
        if self.task == "jumping" and getattr(self, "jump_phase", "") == "THRUST":
            max_d = math.radians(50.0)
        elif self.task in ("walking", "jumping"):
            max_d = math.radians(20.0)
        else:
            max_d = math.radians(10.0)

        delta = np.clip(raw_targets - self.target_angles, -max_d, max_d)
        self.target_angles += delta

    def step(self, action):
        self.current_step += 1

        if self.perturbation_timer > 0:
            self.perturbation_timer -= 1

        if getattr(self, "jump_cooldown", 0) > 0:
            self.jump_cooldown -= 1

        # Application automatique des 10 perturbations réparties pendant l'entraînement standing (force doublée à 2000 N·s)
        if self.task == "standing" and self.current_step in self.perturbation_steps:
            side = np.random.choice(["right", "left"])
            mag = getattr(self, "perturbation_magnitude", 2000.0)
            self.apply_body_perturbation(side=side, magnitude=mag)

        if self.task == "walking" and self.use_cpg:
            self.cpg.step()

        self._apply_action(action)

        # 4 sous-pas d'intégration physique Pymunk (dt = (1/60s) / 4 = 1/240s)
        sub_dt = (1.0 / 60.0) / 4.0
        for _ in range(4):
            self._update_pd_torques()
            self.space.step(sub_dt)

        obs = self._get_obs()

        # Détection d'obstacle par les 3 rayons de vision (ligne de vue) : passage en jumping si distance < 150 px
        all_detected, _ = self.check_vision_rays(threshold=self.vision_threshold)
        if all_detected and self.task == "walking" and getattr(self, "jump_cooldown", 0) == 0:
            # Mémoriser l'obstacle en cours pour éviter tout re-déclenchement intempestif
            upcoming = [bx for bx in getattr(self, "obstacle_positions", []) if bx > (self.torso.position.x - 30.0)]
            if upcoming:
                self.cleared_obstacles.add(min(upcoming))
            self.post_jump_task = "walking"
            self.set_task("jumping")

        is_currently_fallen = self._is_fallen()
        self.just_fallen = is_currently_fallen and not getattr(self, "last_fallen", False)
        self.last_fallen = is_currently_fallen
        self.last_timeout = (self.max_steps is not None and self.current_step >= self.max_steps)

        if self.task == "walking":
            reward = self._reward_walking(action)
        elif self.task == "jumping":
            reward = self._reward_jumping(action)
        else:
            reward = self._reward_standing(action)

        # En standing : ne pas réinitialiser si le robot tombe afin qu'il s'entraîne à se relever
        if self.task == "standing":
            done = self.last_timeout
        elif self.task == "jumping":
            done = self.last_fallen or self.last_timeout or (self.jump_phase == "DONE")
        else:
            done = self.last_fallen or self.last_timeout

        return obs, reward, done

    def _is_fallen(self):
        if self.torso is None:
            return False
        ground_y = self.get_surface_height(self.torso.position.x)
        hauteur = ground_y - self.torso.position.y
        head_angle = self.torso.angle

        # En mode jumping : le robot ne doit JAMAIS être considéré tombé pendant le squat préparatoire (CROUCH),
        # l'impulsion explosive (THRUST) ou la phase aérienne de franchissement (FLIGHT).
        if self.task == "jumping":
            if getattr(self, "jump_phase", "") in ("CROUCH", "THRUST", "FLIGHT"):
                return False
            # En phase LANDING : tombé uniquement si le châssis s'écrase retourné au sol
            return (hauteur < 18.0) and (abs(head_angle) > math.radians(65.0))

        # En mode nominal (standing / walking) : tombé si le châssis touche presque le sol ou bascule
        return (hauteur < 22.0) or (abs(head_angle) > math.radians(45.0))

    def _get_obs(self):
        torso = self.torso
        ground_y = self.get_terrain_height(torso.position.x)
        hauteur = (ground_y - torso.position.y) / 100.0
        sin_torso = math.sin(torso.angle)
        cos_torso = math.cos(torso.angle)
        w_torso = torso.angular_velocity / 10.0
        vx = torso.velocity.x / 100.0
        vy = torso.velocity.y / 100.0

        joint_angles = []
        joint_vels = []
        contacts = []

        for i in range(4):
            th = self.thighs[i]
            sh = self.shins[i]
            rel_h = th.angle - torso.angle
            rel_k = sh.angle - th.angle
            w_h = (th.angular_velocity - torso.angular_velocity) / 10.0
            w_k = (sh.angular_velocity - th.angular_velocity) / 10.0

            joint_angles.extend([rel_h, rel_k])
            joint_vels.extend([w_h, w_k])

            tip_pt = sh.local_to_world((0, 28))
            ground_at_tip = self.get_terrain_height(tip_pt.x)
            is_contact = 1.0 if tip_pt.y >= (ground_at_tip - 11.0) else 0.0
            contacts.append(is_contact)

        cpg_phases = self.cpg.get_phase_features()

        raw_obs = np.array([
            hauteur, sin_torso, cos_torso, w_torso, vx, vy,
            *joint_angles,
            *joint_vels,
            *contacts,
            *cpg_phases
        ], dtype=np.float32)

        if self.normalizer is not None:
            if self.update_normalizer:
                self.normalizer.update(raw_obs)
            return self.normalizer.normalize(raw_obs)

        return raw_obs

    def _reward_standing(self, action):
        ground_y = self.get_terrain_height(self.torso.position.x)
        hauteur = max(0.0, ground_y - self.torso.position.y)
        torso_angle = self.torso.angle
        vx = self.torso.velocity.x

        # Si le robot est tombé (hauteur < 22px ou bascule > 45°) : forte pénalité avec gradient continu de repoussée
        if self._is_fallen():
            base_fall_penalty = -5.0
            # Incitation continue et forte à repousser le corps vers le haut
            push_up_bonus = 3.0 * min(1.0, hauteur / 22.0)
            upright_bonus = 0.5 * math.exp(-(torso_angle ** 2) / (math.radians(30.0) ** 2))
            effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))
            return float(base_fall_penalty + push_up_bonus + upright_bonus - effort_penalty)

        # Maintien strict et prépondérant de la hauteur nominale de base Spot (target_h = 55.0 px)
        target_h = 55.0
        # Double composante :
        # 1. Composante large (Laplace) assurant un gradient positif non nul même en flexion profonde (25 à 45 px)
        score_broad = math.exp(-abs(target_h - hauteur) / 10.0)
        # 2. Composante étroite pour verrouiller la précision millimétrique autour de 55 px
        score_narrow = math.exp(-((target_h - hauteur) ** 2) / 16.0) # sigma = 4.0 px
        height_score = 0.35 * score_broad + 0.65 * score_narrow

        # La hauteur est très largement valorisée (poids prépondérant 6.0)
        height_reward = 6.0 * height_score

        # Pénalité explicite et proportionnelle à tout écart de hauteur (|target_h - hauteur|)
        height_error_penalty = 2.0 * (abs(target_h - hauteur) / target_h)

        # Les autres récompenses sont conditionnées à la bonne tenue de la hauteur :
        # L'alive bonus n'est pas "gratuit" : il n'est perçu que si le robot est debout à bonne hauteur
        alive_bonus = 1.0 * height_score

        # Horizontalité du châssis conditionnée à la hauteur
        sigma_angle = math.radians(12.0)
        chassis_score = math.exp(-(torso_angle ** 2) / (sigma_angle ** 2))
        chassis_reward = 1.5 * chassis_score * height_score

        # Contact des pattes au sol conditionné à la hauteur
        contact_count = sum(
            1.0 for i in range(4)
            if self.shins[i].local_to_world((0, 28)).y >= (self.get_terrain_height(self.shins[i].local_to_world((0, 28)).x) - 11.0)
        )
        contact_bonus = 0.5 * min(1.0, contact_count / 2.0) * height_score

        # Pénalité de dérive bornée (incite à revenir à Vx=0 sans exploser lors d'un choc ponctuel)
        drift_penalty = 0.6 * (1.0 - math.exp(-((vx / 35.0) ** 2)))
        action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
        smoothness_penalty = 0.02 * action_diff
        effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))

        reward = (
            alive_bonus
            + height_reward
            + chassis_reward
            + contact_bonus
            - height_error_penalty
            - drift_penalty
            - smoothness_penalty
            - effort_penalty
        )
        return float(reward)

    def _reward_jumping(self, action):
        """
        Fonction de récompense pour le saut et franchissement d'obstacle (boîte de 120 px) :
          - Élévation du centre de gravité (apogée au-dessus de y=380 px)
          - Progression horizontale vers l'avant (Vx > 0)
          - Maintien du châssis horizontal (stabilité de tangage en vol)
          - Réception stable sur les 4 pattes sans basculer
        """
        if self.torso is None:
            return 0.0

        torso_y = self.torso.position.y
        torso_x = self.torso.position.x
        vx = self.torso.velocity.x
        pitch = self.torso.angle
        w = self.torso.angular_velocity

        # 1. Élévation au-dessus du sol (sommet de la boîte à y=380 px, sol à 500 px)
        height_above_ground = max(0.0, self.y_sol - torso_y)
        elevation_reward = 3.5 * max(0.0, (height_above_ground - 60.0) / 40.0)
        # Bonus d'apogée si le châssis dépasse le sommet de la boîte
        if torso_y < 380.0:
            elevation_reward += 3.0

        # 2. Progression horizontale vers l'avant
        speed_reward = 0.02 * max(-30.0, min(180.0, vx))

        # 3. Bonus de franchissement ou réception sur la boîte (x in [1940, 2060])
        box_bonus = 0.0
        if torso_x >= 1940.0 and torso_y <= 385.0:
            box_bonus = 4.0

        # 4. Pénalité de tangage (assiette stable en vol)
        pitch_penalty = 2.0 * (pitch ** 2) + 0.2 * ((w / 10.0) ** 2)

        # 5. Pénalité de chute
        fall_penalty = 8.0 if self.just_fallen else (2.0 if self.last_fallen else 0.0)

        # 6. Lissage et effort
        smooth_penalty = 0.01 * float(np.sum(np.square(self.current_action - self.prev_action)))
        effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))

        reward = elevation_reward + speed_reward + box_bonus - pitch_penalty - fall_penalty - smooth_penalty - effort_penalty
        return float(reward)

    def _reward_walking(self, action):
        if self._is_fallen():
            return -2.0

        vx = self.torso.velocity.x
        vy = self.torso.velocity.y
        ground_y = self.get_terrain_height(self.torso.position.x)
        hauteur = max(0.0, ground_y - self.torso.position.y)
        torso_angle = self.torso.angle
        w_torso = self.torso.angular_velocity

        # 1. Vitesse longitudinale par noyau exponentiel décroissant (Table I de l'article)
        target_vx = getattr(self, "target_vx", 120.0)
        sigma_vx = 80.0
        val = math.exp(-((target_vx - vx) ** 2) / (sigma_vx ** 2))
        val_zero = math.exp(-(target_vx ** 2) / (sigma_vx ** 2))
        speed_reward = 3.5 * (val - val_zero) / (1.0 - val_zero)
        if vx < 0:
            speed_reward += 1.0 * (vx / 40.0)

        # 2. Maintien de hauteur Spot (Table I)
        target_h = 54.0
        sigma_h = 6.0
        height_reward = 1.5 * math.exp(-((target_h - hauteur) ** 2) / (sigma_h ** 2))

        # 3. Maintien de l'horizontalité du châssis (Table I)
        sigma_angle = math.radians(15.0)
        chassis_reward = 0.8 * math.exp(-(torso_angle ** 2) / (sigma_angle ** 2))

        # 4. Pénalités de vitesse verticale et de rotation (Table I)
        penalty_vy = 0.2 * ((vy / 100.0) ** 2)
        penalty_w = 0.1 * ((w_torso / 10.0) ** 2)

        # 5. Coordination du trot : alternance des paires diagonales (FR+HL vs FL+HR)
        c_list = [
            1.0 if self.shins[i].local_to_world((0, 28)).y >= (self.get_terrain_height(self.shins[i].local_to_world((0, 28)).x) - 11.0) else 0.0
            for i in range(4)
        ]
        c0, c1, c2, c3 = c_list
        pair1_contact = c0 * c3
        pair2_contact = c1 * c2
        trot_coordination = 0.4 * (pair1_contact + pair2_contact - 2.0 * pair1_contact * pair2_contact)

        # 6. Pénalité de trébuchement (frottement / accrochage au sol pendant le swing)
        swing_drag_penalty = 0.0
        if self.use_cpg and self.cpg is not None:
            for i in range(4):
                if math.sin(self.cpg.theta[i]) > 0.15:
                    tip_pt = self.shins[i].local_to_world((0, 28))
                    if tip_pt.y >= (self.get_terrain_height(tip_pt.x) - 11.0):
                        swing_drag_penalty += 0.25

        # 7. Pénalités de puissance et de lissage
        action_diff = float(np.sum(np.square(self.current_action - self.prev_action)))
        smoothness_penalty = 0.02 * action_diff
        effort_penalty = 0.005 * float(np.sum(np.square(self.current_action)))

        reward = (
            speed_reward
            + height_reward
            + chassis_reward
            + trot_coordination
            - penalty_vy
            - penalty_w
            - swing_drag_penalty
            - smoothness_penalty
            - effort_penalty
        )
        return float(reward)

    def set_overlay(self, overlay_info):
        self.overlay_info = overlay_info

    def get_events(self):
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def render(self, overlay_info=None):
        if not self.render_mode:
            return []

        if overlay_info is not None:
            self.overlay_info = overlay_info

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.close()
                exit()
            self.pending_events.append(event)

        if self.torso is not None:
            robot_x = self.torso.position.x
            self.camera_x = robot_x - (self.SCREEN_WIDTH / 2.0)
        else:
            self.camera_x = 0.0

        self.draw_options.transform = pymunk.Transform.translation(-self.camera_x, 0)
        self.screen.fill((255, 255, 255))

        # Rendu du sol (Plat ou Imparfait / Accidenté)
        if getattr(self, "terrain_mode", "flat") == "rough" and getattr(self, "rough_profile_fn", None) is not None:
            # Sol imparfait : contour ondulé et relief rocheux/terreux
            step_px = 6
            screen_pts = []
            for sx in range(0, self.SCREEN_WIDTH + step_px, step_px):
                wx = sx + self.camera_x
                wy = self.get_terrain_height(wx)
                screen_pts.append((sx, int(wy)))

            # Remplissage sous la surface du sol
            poly_pts = [(0, self.SCREEN_HEIGHT)] + screen_pts + [(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)]
            pygame.draw.polygon(self.screen, (242, 236, 228), poly_pts)

            # Ligne de surface texturée
            if len(screen_pts) > 1:
                pygame.draw.lines(self.screen, (125, 90, 60), False, screen_pts, 3)

            # Repères périodiques tous les 100 px calés sur le relief
            cam_start = int(self.camera_x // 100) * 100
            for gx in range(cam_start - 200, cam_start + self.SCREEN_WIDTH + 300, 100):
                sx = int(gx - self.camera_x)
                if 0 <= sx <= self.SCREEN_WIDTH:
                    sy = int(self.get_terrain_height(gx))
                    pygame.draw.circle(self.screen, (140, 100, 65), (sx, sy), 4)
                    pygame.draw.line(self.screen, (180, 150, 120), (sx, sy), (sx, sy + 16), 1)
        else:
            # Sol plat nominal
            cam_start = int(self.camera_x // 100) * 100
            for gx in range(cam_start - 200, cam_start + self.SCREEN_WIDTH + 300, 100):
                sx = int(gx - self.camera_x)
                pygame.draw.circle(self.screen, (215, 220, 230), (sx, int(self.y_sol)), 4)
                pygame.draw.line(self.screen, (230, 235, 240), (sx, int(self.y_sol)), (sx, int(self.y_sol) + 20), 1)

            # Ligne de sol plat
            pygame.draw.line(self.screen, (100, 110, 120), (0, int(self.y_sol)), (self.SCREEN_WIDTH, int(self.y_sol)), 3)

        # Rendu physique Pymunk
        self.space.debug_draw(self.draw_options)

        # Rendu des boîtes d'obstacles carrées (150x150 px, unies toutes grises sans texture ni inscription)
        if getattr(self, "enable_obstacles", False) and hasattr(self, "obstacle_boxes"):
            for body, shape, bx in self.obstacle_boxes:
                screen_x = body.position.x - self.camera_x
                if -200 <= screen_x <= self.SCREEN_WIDTH + 200:
                    half = self.obstacle_size / 2.0
                    rx = int(screen_x - half)
                    ry = int(body.position.y - half)
                    rw = int(self.obstacle_size)
                    rh = int(self.obstacle_size)

                    # Boîte toute grise unie, sans texture ni inscription
                    pygame.draw.rect(self.screen, (135, 135, 135), (rx, ry, rw, rh))
                    pygame.draw.rect(self.screen, (105, 105, 105), (rx, ry, rw, rh), 1)

        # Rendu des 3 rayons semi-transparents rouges de vision (ligne de vue depuis la tête : 0°, -20°, +20°)
        # Visibles UNIQUEMENT quand tous les 3 touchent un objet
        if getattr(self, "rays_visible", False) and hasattr(self, "vision_rays") and self.vision_rays:
            ray_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            all_det = getattr(self, "obstacle_detected", False)
            for ray in self.vision_rays:
                s_world = ray["start"]
                h_world = ray["hit"]
                s_screen = (int(s_world.x - self.camera_x), int(s_world.y))
                h_screen = (int(h_world.x - self.camera_x), int(h_world.y))

                # Couleur du rayon : semi-transparent rouge
                if all_det:
                    halo_col = (255, 30, 30, 95)
                    core_col = (255, 50, 50, 225)
                else:
                    halo_col = (255, 45, 45, 50)
                    core_col = (255, 50, 50, 150)

                pygame.draw.line(ray_surf, halo_col, s_screen, h_screen, 4)
                pygame.draw.line(ray_surf, core_col, s_screen, h_screen, 2)

                # Point d'impact du rayon sur l'objet touché (sol ou boîte)
                dot_color = (255, 20, 20, 240) if all_det else (255, 60, 60, 200)
                pygame.draw.circle(ray_surf, dot_color, h_screen, 4)
                pygame.draw.circle(ray_surf, (255, 255, 255, 230), h_screen, 2)

                # Point d'émission sur la tête
                pygame.draw.circle(ray_surf, (255, 40, 40, 220), s_screen, 3)

            self.screen.blit(ray_surf, (0, 0))

        # Flèche de perturbation visuelle
        if self.perturbation_timer > 0 and self.last_perturbation and self.torso:
            p_info = self.last_perturbation
            side = p_info["side"]
            local_pt = p_info["point"]
            world_pt = self.torso.local_to_world(local_pt)
            screen_x = int(world_pt.x - self.camera_x)
            screen_y = int(world_pt.y)

            arrow_dir = +1 if side == "right" else -1
            arrow_len = 55
            arrow_start_x = screen_x - arrow_dir * arrow_len

            pygame.draw.line(self.screen, (255, 50, 20), (arrow_start_x, screen_y), (screen_x, screen_y), 5)
            tip_p1 = (screen_x - arrow_dir * 14, screen_y - 8)
            tip_p2 = (screen_x - arrow_dir * 14, screen_y + 8)
            pygame.draw.polygon(self.screen, (255, 50, 20), [(screen_x, screen_y), tip_p1, tip_p2])
            pygame.draw.circle(self.screen, (255, 140, 0), (screen_x, screen_y), 12, 2)

        # Affichage texte overlay
        if self.overlay_info:
            if self.font is None:
                self.font = pygame.font.SysFont("monospace", 14, bold=True)
            for idx, text in enumerate(self.overlay_info):
                surf = self.font.render(text, True, (25, 30, 40))
                bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4))
                bg.fill((245, 248, 252))
                bg.set_alpha(225)
                self.screen.blit(bg, (15, 12 + idx * 24))
                self.screen.blit(surf, (20, 14 + idx * 24))

        pygame.display.flip()
        self.clock.tick(60)
        return events

    def close(self):
        if self.render_mode:
            pygame.quit()


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


def load_policy_auto(model_path, state_dim=34, action_dim=8, device="cpu"):
    """Charge un modèle PyTorch pour le robot quadrupède et attache son Normalizer Welford s'il existe."""
    import torch
    from Models import ActorCritic

    loaded = torch.load(model_path, map_location=device)
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded

    policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)

    # Chargement tolérant des poids
    if "lstm.weight_ih_l0" in state_dict:
        loaded_state_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        loaded_action_dim = state_dict["actor_mean.4.weight"].shape[0] if "actor_mean.4.weight" in state_dict else action_dim
        if loaded_state_dim == state_dim and loaded_action_dim == action_dim:
            policy.load_state_dict(state_dict)
        else:
            curr_dict = policy.state_dict()
            for name, param in curr_dict.items():
                if name in state_dict and state_dict[name].shape == param.shape:
                    param.copy_(state_dict[name])
            policy.load_state_dict(curr_dict)
    else:
        curr_dict = policy.state_dict()
        for name, param in curr_dict.items():
            if name in state_dict and state_dict[name].shape == param.shape:
                param.copy_(state_dict[name])
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

