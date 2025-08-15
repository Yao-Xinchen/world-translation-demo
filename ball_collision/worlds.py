import genesis as gs
import torch
from genesis.engine.entities import RigidEntity


class BaseWorld:
    def __init__(self, headless=False, gravity=0.0, friction=0.1, n_envs=1, rand=True):
        gs.init(backend=gs.gpu)

        self.n_envs = n_envs

        # Configure scene based on visualization needs
        show_vis = not headless and self.n_envs == 1

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -gravity),
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=False,
                max_collision_pairs=1000,
                use_gjk_collision=True,
                enable_mujoco_compatibility=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -5.5, 2.5),
                camera_lookat=(0, 0.0, 1.5),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=show_vis,
        )

        init_pos = (0.0, 0.0, 1.0)
        self.init_pos = torch.tensor(init_pos).unsqueeze(0).repeat(self.n_envs, 1)

        self.ball: RigidEntity = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=0.1,
                pos=init_pos,
            ),
            material=gs.materials.rigid.Rigid(
                rho=500.0,
                friction=friction,
                coup_friction=friction,
            )
        )

        box_size = 2.0
        wall_thickness = 0.01

        wall_material = gs.materials.Rigid(
            rho=1000.0,
            friction=friction,
            coup_friction=friction,
        )
        wall_surface = gs.surfaces.Glass(color=(1.0, 1.0, 1.0), opacity=0.1)

        bottom_wall = self.scene.add_entity(
            morph=gs.morphs.Box(size=(box_size, box_size, wall_thickness), fixed=True, pos=(0.0, 0.0, 0.0)),
            material=wall_material,
            surface=wall_surface
        )

        left_wall = self.scene.add_entity(
            morph=gs.morphs.Box(size=(wall_thickness, box_size, box_size), fixed=True,
                                pos=(-box_size / 2, 0.0, box_size / 2)),
            material=wall_material,
            surface=wall_surface
        )

        right_wall = self.scene.add_entity(
            morph=gs.morphs.Box(size=(wall_thickness, box_size, box_size), fixed=True,
                                pos=(box_size / 2, 0.0, box_size / 2)),
            material=wall_material,
            surface=wall_surface
        )

        front_wall = self.scene.add_entity(
            morph=gs.morphs.Box(size=(box_size, wall_thickness, box_size), fixed=True,
                                pos=(0.0, -box_size / 2, box_size / 2)),
            material=wall_material,
            surface=wall_surface
        )

        back_wall = self.scene.add_entity(
            morph=gs.morphs.Box(size=(box_size, wall_thickness, box_size), fixed=True,
                                pos=(0.0, box_size / 2, box_size / 2)),
            material=wall_material,
            surface=wall_surface
        )

        top_cover = self.scene.add_entity(
            morph=gs.morphs.Box(size=(box_size, box_size, wall_thickness), fixed=True, pos=(0.0, 0.0, box_size), ),
            material=gs.materials.Rigid(
                rho=1000.0,
                friction=friction,
            ),
            surface=wall_surface
        )

        self.scene.build(n_envs=self.n_envs)

        self.rand = rand

        # Initialize velocities for each environment
        if self.rand:
            random_directions = torch.randn(self.n_envs, 3)
            random_directions = random_directions / torch.norm(random_directions, dim=1, keepdim=True)
            random_speeds = torch.rand(self.n_envs, 1) * 10 + 5  # uniform between 5 and 15
            batch_velocities = random_directions * random_speeds
        else:
            batch_velocities = torch.tensor([8.0, 0.0, 0.0]).unsqueeze(0).repeat(self.n_envs, 1)

        # Print velocities for each environment
        for i in range(self.n_envs):
            print(
                f"Env {i} ball initial velocity: [{batch_velocities[i, 0]:.2f}, {batch_velocities[i, 1]:.2f}, {batch_velocities[i, 2]:.2f}]")

        self.vel_idx = [0, 1, 2]
        self.ball.set_dofs_velocity(batch_velocities, self.vel_idx)

        self.obs = self.ball.get_dofs_velocity(self.vel_idx)
        self.last_obs = self.obs
        self.step_count = 0

    def physics_step(self):
        self.scene.step()
        self.last_obs = self.obs
        self.obs = self.ball.get_dofs_velocity(self.vel_idx)

    def get_obs(self):
        return self.obs

    def get_last_obs(self):
        return self.last_obs

    def set_obs(self, obs):
        self.obs = obs
        self.ball.set_dofs_velocity(obs, self.vel_idx)

    def get_n_envs(self):
        return self.n_envs

    def reset(self):
        # Reset ball position to initial position
        self.ball.set_dofs_position(self.init_pos, [0, 1, 2])

        # Generate velocities based on rand parameter
        if self.rand:
            random_directions = torch.randn(self.n_envs, 3)
            random_directions = random_directions / torch.norm(random_directions, dim=1, keepdim=True)
            random_speeds = torch.rand(self.n_envs, 1) * 10 + 5  # uniform between 5 and 15
            batch_velocities = random_directions * random_speeds
        else:
            batch_velocities = torch.tensor([5.0, 0.0, 0.0]).unsqueeze(0).repeat(self.n_envs, 1)

        # Set new velocities
        self.ball.set_dofs_velocity(batch_velocities, self.vel_idx)

        # Update observations
        self.obs = self.ball.get_dofs_velocity(self.vel_idx)
        self.last_obs = self.obs

        # Print new velocities for each environment
        for i in range(self.n_envs):
            print(
                f"Reset Env {i} ball velocity: [{batch_velocities[i, 0]:.2f}, {batch_velocities[i, 1]:.2f}, {batch_velocities[i, 2]:.2f}]")


class RealWorld(BaseWorld):
    def __init__(self, rand=True):
        super().__init__(headless=False, gravity=9.81, friction=0.5, n_envs=1, rand=rand)


class SimWorld(BaseWorld):
    def __init__(self, n_envs=256, rand=True):
        _headless = True if n_envs > 1 else False
        super().__init__(headless=_headless, gravity=1.0, friction=0.1, n_envs=n_envs, rand=rand)
