import torch

from worlds import SimWorld
from world_translation.deploy import load_latest_checkpoint, WorldTranslator


def main():
    world = SimWorld(n_envs=1, rand=False)

    models, metadata = load_latest_checkpoint("checkpoints/")
    print(metadata)

    from_world = "sim-world"
    to_world = "real-world"

    translator = WorldTranslator(models, metadata, device=torch.device('cuda'))

    action = torch.zeros(world.get_n_envs(), 0)

    try:
        step_count = 0
        while True:
            # Reset before physics step, so intermediate steps are not collected
            if step_count % 400 == 0:
                world.reset()
                print(f"Resetting world at step {step_count}")

            world.physics_step()
            step_count += 1

            obs = world.get_obs()
            last_obs = world.get_last_obs()

            translated_obs = translator.translate(from_world, to_world, last_obs, action, obs)

            world.set_obs(translated_obs)

    except KeyboardInterrupt:
        print("Stopping sim-world ...")


if __name__ == "__main__":
    main()
