import torch
from world_translation.collect import TransitionCollector
from worlds import RealWorld


def main():
    world = RealWorld(rand=True, headless=True)
    collector = TransitionCollector("./data/data_real", obs_dim=3, action_dim=0, buffer_size=1000, chunk_size=5000)

    collector.start_collection()
    print("Starting data collection...")

    action = torch.zeros(world.get_n_envs(), 0)  # No action in this case

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

            collector.add_transition(last_obs, action, obs)
    except KeyboardInterrupt:
        print("Stopping collection...")
    finally:
        collector.stop_collection()
        print("Collection stopped. Data saved.")


if __name__ == "__main__":
    main()
