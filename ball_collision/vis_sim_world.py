from worlds import SimWorld


def main():
    world = SimWorld(n_envs=1, rand=False)

    try:
        step_count = 0
        while True:
            # Reset before physics step, so intermediate steps are not collected
            if step_count % 400 == 0:
                world.reset()
                print(f"Resetting world at step {step_count}")

            world.physics_step()
            step_count += 1
    except KeyboardInterrupt:
        print("Stopping sim-world ...")


if __name__ == "__main__":
    main()
