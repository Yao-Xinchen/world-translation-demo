from worlds import RealWorld


def main():
    world = RealWorld()

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
        print("Stopping real-world ...")


if __name__ == "__main__":
    main()
