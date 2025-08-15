import torch
from world_translation.train import Trainer


def main():
    trainer = Trainer(
        data_dir_A="./data/data_sim",
        data_dir_B="./data/data_real",
        world_name_A="sim-world",
        world_name_B="real-world",
        latent_dim=256,
        device=torch.device('cuda')
    )

    trainer.configure(batch_size=128)

    trainer.train(
        num_epochs=200,
        save_freq=10,
    )


if __name__ == "__main__":
    main()
