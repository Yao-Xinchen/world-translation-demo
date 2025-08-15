import torch
from world_translation.train import Trainer


def main():
    trainer = Trainer(
        source_data_dir="./data/data_sim",
        target_data_dir="./data/data_real",
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
