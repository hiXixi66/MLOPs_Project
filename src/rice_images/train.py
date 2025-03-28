import os
import matplotlib.pyplot as plt
import torch
from rice_images.data import load_data
from rice_images.model import load_resnet18_timm
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

# Set the device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


@hydra.main(
    config_path="../../configs", config_name="train", version_base="1.1"
)
def train(cfg: DictConfig):
    """Train the ResNet-18 model on the rice images dataset and
    save parameters every epoch_save_interval epochs."""
    print(OmegaConf.to_yaml(cfg))

    # Define the working directories of the original script and Hydra's new one
    # hydra_wd = os.getcwd()
    original_wd = hydra.utils.get_original_cwd()

    # Explicitly define hyperparameters
    lr = cfg.lr
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    epoch_save_interval = cfg.epoch_save_interval
    model_save_path = cfg.model_save_path
    downsample_train = cfg.downsample_train
    # wandb.login(key="f28dead13ccc819e547217762f6e50dbbbb80bec") # Xixi's key
    wandb.init(
        project="rice_classification",  # Replace with your W&B project name
        entity="group-13",  # Replace with your W&B team name
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "model": "ResNet18",
            "downsample_train": downsample_train,
        },
    )

    os.chdir(original_wd)  # change to original directory to load data

    # Create the folder to save model parameters if it doesn't exist
    os.makedirs(f"models/{model_save_path}", exist_ok=True)

    # Load the data
    train_dataset, val_dataset, test_dataset = load_data()
    # Downsample train_dataset to make the code run faster
    num_samples_train = len(train_dataset) // downsample_train
    downsampled_indices = torch.randperm(len(train_dataset))[
        :num_samples_train
    ]
    train_dataset = torch.utils.data.Subset(train_dataset, downsampled_indices)
    num_classes = 5  # There are 5 different grains of rice

    # Load the model
    model = load_resnet18_timm(num_classes=num_classes).to(DEVICE)
    wandb.watch(model, log="all", log_freq=100)

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    """
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    """
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == target).sum().item()
            total += target.size(0)

            if i % 100 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item()}")

        # Calculate and log statistics
        epoch_loss /= len(train_dataloader)
        accuracy = correct / total
        statistics["train_loss"].append(epoch_loss)
        statistics["train_accuracy"].append(accuracy)

        print(
            f"Epoch {epoch} completed. Loss: {epoch_loss}, Accuracy: {
                accuracy * 100:.2f}%"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": accuracy,
            }
        )

        # Save model parameters every epoch_save_interval epochs
        if (epoch + 1) % epoch_save_interval == 0:

            checkpoint_path = (
                f"models/{model_save_path}/resnet18_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model parameters saved at {checkpoint_path}")
            wandb.save(checkpoint_path)  # Log the checkpoint to W&B

    # Save the final trained model
    torch.save(
        model.state_dict(), f"models/{model_save_path}/resnet18_rice_final.pth"
    )

    # Plot statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], label="Train Loss")
    axs[0].set_title("Train Loss")
    axs[0].legend()
    axs[1].plot(statistics["train_accuracy"], label="Train Accuracy")
    axs[1].set_title("Train Accuracy")
    axs[1].legend()
    # Save figure
    os.makedirs(f"reports/figures/{model_save_path}", exist_ok=True)
    fig.savefig(f"reports/figures/{model_save_path}/training_statistics.png")
    print("Training complete and statistics saved.")
    wandb.log(
        {
            "loss_curve": wandb.Image(fig),
            "accuracy_curve": wandb.Image(fig),
        }
    )


if __name__ == "__main__":
    train()
