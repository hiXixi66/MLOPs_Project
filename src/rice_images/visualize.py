import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rice_images.data import load_data
import sys

from rice_images.model import load_resnet18_timm


def visualize(
        model_checkpoint: str,
        figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    # model = MyAwesomeModel().load_state_dict(torch.load(model_checkpoint))
    model = load_resnet18_timm()
    state_dict = torch.load(model_checkpoint)  # Load only the state dictionary
    model.load_state_dict(state_dict)
    model.eval()
    model.fc = torch.nn.Identity()

    train_dataset, val_dataset, test_dataset = load_data()

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(5):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


def main() -> None:
    # typer.run(visualize)
    model_checkpoint = sys.argv[1]
    visualize(model_checkpoint)


if __name__ == "__main__":
    main()
