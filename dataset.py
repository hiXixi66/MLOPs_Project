import typer

# from utils import show_image_and_target


def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics."""
    print("Dataset statistics")


if __name__ == "__main__":
    typer.run(dataset_statistics)
