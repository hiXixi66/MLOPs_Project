import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data():
    train_dataset = torch.load('data/processed/train_dataset.pt')
    val_dataset = torch.load('data/processed/val_dataset.pt')
    test_dataset = torch.load('data/processed/test_dataset.pt')
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def pre_process_data():
    # Define transformations (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((250, 250)),  # Resize to 250x250
        transforms.ToTensor(),         # Convert image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load dataset from folder
    dataset = datasets.ImageFolder(root='data/raw/Rice_Image_Dataset/', transform=transform)

    # Set seed for reproducibility of train, validation and test datasets
    torch.manual_seed(0)
    # Define split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split dataset and save em
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    torch.save(train_dataset, 'data/processed/train_dataset.pt')
    torch.save(val_dataset, 'data/processed/val_dataset.pt')
    torch.save(test_dataset, 'data/processed/test_dataset.pt')


if __name__ =="__main__":
    pre_process_data()


