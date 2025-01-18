import torch
from torchvision import datasets, transforms


# Not sure if downloading works...
def download_data():
    import os
    import requests
    import zipfile

    # Define the folder and file paths
    raw_folder = "data/raw/"
    zip_file_path = os.path.join(raw_folder, "dataset.zip")

    # Ensure the folder exists
    os.makedirs(raw_folder, exist_ok=True)

    # Direct file URL (replace if necessary)
    file_url = "https://www.muratkoklu.com/datasets/vtdhnd09.php"

    # Download the .zip file
    print("Downloading the file...")
    response = requests.get(file_url, allow_redirects=True)
    with open(zip_file_path, "wb") as file:
        file.write(response.content)
    print(f"File downloaded and saved to {zip_file_path}")

    # Extract the .zip file
    print("Extracting the zip file...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(raw_folder)
    print(f"Contents extracted to {raw_folder}")

    # Optional: Remove the .zip file after extraction
    os.remove(zip_file_path)
    print("Zip file removed after extraction.")


def load_data():
    train_dataset = torch.load("data/processed/train_dataset.pt")
    val_dataset = torch.load("data/processed/val_dataset.pt")
    test_dataset = torch.load("data/processed/test_dataset.pt")

    print(
        f"Train: {
            len(train_dataset)}, Validation: {
            len(val_dataset)}, Test: {
                len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def pre_process_data():
    # Define transformations (e.g., resizing, normalization)
    transform = transforms.Compose(
        [
            transforms.Resize((250, 250)),  # Resize to 250x250
            transforms.ToTensor(),  # Convert image to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    # Load dataset from folder
    dataset = datasets.ImageFolder(root='data/raw/Rice_Image_Dataset/Rice_Image_Dataset/', transform=transform)

    # Set seed for reproducibility of train, validation, and test datasets
    torch.manual_seed(0)

    # Define split sizes
    train_size = int(0.7 * len(dataset))  # 70% train
    val_size = int(0.15 * len(dataset))   # 15% validation
    test_size = len(dataset) - train_size - val_size  # Remaining for test

    # Randomly split the dataset into train, validation, and test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Save the splits
    torch.save(train_dataset, 'data/processed/train_dataset.pt')
    torch.save(val_dataset, 'data/processed/val_dataset.pt')
    torch.save(test_dataset, 'data/processed/test_dataset.pt')



if __name__ =="__main__":
    # download_data()
    pre_process_data()
