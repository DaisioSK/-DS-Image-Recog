
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from PIL import Image


def get_emnist_data_loaders():

    # train data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),  
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((28, 28)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((28, 28)),
    ])


    full_train_dataset = torchvision.datasets.EMNIST(root="../data", split="byclass", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.EMNIST(root="../data", split="byclass", train=False, download=True, transform=test_transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size  
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    batch_size = 64
    num_workers = min(4, multiprocessing.cpu_count()) 
    print(f"num_workers: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"training size: {len(train_dataset)}")  
    print(f"validation size: {len(val_dataset)}")  
    print(f"testing size: {len(test_dataset)}")
    print(f"classification size: {len(full_train_dataset.class_to_idx)}")  
    print(f"classifications: {full_train_dataset.class_to_idx}")  

    return train_loader, val_loader, test_loader



def plot_training_curves(train_losses, val_losses, train_accs, val_accs):

    epochs = range(1, len(train_losses) + 1) 

    plt.figure(figsize=(12, 5))

    #  Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label="Train Loss", color='blue')
    plt.plot(epochs, val_losses, marker='s', linestyle='--', label="Validation Loss", color='red', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, marker='o', linestyle='-', label="Train Accuracy", color='blue')
    plt.plot(epochs, val_accs, marker='s', linestyle='--', label="Validation Accuracy", color='red', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def display_emnist_sample_images(images, labels, preds=None):

    emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    fig, axes = plt.subplots(2, 6, figsize=(10, 5))

    images.cpu()
    if not isinstance(images, np.ndarray):
        images = images.numpy()

    for i, ax in enumerate(axes.flat):
        img = images[i]
        img = img.squeeze()
        # img = np.transpose(img, (1, 2, 0))  
        img = np.rot90(img, k=3)  # 270° rotation
        img = np.fliplr(img)  # flip left-right

        ax.imshow(img.squeeze(), cmap="gray")  # grey scale
        title = f"Label: {emnist_classes[labels[i]]}" if preds is None else f"Pred: {emnist_classes[preds[i]]}\nLabel: {emnist_classes[labels[i]]}"
        ax.set_title(title) 
        ax.axis("off")

    plt.show()



def predict(model, image_or_path):
    if isinstance(image_or_path, str):
        image = preprocess_image(image_or_path)  
    elif isinstance(image_or_path, torch.Tensor):
        image = image_or_path  
    else:
        raise ValueError("file path must be (str) or PyTorch Tensor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        return emnist_classes[predicted.item()]



def preprocess_image(image_path, mode="test"):

    image = Image.open(image_path).convert("L")  # greyscale
    image = np.array(image)  # to numpy array

    # color binarization
    if np.mean(image) > 127:
        image = 255 - image

    coords = cv2.findNonZero(255 - image)  # find non-zero pixel coordinates
    x, y, w, h = cv2.boundingRect(coords)
    image = image[y:y+h, x:x+w]  # crop the image

    # padding
    h, w = image.shape
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)

    # EMNIST dataset ajustments
    image = np.rot90(image, k=3)  # 270° rotation
    image = np.fliplr(image)  # flip left-right

    image = Image.fromarray(image).resize((28, 28), resample=Image.LANCZOS)  # resize
    image = np.array(image, dtype=np.uint8).copy()  # numpy array copy

    # augmentation
    if mode == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(15),  
            transforms.RandomAffine(0, translate=(0.1, 0.1)), 
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    image_tensor = transform(image).unsqueeze(0)  # to tensor (1, 1, 28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(Image.open(image_path).convert("L"), cmap="gray")
    axes[0].set_title("Before Processing")
    axes[0].axis("off")

    axes[1].imshow(image.squeeze(), cmap="gray")
    axes[1].set_title("After Processing")
    axes[1].axis("off")

    plt.show()

    return image_tensor


