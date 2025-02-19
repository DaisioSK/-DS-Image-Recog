import torch
# import emnist_utils as eu
# import emnist_cnn as ec

from emnist_utils import plot_training_curves, preprocess_image, predict, display_emnist_sample_images, get_emnist_data_loaders
from emnist_cnn import CNN, train_CNN
import matplotlib.pyplot as plt
import numpy as np


def visualize_testing(model, test_loader, misclassified):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emnist_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    fig, axes = plt.subplots(2, 6, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy().squeeze()
        img = np.rot90(img, k=3) 
        img = np.fliplr(img) 

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Pred: {emnist_classes[predicted[i]]}\nLabel: {emnist_classes[labels[i]]}")
        ax.axis("off")

    plt.show()
    
    # visualize misclassified images
    fig, axes = plt.subplots(2, 6, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img, pred, true = misclassified[i]
        img = img.cpu().numpy().squeeze()
        img = np.rot90(img, k=3)  
        img = np.fliplr(img) 

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Pred: {emnist_classes[pred]}\nTrue: {emnist_classes[true]}")
        ax.axis("off")

    plt.show()  



def main():
    
    train_loader, val_loader, test_loader = get_emnist_data_loaders()

    # visualize the data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    display_emnist_sample_images(images, labels)

    # train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model_path = "../models/emnist_baseline.pth"
    num_epochs = 5
    train_losses, train_accs, val_losses, val_accs = train_CNN(model, model_path, train_loader, val_loader, num_epochs)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)


    # model evaluation
    model.eval()
    correct = 0
    total = 0
    misclassified = []

    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # wrong clsasifications
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append((images[i], predicted[i], labels[i]))

    test_acc = correct / total
    print(f"Test set accuracy: {test_acc:.4f}")
    
    #visualize the testing results
    visualize_testing(model, test_loader, misclassified)
    
    # test on hand-written images from the internet
    image_path = "../images/S.png"
    image_tensor = preprocess_image(image_path, mode="test")
    print("tensor shape: ", image_tensor.shape)

    processed_image = image_tensor
    prediction = predict(model, processed_image)
    print(f"prediction result: {prediction}")




if __name__ == "__main__":
    main()