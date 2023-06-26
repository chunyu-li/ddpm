import torchvision
import matplotlib.pyplot as plt


def show_dataset(dataset, num_samples=20, cols=4):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0])
    plt.show()


train_dataset = torchvision.datasets.ImageFolder(root="./stanford_cars/car_data/car_data/train")
show_dataset(train_dataset)
