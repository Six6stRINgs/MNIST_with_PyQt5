import numpy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageQt


def load_mnist(batch_size=64, root='./dataset'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_custom_image(image_path) -> numpy.ndarray:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(image_path)
    img = transform(img)
    img = img.detach().cpu().numpy()
    img = numpy.expand_dims(img, axis=0)
    return img  # shape: (1, 1, 28, 28)


def np2QPixmap(img):
    img = numpy.squeeze(img)
    img = numpy.uint8(img * 255)
    img = Image.fromarray(img)

    return img.toqpixmap()
