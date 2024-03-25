import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 64
EPOCHS = 3
SAVE_MODEL_PATH = "model.pth"

def download_dataset(download):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=download,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=download,
        transform=ToTensor(),
    )
    return training_data, test_data

def load_data(training_data, test_data, batch_size=BATCH_SIZE):
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # use "break" to just print the data's shape for once
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

def train(dataloader, model, loss_fn, optimizer):
    # Get the total number of samples in the dataset
    size = len(dataloader.dataset)

    # Set the model to training mode
    model.train()

    # Iterate over the batches in the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check if the current batch number is divisible by 100
        # If true, print the training loss and the number of processed samples
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    # Get the total number of samples in the dataset
    size = len(dataloader.dataset)

    # Get the total number of batches in the dataloader
    num_batches = len(dataloader)

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for test loss and correct predictions
    test_loss, correct = 0, 0
    
    # Turn off gradients during testing to save computational resources
    with torch.no_grad():
        for X, y in dataloader:
            # Perform forward pass
            pred = model(X)

            # Compute the test loss
            test_loss += loss_fn(pred, y).item()

            # Count the number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Calculate average test loss
    test_loss /= num_batches

    # Calculate accuracy
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_test_save_model(model, train_dataloader, test_dataloader):
    # set the loss_func and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train and test the model
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    # save the model
    # .pt/.pth/.pkl all the same
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Saved PyTorch Model State to: ", SAVE_MODEL_PATH)


def interferce(model, test_data):
    # loading models
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))

    # interferce
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    # Set the model to evaluation mode
    model.eval()

    # Get the input data and target label for the first sample in the test dataset
    x, y = test_data[0][0], test_data[0][1]

    with torch.no_grad():
        pred = model(x)
        # Get the index of the maximum value along the first dimension of the prediction tensor
        # This index indicates the predicted class label
        predicted = pred[0].argmax(0)
        # Get the predicted and actual class labels using the 'classes' list
        predicted_class, actual_class = classes[predicted], classes[y]
        print(f'Predicted: "{predicted_class}", Actual: "{actual_class}"')



if __name__ == "__main__":
    # download dataset
    training_data, test_data = download_dataset(False)

    # load data with DataLoader
    train_dataloader, test_dataloader = load_data(training_data, test_data)

    # init the model
    model = NeuralNetwork()
    print(model)

    # train、test、save the model
    train_test_save_model(model, train_dataloader, test_dataloader)

    # interferce
    interferce(model, test_data)

