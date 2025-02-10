#import all the necessary libs :)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


LEARNING_RATE = 0.00005

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 4)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc = nn.Linear(16, 10)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), LEARNING_RATE)
        self.relu = nn.ReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.pool1(x)
        x = self.conv3(self.relu(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc(x)
        return x

    def train_model(self, epochs, train_loader):
        self.train()  # Training mode

        for epoch in range(epochs):
            start_time = time.time()
            running_loss = 0.0
            total_batches = 0

            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                result = self.forward(inputs)
                loss = self.loss(result, labels)
                loss.backward()
                with torch.no_grad():
                    self.optimizer.step()

                running_loss += loss.item()
                total_batches += 1 
                if (i + 1) % 8 == 0 or (i + 1) == len(train_loader):
                    print(f"\rEpochs {epoch + 1}/{epochs} - lot {i + 1}/{len(train_loader)} - Loss value : {loss.item():.4f}", end='')

            
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - start_time


            print(f"\nEpochs {epoch + 1}/{epochs} finish - average Loss : {avg_loss:.4f} - Time : {epoch_time:.2f} seconds")

        # change the model_path if you want
        model_path = "model.pth"
        print('path to model:', model_path)
        torch.save(self.state_dict(), model_path)


    def test_model(self, test_loader):
        self.eval()  # Evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on {total} images is : {100 * correct / total:.2f}%')

    def load_weights(self, model_path):
        self.load_state_dict(torch.load(model_path))

model = MNISTModel()

BATCH_SIZE = 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

EPOCHS = 20
model.train_model(EPOCHS, train_loader)
model.test_model(test_loader)