from os import getcwd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lattice_points_ml.LatticePointsDataset import LatticePointsDataset


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.learning_rate = 0.001
        self.transform = transforms.Compose([transforms.ToTensor(),])
        self.criterion = nn.CrossEntropyLoss()
        self.is_predict = False


        #21x21x1
        self.layer1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        #11x11x12
        self.layer2 = nn.Sequential(nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        #6x6x24
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(6 * 6 * 40, 800)
        self.fc2 = nn.Linear(800, 2)
        # self.sm3 = nn.Softmax(0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        if self.is_predict:
            out = out.reshape(1, -1)
        else:
            out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.sm3(out)
        return out

    def load_model(self, filename: str):
        self.load_state_dict(torch.load(filename))

    def train_model(self, num_epochs: int, annot_train_filename: str, model_filename: str = 'model.pt', batch_size: int = 30):
        train_dataset = LatticePointsDataset(annot_train_filename, self.transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_loader.device = self.device
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Прямой запуск
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)
                loss_list.append(loss.item())

                # Обратное распространение и оптимизатор
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Отслеживание точности
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))

        torch.save(self.state_dict(), getcwd() + f'\\lattice_points_ml\\model\\{model_filename}')

    def test_model(self, annot_test_filename: str, batch_size: int = 30):
        test_dataset = LatticePointsDataset(annot_test_filename, self.transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Test Accuracy of the model on the {len(test_loader)*batch_size} test images: {(correct / total) * 100}')

    def predict_model(self, img: np.ndarray):
        self.is_predict = True
        tensor = self.transform(img).to(self.device)
        outputs = self.forward(tensor)
        _, predicted = torch.max(outputs.data, 1)
        self.is_predict = False
        return predicted.to('cpu').tolist()[0]
