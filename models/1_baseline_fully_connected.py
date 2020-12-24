import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import io, transform
from torch import Tensor
from torch.nn import Module, Linear, MSELoss, ModuleList
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary


class DashcamSpeedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = pd.read_csv(os.path.join(root_dir, "dataset.csv"))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        flow_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        flow = io.imread(flow_path)
        speed = self.dataset.iloc[idx, 1]
        speed = np.array([speed])
        sample = {'flow': flow, 'speed': speed}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __call__(self, sample):
        flow, speed = sample['flow'], sample['speed']

        flow = flow[self.x1: self.x2, self.y1: self.y2]

        return {'flow': flow, 'speed': speed}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        flow, speed = sample['flow'], sample['speed']

        h, w = flow.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(flow, (new_h, new_w))

        return {'flow': img, 'speed': speed}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        flow, speed = sample['flow'], sample['speed']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        flow = flow.transpose((2, 0, 1))
        return {'flow': torch.from_numpy(flow).float(),
                'speed': torch.from_numpy(speed).float()}


class BaselineFullyConnected(Module):
    def __init__(self, image_height, image_width):
        super(BaselineFullyConnected, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.linear_1 = Linear(self.image_height * self.image_width * 3, 1024)
        self.hidden_layers = ModuleList([Linear(1024, 1024) for i in range(1)])
        self.linear_2 = Linear(1024, 1)

    def forward(self, x: Tensor):
        x = x.view(-1, self.image_height * self.image_width * 3)
        x = F.relu(self.linear_1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = F.relu(self.linear_2(x))
        return x


if __name__ == '__main__':

    batch_size = 10
    num_epochs = 5
    learning_rate = 0.001
    image_height = 66
    image_width = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_dataset = DashcamSpeedDataset(root_dir="D:\\DashcamSpeed\\data\\output\\training",
                                           transform=transforms.Compose(
                                               [
                                                   # Crop(x1=0, y1=0, x2=1920, y2=900),
                                                   Rescale((image_height, image_width)),
                                                   ToTensor()]))
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=10,
                                     pin_memory=True)

    model = BaselineFullyConnected(image_height, image_width).to(device)
    summary(model, (1, image_width * image_height * 3), batch_size=batch_size)

    total_step = len(training_dataloader)
    loss_x = []
    loss_y = []

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(training_dataloader):
            flow, speed = sample_batched['flow'].to(device), sample_batched['speed'].to(device)

            outputs = model(flow)
            loss = criterion(outputs, speed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i_batch + 1, total_step,
                                                                     loss.item()))

            loss_x.append((epoch * total_step) + (i_batch + 1))
            loss_y.append(loss.item())

    fig, ax = plt.subplots()
    ax.plot(loss_x, loss_y)

    ax.set(xlabel='Step', ylabel='Mean Squared Error',
           title='Mean Squared Error vs. Step')
    ax.grid()
    plt.show()
