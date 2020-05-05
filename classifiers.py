from torch import nn
from torch.nn import functional as F
from torchvision import models


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3)
        self.conv2 = nn.Conv2d(9, 27, kernel_size=3)
        self.conv3 = nn.Conv2d(27, 81, kernel_size=3)

        # max pooling
        self.maxPooling = nn.MaxPool2d(kernel_size=3)

        self.feed_forward_input_size = 81 * 7 * 7

        # feed forward
        self.fc1 = nn.Linear(self.feed_forward_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

        # dropouts
        self.dropout50 = nn.Dropout(0.5)
        self.dropout25 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.maxPooling(F.relu(self.conv1(x)))
        x = self.maxPooling(F.relu(self.conv2(x)))
        x = self.maxPooling(F.relu(self.conv3(x)))

        x = self.dropout50(x)
        x = x.view(-1, self.feed_forward_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        x = F.relu(self.fc2(x))
        x = self.dropout25(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class Vgg16Features(nn.Module):
    def __init__(self):
        super(Vgg16Features, self).__init__()

        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.classifier = nn.Sequential(*list(vgg.classifier)[:-1])
        self.features_count = 4096

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        return self.classifier(x)
