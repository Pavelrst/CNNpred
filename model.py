from torch import nn

class CNNpred(nn.Module):
    def __init__(self, num_features, num_filter, drop):
        super(CNNpred, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=num_filter, kernel_size=(1, num_features))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filter,
                               out_channels=num_filter, kernel_size=(3, 1))
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=num_filter,
                               out_channels=num_filter, kernel_size=(3, 1))
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.drop1 = nn.Dropout(drop)
        self.fc1 = nn.Linear(96, 1)
        self.sig1 = nn.Sigmoid()

    # Defining the forward pass
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.drop1(x)
        x = self.sig1(self.fc1(x))
        return x


class CNNpred_small(nn.Module):
    def __init__(self, num_features, num_filter, drop):
        super(CNNpred_small, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=num_filter, kernel_size=(1, num_features))
        self.relu1 = nn.ReLU()

        self.drop1 = nn.Dropout(drop)
        self.fc1 = nn.Linear(480, 1)
        self.sig1 = nn.Sigmoid()

    # Defining the forward pass
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(x.shape[0], -1)
        x = self.sig1(self.fc1(x))

        return x