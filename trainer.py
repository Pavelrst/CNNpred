from data_utils import IndexDataset
from torch.utils.data import DataLoader
from model import CNNpred, CNNpred_small
from torch import nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

def train():
    ds_train = IndexDataset(root_dir='Dataset_out/train', seq_len=30, target='baseline_target')
    ds_val = IndexDataset(root_dir='Dataset_out/test', seq_len=30, target='baseline_target')
    print(f'dataset len: {len(ds_train)}')
    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=True)

    # model = CNNpred(num_features=82, num_filter=8, drop=0.2)
    model = CNNpred(num_features=82, num_filter=16, drop=0.9)
    print(model)

    loss_fcn = nn.BCELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    num_epochs = 15

    train_loss_list = []
    train_acc = []
    val_loss_list = []
    val_acc = []
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        model.train()

        train_loss_sum = 0
        total_samples = 0
        total_correct = 0
        for idx, (X, y) in enumerate(dl_train):
            # plt.imshow(torch.squeeze(X[1].detach()).numpy())
            # plt.title('Train sample')
            # plt.show()
            y_logit = model.forward(X)
            loss = loss_fcn(y_logit, y)
            y_pred = (y_logit > 0.5).int()
            total_correct += torch.sum((y == y_pred).int()).item()
            total_samples += y.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss_list.append(train_loss_sum/len(dl_train))
        train_acc.append(total_correct/total_samples)

        val_loss_sum = 0
        total_samples = 0
        total_correct = 0
        model.eval()
        for idx, (X, y) in enumerate(dl_val):
            # for s in range(X.shape[0]):
            #     plt.imshow(torch.squeeze(X[s].detach()).numpy())
            #     plt.title('Val sample')
            #     plt.show()
            y_logit = model.forward(X)
            loss = loss_fcn(y_logit, y)
            val_loss_sum += loss.item()

            y_pred = (y_logit > 0.5).int()
            total_correct += torch.sum((y == y_pred).int()).item()
            total_samples += y.shape[0]

        val_loss_list.append(val_loss_sum/len(dl_val))
        val_acc.append(total_correct/total_samples)

    plt.title('Accuracy / Epochs')
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


train()
