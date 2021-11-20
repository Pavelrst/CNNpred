import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

TRAIN_TEST_CUTOFF = '2016-04-21'
TRAIN_VALID_RATIO = 0.75

def add_label_to_the_data():
    for file in [f for f in os.listdir('Dataset') if '.csv' if f and '~' not in f]:
        df = pd.read_csv(os.path.join('Dataset', file), index_col="Date", parse_dates=True)
        df.drop(columns=['Name'], inplace=True)

        # perc_change = df['Close'].pct_change()
        # perc_change.hist(bins=100)
        # stdiv = np.std(perc_change)
        # plt.vlines([-stdiv, stdiv], ymin=0, ymax=80, color='r', label='RMSE')
        # plt.show()

        feature_columns = df.columns

        df["baseline_target"] = (df['Close'].pct_change().shift(-1) > 0).astype(int)

        def set_v1_target(val, stdiv):
            if pd.isna(val):
                return None

            if val > stdiv:
                return 1
            elif val < -stdiv:
                return -1
            else:
                return 0

        df["0.5p_target"] = [set_v1_target(val, 0.005) for val in df['Close'].pct_change().shift(-1)]
        df["1p_target"] = [set_v1_target(val, 0.01) for val in df['Close'].pct_change().shift(-1)]
        # print(df["v1_target"].value_counts(normalize=True)*100)

        df.dropna(inplace=True)

        train_valid_index = df.index[df.index > TRAIN_TEST_CUTOFF]
        train_index = train_valid_index[:int(len(train_valid_index) * TRAIN_VALID_RATIO)]
        scaler = StandardScaler().fit(df.loc[train_index, feature_columns])

        df[feature_columns] = scaler.transform(df[feature_columns])

        test_df = df[df.index > TRAIN_TEST_CUTOFF]
        train_df = df[df.index <= TRAIN_TEST_CUTOFF]

        print(f'Train size: {len(train_df)}, Test size: {len(test_df)}')

        train_df.to_csv(os.path.join('Dataset_out', 'train', file))
        test_df.to_csv(os.path.join('Dataset_out', 'test', file))



class IndexDataset(Dataset):
    def __init__(self, root_dir, seq_len, target='baseline_target',
                 all_targets=['baseline_target', '0.5p_target', '1p_target']):
        self.targets_cols = all_targets
        self.active_target_col = target
        self.dataframes = []
        self.seq_len = seq_len
        for file in [f for f in os.listdir('Dataset') if '.csv' if f and '~' not in f]:
            self.dataframes.append(pd.read_csv(os.path.join(root_dir, file)))
    def __len__(self):
        return len(self.dataframes[0])-self.seq_len

    def __getitem__(self, idx):
        # print(idx)
        index = np.random.randint(0, len(self.dataframes))
        df = self.dataframes[index]

        frame = df.iloc[idx:idx+self.seq_len]
        X = frame[frame.columns.difference(['Date']+self.targets_cols)].to_numpy()

        # the label is the last value
        y = frame[self.active_target_col].to_numpy()[-1]
        X = torch.unsqueeze(torch.Tensor(X), dim=0)
        y = torch.Tensor([y])
        return X, y


# add_label_to_the_data()