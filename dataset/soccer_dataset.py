from torch.utils.data import Dataset
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeoDataLoader

class SoccerDataset(Dataset):
    def __init__(self, actions_df, players_df, k=5, threshold=80):
        self.actions_df = actions_df
        self.players_df = players_df
        self.players_df.index = self.players_df.index.str.lower().str.strip()
        self.k = k
        self.threshold = threshold
        self.most_common_player_values = players_df.mode().iloc[0].values

        # Create a player-to-features dictionary using fuzzy matching
        self.player_dict = self.create_player_dict()

    def create_player_dict(self):
        """Create a dictionary mapping player names to their features using fuzzy matching."""
        player_dict = {}
        player_names_list = [name.strip().lower() for name in self.players_df.index]

        for player_name in player_names_list:
            match_result = process.extractOne(player_name, player_names_list)
            if match_result:
                matched_name, _ = match_result
                player_dict[player_name] = self.players_df.loc[matched_name].values
            else:
                player_dict[player_name] = self.most_common_player_values

        return player_dict

    def get_player_features(self, player_name):
        """Retrieve player features from the pre-built dictionary."""
        player_name = player_name.strip().lower()
        return self.player_dict.get(player_name, self.most_common_player_values)

    def __len__(self):
        return len(self.actions_df)

    def __getitem__(self, idx):
        start_idx = max(0, idx - self.k + 1)
        end_idx = idx + 1
        events = self.actions_df.iloc[start_idx:end_idx]

        player_names = events['player_name'].unique()
        node_features = []
        for name in player_names:
            node_features.append(self.get_player_features(name))

        edge_index = []
        edge_attr = []
        for _, event in events.iterrows():
            receiver = event['player_name_recipient'] if pd.notnull(event['player_name_recipient']) else event['player_name']
            if receiver not in player_names:
                node_features = np.append(node_features, [self.get_player_features(receiver)], axis=0)
                player_names = np.append(player_names, receiver)

            edge_index.append([list(player_names).index(event['player_name']), list(player_names).index(receiver)])
            edge_attr.append([
                event['period_id'],
                event['time_seconds'],
                event['team_id'],
                event['start_x'],
                event['start_y'],
                event['end_x'],
                event['end_y'],
                event['type_id'],
                event['result_id'],
                event['bodypart_id'],
                event['action_id']
            ])

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_features = np.array(node_features)
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)

        y = torch.tensor(events.iloc[-1]['xT_change'], dtype=torch.float)

        data = Data(x=node_features_tensor,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    player_names=player_names
                    )


        return data

def collate_fn(batch):
    return batch

def create_dataloaders(actions_df, players_df, batch_size=32, k=5, val_split=0.2):
    dataset = SoccerDataset(actions_df, players_df, k=k)

    # Determine sizes for training and validation sets
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = GeoDataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    return train_loader, val_loader

def standardize_data(loader):
    for data_item in loader:
        # Standardize node features
        node_mean = data_item.x.mean(dim=0, keepdim=True)
        node_std = data_item.x.std(dim=0, keepdim=True)
        data_item.x = (data_item.x - node_mean) / (node_std + 1e-5)

        # Standardize edge features
        edge_mean = data_item.edge_attr.mean(dim=0, keepdim=True)
        edge_std = data_item.edge_attr.std(dim=0, keepdim=True)
        data_item.edge_attr = (data_item.edge_attr - edge_mean) / (edge_std + 1e-5)
