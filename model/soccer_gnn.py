import torch.nn as nn
import torch
import torch_geometric.nn as pyg_nn
import torchmetrics
import wandb 
import pytorch_lightning as pl

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # Xavier initialization for fully connected layers
        print(f"Initialized Linear layer {m}: Weight mean = {m.weight.mean().item()}, std = {m.weight.std().item()}")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, pyg_nn.GCNConv):  # Initialization for GCNConv layers
        torch.nn.init.kaiming_uniform_(m.lin.weight, nonlinearity='leaky_relu')  # Kaiming initialization for ReLU variants
        print(f"Initialized GCN layer {m}: Weight mean = {m.lin.weight.mean().item()}, std = {m.lin.weight.std().item()}")


class SoccerGNN(pl.LightningModule):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, output_dim, lr=1e-3):
        super(SoccerGNN, self).__init__()
        self.save_hyperparameters()

        # Model layers
        self.conv1 = pyg_nn.GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Global pooling layer (mean pooling)
        self.global_pool = pyg_nn.global_mean_pool

        # Hyperparameters
        self.lr = lr

        # Initialize the weights
        self.apply(init_weights)

        # Regression metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self.edge_mlp(edge_attr)

        # GCN layers with Leaky ReLU (Batch Normalization removed)
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.02)
        # print(f"After conv1: {x.mean().item()}, {x.std().item()}")

        x = self.conv2(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.02)
        # print(f"After conv2: {x.mean().item()}, {x.std().item()}")

        # Node embeddings
        node_embeddings = torch.relu(self.fc1(x))  # These are the node embeddings

        # Pool node embeddings to get graph-level output (for training)
        graph_embedding = self.global_pool(node_embeddings, batch)

        # Final output for graph (xG prediction)
        out = self.fc2(graph_embedding)

        return out, node_embeddings

    def training_step(self, batch, batch_idx):
        # The forward pass returns (out, node_embeddings)
        out, _ = self(batch)
        out = out.squeeze(-1)  # Apply squeeze to the graph-level prediction
        loss = nn.MSELoss()(out, batch.y)

        batch_size = batch.y.size(0)

        # Compute the actual values of the metrics
        self.train_mse(out, batch.y)
        self.train_mae(out, batch.y)

        # Log computed metric values
        train_mse_value = self.train_mse.compute()
        train_mae_value = self.train_mae.compute()

        self.log('train_loss', loss, batch_size=batch_size)
        self.log('train_mse', train_mse_value, batch_size=batch_size)
        self.log('train_mae', train_mae_value, batch_size=batch_size)

        # Log to W&B
        wandb.log({
            'train_loss': loss,
            'train_mse': train_mse_value,
            'train_mae': train_mae_value
        })

        return loss

    def validation_step(self, batch, batch_idx):
        # The forward pass returns (out, node_embeddings)
        out, _ = self(batch)
        out = out.squeeze(-1)  # Apply squeeze to the graph-level prediction
        loss = nn.MSELoss()(out, batch.y)

        batch_size = batch.y.size(0)

        # Compute the actual values of the metrics
        self.val_mse(out, batch.y)
        self.val_mae(out, batch.y)

        # Log computed metric values
        val_mse_value = self.val_mse.compute()
        val_mae_value = self.val_mae.compute()

        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('val_mse', val_mse_value, prog_bar=True, batch_size=batch_size)
        self.log('val_mae', val_mae_value, prog_bar=True, batch_size=batch_size)

        # Log to W&B
        wandb.log({
            'val_loss': loss,
            'val_mse': val_mse_value,
            'val_mae': val_mae_value
        })


    def configure_optimizers(self):
        # Change optimizer and learning rate here
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # If you want to add learning rate schedulers, you can do that too
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

