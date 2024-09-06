from model.soccer_gnn import SoccerGNN
from dataset.event_data import get_event_data
from dataset.player_data import get_player_data
from dataset.soccer_dataset import create_dataloaders, standardize_data
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser(description='Deep learning model for player evaluation')
parser.add_argument('-c', '--comp', default="Premier League", type=str,
                    help='competition name')
parser.add_argument('-s', '--season', default="2015/2016", type=str,
                    help='season name')
parser.add_argument('-p', '--path', default='', type=str,
                    help='path to player data')
parser.add_argument('-o', '--out_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/ckpt.pth', type=str,
                    help='path to save model')
parser.add_argument('-ed', '--edge_dim', default=11, type=int,
                    help='edge feature dim')
parser.add_argument('-hd', '--hidden_dim', default=32, type=int,
                    help='hidden dimension')
parser.add_argument('-od', '--output_dim', default=32, type=int,
                    help='output dimension')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='batch size')
parser.add_argument('-vs', '--val_size', default=0.2, type=float,
                    help='validation size')
parser.add_argument('-w', '--wandb', default='CMSAC', type=str,
                    help='wandb project name')
parser.add_argument('-e', '--epoch', default=10, type=int,
                    help='epoch number for training')
parser.add_argument('-d', '--device', default=0, type=int,
                    help='device number')

def main(args):
    competition = args.c
    season = args.s
    players = args.p

    save_path = args.o
    edge_dim = args.ed
    hidden_dim = args.hd
    output_dim = args.od

    lr = args.lr
    batch_size = args.bs
    val_size = args.vs

    wandb_name = args.w 
    epoch = args.e
    device = args.d

    events = get_event_data(competition, season)
    players = get_player_data(players)

    train_loader, val_loader = create_dataloaders(events, players, batch_size=batch_size, val_split=val_size)
    standardize_data(train_loader)
    standardize_data(val_loader)

    node_feature_dim = len(players.columns)

    model = SoccerGNN(node_feature_dim, edge_dim, hidden_dim, output_dim, lr=lr)

    wandb.init(project=wandb_name, reinit=True)
    wandb_logger = WandbLogger(project=wandb_name, log_model=True)

    # PyTorch Lightning Trainer setup
    trainer = pl.Trainer(
        max_epochs=epoch,
        logger=wandb_logger,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=device
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)