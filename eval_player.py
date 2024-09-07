import argparse
from model.soccer_gnn import SoccerGNN
from dataset.event_data import get_event_data
from dataset.player_data import get_player_data
from dataset.soccer_dataset import create_dataloaders, standardize_data
import torch
from model.player_rank import distribute_xthreat, rank_players_by_xthreat

parser = argparse.ArgumentParser(description='Deep learning model for player evaluation')
parser.add_argument('-c', '--comp', default="Premier League", type=str,
                    help='competition name')
parser.add_argument('-s', '--season', default="2015/2016", type=str,
                    help='season name')
parser.add_argument('-p', '--path', default='', type=str,
                    help='path to player data')
parser.add_argument('-m', '--model_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/ckpt.pth', type=str,
                    help='path to load model')
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
parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='batch size')
parser.add_argument('-o', '--output_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/output.csv', type=str,
                    help='path to load model')

def main(args):
    competition = args.c
    season = args.s
    players = args.p

    save_path = args.o
    edge_dim = args.ed
    hidden_dim = args.hd
    output_dim = args.od
    model_path = args.m

    lr = args.lr
    batch_size = args.bs

    events = get_event_data(competition, season)
    players = get_player_data(players)

    loader, _ = create_dataloaders(events, players, batch_size=batch_size, val_split=0)
    standardize_data(loader)
    
    node_feature_dim = len(players.columns)

    model = SoccerGNN(node_feature_dim, edge_dim, hidden_dim, output_dim, lr=lr)
    model.load_state_dict(torch.load(model_path))

    player_xthreat_contributions = distribute_xthreat(model, loader)
    
    df_ranked_players = rank_players_by_xthreat(player_xthreat_contributions)

    df_ranked_players.to_csv(save_path, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)