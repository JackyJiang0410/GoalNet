# Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System
This project uses Graph Neural Networks (GNNs) to analyze soccer player interactions and assign **Expected Threat (xT)** values to individual players. By leveraging player interaction data, this model highlights the hidden contributions of players who may not directly contribute to goals but play crucial roles in ball progression and team coordination.

## Table of Contents
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Player Evaluation](#player-evaluation)
- [Results](#results)

## Installation

First, install all the required dependencies listed in the `requirements.txt` file. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Project Workflow
1. Download Player Information
Use the dataset/sofascore.ipynb Jupyter notebook to download player information from SofaScore. Follow these steps:
Open dataset/sofascore.ipynb in your Jupyter environment.
Execute the cells to download the player data.
Save the player data to the desired directory for use in training and evaluation.
2. Train the Model
Once the player data is prepared, proceed to train the GNN model by running the train.py script. Before training, you can update the script's arguments to fit your competition, season, and other configurations.
```bash
parser.add_argument('-c', '--comp', default="Premier League", type=str, help='competition name')
parser.add_argument('-s', '--season', default="2015/2016", type=str, help='season name')
parser.add_argument('-p', '--path', default='', type=str, help='path to player data')
parser.add_argument('-o', '--out_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/ckpt.pth', type=str, help='path to save model')
parser.add_argument('-ed', '--edge_dim', default=11, type=int, help='edge feature dim')
parser.add_argument('-hd', '--hidden_dim', default=32, type=int, help='hidden dimension')
parser.add_argument('-od', '--output_dim', default=32, type=int, help='output dimension')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size')
parser.add_argument('-vs', '--val_size', default=0.2, type=float, help='validation size')
parser.add_argument('-w', '--wandb', default='CMSAC', type=str, help='wandb project name')
parser.add_argument('-e', '--epoch', default=10, type=int, help='epoch number for training')
parser.add_argument('-d', '--device', default=0, type=int, help='device number')
```

Example Training Command
```bash
python train.py -c "Premier League" -s "2015/2016" -p /path/to/player_data -o /path/to/save_model.ckpt
```
This will train the GNN model and save the trained weights to the specified out_path.

3. Evaluate Player xT Contributions
After training, use eval_player.py to evaluate the xT contributions for each player. Update the script’s arguments to match your setup.

```bash
parser.add_argument('-c', '--comp', default="Premier League", type=str, help='competition name')
parser.add_argument('-s', '--season', default="2015/2016", type=str, help='season name')
parser.add_argument('-p', '--path', default='', type=str, help='path to player data')
parser.add_argument('-m', '--model_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/ckpt.pth', type=str, help='path to load model')
parser.add_argument('-ed', '--edge_dim', default=11, type=int, help='edge feature dim')
parser.add_argument('-hd', '--hidden_dim', default=32, type=int, help='hidden dimension')
parser.add_argument('-od', '--output_dim', default=32, type=int, help='output dimension')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size')
parser.add_argument('-o', '--output_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/output.csv', type=str, help='path to save evaluation results')
```
Example Evaluation Command
```bash
python eval_player.py -c "Premier League" -s "2015/2016" -p /path/to/player_data -m /path/to/save_model.ckpt -o /path/to/output.csv
```
This command will evaluate the xT contributions of each player using the trained GNN model and save the results to the specified output_path.

Files
train.py: Used to train the GNN model.
eval_player.py: Used to evaluate the xT contributions of players.
dataset/sofascore.ipynb: Jupyter notebook to download player information from SofaScore.
requirements.txt: Contains all the necessary packages and dependencies required for the project.
Output
The trained model is saved to the specified path (e.g., ckpt.pth).
The player xT contributions are saved to a CSV file (e.g., output.csv).
