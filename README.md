# Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System
\documentclass{article}
\usepackage{hyperref}

\title{Unveiling Hidden Pivotal Players: A GNN-Based Soccer Player Evaluation System}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Overview}

This project uses Graph Neural Networks (GNN) to evaluate soccer players' contributions, beyond traditional metrics like goals and assists, by analyzing their involvement in ball progression and defensive maneuvers. This system leverages player and event data from soccer matches to assign Expected Threat (xT) contributions to players using node embeddings learned through the GNN model.

\section*{Installation}

Before running the code, you need to install all required packages. You can do this by using the provided \texttt{requirements.txt} file:

\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}

\section*{Steps}

\subsection*{1. Download Player Information}

Use the Jupyter notebook \texttt{dataset/sofascore.ipynb} to download player information. You can save this data to any directory of your choice. 

\subsection*{2. Train the Model}

Once you have downloaded the player data, update the arguments in \texttt{train.py} to set up your training environment. The following are the arguments you can modify:

\begin{verbatim}
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
\end{verbatim}

Once you have updated the necessary arguments, you can train the model and save it to your desired directory by running the following command:

\begin{verbatim}
python train.py
\end{verbatim}

The trained model will be saved to the specified \texttt{--out\_path} (default: \texttt{ckpt.pth}).

\subsection*{3. Evaluate Player Contributions}

After training the model, you can extract the xT contributions for each player using the \texttt{eval\_player.py} script. The following are the arguments you can modify:

\begin{verbatim}
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
parser.add_argument('-o', '--output_path', default='Unveiling-Hidden-Pivotal-Players-A-GNN-Based-Soccer-Player-Evaluation-System/output.csv', type=str,
                    help='path to save xT contributions')
\end{verbatim}

You can evaluate player contributions and save the results to your desired directory by running the following command:

\begin{verbatim}
python eval_player.py
\end{verbatim}

The player xT contributions will be saved to the specified \texttt{--output\_path} (default: \texttt{output.csv}).

\section*{Summary of Commands}

\begin{enumerate}
    \item Install dependencies: \texttt{pip install -r requirements.txt}
    \item Download player data: Run \texttt{sofascore.ipynb}
    \item Train the model: \texttt{python train.py}
    \item Evaluate player xT: \texttt{python eval\_player.py}
\end{enumerate}

\section*{License}

This project is licensed under the MIT License.

\end{document}
