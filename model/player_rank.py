import torch
import pandas as pd

def distribute_xthreat(model, loader):
    model.eval()
    player_xthreat_contributions = {}

    with torch.no_grad():
        for batch in loader:
            # Get xG prediction and node embeddings
            xg_predictions, node_embeddings = model(batch)
            xg_predictions = xg_predictions.squeeze(-1)
            batch_indices = batch.batch.cpu().numpy()

            for i in range(len(xg_predictions)):
                event_xg = xg_predictions[i].item()
                event_embeddings = node_embeddings[batch_indices == i]
                # print(event_embeddings)

                current_player_names = batch.player_names[i]

                # Compute importance of each node in this event
                node_magnitudes = torch.norm(event_embeddings, dim=1)
                total_magnitude = node_magnitudes.sum().item()

                if total_magnitude == 0:
                    print(f"Warning: Total magnitude for event {i} is zero. Skipping event.")
                    continue
                    break

                # Distribute xG based on node embedding magnitude
                for j, player_name in enumerate(current_player_names):
                    contribution = (node_magnitudes[j].item() / total_magnitude) * event_xg
                    if player_name not in player_xthreat_contributions:
                        player_xthreat_contributions[player_name] = 0
                    player_xthreat_contributions[player_name] += contribution

    return player_xthreat_contributions

def summarize_player_xthreat_contributions(player_xthreat_contributions):
    for player_name, xthreat_contribution in player_xthreat_contributions.items():
        print(f"Player: {player_name}, xThreat Contribution: {xthreat_contribution:.3f}")

def rank_players_by_xthreat(player_xthreat_contributions):
    # Convert the dictionary to a sorted list of tuples (player_name, xthreat_contribution)
    ranked_players = sorted(player_xthreat_contributions.items(), key=lambda x: x[1], reverse=True)

    # Convert it to a DataFrame for better handling
    df = pd.DataFrame(ranked_players, columns=["Player", "xThreat Contribution"])

    # Add a ranking column
    df['Rank'] = df['xThreat Contribution'].rank(ascending=False, method='first').astype(int)

    return df

def generate_latex_table(df):
    # Generate LaTeX table from the DataFrame
    latex_table = df[['Rank', 'Player', 'xThreat Contribution']].to_latex(index=False, float_format="%.3f")
    return latex_table