import pandas as pd

def compute_accuracy_by_season(pred_file, labels_file, test_file):
    # Load predicted data from predictions CSV
    pred_df = pd.read_csv(pred_file)
    
    # Load test labels, skipping the first row with "removed_value"
    labels_df = pd.read_csv(labels_file, header=None, skiprows=1, names=["actual_player"])
    
    # Load the original test file (which contains the season column)
    test_df = pd.read_csv(test_file)
    if "season" not in test_df.columns:
        raise ValueError("No 'season' column found in the test file. Cannot compute accuracy by season.")
    
    # Check that the number of predictions, labels, and test rows are aligned.
    # We'll assume that the predictions and labels correspond in order to the test file rows.
    min_len = min(len(pred_df), len(labels_df), len(test_df))
    pred_df = pred_df.iloc[:min_len].reset_index(drop=True)
    labels_df = labels_df.iloc[:min_len].reset_index(drop=True)
    test_df = test_df.iloc[:min_len].reset_index(drop=True)
    
    # Add season from test file to the predictions DataFrame
    pred_df["season"] = test_df["season"]
    
    # Attach the actual labels to the predictions DataFrame
    pred_df["actual_player"] = labels_df["actual_player"]
    
    # Now, group by season and compute accuracy for each season
    grouped = pred_df.groupby("season")
    
    for season, group in grouped:
        correct = (group["Predicted_player"] == group["actual_player"]).sum()
        total = len(group)
        accuracy = correct / total if total > 0 else 0
        print(f"Season {season}: Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    compute_accuracy_by_season("predicted_fifth_players.csv", "NBA_test_labels.csv", "NBA_test.csv")
