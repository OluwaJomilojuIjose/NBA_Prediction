import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump

CHUNKSIZE = 10000

# -----------------------------
# Helper Functions
# -----------------------------

def load_and_augment_training_data(data_folder, exclude_files=[], chunksize=CHUNKSIZE):
    print("Processing training data in chunks from folder:", data_folder)
    augmented_chunks = []
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    for file in csv_files:
        if any(exclude in file for exclude in exclude_files):
            continue
        print("Processing file:", file)
        for chunk in pd.read_csv(file, chunksize=chunksize):
            chunk = preprocess_data(chunk, allowed_features)
            augmented_chunk = augment_training_data(chunk, home_cols=["home_0", "home_1", "home_2", "home_3", "home_4"])
            augmented_chunks.append(augmented_chunk)
            print(f"Processed chunk with {len(chunk)} rows; augmented to {len(augmented_chunk)} rows.")
    if augmented_chunks:
        full_augmented_df = pd.concat(augmented_chunks, ignore_index=True)
        print("Finished processing training data. Total augmented data shape:", full_augmented_df.shape)
        return full_augmented_df
    else:
        print("No training data found.")
        return pd.DataFrame()

def load_allowed_features(allowed_features_path):
    print("Loading allowed features from:", allowed_features_path)
    with open(allowed_features_path, 'r') as f:
        allowed_features = [line.strip() for line in f if line.strip()]
    print("Allowed features loaded:", allowed_features)
    return allowed_features

def load_metadata(metadata_path):
    print("Loading metadata from:", metadata_path)
    metadata_df = pd.read_excel(metadata_path)
    print("Metadata loaded. Columns:", metadata_df.columns.tolist())
    return metadata_df

def preprocess_data(df, feature_list):
    print("Preprocessing data: dropping rows with missing values for features:", feature_list)
    home_slots = {"home_0", "home_1", "home_2", "home_3", "home_4"}
    features_to_check = [col for col in feature_list if col not in home_slots and col in df.columns]
    df = df.dropna(subset=features_to_check)
    print("Preprocessing complete. Data shape:", df.shape)
    return df

def augment_training_data(df, home_cols=["home_0", "home_1", "home_2", "home_3", "home_4"]):
    print("Augmenting data for", len(df), "rows.")
    instances = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        for col in home_cols:
            instance = row_dict.copy()
            instance["target"] = instance[col]
            instance[col] = "MASK"
            instance["mask_pos"] = col
            instances.append(instance)
    augmented_df = pd.DataFrame(instances)
    print("Augmentation complete. Augmented data shape:", augmented_df.shape)
    return augmented_df

def prepare_training_data(df, features_to_use, target_col="target"):
    print("Preparing training data...")
    X = df[features_to_use].copy()
    y = df[target_col].copy()
    print("Training data prepared. X shape:", X.shape, "y shape:", y.shape)
    return X, y

def encode_categorical_columns(X, categorical_features):
    print("Encoding categorical features:", categorical_features)
    encoders = {}
    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            print(f"Encoded column: {col}")
    return X, encoders

def encode_target(y):
    print("Encoding target variable.")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    print("Target encoding complete.")
    return y_encoded, le

def train_model(X, y):
    print("Training model using RandomForestClassifier with limited parallelism and depth...")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=1
    )
    clf.fit(X, y)
    print("Model training complete.")
    return clf

def predict_missing_player_for_row(model, row, features_to_use, categorical_features, feature_encoders, target_encoder):
    row = row.copy()
    missing_col = None
    for col in ["home_0", "home_1", "home_2", "home_3", "home_4"]:
        if pd.isna(row[col]) or (isinstance(row[col], str) and row[col].strip() == "?"):
            missing_col = col
            break
    if missing_col is None:
        return row
    row[missing_col] = "MASK"
    X_instance = {}
    for col in features_to_use:
        value = str(row[col]) if col in row else (missing_col if col == "mask_pos" else "unknown")
        if col in categorical_features and col in feature_encoders:
            le = feature_encoders[col]
            if value not in le.classes_:
                value = "unknown"
                if "unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "unknown")
            X_instance[col] = le.transform([value])[0]
        else:
            try:
                X_instance[col] = float(value)
            except:
                X_instance[col] = 0.0
    feature_vector = [X_instance[col] for col in features_to_use]
    pred_encoded = model.predict([feature_vector])[0]
    pred = target_encoder.inverse_transform([pred_encoded])[0]
    row[missing_col] = pred
    row["Predicted_missing_slot"] = missing_col
    return row

def predict_missing_player(model, test_df, features_to_use, categorical_features, feature_encoders, target_encoder):
    print("Generating predictions on test data...")
    predicted_rows = []
    for idx, row in test_df.iterrows():
        row_pred = predict_missing_player_for_row(model, row, features_to_use, categorical_features, feature_encoders, target_encoder)
        predicted_rows.append(row_pred)
    pred_df = pd.DataFrame(predicted_rows)
    print("Predictions generated.")
    return pred_df

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("Starting pipeline...")
    data_folder = "data"
    allowed_features_path = os.path.join(data_folder, "allowed_features.txt")
    metadata_path = os.path.join(data_folder, "Matchup-metadata.xlsx")

    global allowed_features
    allowed_features = load_allowed_features(allowed_features_path)

    test_file = os.path.join(data_folder, "NBA_test.csv")
    test_df = pd.read_csv(test_file)
    print("Test data shape:", test_df.shape)

    common_features = [feat for feat in allowed_features if feat in test_df.columns]
    features_to_use = common_features.copy()
    if "mask_pos" not in features_to_use:
        features_to_use.append("mask_pos")
    print("Features to use for training/prediction:", features_to_use)

    load_metadata(metadata_path)

    print("Loading and augmenting training data...")
    training_augmented_df = load_and_augment_training_data(data_folder, exclude_files=["NBA_test"], chunksize=CHUNKSIZE)
    X, y = prepare_training_data(training_augmented_df, features_to_use, target_col="target")

    candidate_categorical = ['game', 'season', 'home_team', 'away_team',
                             'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                             'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 'mask_pos']
    categorical_features = [feat for feat in candidate_categorical if feat in features_to_use]

    print("Encoding categorical features...")
    X, feature_encoders = encode_categorical_columns(X, categorical_features)

    print("Encoding target variable...")
    y_encoded, target_encoder = encode_target(y)

    print("Splitting training data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    print("Training set shape:", X_train.shape, "Validation set shape:", X_val.shape)

    print("Training the model...")
    clf = train_model(X_train, y_train)

    if len(X_train) > 50000:
        subsample = X_train.sample(n=50000, random_state=42)
        subsample_index = subsample.index.to_numpy()
        X_train_cv = subsample
        y_train_cv = y_train[subsample_index]
    else:
        X_train_cv = X_train
        y_train_cv = y_train

    print("Starting cross-validation with 3 folds (n_jobs=1)...")
    cv_scores = cross_val_score(clf, X_train_cv, y_train_cv, cv=3, n_jobs=1)
    print(f"Cross-validation scores: {cv_scores}")

    print("Saving model and encoders to model.pkl...")
    dump({
        'model': clf,
        'feature_encoders': feature_encoders,
        'target_encoder': target_encoder,
        'allowed_features': features_to_use,
        'categorical_features': categorical_features
    }, "model.pkl")
    print("Model and encoders saved to model.pkl")

    print("Replacing '?' with NaN in test data...")
    test_df.replace('?', np.nan, inplace=True)

    allowed_cols = ['season', 'home_team', 'away_team', 'starting_min',
                    'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                    'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
    common_cols = [col for col in allowed_cols if col in test_df.columns]
    print("Using these columns from test data:", common_cols)
    test_df = test_df[common_cols]

    print("Preprocessing test data...")
    test_df = preprocess_data(test_df, features_to_use)

    print("Generating predictions on test data...")
    test_predictions = predict_missing_player(clf, test_df, features_to_use, categorical_features, feature_encoders, target_encoder)

    results = []
    for idx, row in test_predictions.iterrows():
        missing_slot = row.get("Predicted_missing_slot", None)
        predicted_player = row.get(missing_slot, None) if missing_slot is not None else None
        results.append({
            "home_team": row["home_team"],
            "Predicted_missing_slot": missing_slot,
            "Predicted_player": predicted_player
        })
    final_output = pd.DataFrame(results)

    labels_file = os.path.join(data_folder, "NBA_test_labels.csv")

    if os.path.exists(labels_file):
        print("Loading test labels from:", labels_file)
        test_labels = pd.read_csv(labels_file, skiprows=1, header=None)
        test_labels.columns = ["Actual_player"]

        min_len = min(len(final_output), len(test_labels), len(test_df))
        final_output = final_output.iloc[:min_len].reset_index(drop=True)
        test_labels = test_labels.iloc[:min_len].reset_index(drop=True)
        test_df = test_df.iloc[:min_len].reset_index(drop=True)

        final_output["Actual_player"] = test_labels["Actual_player"]

        if "season" in test_df.columns:
            final_output["season"] = test_df["season"]
            grouped = final_output.groupby("season")
            print("\nAccuracy by season:")
            for season, group in grouped:
                correct = (group["Predicted_player"] == group["Actual_player"]).sum()
                total = len(group)
                acc = correct / total if total > 0 else 0
                print(f"Season {season}: Accuracy = {acc:.4f}")
        else:
            print("\nNo 'season' column in test data; cannot compute per-season accuracy.")

        correct_total = (final_output["Predicted_player"] == final_output["Actual_player"]).sum()
        total_predictions = len(final_output)
        overall_accuracy = correct_total / total_predictions if total_predictions > 0 else 0
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

        print("\nFirst 20 predictions vs actual:")
        print(final_output.head(20))

    else:
        print("No test labels file found. Final predictions (first 20 rows):")
        print(final_output.head(20))

    print("\nSaving final predictions to predicted_fifth_players.csv...")
    final_output.to_csv("predicted_fifth_players.csv", index=False)
    print("Predictions saved to predicted_fifth_players.csv")

    print("\nPipeline complete. Exiting.")

if __name__ == "__main__":
    main()
