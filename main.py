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
    """
    Load and process training CSV files in chunks.
    For each chunk, perform augmentation by masking each home slot.
    Returns a concatenated DataFrame of augmented training data.
    """
    print("Processing training data in chunks from folder:", data_folder)
    augmented_chunks = []
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    for file in csv_files:
        if any(exclude in file for exclude in exclude_files):
            continue
        print("Processing file:", file)
        for chunk in pd.read_csv(file, chunksize=chunksize):
            # Preprocess the chunk (drop rows missing non-home features)
            chunk = preprocess_data(chunk, allowed_features)
            # Augment the chunk by masking each home slot
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
    """Load allowed feature names from a text file (one per line)."""
    print("Loading allowed features from:", allowed_features_path)
    with open(allowed_features_path, 'r') as f:
        allowed_features = [line.strip() for line in f if line.strip()]
    print("Allowed features loaded:", allowed_features)
    return allowed_features

def load_metadata(metadata_path):
    """Load metadata from an Excel file (optional)."""
    print("Loading metadata from:", metadata_path)
    metadata_df = pd.read_excel(metadata_path)
    print("Metadata loaded. Columns:", metadata_df.columns.tolist())
    return metadata_df

def preprocess_data(df, feature_list):
    """
    Drop rows with missing values for the specified features.
    For test data, we do NOT want to drop rows missing home slot values.
    """
    print("Preprocessing data: dropping rows with missing values for features:", feature_list)
    home_slots = {"home_0", "home_1", "home_2", "home_3", "home_4"}
    features_to_check = [col for col in feature_list if col not in home_slots]
    df = df.dropna(subset=features_to_check)
    print("Preprocessing complete. Data shape:", df.shape)
    return df

def augment_training_data(df, home_cols=["home_0", "home_1", "home_2", "home_3", "home_4"]):
    """
    For each row in the dataframe, create one training instance per home slot by replacing
    that slot with "MASK" and using the original value as the target.
    """
    print("Augmenting data for", len(df), "rows.")
    instances = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        for col in home_cols:
            instance = row_dict.copy()
            instance["target"] = instance[col]  # Save original value as target
            instance[col] = "MASK"              # Mask the current home slot
            instance["mask_pos"] = col          # Record which slot was masked
            instances.append(instance)
    augmented_df = pd.DataFrame(instances)
    print("Augmentation complete. Augmented data shape:", augmented_df.shape)
    return augmented_df

def prepare_training_data(df, features_to_use, target_col="target"):
    """
    Prepare training data using the provided features (which should match between training and prediction)
    and the target column.
    """
    print("Preparing training data...")
    X = df[features_to_use].copy()
    y = df[target_col].copy()
    print("Training data prepared. X shape:", X.shape, "y shape:", y.shape)
    return X, y

def encode_categorical_columns(X, categorical_features):
    """Encode categorical features using LabelEncoder."""
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
    """Encode the target variable using LabelEncoder."""
    print("Encoding target variable.")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    print("Target encoding complete.")
    return y_encoded, le

def train_model(X, y):
    """Train a RandomForest classifier with reduced memory usage."""
    print("Training model using RandomForestClassifier with limited parallelism and depth...")
    clf = RandomForestClassifier(
        n_estimators=50,       # Reduceded number of trees
        max_depth=15,          # Limited tree depth
        random_state=42,
        n_jobs=1               # Used one core to reduce parallel overhead
    )
    clf.fit(X, y)
    print("Model training complete.")
    return clf

def predict_missing_player_for_row(model, row, features_to_use, categorical_features, feature_encoders, target_encoder):
    """
    For a single test row, detect the first home slot with missing data ('?' or NaN),
    replace it with "MASK", encode features, and predict the missing player.
    """
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
        value = str(row[col])
        if col in categorical_features and col in feature_encoders:
            le = feature_encoders[col]
            if value not in le.classes_:
                value = "unknown"
                if "unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "unknown")
            X_instance[col] = le.transform([value])[0]
        else:
            X_instance[col] = value
    feature_vector = [X_instance[col] for col in features_to_use]
    pred_encoded = model.predict([feature_vector])[0]
    pred = target_encoder.inverse_transform([pred_encoded])[0]
    row[missing_col] = pred
    row["Predicted_missing_slot"] = missing_col
    return row

def predict_missing_player(model, test_df, features_to_use, categorical_features, feature_encoders, target_encoder):
    """
    Generate predictions for missing home players in test data.
    Assumes each test row has at most one missing home slot.
    """
    print("Generating predictions on test data...")
    predicted_rows = []
    for idx, row in test_df.iterrows():
        row_pred = predict_missing_player_for_row(model, row, features_to_use, categorical_features, feature_encoders, target_encoder)
        predicted_rows.append(row_pred)
    pred_df = pd.DataFrame(predicted_rows)
    print("Predictions generated.")
    return pred_df

def filter_ineligible_players(pred_df, metadata_df):
    """
    Filter out predictions for players who are ineligible.
    Expects metadata_df to have columns 'Player' and 'Eligibility'.
    """
    print("Filtering ineligible players...")
    required_columns = {'Player', 'Eligibility'}
    if not required_columns.issubset(set(metadata_df.columns)):
        print("Warning: Metadata does not have required columns", required_columns, ". Skipping filtering.")
        return pred_df
    pred_df["Predicted_player"] = None
    for col in ["home_0", "home_1", "home_2", "home_3", "home_4"]:
        mask = pred_df["Predicted_missing_slot"] == col
        pred_df.loc[mask, "Predicted_player"] = pred_df.loc[mask, col]
    merged_df = pred_df.merge(metadata_df[['Player', 'Eligibility']],
                              left_on='Predicted_player', right_on='Player', how='left')
    eligible_df = merged_df[merged_df['Eligibility'] == 'Eligible']
    print("Filtering complete. Eligible predictions shape:", eligible_df.shape)
    return eligible_df

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("Starting pipeline...")
    data_folder = "data"
    allowed_features_path = os.path.join(data_folder, "allowed_features.txt")
    metadata_path = os.path.join(data_folder, "Matchup-metadata.xlsx")
    
    # Load allowed features.
    global allowed_features  # so that preprocess_data can access it
    allowed_features = load_allowed_features(allowed_features_path)
    
    # Load test data.
    test_file = os.path.join(data_folder, "NBA_test.csv")
    test_df = pd.read_csv(test_file)
    print("Test data shape:", test_df.shape)
    
    # Compute intersection: allowed features present in test data.
    common_features = [feat for feat in allowed_features if feat in test_df.columns]
    features_to_use = common_features.copy()
    print("Features to use for training/prediction:", features_to_use)
    
    metadata_df = load_metadata(metadata_path)
    
    # -----------------------------
    # Training Data Preparation with Chunking and Augmentation
    # -----------------------------
    print("Loading and augmenting training data...")
    training_augmented_df = load_and_augment_training_data(data_folder, exclude_files=["NBA_test"], chunksize=CHUNKSIZE)
    
    X, y = prepare_training_data(training_augmented_df, features_to_use, target_col="target")
    
    candidate_categorical = ['game', 'season', 'home_team', 'away_team',
                             'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                             'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
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
    
    # Optional cross-validation on a subsample.
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
    
    # -----------------------------
    # Prediction on Entire Test File
    # -----------------------------
    print("Replacing '?' with NaN in test data...")
    test_df.replace('?', np.nan, inplace=True)
    
    allowed_cols = ['game', 'season', 'home_team', 'away_team', 'starting_min',
                    'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                    'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
    common_cols = [col for col in allowed_cols if col in test_df.columns]
    print("Using these columns from test data:", common_cols)
    test_df = test_df[common_cols]
    
    print("Preprocessing test data...")
    test_df = preprocess_data(test_df, features_to_use)
    
    print("Generating predictions on test data...")
    test_predictions = predict_missing_player(clf, test_df, features_to_use, categorical_features, feature_encoders, target_encoder)
    
    # Build final output with home_team, predicted missing slot, and predicted player.
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
    
    # -----------------------------
    # Compare Predictions with Actual Labels
    # -----------------------------
    labels_file = os.path.join(data_folder, "NBA_test_labels.csv")
    if os.path.exists(labels_file):
        print("Loading test labels from:", labels_file)
        test_labels = pd.read_csv(labels_file)
        # Look for a column with actual missing player info: first try "Actual_player", then "home_4"
        if "Actual_player" in test_labels.columns and len(test_labels) >= len(final_output):
            final_output["Actual_player"] = test_labels["Actual_player"].iloc[:len(final_output)].values
        elif "home_4" in test_labels.columns and len(test_labels) >= len(final_output):
            final_output["Actual_player"] = test_labels["home_4"].iloc[:len(final_output)].values
        else:
            print("Warning: Test labels file does not have an expected column ('Actual_player' or 'home_4') or enough rows.")
        print("Comparison of predicted vs actual (first 20 rows):")
        print(final_output.head(20))
    else:
        print("No test labels file found. Final predictions:")
        print(final_output.head(20))
    
    print("Saving final predictions to predicted_fifth_players.csv...")
    final_output.to_csv("predicted_fifth_players.csv", index=False)
    print("Predictions saved to predicted_fifth_players.csv")
    
    print("Pipeline complete. Exiting.")

if __name__ == "__main__":
    main()
