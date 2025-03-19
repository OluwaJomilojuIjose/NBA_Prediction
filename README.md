NBA Missing Player Predictor

Predicts the missing player in an NBA team's starting lineup using RandomForestClassifier.

Project Objective
This project aims to predict missing players from NBA starting lineups based on historical game data. It leverages machine learning, specifically a RandomForest model, to identify which player is missing from the home team lineup.

The pipeline:

Prepares and augments training data.
Trains a RandomForestClassifier on augmented data.
Predicts the missing player in test data.
Computes seasonal and overall accuracy.

Final Results:
The model achieved the following accuracy by season:

Season	Accuracy
2007	60.00%
2008	63.00%
2009	55.00%
2010	61.00%
2011	67.00%
2012	62.00%
2013	62.00%
2014	70.00%
2015	57.00%
2016	24.00%


Folder Structure:
bash
Copy
Edit
project-root/
├── data/
│   ├── NBA_train_*.csv      # Training data files
│   ├── NBA_test.csv         # Test data (incomplete lineups)
│   ├── NBA_test_labels.csv  # Actual missing players for accuracy checking
│   ├── allowed_features.txt # List of allowed feature names (one per line)
│   └── Matchup-metadata.xlsx # Optional metadata (not currently used)
├── model.pkl                # Saved model and encoders
├── predicted_fifth_players.csv # Output predictions with actual labels
├── your_script.py           # Main Python pipeline (rename appropriately)
└── README.md


Requirements:

Python 3.7+
pandas
numpy
scikit-learn
joblib
openpyxl (for loading Excel files)

Install them with:

pip install pandas numpy scikit-learn joblib openpyxl


How to Run the Pipeline:

Prepare Data Folder

Place your NBA_train_*.csv files and NBA_test.csv in the data/ folder.
Include allowed_features.txt (features to use).
Include NBA_test_labels.csv for accuracy checking.
Run the Python Script

Outputs:

model.pkl: Trained model and encoders.
predicted_fifth_players.csv: Predictions along with actual players (if labels provided).

Console outputs show:

Accuracy by season
Overall accuracy

Data Explanation:
Training CSVs: Complete game data with all home and away player slots filled.
Test CSV: Games with one missing player from home_0 through home_4.
Labels CSV: Correct Values for the missing player in Test CSV file (aligned by row).

What the Code Does:

Preprocess Training Data:
Keeps lineup slots even if incomplete.
Augment Training Data: Creates training examples by masking each home player one at a time and storing the original as the target.
Categorical features like team names, players, etc. are encoded.
RandomForestClassifier is trained on augmented data.
Identifies the missing home slot in the Test data.
Predicts the player who should be in that slot.
Compares predictions to actual missing players.
Reports per-season accuracy and overall accuracy.
