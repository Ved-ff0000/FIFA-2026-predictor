import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:\\Users\\Vedit\\OneDrive\\Desktop\\world_cup_last_50_years.csv")

# Encode teams
team_encoder = LabelEncoder()

all_teams = pd.concat([
    df["home_team"],
    df["away_team"]
]).unique()

team_encoder.fit(all_teams)

df["home_team_enc"] = team_encoder.transform(df["home_team"])
df["away_team_enc"] = team_encoder.transform(df["away_team"])

# Encode stage
stage_encoder = LabelEncoder()
df["stage_enc"] = stage_encoder.fit_transform(df["stage"])

# Create result label
def get_result(row):
    if row["home_goals"] > row["away_goals"]:
        return "Home Win"
    elif row["home_goals"] < row["away_goals"]:
        return "Away Win"
    else:
        return "Draw"

df["result"] = df.apply(get_result, axis=1)

# Encode target
result_encoder = LabelEncoder()
df["result_enc"] = result_encoder.fit_transform(df["result"])

# Features
X = df[
    [
        "home_team_enc",
        "away_team_enc",
        "stage_enc",
        "year"
    ]
]

y = df["result_enc"]

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

# Save everything
joblib.dump(model, "fifa_predictor.pkl")
joblib.dump(team_encoder, "team_encoder.pkl")
joblib.dump(stage_encoder, "stage_encoder.pkl")
joblib.dump(result_encoder, "result_encoder.pkl")

print("Saved successfully!")