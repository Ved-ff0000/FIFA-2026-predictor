import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fifa_predictor.pkl")
team_encoder = joblib.load("team_encoder.pkl")
stage_encoder = joblib.load("stage_encoder.pkl")
result_encoder = joblib.load("result_encoder.pkl")

st.title("⚽ FIFA World Cup 2026 Predictor")

teams = sorted(team_encoder.classes_)
stages = sorted(stage_encoder.classes_)

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)
stage = st.selectbox("Stage", stages)

year = st.number_input(
    "Tournament Year",
    min_value=1974,
    max_value=2030,
    value=2026
)

if st.button("Predict Match"):

    home_enc = team_encoder.transform([home_team])[0]
    away_enc = team_encoder.transform([away_team])[0]
    stage_enc = stage_encoder.transform([stage])[0]

    data = pd.DataFrame(
        [[home_enc, away_enc, stage_enc, year]],
        columns=[
            "home_team_enc",
            "away_team_enc",
            "stage_enc",
            "year"
        ]
    )

    prediction = model.predict(data)[0]

    result = result_encoder.inverse_transform(
        [prediction]
    )[0]

    st.success(f"Prediction: {result}")

    probs = model.predict_proba(data)[0]

    st.subheader("Probabilities")

    for label, prob in zip(
        result_encoder.classes_,
        probs
    ):
        st.write(
            f"{label}: {prob*100:.2f}%"
        )

    import matplotlib.pyplot as plt

    # Create dataframe for plotting
    prob_df = pd.DataFrame({
        "Result": result_encoder.classes_,
        "Probability": probs * 100
    })

    st.subheader("📊 Match Outcome Probabilities")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        prob_df["Result"],
        prob_df["Probability"]
    )

    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Outcome")
    ax.set_title("Predicted Match Outcome")

    st.pyplot(fig)