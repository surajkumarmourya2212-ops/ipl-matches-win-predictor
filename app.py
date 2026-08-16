import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as file:
    pipe = pickle.load(file)


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 IPL Win Probability Predictor")
st.write("Predict the winning probability of the batting team.")


# -----------------------------
# Teams
# -----------------------------
teams = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]


# -----------------------------
# User inputs
# -----------------------------
batting_team = st.selectbox(
    "Batting Team",
    sorted(teams)
)

bowling_team = st.selectbox(
    "Bowling Team",
    sorted(teams)
)

city = st.selectbox(
    "City",
    [
        "Mumbai",
        "Delhi",
        "Kolkata",
        "Chennai",
        "Bangalore",
        "Hyderabad",
        "Ahmedabad",
        "Jaipur",
        "Pune",
        "Lucknow"
    ]
)

target = st.number_input(
    "Target Score",
    min_value=1,
    value=180
)

score = st.number_input(
    "Current Score",
    min_value=0,
    value=50
)

overs = st.number_input(
    "Overs Completed",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.1
)

wickets = st.number_input(
    "Wickets Lost",
    min_value=0,
    max_value=10,
    value=2
)


# -----------------------------
# Calculate features
# -----------------------------
runs_left = target - score

balls_left = 120 - int(overs * 6)

wickets_remaining = 10 - wickets

if overs > 0:
    crr = score / overs
else:
    crr = 0

if balls_left > 0:
    rrr = (runs_left * 6) / balls_left
else:
    rrr = 0


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Win Probability"):

    if batting_team == bowling_team:
        st.error("Batting team and bowling team cannot be the same.")

    elif balls_left <= 0:
        st.error("The innings is over.")

    else:

        input_df = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [city],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets": [wickets_remaining],
            "total_runs_x": [target],
            "crr": [crr],
            "rrr": [rrr]
        })

        # Prediction probability
        probabilities = pipe.predict_proba(input_df)[0]

        # Probability of class 0 and class 1
        team1_probability = probabilities[0] * 100
        team2_probability = probabilities[1] * 100

        st.success("Prediction completed!")

        st.subheader("🏆 Win Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                batting_team,
                f"{team2_probability:.2f}%"
            )

        with col2:
            st.metric(
                bowling_team,
                f"{team1_probability:.2f}%"
            )

        st.progress(
            int(team2_probability)
        )

        st.write(
            f"**Current Run Rate:** {crr:.2f}"
        )

        st.write(
            f"**Required Run Rate:** {rrr:.2f}"
        )

        st.write(
            f"**Runs Left:** {runs_left}"
        )

        st.write(
            f"**Balls Left:** {balls_left}"
        )

        st.write(
            f"**Wickets Remaining:** {wickets_remaining}"
        )
