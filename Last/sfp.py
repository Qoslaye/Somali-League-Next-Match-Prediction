import streamlit as st
import pandas as pd
import pickle
import os
from scipy.stats import poisson
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler

# Add enhanced CSS styling
st.markdown("""
    <style>
    .centered-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .button-container {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin: 20px 0;
    }

    .prediction-results {
        background: white;
        padding: 30px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .section-title {
        color: #2c3e50;
        font-size: 24px;
        font-weight: 600;
        text-align: center;
        margin: 25px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #eef2f7;
    }
    
    .prediction-bar-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .prediction-bar-wrapper {
        height: 12px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .home-segment {
        background: #2563eb;
    }
    
    .draw-segment {
        background: #7c3aed;
    }
    
    .away-segment {
        background: #16a34a;
    }
    
    .match-result {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    
    .match-result:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .result-indicator {
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    
    .result-win {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    
    .result-loss {
        background-color: #ffebee;
        color: #c62828;
    }
    
    .result-draw {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin: 20px 0;
        text-align: center;
    }
    
    .stat-item {
        padding: 15px 25px;
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 14px;
    }
    
    .team-name {
        font-weight: 600;
        color: #2c3e50;
    }
    
    .percentage-labels {
        display: flex;
        justify-content: space-between;
        padding: 10px 20px;
        color: #64748b;
        font-size: 14px;
        font-weight: 500;
    }
    
    .prediction-percentages {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        font-weight: 600;
    }
    
    .percentage-label {
        font-size: 14px;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .home-percentage {
        color: #2563eb;
    }
    
    .draw-percentage {
        color: #7c3aed;
    }
    
    .away-percentage {
        color: #16a34a;
    }
    </style>
""", unsafe_allow_html=True)

# Wrap your existing content in the centered container
st.markdown('<div class="centered-container">', unsafe_allow_html=True)

# Load df_team_strength from pickle
try:
    with open('df_team_strength.pkl', 'rb') as file:
        df_team_strength = pickle.load(file)
except FileNotFoundError:
    st.error("Could not find 'df_team_strength.pkl'. Please ensure the file exists in the script directory.")
    st.stop()

# Ensure index is clean
df_team_strength.index = df_team_strength.index.str.replace(r'\s+', ' ', regex=True).str.strip()

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Cleaned4_Somali_league_datasets.csv")

try:
    df = pd.read_csv(file_path)
    print("✅ CSV Loaded Successfully!")
except FileNotFoundError:
    st.error("Could not find 'Cleaned4_Somali_league_datasets.csv'. Please ensure the file exists in the script directory.")
    st.stop()

# Load Logistic Regression model and scaler
try:
    with open('logistic_regression.pkl', 'rb') as f:
        log_reg = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Could not load a required file: {e}. Please ensure 'logistic_regression.pkl' and 'scaler.pkl' are present.")
    st.stop()

# Define feature columns
feature_columns = [
    'Home_Score', 'Away_Score', 'Home_Goal_Diff', 'Away_Goal_Diff', 
    'Home_Last_5_Wins', 'Away_Last_5_Wins', 'Home_Last_5_Points', 
    'Away_Last_5_Points', 'H2H_Home_Wins', 'H2H_Home_Losses', 
    'H2H_Away_Wins', 'H2H_Away_Losses', 'H2H_Draws'
]

# Functions for validation and data processing
def validate_teams(home_team, away_team, df):
    home_exists = home_team in df['Home_Team'].values or home_team in df['Away_Team'].values
    away_exists = away_team in df['Away_Team'].values or away_team in df['Home_Team'].values
    if not home_exists:
        print(f"Error: '{home_team}' is not a valid team in the Somali League dataset.")
        return False
    if not away_exists:
        print(f"Error: '{away_team}' is not a valid team in the Somali League dataset.")
        return False
    if home_team == away_team:
        print("Error: Home Team and Away Team cannot be the same.")
        return False
    return True

def get_h2h_info(home_team, away_team, df):
    h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                     ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))]
    
    if h2h_matches.empty:
        return {'total_matches': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0}
    
    h2h_home = df[(df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)]
    if not h2h_home.empty:
        latest_h2h = h2h_home.iloc[-1]
        home_wins = int(latest_h2h['H2H_Home_Wins'])
        home_losses = int(latest_h2h['H2H_Home_Losses'])
        away_wins = int(latest_h2h['H2H_Away_Wins'])
        away_losses = int(latest_h2h['H2H_Away_Losses'])
        draws = int(latest_h2h['H2H_Draws'])
        total_matches = home_wins + home_losses + draws
    else:
        total_matches = len(h2h_matches)
        home_wins_as_home = len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 0)])
        away_wins_as_away = len(h2h_matches[(h2h_matches['Away_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
        draws = len(h2h_matches[h2h_matches['Result_Encoded'] == 1])
        home_wins = home_wins_as_home + len(h2h_matches[(h2h_matches['Home_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
        away_wins = away_wins_as_away + len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 2)])

    return {'total_matches': total_matches, 'home_wins': home_wins, 'away_wins': away_wins, 'draws': draws}

def get_team_form(team, df):
    team_matches = df[(df['Home_Team'] == team) | (df['Away_Team'] == team)].sort_index(ascending=False).head(5)
    
    if team_matches.empty:
        return {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0}
    
    wins = 0
    draws = 0
    losses = 0
    points = 0
    
    for _, match in team_matches.iterrows():
        if match['Home_Team'] == team:
            if match['Result_Encoded'] == 0:  # Home win
                wins += 1
                points += 3
            elif match['Result_Encoded'] == 1:  # Draw
                draws += 1
                points += 1
            else:  # Away win (loss for home team)
                losses += 1
        else:  # Team is Away_Team
            if match['Result_Encoded'] == 2:  # Away win
                wins += 1
                points += 3
            elif match['Result_Encoded'] == 1:  # Draw
                draws += 1
                points += 1
            else:  # Home win (loss for away team)
                losses += 1
    
    return {'wins': wins, 'draws': draws, 'losses': losses, 'points': points}

def get_recent_match_result(home_team, away_team, df):
    h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                     ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))].sort_index(ascending=False)
    
    if h2h_matches.empty:
        return None
    
    latest_match = h2h_matches.iloc[0]
    home_team_match = latest_match['Home_Team']
    away_team_match = latest_match['Away_Team']
    home_score = int(latest_match['Home_Score'])
    away_score = int(latest_match['Away_Score'])
    season = latest_match['Season']
    
    return {'home_team': home_team_match, 'away_team': away_team_match, 'home_score': home_score, 'away_score': away_score, 'season': season}

def prepare_input(home_team, away_team, df, feature_columns):
    home_data = df[df['Home_Team'] == home_team]
    away_data = df[df['Away_Team'] == away_team]
    
    if home_data.empty:
        home_data = df
        print(f"Warning: No historical data for {home_team} as Home Team. Using dataset averages.")
    if away_data.empty:
        away_data = df
        print(f"Warning: No historical data for {away_team} as Away Team. Using dataset averages.")
    
    home_means = home_data.mean(numeric_only=True)
    away_means = away_data.mean(numeric_only=True)
    
    features_dict = {}
    for col in feature_columns:
        if 'Home' in col:
            features_dict[col] = home_means.get(col, df[col].mean()) if col in home_means else df[col].mean()
        elif 'Away' in col:
            features_dict[col] = away_means.get(col, df[col].mean()) if col in away_means else df[col].mean()
        else:
            features_dict[col] = (home_means.get(col, 0) + away_means.get(col, 0)) / 2 if col in home_means else df[col].mean()
    
    input_data = pd.DataFrame([features_dict], columns=feature_columns)
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

def predict_points(home, away):
    if home in df_team_strength.index and away in df_team_strength.index:
        lamb_home = df_team_strength.at[home, 'GoalsScored'] * df_team_strength.at[away, 'GoalsConceded']
        lamb_away = df_team_strength.at[away, 'GoalsScored'] * df_team_strength.at[home, 'GoalsConceded']
        prob_home, prob_away, prob_draw = 0, 0, 0
        
        for x in range(0, 11):  # Number of goals home team
            for y in range(0, 11):  # Number of goals away team
                p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                if x == y:
                    prob_draw += p
                elif x > y:
                    prob_home += p
                else:
                    prob_away += p
        
        points_home = 3 * prob_home + prob_draw
        points_away = 3 * prob_away + prob_draw
        return points_home, points_away, prob_home, prob_draw, prob_away
    else:
        return 0, 0, 0, 0, 0

# Function to get team logo path
def get_team_logo(team_name):
    logo_folder = "TeamLogo"
    
    if team_name == "Heegan S.C":
        team_name = "Heegan SC"
    elif team_name == "Horseed S.C":
        team_name = "Horseed SC"
    
    for ext in ["png", "jpeg", "jpg"]:
        logo_path = os.path.join(logo_folder, f"{team_name}.{ext}")
        if os.path.exists(logo_path):
            return logo_path
    
    return None

# Function to create circular logos
def create_circular_logo(image_path, size=(150, 150)):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize(size, Image.LANCZOS)
    
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    
    circular_image = Image.new("RGBA", size, (255, 255, 255, 0))
    circular_image.paste(image, (0, 0), mask)
    
    return circular_image

# Function to get recent matches
def get_recent_matches(team, df, num_matches=5):
    # Get matches where the team played (either home or away)
    team_matches = df[(df['Home_Team'] == team) | (df['Away_Team'] == team)].sort_index(ascending=False).head(num_matches)
    recent_matches = []
    
    for _, match in team_matches.iterrows():
        is_home = match['Home_Team'] == team
        opponent = match['Away_Team'] if is_home else match['Home_Team']
        score = f"{match['Home_Score']}-{match['Away_Score']}"
        
        # Determine the result from the team's perspective
        if is_home:
            if match['Home_Score'] > match['Away_Score']:
                result = 'win'
            elif match['Home_Score'] < match['Away_Score']:
                result = 'loss'
            else:
                result = 'draw'
        else:
            if match['Home_Score'] < match['Away_Score']:
                result = 'win'
            elif match['Home_Score'] > match['Away_Score']:
                result = 'loss'
            else:
                result = 'draw'
        
        recent_matches.append({
            'opponent': opponent,
            'score': score,
            'result': result,
            'season': match['Season'],
            'is_home': is_home
        })
    
    return recent_matches

# Streamlit UI
st.title("Somali Football Match Predictor")

team_names = ["Select a team"] + df_team_strength.index.tolist()

if "home_team" not in st.session_state:
    st.session_state.home_team = "Select a team"

if "away_team" not in st.session_state:
    st.session_state.away_team = "Select a team"

col1, col2 = st.columns(2)

with col1:
    home_logo = get_team_logo(st.session_state.home_team) if st.session_state.home_team != "Select a team" else None
    if home_logo:
        st.image(create_circular_logo(home_logo), caption=st.session_state.home_team, use_container_width=False)

with col2:
    away_logo = get_team_logo(st.session_state.away_team) if st.session_state.away_team != "Select a team" else None
    if away_logo:
        st.image(create_circular_logo(away_logo), caption=st.session_state.away_team, use_container_width=False)

sel1, sel2 = st.columns(2)

with sel1:
    home_team = st.selectbox("Select Home Team", options=team_names, index=team_names.index(st.session_state.home_team), key='home_team')

if home_team != "Select a team":
    away_team_options = ["Select a team"] + [team for team in df_team_strength.index if team != home_team]
    if st.session_state.away_team == home_team or st.session_state.away_team not in away_team_options:
        st.session_state.away_team = away_team_options[1] if len(away_team_options) > 1 else "Select a team"
else:
    away_team_options = ["Select a team"] + df_team_strength.index.tolist()

with sel2:
    away_team = st.selectbox("Select Away Team", options=away_team_options, index=away_team_options.index(st.session_state.away_team), key='away_team')

def clear_selection():
    st.session_state.home_team = "Select a team"
    st.session_state.away_team = "Select a team"

# Center the buttons
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("Predict Match Outcome")
with col2:
    clear_button = st.button("Clear Selection", on_click=clear_selection)
st.markdown('</div>', unsafe_allow_html=True)

# Show prediction results
if predict_button:
    if home_team != "Select a team" and away_team != "Select a team" and validate_teams(home_team, away_team, df):
        st.markdown('<div class="prediction-results">', unsafe_allow_html=True)
        
        # Calculate predictions
        h2h_info = get_h2h_info(home_team, away_team, df)
        home_form = get_team_form(home_team, df)
        away_form = get_team_form(away_team, df)
        
        # Calculate probabilities
        input_data = prepare_input(home_team, away_team, df, feature_columns)
        probabilities = log_reg.predict_proba(input_data)[0]
        home_prob, draw_prob, away_prob = probabilities * 100

        # Display match prediction with clear percentages
        st.markdown('<div class="section-title">Match Prediction</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="prediction-bar-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span class="team-name">{home_team}</span>
                    <span class="team-name">{away_team}</span>
                </div>
                <div class="prediction-bar-wrapper">
                    <div class="prediction-segment home-segment" style="width: {home_prob}%;"></div>
                    <div class="prediction-segment draw-segment" style="width: {draw_prob}%;"></div>
                    <div class="prediction-segment away-segment" style="width: {away_prob}%;"></div>
                </div>
                <div class="prediction-percentages">
                    <span class="percentage-label home-percentage">Home Win: {home_prob:.1f}%</span>
                    <span class="percentage-label draw-percentage">Draw: {draw_prob:.1f}%</span>
                    <span class="percentage-label away-percentage">Away Win: {away_prob:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Display Head to Head
        if h2h_info['total_matches'] > 0:
            st.markdown('<div class="section-title">Head to Head</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="stats-container">
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['home_wins']}</div>
                        <div class="stat-label">{home_team} Wins</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['draws']}</div>
                        <div class="stat-label">Draws</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['away_wins']}</div>
                        <div class="stat-label">{away_team} Wins</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Display Recent Form
        st.markdown('<div class="section-title">Recent Form</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div style="font-weight: 600; margin-bottom: 15px; color: #2c3e50; font-size: 16px;">{home_team} Last 5 Matches</div>', unsafe_allow_html=True)
            home_recent = get_recent_matches(home_team, df)
            for match in home_recent:
                result_class = f"result-{match['result']}"
                home_away_indicator = "HOME" if match['is_home'] else "AWAY"
                if not match['is_home']:
                    score_parts = match['score'].split('-')
                    match_score = f"{score_parts[1]}-{score_parts[0]}"
                else:
                    match_score = match['score']
                
                st.markdown(f"""
                    <div class="match-result">
                        <div style="display: flex; flex-direction: column; flex: 1;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: 600; color: #2c3e50;">{home_team if match['is_home'] else match['opponent']}</span>
                                <span style="font-weight: 600; color: #2c3e50;">{match_score}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 600; color: #2c3e50;">{match['opponent'] if match['is_home'] else home_team}</span>
                                <span style="color: #64748b; font-size: 12px;">{home_away_indicator}</span>
                            </div>
                        </div>
                        <span class="result-indicator {result_class}" style="margin-left: 10px;">{match['result'].upper()}</span>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div style="font-weight: 600; margin-bottom: 15px; color: #2c3e50; font-size: 16px;">{away_team} Last 5 Matches</div>', unsafe_allow_html=True)
            away_recent = get_recent_matches(away_team, df)
            for match in away_recent:
                result_class = f"result-{match['result']}"
                home_away_indicator = "HOME" if match['is_home'] else "AWAY"
                if not match['is_home']:
                    score_parts = match['score'].split('-')
                    match_score = f"{score_parts[1]}-{score_parts[0]}"
                else:
                    match_score = match['score']
                
                st.markdown(f"""
                    <div class="match-result">
                        <div style="display: flex; flex-direction: column; flex: 1;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: 600; color: #2c3e50;">{away_team if match['is_home'] else match['opponent']}</span>
                                <span style="font-weight: 600; color: #2c3e50;">{match_score}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 600; color: #2c3e50;">{match['opponent'] if match['is_home'] else away_team}</span>
                                <span style="color: #64748b; font-size: 12px;">{home_away_indicator}</span>
                            </div>
                        </div>
                        <span class="result-indicator {result_class}" style="margin-left: 10px;">{match['result'].upper()}</span>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close prediction-results
    else:
        st.error("Please select valid teams.")

st.markdown('</div>', unsafe_allow_html=True)  # Close centered-container