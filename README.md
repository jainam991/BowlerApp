# 🏏 Bowler Workload Platform

> Real-data cricket analytics platform for fast-bowler performance, workload monitoring, matchup analysis, injury/fitness management and decision support.

---

## 📌 Overview

**Bowler Workload Platform** is an interactive cricket analytics application built using Python and Streamlit.

The project takes real ball-by-ball cricket data and transforms it into a structured analytics platform focused primarily on **bowler performance and workload management**.

Instead of only showing traditional cricket statistics such as wickets and economy, the application combines:

- Performance analytics
- Workload analytics
- ACWR-based workload monitoring
- Rest and recent workload tracking
- Matchup analysis
- Player comparison
- Team analysis
- Injury and fitness management
- Return-to-play planning
- Squad-level workload impact
- Bowler selection support
- Machine-learning matchup predictions
- Rule-based data assistant
- CSV exports

The overall objective is to move from:

> **"What did the bowler do?"**

towards:

> **"How is the bowler performing, how much workload are they carrying, what could happen if their workload increases, and what decision should the team consider?"**

---

# 🎯 Project Objective

The main objective is to build a **Fast Bowler Workload, Performance & Fitness Decision-Support Platform**.

Fast bowling involves repeated high physical workloads. Therefore, simply looking at wickets or economy is not enough.

The platform attempts to bring together three major areas:

```text
                  BOWLER ANALYTICS
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     PERFORMANCE      WORKLOAD       FITNESS
          │              │              │
     Wickets         ACWR           Injury Log
     Economy         Rest           Status
     Matchups        Monotony       Rehab
     Form            Strain         Return-to-play
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                DECISION SUPPORT
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Selection      Risk Alerts    Squad Impact
```

---

# 🧠 What Makes This Project Different?

Most basic cricket dashboards focus on:

- Runs
- Wickets
- Economy
- Strike rate
- Match statistics

This project goes further by combining **performance + workload + fitness + decision support**.

For example:

A bowler may have excellent historical performance but currently have a high ACWR.

Instead of simply saying:

> "This bowler is performing well."

the system can show:

> "This bowler performs well, but their current workload is significantly above their recent normal."

The project therefore tries to provide **context around performance**, rather than performance statistics alone.

---

# 📊 Data Pipeline

The application is designed around real ball-by-ball cricket data.

The high-level pipeline is:

```text
Raw Ball-by-Ball Dataset
          │
          ▼
    preprocess.py
          │
          ▼
 ┌─────────────────────────┐
 │ Processed CSV datasets  │
 ├─────────────────────────┤
 │ bowler_master.csv       │
 │ bowler_match_summary.csv│
 │ bowler_vs_team_summary.csv
 └─────────────────────────┘
          │
          ▼
       app.py
          │
          ▼
   Streamlit Dashboard
          │
          ▼
 Analytics + Decision Support
```

The Streamlit application expects three main processed CSV files:

```text
bowler_match_summary.csv
bowler_vs_team_summary.csv
bowler_master.csv
```

These are either loaded from the same directory as `app.py` or uploaded through the sidebar.

---

# 📁 Project Structure

Recommended repository structure:

```text
Bowlerapp/
│
├── app.py
├── preprocess.py
├── requirements.txt
├── README.md
│
├── bowler_master.csv
├── bowler_match_summary.csv
├── bowler_vs_team_summary.csv
├── injury_log.csv
│
├── .gitignore
│
└── venv/
```

### Important

`venv/` should **NOT** be uploaded to GitHub.

Recommended `.gitignore`:

```text
venv/
__pycache__/
*.pyc
.DS_Store
.env
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing and aggregation |
| NumPy | Numerical calculations |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive charts |
| Matplotlib | Custom visualisations |
| Seaborn | Statistical visualisations |
| Scikit-learn | Machine-learning matchup prediction |
| CSV | Processed data storage |
| HTML/CSS | Custom dashboard styling |

---

# 📦 Main Python Libraries

The application imports:

```python
numpy
pandas
matplotlib
seaborn
plotly
streamlit
scikit-learn
```

The scikit-learn components currently used are:

```python
RandomForestRegressor
train_test_split
mean_absolute_error
r2_score
LabelEncoder
```

---

# 🚀 Running the Project

## 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd Bowlerapp
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install requirements

```bash
pip install -r requirements.txt
```

If requirements.txt needs to be recreated:

```bash
pip install pandas numpy streamlit matplotlib seaborn plotly scikit-learn
```

---

# ▶️ Running the Application

Once the processed CSV files are available:

```bash
streamlit run app.py
```

The application will open locally in the browser.

Usually:

```text
http://localhost:8501
```

---

# 📂 Required Data Files

The application requires:

### 1. `bowler_match_summary.csv`

This is the main analytical dataset.

It contains bowler-match level information used for:

- Match statistics
- Workload
- ACWR
- Rest days
- Recent match count
- Economy
- Wickets
- Opponent analysis
- Player profiles
- Team analysis
- ML features

---

### 2. `bowler_vs_team_summary.csv`

This contains bowler-vs-opponent information.

The application also recomputes the opponent summary dynamically from `bowler_match_summary.csv` so that the selected format/country filters are correctly reflected.

The core calculation groups data by:

```text
bowler
+
opponent team
```

and calculates:

```text
Average economy
Average wickets
Matches played
Performance score
```

---

### 3. `bowler_master.csv`

This is the player/master dataset.

It provides information such as:

```text
Bowler ID
Full name
Country
Bowling style
Image URL
Team information
```

The application merges this information with the match summary to create the display name and player metadata.

---

### 4. `injury_log.csv`

This is the project's persistent injury/fitness record.

It contains:

```text
log_id
bowler
display_name
date_reported
status
injury_type
body_part
expected_return_date
notes
source
```

Unlike normal calculated analytics, this file is intended to be updated by users/staff over time.

---

# 🔄 Data Loading Architecture

The application first checks whether the required CSVs exist locally.

If a file is missing, the sidebar provides a CSV upload option.

Therefore there are two possible workflows:

### Local files

```text
app.py
bowler_master.csv
bowler_match_summary.csv
bowler_vs_team_summary.csv
```

### Sidebar upload

```text
Open application
       ↓
Missing file detected
       ↓
Upload CSV from sidebar
       ↓
Application loads uploaded data
```

This makes the application easier to demonstrate on another computer.

---

# 🏏 Application Navigation

The application currently contains the following major pages:

```text
Home
Injury & Fitness
Squad Impact Engine
ACWR Engine
Dataset Overview
Player Profile
Compare Players
Advanced Search
Leaderboard
Bowler Selection
Workload Monitor
Team Overview
Ask the Data
Export Data
Methodology
```

---

# 1️⃣ Home

The Home page acts as the main dashboard.

It provides high-level KPIs:

- Number of bowlers
- Number of matches
- Total wickets
- Average economy
- Number of bowlers currently in High ACWR risk

It also shows:

### Top wicket-takers

Ranks bowlers according to total wickets.

### Most economical bowlers

Shows bowlers with the lowest average economy, with a minimum of 3 matches.

### Bowling style distribution

Shows the distribution of bowling styles.

### Matches per season

Shows how many matches exist across seasons.

The top-wicket and economy charts are interactive and can be used to jump directly to a player's profile.

---

# 2️⃣ ACWR Engine

The ACWR Engine is one of the core workload-analysis components.

## What is ACWR?

ACWR stands for:

> **Acute:Chronic Workload Ratio**

The application uses:

```text
ACWR = Acute workload / Chronic workload
```

where:

```text
Acute workload
=
Overs bowled in the latest match

Chronic workload
=
Average overs bowled over the previous 4 matches
```

---

## ACWR Zones

The application currently categorizes ACWR as:

| ACWR | Tier |
|---:|---|
| `< 0.8` | Undertrained |
| `0.8 – 1.3` | Low |
| `1.3 – 1.5` | Moderate |
| `> 1.5` | High |

These thresholds are used consistently throughout the application.

---

## ACWR Gauge

A custom semi-circle gauge visualizes the current ACWR.

The gauge contains:

```text
Undertrained
      ↓
Low / Sweet Spot
      ↓
Moderate
      ↓
High
```

---

## What-if Simulator

The user can select a bowler and enter hypothetical overs for the next match.

The application calculates:

```text
Hypothetical ACWR
=
Hypothetical overs
/
Recent 4-match average overs
```

This lets the user see how a potential bowling workload would affect the bowler's workload tier.

---

## Newly Flagged Bowlers

The system compares the previous and current ACWR tier.

Example:

```text
Previous: Low
Current:  High
```

This is flagged as a newly worsened risk state.

This is useful because a static ACWR value does not always show that the risk **just changed**.

---

# 3️⃣ Workload Monitor

The Workload Monitor focuses specifically on workload risk.

It provides:

- ACWR risk distribution
- ACWR distribution across bowlers
- List of currently High-risk bowlers
- Latest overs
- Rest days before latest match

High-risk means:

```text
ACWR > 1.5
```

The purpose is to quickly identify bowlers whose current match workload is substantially above their recent normal.

---

# 4️⃣ Player Profile

The Player Profile page provides a detailed view of one bowler.

It includes:

### Player information

- Name
- Country
- Bowling style
- Profile image if available
- Current fitness status
- Current ACWR tier

### Current workload

- ACWR
- Safe overs ceiling
- Latest rest days
- Matches on record
- Wickets
- Economy

### Milestones

The application identifies:

- Best wicket-taking match
- Best economy match

### Workload history

Shows:

```text
Overs bowled over time
+
ACWR over time
```

This allows us to see workload spikes instead of only looking at the latest number.

### Seasonal performance

For players with season information, the application calculates:

- Wickets by season
- Average economy by season
- Cumulative wickets

### Performance against opponents

The player's performance against different opposition is shown using:

- Average economy
- Average wickets
- Number of matches
- Performance score

---

# 5️⃣ Compare Players

This page allows comparison of 2 or 3 bowlers.

For two-player comparisons, the system compares:

- Wickets
- Economy
- Latest overs
- ACWR safety

For three-player comparisons, it provides a table.

---

## Radar Comparison

Players are compared across:

```text
Wickets
Economy
Overs bowled
ACWR safety
Consistency
```

The values are normalized from 0–1 relative to the currently selected player pool.

Important:

> Radar scores are relative to the current comparison group, not absolute career ratings.

---

# 6️⃣ Advanced Search

Advanced Search allows the user to filter bowlers using multiple criteria.

Available filters include:

- Country
- Bowling style
- Minimum matches
- Minimum wickets
- Economy range
- ACWR risk tier

The resulting player table contains:

```text
Player
Country
Bowling style
Matches
Wickets
Average economy
ACWR
ACWR tier
```

The filtered results can also be downloaded as CSV.

---

# 7️⃣ Leaderboard

The Leaderboard contains six different ranking systems.

## Best Performers

Ranks players using average matchup performance score.

---

## Safest ACWR

Shows bowlers currently in the Low ACWR tier, sorted by ACWR.

---

## Highest Workload

Ranks bowlers by current ACWR.

---

## Most Wickets

Ranks players according to total wickets in the current filter.

---

## Most Consistent

Consistency is calculated using:

```text
Standard deviation of economy across matches
```

Lower standard deviation:

```text
More consistent
```

Minimum:

```text
3 matches
```

---

## Rising Stars

The application compares:

```text
First recorded season average economy
            vs
Latest recorded season average economy
```

A reduction in economy is considered improvement.

---

# 8️⃣ Matchup-Aware Bowler Selection

This is one of the major decision-support features.

Instead of simply asking:

> "Who has the most wickets?"

the system asks:

> "Which bowlers are strong against this opponent AND currently have workload capacity?"

The user selects an opponent.

The application calculates:

### Performance score

```text
Performance Score
=
(avg wickets × 2)
-
(avg economy × 0.5)
```

Then it normalizes the performance.

---

## Safety Score

The application evaluates ACWR relative to the desired zone.

The selection score combines:

```text
Selection Score
=
Performance Weight × Performance Score
+
Safety Weight × Safety Score
```

The user controls the performance weight.

For example:

```text
Performance Weight = 0.60
Safety Weight      = 0.40
```

This allows the user to decide whether the selection should prioritize:

```text
Performance
```

or:

```text
Workload safety
```

---

# 🤖 Machine Learning Matchup Predictor

The application contains **one genuinely trained ML component**.

Everything else such as ACWR, leaderboards and selection scoring is formula-based analytics.

The ML component uses:

```text
RandomForestRegressor
```

Two separate models are trained:

```text
Model 1 → Predict expected economy
Model 2 → Predict expected wickets
```

---

## ML Features

The model uses:

```text
career_avg_economy_before
career_avg_wickets_before
matches_played_before
rest_days_before
matches_last_30_days
chronic_avg_overs
format
opponent
```

---

## Leakage Protection

The historical career features are deliberately calculated using only previous matches.

For example:

```python
shift(1).expanding().mean()
```

is used for historical economy and wickets.

This prevents the target match's own performance from being directly included in its historical features.

---

## Model Configuration

Each Random Forest currently uses:

```text
n_estimators = 200
max_depth = 8
random_state = 42
```

The dataset is divided into:

```text
80% training
20% testing
```

The application reports:

### Economy

- MAE
- R²

### Wickets

- MAE
- R²

These metrics are displayed inside the application instead of hiding model performance.

---

## Important ML Note

The current evaluation uses a random train/test split.

Although the feature construction avoids using the target match's own values, the evaluation is **not a strict chronological/forward-time validation**.

For a future research version, we should consider:

```text
Time-based train/test split
or
Walk-forward validation
```

This would provide a more realistic estimate of how the model performs on genuinely future matches.

---

# 9️⃣ Injury & Fitness Management

This is the flagship feature of the project.

The reason it exists is important:

There is no reliable public dataset in the project containing real bowler injury outcomes that can be directly used as ground truth.

Therefore, instead of pretending to have a validated injury-prediction model, the application creates a persistent injury/fitness log.

---

## Injury Statuses

The system supports:

```text
Fit
Managed
Injured
Rehab
```

The latest manually logged status becomes the current status of the player.

---

## Logging a Case

A user can record:

```text
Bowler
Status
Body part
Injury type
Expected return date
Notes
```

The entry is saved to:

```text
injury_log.csv
```

Therefore, the information persists across application restarts.

---

# 🔔 Injury Alerts

The alert system combines multiple signals.

Current alert types include:

### High ACWR

The bowler is currently above:

```text
ACWR > 1.5
```

---

### Deconditioning Risk

The bowler has had a long gap since their previous recorded match.

---

### High Monotony

Recent workload is unusually repetitive.

---

### Status Conflict

Example:

```text
Injury status = Injured
ACWR = Moderate/High
```

This indicates that the recorded status and workload data should be manually checked.

---

# 📈 Monotony and Strain

The application also implements workload concepts beyond ACWR.

## Monotony

```text
Monotony
=
Mean recent workload
/
Standard deviation of recent workload
```

The current implementation uses the player's latest 7 matches.

---

## Monotony tiers

```text
< 1.5       → Varied
1.5 – 2.0   → Moderate
> 2.0       → High
```

---

## Strain

```text
Strain
=
Monotony × Total recent workload
```

This gives a second perspective on workload.

A bowler can therefore have elevated strain because:

- workload is high
- workload is repetitive
- or both

---

# 🩺 Assessment Guide

The Injury & Fitness page contains a structured assessment guide.

Important:

> This is NOT an AI medical diagnosis system.

It is a fixed, rule-based knowledge system.

The user selects:

```text
Bowler
Body part
Severity
Symptoms
```

The system then returns:

- General mechanism
- General considerations
- POLICE first-response framework
- Severity-specific guidance
- Red-flag warnings
- Similar logged cases

---

## Supported Body Areas

The current knowledge base contains guidance for:

```text
Lower back
Shoulder
Knee
Ankle
Hamstring
Quadriceps
Groin
Side/abdominal
Elbow
Foot/stress fracture
Other
```

---

# ⚠️ Medical Disclaimer

The assessment guide is educational only.

It does not:

- Diagnose an injury
- Prescribe treatment
- Replace a doctor
- Replace a physiotherapist
- Predict medical outcomes

Actual medical decisions must be made by qualified professionals.

---

# 📈 Return-to-Play Planner

The application includes a graded bowling-load return plan.

The standard ramp currently uses:

```text
Week 1 → 25% of normal workload
Week 2 → 50%
Week 3 → 75%
Week 4 → 100%
```

The baseline is normally calculated using the bowler's recent workload.

If an injury case exists, the application attempts to use the four matches before the case as the baseline rather than using reduced post-injury workload.

This is important because:

```text
Reduced workload after injury
          ↓
should not become
          ↓
the definition of "normal workload"
```

---

# 📇 Fitness Passport

The application can generate an exportable HTML fitness passport for a player.

It contains:

- Player name
- Country
- Bowling style
- Current status
- ACWR
- Total wickets
- Total overs
- Recent match history
- Economy
- Injury/status history

The generated HTML can be:

```text
Viewed in browser
        ↓
Printed
        ↓
Saved as PDF
```

This is intended as a compact player summary that can be shared with coaching/medical staff.

---

# 🔄 Squad Impact Engine

The Squad Impact Engine extends the analysis from:

```text
Individual bowler
```

to:

```text
Entire bowling unit
```

It contains three major components.

---

# 🌊 Cascade Risk

If a bowler becomes unavailable, their workload does not disappear.

Other bowlers may have to absorb those overs.

The application estimates how the missing workload could be distributed among fit teammates.

It then calculates:

```text
Current workload
+
Projected additional workload
=
Projected workload
```

and estimates the resulting ACWR.

This can identify teammates whose risk tier could worsen.

Example:

```text
Bowler A unavailable
       ↓
8 overs/match need redistribution
       ↓
Bowler B +3 overs
Bowler C +2 overs
Bowler D +3 overs
       ↓
Projected ACWR calculated
```

This is a projection, not a prediction of what the coach will actually do.

---

# 💳 Recovery Debt Ledger

This is a custom project concept.

The idea is:

> Time away from bowling can create a conditioning gap, and the size of that gap depends partly on how much the player was bowling before the injury.

The application calculates:

```text
Days out
+
Pre-injury workload
        ↓
Conditioning debt score
        ↓
Recommended ramp length
```

The current debt formula is an internally defined scoring system.

It is **not a published clinical measurement**.

It is intended for:

```text
Comparison
Ranking
Decision support
```

rather than claiming an absolute medical meaning.

---

# 🔁 Substitution Finder

If a bowler is:

```text
Injured
Managed
High ACWR
```

the application can search for fit teammates.

Candidates are ranked primarily by:

```text
ACWR headroom
```

and then:

```text
Recent economy
```

This turns:

```text
"Player is risky"
```

into:

```text
"Here are possible teammates who have workload capacity."
```

---

# 🏟️ Team Overview

The Team Overview page provides team-level analytics.

For a selected team:

- Number of bowlers used
- Matches
- Total wickets
- Top bowlers
- Average economy
- Seasonal economy trend

---

## Head-to-Head

Two teams can be selected.

The application calculates:

- Matches between the teams
- Total wickets
- Average economy
- Top wicket-taking bowlers

This gives a historical team-vs-team bowling perspective.

---

# 💬 Ask the Data

The application contains a built-in data assistant.

Important:

> This is **not an LLM chatbot**.

It is a rule-based system using:

```text
Pandas
Regex
difflib
Session state
```

There are:

- No external AI APIs
- No language model
- No generated statistics

The chatbot directly looks up information from the loaded datasets.

---

## Example Questions

```text
wickets for Bumrah

what's Rashid Khan's economy?

how many overs has Shami bowled?

rest days for Starc

ACWR for Starc

is Boult at risk?

monotony for Rabada

how many overs can Shami safely bowl?

is Archer injured?

how does Bumrah do against Australia?

compare Bumrah and Starc

top 10 wicket takers

best economy

most consistent bowlers

rising stars

who is highest risk

who is injured

how many bowlers are there?

how many matches?

which teams?

what is ACWR?

what does monotony mean?
```

---

# 🧠 Chatbot Player Recognition

The chatbot has three levels of player-name detection.

### Level 1

Exact full name.

### Level 2

Token-based matching.

For example:

```text
"wickets for Bumrah"
```

can identify:

```text
Jasprit Bumrah
```

### Level 3

Fuzzy matching for typos.

This allows small mistakes in names to be handled without using an external AI model.

---

# 🔁 Conversation Context

The chatbot remembers the most recently referenced player within the current Streamlit session.

Therefore:

```text
User:
What is Bumrah's economy?

Assistant:
...

User:
And his ACWR?

Assistant:
Bumrah's ACWR is ...
```

The second question can resolve "his" using session state.

---

# 📤 Export Data

The Export Data page allows users to download processed tables.

Available exports include:

```text
Bowler match summary
Bowler vs team summary
Player master list
Advanced search aggregate
```

The exported files are CSVs.

---

# 🎨 UI / UX

The application uses a custom dark cricket-themed visual system.

Major UI features include:

- Dark dashboard
- Custom CSS
- Cricket-inspired color palette
- KPI cards
- Interactive charts
- Custom ACWR gauge
- Radar charts
- Player cards
- Risk badges
- Status badges
- Sidebar navigation
- Format-specific landing screen
- Animated introduction
- Responsive layouts
- Floating data assistant

The project deliberately avoids looking like a default Streamlit application.

---

# 🧭 Format Selection

When the application starts, users can choose:

```text
T20
ODI
TEST
All Formats
```

The format selection affects the analytics shown throughout the application.

There is also a sidebar option to switch format later.

---

# 🌍 Country Filtering

The sidebar provides an optional country filter.

This filter affects the main analytical dataset used by the dashboard.

Therefore, many pages can be explored using:

```text
Format
+
Country
```

as global filters.

---

# ⚡ Performance Optimization

The application contains several performance optimizations.

Streamlit caching is used for expensive calculations such as:

```text
compute_vs_team()
compute_consistency()
compute_rising_stars()
compute_monotony_strain_all()
prepare_ml_training_data()
train_performance_models()
compute_alerts()
```

The application also avoids repeatedly rendering individual HTML components where possible by batching cards into grids.

This becomes important when working with large cricket datasets.

---

# 🧮 Main Calculations Summary

## Overs

```text
Overs = Legal deliveries / 6
```

---

## Economy

```text
Economy = Runs conceded / Overs bowled
```

---

## ACWR

```text
ACWR = Current match overs / Average overs from previous 4 matches
```

---

## Monotony

```text
Monotony = Mean recent workload / Standard deviation
```

---

## Strain

```text
Strain = Monotony × Total recent workload
```

---

## Performance Score

```text
Performance Score
=
(Avg wickets × 2)
-
(Avg economy × 0.5)
```

---

## Consistency

```text
Consistency
=
Standard deviation of economy
```

Lower:

```text
More consistent
```

---

# 🔬 What Is ML vs What Is Formula-Based?

This distinction is important when presenting the project.

## Formula / Analytics Based

The following are NOT machine-learning models:

```text
ACWR
Monotony
Strain
Performance Score
Selection Score
Safe overs ceiling
Recovery Debt
Cascade Risk
Consistency
Rising Stars
Risk tiers
Return-to-play ramp
```

They are explicit mathematical/rule-based calculations.

---

## Machine Learning

The only trained ML component currently is:

```text
Random Forest Regression
```

with two targets:

```text
Expected economy
Expected wickets
```

This distinction should be maintained in project presentations.

Do NOT describe the entire application as an "AI injury prediction system."

---

# ⚠️ Important Scientific Limitation

The application intentionally does **not** claim:

```text
ACWR = Injury Prediction
```

Instead:

```text
ACWR
  ↓
Workload warning signal
  ↓
Human decision
```

The current system does not have a sufficiently reliable injury-outcome dataset to validate a true injury prediction model.

Therefore:

> ACWR should be treated as an early-warning workload heuristic and not as a medical diagnosis or validated injury predictor.

---

# 🧪 Demo Injury Data

The application includes a demo-data generator.

This is useful when presenting the application before real injury records are available.

Demo records are explicitly marked:

```text
source = "demo"
```

Real manually entered cases are:

```text
source = "manual"
```

Therefore demo data can be removed without deleting real entries.

---

# 🤝 Working With Your Project Partner

Before modifying anything:

```bash
git pull
```

After completing a feature:

```bash
git add .
git commit -m "Describe your change"
git push
```

---

# 🧩 Which File Should I Modify?

## `app.py`

Modify this when working on:

- Dashboard UI
- Navigation
- Charts
- Player profiles
- ACWR logic
- Injury management
- Squad Impact Engine
- Chatbot
- ML prediction
- Export functionality

The current `app.py` contains most of the application's business logic and UI.

---

## `preprocess.py`

Modify this when working on:

- Raw ball-by-ball processing
- Data cleaning
- Feature engineering
- Match-level aggregation
- Bowler-level aggregation
- Workload features
- Rest-day calculations
- Rolling workload
- New dataset columns

If a new metric needs to be generated from raw data, it should generally be added to preprocessing rather than manually editing the processed CSV.

---

## CSV files

The processed CSVs are application inputs.

Avoid manually changing calculated values unless there is a specific reason.

Prefer:

```text
Change preprocessing
        ↓
Regenerate CSV
        ↓
Run application
```

---

# 🔄 Recommended Development Workflow

```text
1. Understand the metric
        ↓
2. Decide whether it belongs in preprocessing or app.py
        ↓
3. Implement
        ↓
4. Run the application
        ↓
5. Test using multiple bowlers
        ↓
6. Check edge cases
        ↓
7. Verify calculations
        ↓
8. Test UI
        ↓
9. Commit
        ↓
10. Push
```

---

# 🧪 Things To Test Before Committing

Whenever modifying the application, test:

### Data

- Missing CSV
- Empty CSV
- Missing columns
- Unknown player
- Unknown opponent

### Workload

- Player with only 1 match
- Player with 2 matches
- Player with 4+ matches
- Very high ACWR
- Very low ACWR
- Zero/near-zero workload

### Injury

- No injury records
- Demo injury records
- Real/manual records
- Clearing demo records
- Updating a player to Fit
- Player with multiple historical cases

### ML

- Dataset with enough rows
- Dataset with insufficient rows
- Unknown opponent
- Unknown format
- Player with little history

### UI

- T20
- ODI
- Test
- All formats
- Country filters
- Sidebar navigation
- Player search
- Export buttons

---

# 🚧 Current Limitations

The current version has several limitations that should be known before presenting it as a production system.

### 1. ACWR is a heuristic

It is not a clinically validated injury prediction model.

### 2. Injury ground truth is limited

Real injury records must be manually logged.

### 3. Demo injury data is synthetic

Demo records are for presentation/testing only.

### 4. Recovery Debt is a custom metric

The debt score is internally defined and should not be presented as a clinically validated measurement.

### 5. Cascade Risk is a projection

The application assumes workload redistribution based on current workload proportions. Actual coaching decisions may distribute overs differently.

### 6. Return-to-play is a general template

It is not a medical prescription.

### 7. ML validation can be improved

The current model uses a random 80/20 train-test split. A future version should use chronological validation or walk-forward testing.

### 8. Large datasets

Very large raw ball-by-ball datasets should not be stored directly in GitHub.

---

# 🔮 Future Development

Potential next-stage improvements:

## Workload

- Daily workload data
- Training-session workload
- Gym workload
- Bowling intensity
- Speed data
- Session-RPE
- GPS/accelerometer data

## Injury Analytics

Once sufficient real injury records exist:

```text
Workload features
+
Injury history
+
Training data
+
Recovery data
        ↓
Validated injury-risk model
```

Potential models:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Time-series models

---

## ML Improvements

Future versions should investigate:

- Time-series validation
- Walk-forward validation
- Feature importance
- SHAP explanations
- Calibration
- Cross-validation
- Separate models by format
- Player-specific models
- Opponent-specific features

---

## Advanced Workload Features

Potential future features:

```text
7-day workload
14-day workload
28-day workload
Acute workload
Chronic workload
Workload monotony
Workload strain
Rest-day patterns
Bowling intensity
Spell length
Consecutive match load
Travel/rest effects
```

---

# 🏗️ High-Level Architecture

```text
                   RAW CRICKET DATA
                          │
                          ▼
                  ┌───────────────┐
                  │ preprocess.py │
                  └───────┬───────┘
                          │
             ┌────────────┼─────────────┐
             ▼            ▼             ▼
        Master Data   Match Summary   VS Team
             │            │             │
             └────────────┼─────────────┘
                          ▼
                     ┌─────────┐
                     │ app.py  │
                     └────┬────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   Performance         Workload          Fitness
   Analytics           Analytics         Management
        │                 │                 │
        ├────────────┬────┴────┬────────────┤
        │            │         │            │
        ▼            ▼         ▼            ▼
    Matchups       ACWR     Injury      Squad Impact
        │            │      System          │
        ▼            ▼         ▼            ▼
   ML Predictor   Alerts   RTP Plans   Substitution
        │            │         │            │
        └────────────┴─────────┴────────────┘
                          │
                          ▼
                  DECISION SUPPORT
```

---

# 👥 Team Responsibilities

Suggested division of work:

### Person 1 — Data / Backend

Focus on:

```text
preprocess.py
Data cleaning
Feature engineering
CSV generation
Workload calculations
```

### Person 2 — Dashboard / Frontend

Focus on:

```text
app.py UI
Streamlit pages
Charts
Navigation
UX
```

### Shared

Both members should work together on:

```text
ML model
ACWR methodology
Injury system
Testing
Documentation
Research
```

---

# 📌 Important Project Principle

The project should always distinguish between:

```text
FACT
```

and:

```text
ESTIMATE
```

### Fact

Computed directly from historical data.

Example:

```text
A bowler bowled 18 overs across the selected matches.
```

### Formula-based estimate

Example:

```text
Projected ACWR = 1.6
```

### ML prediction

Example:

```text
Predicted economy = 6.8
Predicted wickets = 1.4
```

### Medical decision

Should always remain with:

```text
Qualified medical/coaching staff
```

---

# 🏁 Current Project Status

### Completed

- [x] Real ball-by-ball data integration
- [x] Bowler match-level analytics
- [x] Bowler master dataset integration
- [x] Bowler-vs-team analytics
- [x] T20 / ODI / Test filtering
- [x] Country filtering
- [x] Home dashboard
- [x] Player profiles
- [x] Player comparison
- [x] Radar comparison
- [x] Advanced search
- [x] Multiple leaderboards
- [x] ACWR engine
- [x] Workload monitor
- [x] Injury & fitness management
- [x] Injury alerts
- [x] Monotony and strain
- [x] Return-to-play planner
- [x] Fitness passport
- [x] Squad Impact Engine
- [x] Recovery Debt Ledger
- [x] Substitution Finder
- [x] Team overview
- [x] Head-to-head analysis
- [x] Rule-based Ask the Data assistant
- [x] CSV export
- [x] Random Forest matchup prediction
- [x] Methodology page
- [x] Custom Streamlit UI

---

# 🚀 Final Vision

The long-term goal is to turn BowlerApp into a genuine **fast-bowler workload and decision-support platform**.

The progression is:

```text
Raw cricket data
        ↓
Performance analytics
        ↓
Workload analytics
        ↓
Fitness information
        ↓
Risk signals
        ↓
Squad impact
        ↓
Matchup prediction
        ↓
Decision support
```

The platform is not intended to replace a coach, analyst, physiotherapist or doctor.

Instead, its purpose is to give them **more structured information before making a decision**.

---

# ⚠️ Disclaimer

Bowler Workload Platform is an analytics and decision-support project.

ACWR, workload scores, recovery debt, cascade projections and return-to-play plans should not be interpreted as medical diagnoses or prescriptions.

Machine-learning predictions are estimates based on historical data and should not be treated as guaranteed outcomes.

Final sporting and medical decisions should always involve qualified professionals.

---

## 🏏 Built With

**Python · Pandas · NumPy · Streamlit · Plotly · Matplotlib · Seaborn · Scikit-learn**

---

### Project Status

**Active Development 🚧**

The platform is continuously being improved with additional workload metrics, better validation, more real-world injury data and stronger decision-support capabilities.
