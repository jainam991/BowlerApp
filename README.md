# 🏏 BowlerApp — Cricket Fast Bowler Analysis & Workload Dashboard

## 📌 Project Overview

**BowlerApp** is a Python-based cricket analytics application designed to analyse **fast bowler performance, workload, match activity, team-wise performance, and injury/recovery-related information**.

The project processes ball-by-ball cricket data and converts it into structured datasets that can be used by the dashboard for player-level and team-level analysis.

The main goal is to help analyse a fast bowler's workload and performance across different matches and teams, providing useful insights for **performance analysis and decision-making**.

---

## 🎯 Main Objectives

* Analyse individual fast bowler performance.
* Track bowler workload across matches.
* Analyse bowler performance against different teams.
* Generate match-level bowling summaries.
* Maintain a master dataset containing bowler information.
* Incorporate injury-related information.
* Provide an interactive dashboard using **Streamlit**.
* Convert raw ball-by-ball data into smaller, analysis-ready datasets.

---

## 🧠 Project Workflow

```text
Raw Ball-by-Ball Cricket Data
            │
            ▼
      preprocess.py
            │
            ▼
 ┌──────────────────────────┐
 │ Processed CSV Datasets   │
 ├──────────────────────────┤
 │ bowler_master.csv        │
 │ bowler_match_summary.csv │
 │ bowler_vs_team_summary.csv│
 │ injury_log.csv            │
 └──────────────────────────┘
            │
            ▼
          app.py
            │
            ▼
   Streamlit Analytics Dashboard
            │
            ▼
   Bowler / Match / Team Insights
```

---

# 📂 Project Structure

```text
Bowlerapp/
│
├── app.py
│       └── Main Streamlit dashboard/application
│
├── preprocess.py
│       └── Processes raw ball-by-ball cricket data
│
├── requirements.txt
│       └── Python dependencies required for the project
│
├── bowler_master.csv
│       └── Main bowler-level dataset
│
├── bowler_master_2.csv
│       └── Additional/updated bowler master dataset
│
├── bowler_match_summary.csv
│       └── Match-level bowling statistics
│
├── bowler_match_summary_2.csv
│       └── Additional/updated match summary
│
├── bowler_vs_team_summary.csv
│       └── Bowler performance against different teams
│
├── bowler_vs_team_summary_2.csv
│       └── Additional/updated team-wise summary
│
├── injury_log.csv
│       └── Injury-related records used for analysis
│
└── venv/
        └── Local Python virtual environment
        (DO NOT upload this folder to GitHub)
```

---

# ⚙️ Technologies Used

| Technology | Purpose                      |
| ---------- | ---------------------------- |
| Python     | Core programming language    |
| Pandas     | Data processing and analysis |
| NumPy      | Numerical calculations       |
| Streamlit  | Interactive dashboard        |
| Plotly     | Interactive visualisations   |
| Matplotlib | Data visualisation           |
| Seaborn    | Statistical visualisation    |
| CSV        | Data storage                 |

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd Bowlerapp
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

## 3. Activate the virtual environment

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not updated yet, the main packages required are:

```bash
pip install pandas numpy streamlit matplotlib seaborn plotly
```

---

# 📊 Data Preprocessing

The project uses `preprocess.py` to convert the raw ball-by-ball cricket dataset into structured datasets required by the dashboard.

Example:

```bash
python3 preprocess.py ball_by_ball_data.csv
```

The preprocessing stage generates datasets such as:

```text
bowler_master.csv
bowler_match_summary.csv
bowler_vs_team_summary.csv
```

These processed files are then used by `app.py`.

---

# 🖥️ Running the Dashboard

After preprocessing and installing the requirements:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open that URL in a browser to access the dashboard.

---

# 📈 Current Analysis

The application is intended to provide analysis such as:

### 👤 Bowler Analysis

* Bowler statistics
* Number of matches
* Overs bowled
* Runs conceded
* Wickets
* Economy
* Bowling performance

### 🏏 Match Analysis

* Match-by-match workload
* Bowling activity
* Overs bowled in individual matches
* Performance trends

### 🆚 Team Analysis

* Performance against different opposition
* Overs against each team
* Runs conceded
* Wickets
* Economy
* Team-specific performance patterns

### 🩹 Injury Information

`injury_log.csv` is maintained separately so injury-related information can be incorporated into the analysis and eventually connected with workload/recovery analysis.

---

# 🔬 Project Direction

The long-term objective of BowlerApp is to move beyond basic cricket statistics and develop a more advanced **Fast Bowler Workload, Fatigue & Recovery Analysis System**.

Potential future analytics include:

* Bowling workload trends
* Recent workload accumulation
* Match-to-match workload
* Short-term vs long-term workload
* Workload spikes
* Rest-period analysis
* Bowling intensity indicators
* Injury-risk indicators
* Recovery recommendations
* Player workload comparison
* Team-level workload monitoring

The aim is to make the system useful not only for displaying statistics but also for supporting **better cricket performance and workload-management decisions**.

---

# 🤝 Working With This Repository

### For my project partner

The main files you should understand first are:

### `app.py`

This is the **main application/dashboard**.

If you want to change:

* UI
* dashboard layout
* filters
* charts
* player selection
* displayed statistics
* analytics

start here.

### `preprocess.py`

This controls how the raw cricket data is transformed into the processed datasets.

If the raw dataset changes or we want to calculate new metrics, this is one of the main files to modify.

### CSV files

These are the processed datasets consumed by the application.

**Avoid manually changing them unless necessary.**

If a new metric needs to be added, preferably modify the preprocessing logic and regenerate the datasets.

---

# ⚠️ Important Notes

## Do NOT upload `venv/`

The `venv` folder contains the local Python environment and should not be committed to GitHub.

Add this to `.gitignore`:

```text
venv/
__pycache__/
*.pyc
.DS_Store
.env
```

---

## Large Datasets

The original raw ball-by-ball dataset may be very large.

Do **not** upload extremely large raw datasets directly to GitHub.

GitHub is being used primarily for:

* Source code
* Configuration
* Processed/smaller datasets
* Documentation
* Collaboration

The large raw dataset can be stored separately and shared between project members when required.

---

# 🔄 Development Workflow

When making changes:

```text
1. Pull latest changes
       ↓
2. Make changes
       ↓
3. Test locally
       ↓
4. Check dashboard
       ↓
5. Commit changes
       ↓
6. Push to GitHub
```

Example:

```bash
git pull
```

After making changes:

```bash
git add .
git commit -m "Describe your changes"
git push
```

---

# 📝 Commit Message Examples

Use clear commit messages so both partners understand what changed.

```text
Add bowler workload calculation
```

```text
Update match summary preprocessing
```

```text
Add team-wise performance chart
```

```text
Fix Streamlit sidebar filters
```

```text
Update injury analysis
```

```text
Improve dashboard UI
```

---

# 👥 Collaboration

Both partners should work on the same GitHub repository.

Before starting work:

```bash
git pull
```

After completing a feature:

```bash
git add .
git commit -m "Describe change"
git push
```

For major features, it is recommended to create a separate branch instead of directly modifying `main`.

Example:

```bash
git checkout -b workload-analysis
```

---

# 🏁 Current Status

**Project Stage:** Development

### Currently available

* Ball-by-ball data processing
* Bowler master dataset
* Match-level summary
* Bowler vs team analysis
* Injury log
* Streamlit application
* Data visualisation

### Planned

* Advanced workload metrics
* Fatigue analysis
* Recovery analysis
* Workload spike detection
* Injury-risk modelling
* Player comparison
* Decision-support features
* Improved dashboard UI/UX

---

# 👨‍💻 Project

**BowlerApp**

A cricket analytics project focused on **fast bowler performance, workload, fatigue and recovery analysis**.

Built with:

**Python + Pandas + Streamlit + Plotly**

> The project is currently under active development. Features, datasets and analytical methods may change as the project evolves.
