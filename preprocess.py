"""
Cricket Dataset Preprocessor
------------------------------
Run this LOCALLY against your full ~4GB CSV file. It streams the file
in chunks (never loads the whole thing into memory), so it works fine
even on a laptop. It outputs three small files (a few MB, not GB):

    bowler_match_summary.csv   - one row per (bowler, match): overs
                                  bowled, economy, wickets, opponent,
                                  format, date
    bowler_vs_team_summary.csv - one row per (bowler, opponent team):
                                  aggregated performance stats
    bowler_master.csv          - one row per bowler: real name,
                                  country, bowling style, playing
                                  role, photo URL

Usage:
    python preprocess.py /path/to/your/full_dataset.csv

These three output files are what the Streamlit app actually loads —
they're small enough to upload or keep alongside app.py.
"""

import sys
import csv
import pandas as pd
import numpy as np

USECOLS = [
    'match_id', 'season', 'start_date', 'bowling_team', 'batting_team', 'bowler',
    'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
    'wicket', 'wicket_type', 'format',
    'full name_bowler', 'country_bowler', 'bowling style_bowler',
    'playing role_bowler', 'image url_bowler'
]

NON_BOWLER_WICKET_TYPES = {'run out', 'retired hurt', 'retired out', 'obstructing the field'}


def detect_delimiter(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    for candidate in ['\t', ',', ';', '|']:
        if candidate in first_line:
            return candidate
    return ','


def process_file(path, chunksize=200_000):
    sep = detect_delimiter(path)
    print(f"Detected delimiter: {sep!r}")

    match_bowler_records = {}   # (match_id, bowler) -> accumulated stats dict
    master_records = {}         # bowler short-name -> {name, country, style, role, image}

    total_rows = 0
    for chunk_num, chunk in enumerate(
        pd.read_csv(path, sep=sep, usecols=lambda c: c in USECOLS,
                    chunksize=chunksize, low_memory=False)
    ):
        total_rows += len(chunk)

        chunk['is_legal'] = chunk['wides'].isna() & chunk['noballs'].isna()
        chunk['runs_conceded'] = (
            chunk['runs_off_bat'].fillna(0) + chunk['wides'].fillna(0) + chunk['noballs'].fillna(0)
        )
        wicket_col = chunk['wicket'].fillna(0)
        wtype_col = chunk['wicket_type'].fillna('')
        chunk['bowler_wicket'] = (
            (wicket_col == 1) & (~wtype_col.isin(NON_BOWLER_WICKET_TYPES))
        )

        group_cols = ['match_id', 'bowler', 'bowling_team', 'batting_team', 'format', 'season', 'start_date']
        grouped = chunk.groupby(group_cols, dropna=False).agg(
            legal_balls=('is_legal', 'sum'),
            runs_conceded=('runs_conceded', 'sum'),
            wickets=('bowler_wicket', 'sum'),
        ).reset_index()

        for row in grouped.itertuples(index=False):
            key = (row.match_id, row.bowler)
            if key not in match_bowler_records:
                match_bowler_records[key] = {
                    'match_id': row.match_id, 'bowler': row.bowler,
                    'bowling_team': row.bowling_team, 'batting_team': row.batting_team,
                    'format': row.format, 'season': row.season, 'start_date': row.start_date,
                    'legal_balls': row.legal_balls, 'runs_conceded': row.runs_conceded,
                    'wickets': row.wickets,
                }
            else:
                rec = match_bowler_records[key]
                rec['legal_balls'] += row.legal_balls
                rec['runs_conceded'] += row.runs_conceded
                rec['wickets'] += row.wickets

        # capture player master info (name/country/style/role/photo) once per bowler
        master_cols = ['bowler', 'full name_bowler', 'country_bowler',
                        'bowling style_bowler', 'playing role_bowler', 'image url_bowler']
        available_master_cols = [c for c in master_cols if c in chunk.columns]
        if len(available_master_cols) == len(master_cols):
            for row in chunk[master_cols].dropna(subset=['bowler']).drop_duplicates('bowler').itertuples(index=False):
                if row.bowler not in master_records:
                    master_records[row.bowler] = {
                        'bowler': row.bowler,
                        'full_name': row[1] if pd.notna(row[1]) else row.bowler,
                        'country': row[2] if pd.notna(row[2]) else 'Unknown',
                        'bowling_style': row[3] if pd.notna(row[3]) else 'Unknown',
                        'playing_role': row[4] if pd.notna(row[4]) else 'Bowler',
                        'image_url': row[5] if pd.notna(row[5]) else '',
                    }

        print(f"  processed chunk {chunk_num + 1}, rows so far: {total_rows:,}")

    print(f"\nTotal rows processed: {total_rows:,}")
    print(f"Unique (match, bowler) pairs: {len(match_bowler_records):,}")
    print(f"Unique bowlers with master info: {len(master_records):,}")

    match_summary = pd.DataFrame(list(match_bowler_records.values()))
    match_summary['overs_bowled'] = match_summary['legal_balls'] / 6.0
    match_summary['economy'] = np.where(
        match_summary['overs_bowled'] > 0,
        match_summary['runs_conceded'] / match_summary['overs_bowled'],
        0.0
    )
    match_summary['start_date'] = pd.to_datetime(match_summary['start_date'], errors='coerce', dayfirst=True)
    match_summary = match_summary.sort_values(['bowler', 'start_date'])

    # ---- real workload features: rest days + matches in rolling 30-day window ----
    match_summary['rest_days_before'] = (
        match_summary.groupby('bowler')['start_date'].diff().dt.days
    )
    match_summary['rest_days_before'] = match_summary['rest_days_before'].fillna(30).clip(0, 60)

    def count_recent_matches(group):
        dates = group['start_date'].values
        counts = []
        for i, d in enumerate(dates):
            window_start = d - np.timedelta64(30, 'D')
            counts.append(int(((dates > window_start) & (dates < d)).sum()))
        return pd.Series(counts, index=group.index)

    match_summary['matches_last_30_days'] = (
        match_summary.groupby('bowler', group_keys=False).apply(count_recent_matches)
    )

    # ---- ACWR (Acute:Chronic Workload Ratio) — real sports-science heuristic ----
    # acute = this match's overs bowled; chronic = rolling average overs over
    # the player's last 4 matches (a common proxy when exact weekly training
    # load isn't available).
    match_summary['chronic_avg_overs'] = (
        match_summary.groupby('bowler')['overs_bowled']
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean().shift(1))
    )
    match_summary['chronic_avg_overs'] = match_summary['chronic_avg_overs'].fillna(match_summary['overs_bowled'])
    match_summary['acwr'] = np.where(
        match_summary['chronic_avg_overs'] > 0,
        match_summary['overs_bowled'] / match_summary['chronic_avg_overs'],
        1.0
    )

    match_summary.to_csv('bowler_match_summary.csv', index=False)
    print("Wrote bowler_match_summary.csv")

    # ---- bowler vs team aggregation (real matchup data) ----
    vs_team = match_summary.groupby(['bowler', 'batting_team']).agg(
        avg_economy=('economy', 'mean'),
        avg_wickets=('wickets', 'mean'),
        matches_played=('match_id', 'nunique'),
    ).reset_index().rename(columns={'batting_team': 'opponent_team'})
    vs_team['performance_score'] = vs_team['avg_wickets'] * 2 - vs_team['avg_economy'] * 0.5
    vs_team.to_csv('bowler_vs_team_summary.csv', index=False)
    print("Wrote bowler_vs_team_summary.csv")

    # ---- player master table ----
    master_df = pd.DataFrame(list(master_records.values()))
    master_df.to_csv('bowler_master.csv', index=False)
    print("Wrote bowler_master.csv")

    print("\nDone. Upload/copy these 3 CSV files alongside app.py.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py /path/to/full_dataset.csv")
        sys.exit(1)
    process_file(sys.argv[1])
