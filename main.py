import streamlit as st
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np

STATS_9CAT = ["PTS", "REB", "AST", "ST", "BLK", "3PTM", "FG%", "FT%", "TO"]
POSITIVE_STATS = ["PTS", "REB", "AST", "ST", "BLK", "3PTM", "FG%", "FT%"]
NEGATIVE_STATS = ["TO"]
PCT_STATS = ["FG%", "FT%"]
VOLUME_COLS = {"FG%": ("FGM", "FGA"), "FT%": ("FTM", "FTA")}
UNWANTED_COLS = ["Id", "Opponent", "ADP", "Ros%", "+/-", "GP"]

Z_COLOR_BANDS = [
    (4, "#006400"),
    (3, "#228B22"),
    (2, "#32CD32"),
    (1, "#90EE90"),
    (0, "#FFFFFF"),
    (-1, "#FFB6C1"),
    (-2, "#FF69B4"),
    (-3, "#FF1493"),
    (-4, "#C71585"),
]


def load_data(selected_path, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(selected_path)


def compute_means(data: pd.DataFrame) -> Dict[str, float]:
    return {stat: data[stat].mean() for stat in STATS_9CAT}


def compute_stds(data: pd.DataFrame) -> Dict[str, float]:
    return {stat: data[stat].std() for stat in STATS_9CAT}


def compute_avg_pcts(data: pd.DataFrame) -> Dict[str, float]:
    avg_pcts = {}
    for pct in PCT_STATS:
        makes, att = VOLUME_COLS[pct]
        total_makes = data[makes].sum()
        total_att = data[att].sum()
        avg_pcts[pct] = total_makes / total_att if total_att > 0 else 0.0
    return avg_pcts


def z_score(val: float, mean: float, std: float, is_negative: bool = False) -> float:
    base_z = (val - mean) / std if std != 0 else 0.0
    return -base_z if is_negative else base_z


def add_basic_z_scores(
    df: pd.DataFrame, means: Dict[str, float], stds: Dict[str, float]
) -> pd.DataFrame:
    for stat in STATS_9CAT:
        if stat in PCT_STATS:
            continue
        is_neg = stat in NEGATIVE_STATS
        df[f"Z_{stat}"] = df[stat].apply(
            lambda v: z_score(v, means[stat], stds[stat], is_neg)
        )
    return df


def add_z_scores(data: pd.DataFrame) -> pd.DataFrame:
    means = compute_means(data)
    stds = compute_stds(data)
    avg_pcts = compute_avg_pcts(data)
    df = data.copy()
    df = add_basic_z_scores(df, means, stds)
    for pct in PCT_STATS:
        makes, att = VOLUME_COLS[pct]
        avg_pct = avg_pcts[pct]
        df[f"impact_{pct}"] = df[makes] - avg_pct * df[att]
    impact_means = {pct: df[f"impact_{pct}"].mean() for pct in PCT_STATS}
    impact_stds = {pct: df[f"impact_{pct}"].std() for pct in PCT_STATS}
    for pct in PCT_STATS:
        df[f"Weighted_Z_{pct}"] = df[f"impact_{pct}"].apply(
            lambda v: z_score(v, impact_means[pct], impact_stds[pct])
        )
    return df


def get_teams(data: pd.DataFrame) -> List[str]:
    return sorted(data["Status"].unique().tolist())


def is_real_team(team: str) -> bool:
    return team != "FA" and not team.startswith("W ")


def filter_team_players(data: pd.DataFrame, team: str) -> pd.DataFrame:
    return data[data["Status"] == team]


def get_team_z_dict(team_data: pd.DataFrame, agg_func) -> Dict[str, float]:
    z_dict = {}
    for stat in STATS_9CAT:
        col = f"Weighted_Z_{stat}" if stat in PCT_STATS else f"Z_{stat}"
        z_dict[stat] = agg_func(team_data[col])
    return z_dict


def compute_team_z_totals(team_data: pd.DataFrame) -> Dict[str, float]:
    return get_team_z_dict(team_data, np.sum)


def compute_team_z_averages(team_data: pd.DataFrame) -> Dict[str, float]:
    return get_team_z_dict(team_data, np.mean)


def compute_team_raw_totals(team_data: pd.DataFrame) -> Dict[str, float]:
    raw_dict = {}
    for stat in STATS_9CAT:
        if stat in PCT_STATS:
            makes, att = VOLUME_COLS[stat]
            total_makes = team_data[makes].sum()
            total_att = team_data[att].sum()
            raw_dict[stat] = (total_makes / total_att) if total_att > 0 else 0
        else:
            raw_dict[stat] = team_data[stat].sum()
    return raw_dict


def compute_team_avg_raw(team_data: pd.DataFrame) -> Dict[str, float]:
    avg_dict = {}
    for stat in STATS_9CAT:
        if stat in PCT_STATS:
            makes, att = VOLUME_COLS[stat]
            total_m = team_data[makes].sum()
            total_a = team_data[att].sum()
            avg_dict[stat] = total_m / total_a if total_a > 0 else 0.0
        else:
            avg_dict[stat] = team_data[stat].mean()
    return avg_dict


def compute_ranks(averages_df: pd.DataFrame, is_z: bool) -> pd.DataFrame:
    ranks_df = pd.DataFrame(index=averages_df.index, columns=averages_df.columns)
    for stat in STATS_9CAT:
        if is_z:
            ascending = False
        else:
            ascending = stat in NEGATIVE_STATS
        ranks_df[stat] = (
            averages_df[stat].rank(ascending=ascending, method="min").astype(int)
        )
    return ranks_df


def simulate_trade(
    data: pd.DataFrame,
    team_a: str,
    team_b: str,
    a_players: List[str],
    b_players: List[str],
) -> Tuple:
    df = data.copy()
    a_mask = df["Player"].isin(a_players) & (df["Status"] == team_a)
    b_mask = df["Player"].isin(b_players) & (df["Status"] == team_b)
    df.loc[a_mask, "Status"] = team_b
    df.loc[b_mask, "Status"] = team_a
    new_a_data = filter_team_players(df, team_a)
    new_b_data = filter_team_players(df, team_b)
    pre_a_z = compute_team_z_totals(filter_team_players(data, team_a))
    pre_b_z = compute_team_z_totals(filter_team_players(data, team_b))
    post_a_z = compute_team_z_totals(new_a_data)
    post_b_z = compute_team_z_totals(new_b_data)
    pre_a_raw = compute_team_raw_totals(filter_team_players(data, team_a))
    pre_b_raw = compute_team_raw_totals(filter_team_players(data, team_b))
    post_a_raw = compute_team_raw_totals(new_a_data)
    post_b_raw = compute_team_raw_totals(new_b_data)
    return (
        pre_a_z,
        post_a_z,
        pre_b_z,
        post_b_z,
        pre_a_raw,
        post_a_raw,
        pre_b_raw,
        post_b_raw,
    )


def get_z_color(z_val: float) -> str:
    for threshold, color in Z_COLOR_BANDS:
        if z_val >= threshold:
            return color
    return Z_COLOR_BANDS[-1][1]


def apply_z_color(val: float) -> str:
    color = get_z_color(val)
    return f"background-color: {color}"


def apply_colors(df, ranks_df):
    n = len(df.index)
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in df.columns:
        for row in df.index:
            rk = ranks_df.loc[row, col]
            if n <= 1:
                color = "#FFFFFF"
            else:
                normalized = (n - rk) / (n - 1)
                z_mapped = 8 * normalized - 4
                color = get_z_color(z_mapped)
            styles.loc[row, col] = f"background-color: {color}"
    return styles


def color_diff(col: pd.Series, is_z: bool, stat: str) -> List[str]:
    pre_val = col["Pre"]
    post_val = col["Post"]
    if is_z:
        improved = post_val > pre_val
    else:
        if stat in NEGATIVE_STATS:
            improved = post_val < pre_val
        else:
            improved = post_val > pre_val
    post_style = (
        "background-color: #90EE90"
        if improved
        else "background-color: #FFB6C1"
        if not improved
        else ""
    )
    return ["", post_style]


def style_trade_df(
    pre: Dict[str, float], post: Dict[str, float], is_z: bool
) -> pd.DataFrame:
    df = pd.DataFrame({"Pre": pre, "Post": post}).T

    def apply_color(col: pd.Series, stat: str) -> List[str]:
        return color_diff(col, is_z, stat)

    styled = df.style
    for stat in df.columns:
        styled = styled.apply(
            lambda col: apply_color(col, stat), axis=0, subset=pd.IndexSlice[:, stat]
        )
    return styled


def prepare_display_df(df: pd.DataFrame, show_z: bool) -> pd.DataFrame:
    df = df.drop(columns=UNWANTED_COLS, errors="ignore")
    if show_z:
        cols = ["Player", "Status"] + [
            f"Weighted_Z_{stat}" if stat in PCT_STATS else f"Z_{stat}"
            for stat in STATS_9CAT
        ]
        display_df = df[cols]
        rename_dict = {f"Weighted_Z_{stat}": f"Z_{stat}" for stat in PCT_STATS}
        display_df = display_df.rename(columns=rename_dict)
        return display_df
    else:
        cols = ["Player", "Status"] + STATS_9CAT + ["FGM", "FGA", "FTM", "FTA"]
        display_df = df[list(set(cols) & set(df.columns))]
        return display_df


def display_h2h_analysis(data_with_z: pd.DataFrame, real_teams: List[str]):
    st.subheader("H2H Analysis")
    selected_team = st.selectbox("Select Team for H2H", real_teams)
    if not selected_team:
        return
    raw_totals = {
        team: compute_team_raw_totals(filter_team_players(data_with_z, team))
        for team in real_teams
    }
    for other in sorted([t for t in real_teams if t != selected_team]):
        wins, losses, ties = 0, 0, 0
        stat_results = []
        for stat in STATS_9CAT:
            a_val = raw_totals[selected_team][stat]
            b_val = raw_totals[other][stat]
            if a_val == b_val:
                color = "black"
                ties += 1
            elif stat in NEGATIVE_STATS:
                if a_val < b_val:
                    color = "green"
                    wins += 1
                else:
                    color = "red"
                    losses += 1
            else:
                if a_val > b_val:
                    color = "green"
                    wins += 1
                else:
                    color = "red"
                    losses += 1
            stat_results.append(f'<span style="color:{color}">{stat}</span>')
        score = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
        overall_color = (
            "green" if wins > losses else "red" if losses > wins else "black"
        )
        st.markdown(
            f'**<span style="color:{overall_color}">{score}</span>** vs {other}',
            unsafe_allow_html=True,
        )
        st.markdown(" ".join(stat_results), unsafe_allow_html=True)


def display_team_summaries(data_with_z: pd.DataFrame):
    real_teams = sorted([t for t in get_teams(data_with_z) if is_real_team(t)])
    view_mode = st.radio(
        "View Team Summaries as", ["Average Z-Scores", "Average Stats"], index=0
    )
    is_z = view_mode == "Average Z-Scores"
    st.subheader(f"Team {view_mode}")
    if is_z:
        summaries = {
            team: compute_team_z_averages(filter_team_players(data_with_z, team))
            for team in real_teams
        }
    else:
        summaries = {
            team: compute_team_avg_raw(filter_team_players(data_with_z, team))
            for team in real_teams
        }
    summaries_df = pd.DataFrame.from_dict(summaries, orient="index")
    ranks_df = compute_ranks(summaries_df, is_z)
    display_df = pd.DataFrame(
        index=summaries_df.index, columns=summaries_df.columns, dtype=object
    )
    for stat in STATS_9CAT:
        for team in summaries_df.index:
            rk = ranks_df.at[team, stat]
            val = summaries_df.at[team, stat]
            if is_z:
                display_df.at[team, stat] = f"rk {rk} - {val:.2f}"
            else:
                fmt = ".4f" if stat in PCT_STATS else ".2f"
                display_df.at[team, stat] = f"rk {rk} - {val:{fmt}}"
    styled = display_df.style.apply(apply_colors, ranks_df=ranks_df, axis=None)
    st.dataframe(styled, hide_index=False)
    display_h2h_analysis(data_with_z, real_teams)


def display_player_data(data_with_z: pd.DataFrame):
    st.subheader("Player Data")
    view_mode = st.radio("View", ["Z-Scores", "Raw Stats"], index=0, key="full_view")
    show_z = view_mode == "Z-Scores"
    display_df = prepare_display_df(data_with_z, show_z)
    if show_z:
        z_cols = [f"Z_{stat}" for stat in STATS_9CAT]
        styled = display_df.style.applymap(apply_z_color, subset=z_cols)
    else:
        styled = display_df.style
    st.dataframe(
        styled,
        column_config={"Player": st.column_config.Column(pinned=True)},
        hide_index=True,
    )


def setup_trade_teams(data_with_z: pd.DataFrame) -> Tuple[str, str, List[str]]:
    all_teams = get_teams(data_with_z)
    teams = [t for t in all_teams if is_real_team(t)]
    if len(teams) < 2:
        st.warning("Need at least two teams in Status column.")
        return None, None, teams
    team_a = st.selectbox("Team A", teams)
    team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)
    if team_a == team_b:
        st.warning("Select different teams.")
        return None, None, teams
    return team_a, team_b, teams


def display_team_players(
    data_with_z: pd.DataFrame, team: str, key_suffix: str
) -> pd.DataFrame:
    st.subheader(f"{team} Players")
    view_mode = st.radio(
        "View", ["Z-Scores", "Raw Stats"], index=0, key=f"{key_suffix}_view"
    )
    show_z = view_mode == "Z-Scores"
    team_data = filter_team_players(data_with_z, team)
    display_df = prepare_display_df(team_data, show_z)
    if show_z:
        z_cols = [f"Z_{stat}" for stat in STATS_9CAT]
        styled = display_df.style.applymap(apply_z_color, subset=z_cols)
    else:
        styled = display_df.style
    st.dataframe(
        styled,
        column_config={"Player": st.column_config.Column(pinned=True)},
        hide_index=True,
    )
    return team_data


def handle_trade_simulation(
    team_a: str,
    team_b: str,
    a_to_trade: List[str],
    b_to_trade: List[str],
    data_with_z: pd.DataFrame,
):
    if "trade_results" not in st.session_state:
        st.session_state.trade_results = None
    if st.button("Simulate Trade"):
        results = simulate_trade(data_with_z, team_a, team_b, a_to_trade, b_to_trade)
        st.session_state.trade_results = results
    if st.session_state.trade_results:
        view_mode = st.radio(
            "View", ["Z-Scores", "Raw Stats"], index=0, key="trade_view"
        )
        show_z = view_mode == "Z-Scores"
        if show_z:
            pre_a, post_a, pre_b, post_b = st.session_state.trade_results[:4]
        else:
            pre_a, post_a, pre_b, post_b = st.session_state.trade_results[4:]
        st.subheader(f"{team_a} {view_mode}: Pre vs Post")
        a_styled = style_trade_df(pre_a, post_a, show_z)
        st.dataframe(a_styled)
        st.subheader(f"{team_b} {view_mode}: Pre vs Post")
        b_styled = style_trade_df(pre_b, post_b, show_z)
        st.dataframe(b_styled)


def display_z_explanation():
    st.markdown(
        """
        **Z-scores standardize stats**: $$ z = \\frac{value - \\mu}{\\sigma} $$.
        For TO, inverted since lower is better.

         **Weighted Z for FG%/FT%**: Z-score of impact, where impact = makes - (league_avg_% * attempts).
               Explains value of % with volume - high % on many shots > high % on few.
        """
    )


def main():
    st.set_page_config(layout="wide")
    st.title("Fantasy Basketball Trade Machine")
    st.markdown(
        "Upload CSV from Fantrax with 'Standard' Stats selected. Select data source below or override with upload."
    )
    data_source = st.radio("Data Source", ["Last Season", "This Season So Far"])
    selected_path = (
        "last_season.csv" if data_source == "Last Season" else "this_season_so_far.csv"
    )
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    data = load_data(selected_path, uploaded_file)
    data_with_z = add_z_scores(data)
    display_z_explanation()
    display_team_summaries(data_with_z)
    display_player_data(data_with_z)
    team_a, team_b, teams = setup_trade_teams(data_with_z)
    if not team_a:
        return
    st.subheader("Trade Simulator")
    a_data = display_team_players(data_with_z, team_a, "a")
    b_data = display_team_players(data_with_z, team_b, "b")
    a_players_all = a_data["Player"].tolist()
    b_players_all = b_data["Player"].tolist()
    a_to_trade = st.multiselect(f"Players from {team_a} to trade", a_players_all)
    b_to_trade = st.multiselect(f"Players from {team_b} to trade", b_players_all)
    handle_trade_simulation(team_a, team_b, a_to_trade, b_to_trade, data_with_z)


if __name__ == "__main__":
    main()
