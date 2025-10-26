import pandas as pd
from typing import Dict, Tuple, List
import numpy as np
import streamlit as st

STATS_9CAT: List[str] = ["PTS", "REB", "AST", "ST", "BLK", "3PTM", "FG%", "FT%", "TO"]
POSITIVE_STATS: List[str] = ["PTS", "REB", "AST", "ST", "BLK", "3PTM", "FG%", "FT%"]
NEGATIVE_STATS: List[str] = ["TO"]
PCT_STATS: List[str] = ["FG%", "FT%"]
VOLUME_COLS: Dict[str, Tuple[str, str]] = {"FG%": ("FGM", "FGA"), "FT%": ("FTM", "FTA")}
UNWANTED_COLS: List[str] = ["Id", "Opponent", "ADP", "Ros%", "+/-", "GP"]

Z_COLOR_BANDS: List[Tuple[int, str]] = [
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


def load_data(selected_path: str, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(selected_path)


def data_selector() -> pd.DataFrame:
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Last Season"
    index = 0 if st.session_state.data_source == "Last Season" else 1
    data_source = st.radio(
        "Data Source", ["Last Season", "This Season So Far"], index=index
    )
    st.session_state.data_source = data_source
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if "use_uploaded" not in st.session_state:
        st.session_state.use_uploaded = False
    if st.session_state.use_uploaded:
        st.info("Using uploaded data")
        if st.button("Clear uploaded data"):
            st.session_state.use_uploaded = False
            if "uploaded_data" in st.session_state:
                del st.session_state["uploaded_data"]
            st.rerun()
    if uploaded_file is not None:
        data = load_data("", uploaded_file)
        st.session_state.uploaded_data = data
        st.session_state.use_uploaded = True
    elif st.session_state.use_uploaded:
        data = st.session_state.uploaded_data
    else:
        selected_path = (
            "last_season.csv"
            if st.session_state.data_source == "Last Season"
            else "this_season_so_far.csv"
        )
        data = load_data(f"data/{selected_path}", None)
    data_with_z = add_z_scores(data)
    st.session_state.data_with_z = data_with_z
    return data_with_z


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
    pre_a_z = compute_team_z_averages(filter_team_players(data, team_a))
    pre_b_z = compute_team_z_averages(filter_team_players(data, team_b))
    post_a_z = compute_team_z_averages(new_a_data)
    post_b_z = compute_team_z_averages(new_b_data)
    pre_a_raw = compute_team_avg_raw(filter_team_players(data, team_a))
    pre_b_raw = compute_team_avg_raw(filter_team_players(data, team_b))
    post_a_raw = compute_team_avg_raw(new_a_data)
    post_b_raw = compute_team_avg_raw(new_b_data)
    return (
        pre_a_z,
        post_a_z,
        pre_b_z,
        post_b_z,
        pre_a_raw,
        post_a_raw,
        pre_b_raw,
        post_b_raw,
        df,
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
        if post_val != pre_val
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
