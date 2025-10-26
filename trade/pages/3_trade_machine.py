import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict
from utils import (
    STATS_9CAT,
    NEGATIVE_STATS,
    PCT_STATS,
    get_teams,
    is_real_team,
    filter_team_players,
    simulate_trade,
    prepare_display_df,
    apply_z_color,
    style_trade_df,
    compute_team_raw_totals,
    data_selector,
)


def setup_trade_teams(data_with_z) -> Tuple[str, str, List[str]]:
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


def display_team_players(data_with_z, team: str, key_suffix: str) -> pd.DataFrame:
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


def process_z_dict_for_punt(
    z_dict: Dict[str, float], selected_stats: List[str]
) -> Dict[str, float]:
    filtered = {k: z_dict[k] for k in selected_stats}
    total = sum(filtered.values())
    filtered["Total"] = total
    return filtered


def display_h2h_analysis(
    selected_team: str, raw_totals: Dict[str, Dict[str, float]], title: str
):
    st.subheader(title)
    for other in sorted([t for t in raw_totals.keys() if t != selected_team]):
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


def handle_trade_simulation(
    team_a: str,
    team_b: str,
    a_to_trade: List[str],
    b_to_trade: List[str],
    data_with_z,
):
    if "trade_results" not in st.session_state:
        st.session_state.trade_results = None
    if st.button("Simulate Trade"):
        results = simulate_trade(data_with_z, team_a, team_b, a_to_trade, b_to_trade)
        st.session_state.trade_results = results
    if st.session_state.trade_results:
        view_mode = st.radio(
            "View", ["Average Z-Scores", "Average Stats"], index=0, key="trade_view"
        )
        show_z = view_mode == "Average Z-Scores"
        real_teams = [t for t in get_teams(data_with_z) if is_real_team(t)]
        pre_a, post_a, pre_b, post_b = st.session_state.trade_results[:4]
        _, _, _, _, pre_a_raw, post_a_raw, pre_b_raw, post_b_raw, post_df = (
            st.session_state.trade_results
        )
        pre_raw_totals = {
            team: compute_team_raw_totals(filter_team_players(data_with_z, team))
            for team in real_teams
        }
        post_raw_totals = {
            team: compute_team_raw_totals(filter_team_players(post_df, team))
            for team in real_teams
        }
        if show_z:
            punt_stats = st.multiselect(
                "Select stats to punt (exclude from total and hide)",
                STATS_9CAT,
                default=[],
            )
            selected_stats = [s for s in STATS_9CAT if s not in punt_stats]
            pre_a = process_z_dict_for_punt(pre_a, selected_stats)
            post_a = process_z_dict_for_punt(post_a, selected_stats)
            pre_b = process_z_dict_for_punt(pre_b, selected_stats)
            post_b = process_z_dict_for_punt(post_b, selected_stats)
        else:
            pre_a, post_a, pre_b, post_b = pre_a_raw, post_a_raw, pre_b_raw, post_b_raw
        st.subheader(f"{team_a} {view_mode}: Pre vs Post")
        a_styled = style_trade_df(pre_a, post_a, show_z)
        st.dataframe(a_styled)
        st.subheader(f"{team_b} {view_mode}: Pre vs Post")
        b_styled = style_trade_df(pre_b, post_b, show_z)
        st.dataframe(b_styled)

        st.subheader(f"H2H Analysis for {team_a}")
        col1, col2 = st.columns(2)
        with col1:
            display_h2h_analysis(team_a, pre_raw_totals, "Pre-Trade")
        with col2:
            display_h2h_analysis(team_a, post_raw_totals, "Post-Trade")

        st.subheader(f"H2H Analysis for {team_b}")
        col1, col2 = st.columns(2)
        with col1:
            display_h2h_analysis(team_b, pre_raw_totals, "Pre-Trade")
        with col2:
            display_h2h_analysis(team_b, post_raw_totals, "Post-Trade")


def main():
    st.set_page_config(layout="wide")
    st.title("Fantasy Basketball Trade Machine")
    st.markdown(
        "### Upload CSV from Fantrax with 'Standard' Stats selected. Select data source below or override with upload."
    )
    st.components.v1.html(
        """
    <script>
        const doc = window.parent.document;
        doc.body.setAttribute('data-theme', 'light');
    </script>
    """,
        height=0,
    )
    data_with_z = data_selector()
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
