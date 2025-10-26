import streamlit as st
from typing import List
import pandas as pd
from utils import (
    STATS_9CAT,
    NEGATIVE_STATS,
    PCT_STATS,
    get_teams,
    is_real_team,
    filter_team_players,
    compute_team_z_averages,
    compute_team_avg_raw,
    compute_ranks,
    compute_team_raw_totals,
    apply_colors,
    data_selector,
)


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
    display_columns = list(STATS_9CAT)
    if is_z:
        rank_sums = ranks_df.sum(axis=1)
        sum_ranks = rank_sums.rank(ascending=True, method="min").astype(int)
        ranks_df["Rank Sum"] = sum_ranks
        display_columns = ["Rank Sum"] + display_columns
    display_df = pd.DataFrame(
        index=summaries_df.index, columns=display_columns, dtype=object
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
    if is_z:
        for team in summaries_df.index:
            sum_val = int(rank_sums.at[team])
            sum_rk = sum_ranks.at[team]
            display_df.at[team, "Rank Sum"] = f"rk {sum_rk} - {sum_val}"
    styled = display_df.style.apply(apply_colors, ranks_df=ranks_df, axis=None)
    st.dataframe(styled, hide_index=False)
    display_h2h_analysis(data_with_z, real_teams)


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
    display_team_summaries(data_with_z)


if __name__ == "__main__":
    main()
