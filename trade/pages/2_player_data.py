import streamlit as st
from typing import List
from utils import (
    STATS_9CAT,
    PCT_STATS,
    prepare_display_df,
    apply_z_color,
    data_selector,
)


def filter_by_search(df, search: str):
    if search:
        return df[df["Player"].str.contains(search, case=False)]
    return df


def compute_total_z(display_df, z_cols: List[str], punted: List[str]):
    active_cols = [c for c in z_cols if c.split("_")[1] not in punted]
    return display_df[active_cols].sum(axis=1)


def add_total_z(display_df, z_cols: List[str], punted: List[str]):
    display_df = display_df.copy()
    display_df.insert(2, "Z_Total", compute_total_z(display_df, z_cols, punted))
    return display_df


def display_player_data(data_with_z):
    st.subheader("Player Data")
    view_mode = st.radio("View", ["Z-Scores", "Raw Stats"], index=0, key="full_view")
    show_z = view_mode == "Z-Scores"
    search = st.text_input("Search Player", "")
    punted = st.multiselect("Punt Categories", STATS_9CAT, []) if show_z else []
    display_df = prepare_display_df(data_with_z, show_z)
    display_df = filter_by_search(display_df, search)
    if show_z:
        z_cols = [f"Z_{stat}" for stat in STATS_9CAT]
        display_df = add_total_z(display_df, z_cols, punted)
        styled = display_df.style.applymap(apply_z_color, subset=z_cols + ["Z_Total"])
    else:
        styled = display_df.style
    st.dataframe(
        styled,
        column_config={"Player": st.column_config.Column(pinned=True)},
        hide_index=True,
        height=600,
        use_container_width=True,
    )


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
    display_player_data(data_with_z)


if __name__ == "__main__":
    main()
