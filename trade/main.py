import streamlit as st
from utils import data_selector


def main() -> None:
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
    st.markdown(
        """
        **Z-scores standardize stats**: $$ z = \\frac{value - \\mu}{\\sigma} $$.
        For TO, inverted since lower is better.

         **Weighted Z for FG%/FT%**: Z-score of impact, where impact = makes - (league_avg_% * attempts).
               Explains value of % with volume - high % on many shots > high % on few.
        """
    )


if __name__ == "__main__":
    main()
