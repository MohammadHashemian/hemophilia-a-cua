import polars as pl

from app.notebook.psa.presentation import render_table, table_sections


def test_wide_tables_are_split_without_losing_calculated_columns():
    frame = pl.DataFrame(
        {
            "comparison": ["A vs B"],
            "paired_iterations": [100],
            "delta_cost": [10.0],
            "delta_cost_ci_low": [8.0],
            "delta_cost_ci_high": [12.0],
            "delta_qaly": [0.5],
            "delta_qaly_ci_low": [0.4],
            "delta_qaly_ci_high": [0.6],
            "icer": [20.0],
            "interpretation": ["Cost-effective"],
            "delta_nmb": [30.0],
            "probability_cost_effective": [0.9],
        }
    )

    sections = table_sections({"ICER and NMB": frame})

    assert [section.title for section in sections] == [
        "Incremental costs",
        "Incremental QALYs",
        "Cost-effectiveness decision",
    ]
    assert max(section.data.width for section in sections) <= 5
    assert set().union(*(set(section.data.columns) for section in sections)) == set(
        frame.columns
    )


def test_renderer_prevents_wrapping_and_allows_horizontal_scrolling():
    html = render_table(pl.DataFrame({"scenario": ["A"], "value": [1234.56789]}))

    assert "overflow-x: auto" in html
    assert "white-space: nowrap" in html
    assert "1,234.568" in html


def test_renderer_supports_notebook_and_system_color_themes():
    html = render_table(pl.DataFrame({"scenario": ["A"], "value": [1.0]}))

    assert "--jp-layout-color1" in html
    assert "--jp-ui-font-color1" in html
    assert "@media (prefers-color-scheme: dark)" in html
    assert "--psa-table-bg: #161b22" in html
