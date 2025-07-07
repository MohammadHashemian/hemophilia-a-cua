from ..utils.logger import get_logger
from time import time
from pathlib import Path
import pandas as pd
import numpy as np

logger = get_logger()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Settings
pd.set_option("display.max_colwidth", None)
logger.debug("Set display.max_colwidth to None")


def load_amarnameh(
    io_amarnameh_1400: Path = PROJECT_ROOT / "data" / "raw" / "amarnameh1400.xlsx",
    io_amarnameh_1399: Path = PROJECT_ROOT / "data" / "raw" / "amarnameh1399.xlsx",
    sheet_name: str = "Dataset",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Amarnameh 1400 and 1399 Excel datasets with renamed columns.
    Returns:
        Tuple of cleaned DataFrames (df_1400, df_1399)
    """
    if not io_amarnameh_1400.exists():
        raise FileNotFoundError(f"{io_amarnameh_1400} does not exist!")

    if not io_amarnameh_1399.exists():
        raise FileNotFoundError(f"{io_amarnameh_1399} does not exist!")

    rename_1400 = {
        "نام شرکت تولید کننده": "manufacturer_company_name",
        "نام شرکت تامین کننده": "supplier_company_name",
        "نام صاحب برند": "brand_owner",
        "توزیع کننده": "supplier",
        "کشور تولید  کننده": "manufacturer_country",
        "نام برند": "brand_name",
        "نام لاتین برند": "brand_latin_name",
        "نام ژنریک": "generic_name",
        "نام ماده موثره": "active_ingredient",
        "تعداد فروش (بسته)": "count_of_package_sold",
        "تعداد در بسته": "count_in_package",
        "فروش عددی": "sale_count",
        "فروش ریالی مصرف کننده": "sale_amount",
        "کد ژنریک": "generic_code",
        "OTC": "otc",
        "بیولوژیک": "biologic",
        "ATC Code": "atc_code",
    }

    logger.info(f"Reading {io_amarnameh_1400}")
    start = time()
    df_1400 = pd.read_excel(io_amarnameh_1400, sheet_name=sheet_name)
    df_1400.rename(columns=rename_1400, inplace=True)
    logger.info(f"Loaded 1400 data in {round(time() - start, 2)}s")

    rename_1399 = {
        "صاحب پروانه": "brand_owner",
        "توزیع کننده": "supplier",
        "نام فرآورده (برند)": "brand_name",
        "تولیدی/وارداتی": "import_or_manufactured",
        "کشور تولید کننده": "manufacturer_country",
        "تحت لیسانس": "licensed",
        "نام ژنریک": "generic_name",
        "کد ژنریک": "generic_code",
        "ماده موثره": "active_ingredient",
        "تعداد فروش (بسته)": "count_of_package_sold",
        "تعداد در بسته": "count_in_package",
        "فروش عددی ": "sale_count",
        "فروش ریالی مصرف کننده ": "sale_amount",
        "OTC": "otc",
        "بیولوژیک": "biologic",
        "ATC Code": "atc_code",
        "IRC": "irc",
    }

    logger.info(f"Reading {io_amarnameh_1399}")
    start = time()
    df_1399 = pd.read_excel(io_amarnameh_1399, sheet_name=sheet_name)
    df_1399.rename(columns=rename_1399, inplace=True)
    logger.info(f"Loaded 1399 data in {round(time() - start, 2)}s")
    logger.info("-" * 64)
    return df_1400, df_1399


def filter_factor_viii(
    df: pd.DataFrame, exclude_recombinant: bool = False
) -> pd.DataFrame:
    """
    Filters for Factor VIII records excluding Von Willebrand and optionally recombinant.
    """
    mask = df["active_ingredient"].str.contains(
        "factor viii", case=False, na=False
    ) & ~df["active_ingredient"].str.contains("von willebrand", case=False, na=False)
    if exclude_recombinant:
        mask &= ~df["generic_name"].str.contains("recombinant", case=False, na=False)
    return df[mask]


def filter_recombinant_factor_viii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters recombinant Factor VIII products and adds price_per_package.
    """
    filtered = df[
        df["active_ingredient"].str.contains("factor viii", case=False, na=False)
        & ~df["active_ingredient"].str.contains("von willebrand", case=False, na=False)
        & df["generic_name"].str.contains("recombinant", case=False, na=False)
    ].copy()

    filtered["price_per_package"] = round(
        filtered["sale_amount"] / filtered["sale_count"]
    )
    return filtered


def aggregate_sales(df: pd.DataFrame, count_column: str = "sale_count") -> pd.DataFrame:
    """
    Groups by generic_name and computes total sales and price per package.
    """
    agg_df = (
        df.groupby("generic_name")
        .agg(sale_count=(count_column, "sum"), sale_amount=("sale_amount", "sum"))
        .reset_index()
    )
    agg_df["price_per_package"] = round(agg_df["sale_amount"] / agg_df["sale_count"])
    return agg_df


def process_amarnameh():
    """
    Cleans amarnameh 1400 and 1399, returns ((factors_1400, factors_1399), recombinant_1400)
    """
    df_1400, df_1399 = load_amarnameh()

    drop_cols = ["supplier", "otc", "biologic", "brand_name"]
    df_1400.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_1399.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ------- 1400 Analysis -------
    logger.info("[Factor VIII] [1400]")
    df_fviii_1400 = filter_factor_viii(df_1400, exclude_recombinant=False)

    dosage_forms = set(df_fviii_1400["generic_name"].dropna())
    recombinant = [d for d in dosage_forms if "recombinant" in d.lower()]

    if recombinant:
        logger.info(f"Found {len(recombinant)} recombinant products.")
        df_recombinant_1400 = filter_recombinant_factor_viii(df_1400)
        logger.info(
            f"{df_recombinant_1400.shape[0]} recombinant rows with price calculated."
        )
        df_fviii_1400 = filter_factor_viii(df_1400, exclude_recombinant=True)
    else:
        logger.info("No recombinant Factor VIII found.")

    logger.info(f"{df_fviii_1400.shape[0]} human factor rows found.")
    df_agg_1400 = aggregate_sales(df_fviii_1400, count_column="sale_count")

    logger.info("-" * 64)

    # ------- 1399 Analysis -------
    logger.info("[Factor VIII] [1399]")
    df_fviii_1399 = filter_factor_viii(df_1399, exclude_recombinant=False)
    logger.info(f"{df_fviii_1399.shape[0]} total rows matched.")

    dosage_forms_1399 = set(df_fviii_1399["generic_name"].dropna())
    recombinant_1399 = [d for d in dosage_forms_1399 if "recombinant" in d.lower()]
    if not recombinant_1399:
        logger.info("No recombinant Factor VIII found in 1399.")

    df_agg_1399 = aggregate_sales(df_fviii_1399, count_column="count_of_package_sold")
    logger.info("Aggregation complete.")

    return ((df_agg_1400, df_agg_1399), (df_recombinant_1400))  # type: ignore


def merge_common_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames on their common columns.
    Returns only the common columns after outer merge.
    """
    common_cols = list(set(df1.columns) & set(df2.columns))
    merged = pd.merge(df1, df2, on=common_cols, how="outer")
    return merged[common_cols]


def merge_and_save(
    path: Path,
    df_agg_1400: pd.DataFrame,
    df_agg_1399: pd.DataFrame,
    df_recombinant_1400: pd.DataFrame,
):
    """
    Saves the analysis results to an Excel file with multiple sheets.
    """
    df_merged = merge_common_columns(df_agg_1400, df_recombinant_1400)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_merged.to_excel(writer, sheet_name="dataset", index=False)
        df_agg_1400.to_excel(writer, sheet_name="1400", index=False)
        df_agg_1399.to_excel(writer, sheet_name="1399", index=False)
        df_recombinant_1400.to_excel(writer, sheet_name="recombinant 1400", index=False)

    logger.info(f"Factor VIII pricing statistics saved at: {path}")


def save(df: pd.DataFrame, path: Path, **kwargs):
    """
    kwargs: passes through pd.to_excel() arguments
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, **kwargs)


def load_global_hemophilia_data(
    path_a: Path = PROJECT_ROOT / "data" / "raw" / "hemophilia-a-2023.csv",
    path_b: Path = PROJECT_ROOT / "data" / "raw" / "hemophilia-b-2023.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and cleans global hemophilia A & B data for analysis.
    Converts percentage and comma-formatted columns to numeric, and filters unknown ages.
    """
    logger.info("Reading world hemophilia tubular hemophilia distribution data A & B.")
    df_ha = pd.read_csv(path_a)
    df_hb = pd.read_csv(path_b)

    logger.info("Converting string columns to numeric...")

    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        # Handle percentage columns
        percent_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" and df[col].str.contains("%").any()
        ]
        for col in percent_cols:
            df[col] = (
                df[col]
                .str.replace(r"[%\s]", "", regex=True)
                .apply(pd.to_numeric, errors="coerce")
                / 100
            )

        # Handle comma-formatted numeric columns
        comma_cols = [
            col
            for col in df.columns
            if col not in percent_cols
            and col != "Country"
            and df[col].dtype == "object"
        ]
        for col in comma_cols:
            df[col] = (
                df[col]
                .str.replace(",", "", regex=False)
                .apply(pd.to_numeric, errors="coerce")
            )

        return df

    df_ha = clean_dataframe(df_ha)
    df_hb = clean_dataframe(df_hb)

    logger.info("Dropping countries with unknown age distribution...")
    df_known_a = df_ha[df_ha["Age not known"] == 0.0].reset_index(drop=True)
    df_known_b = df_hb[df_hb["Age not known"] == 0.0].reset_index(drop=True)

    logger.info("World hemophilia data cleaned successfully.")
    return df_known_a, df_known_b


def load_population_pyramids(
    start: int = 2014,
    stop: int = 2025,
    path: Path = PROJECT_ROOT / "data" / "raw" / "Iran_*.csv",
    dropF: bool = True,
) -> dict[int, pd.DataFrame]:
    """Loads existing csv files and cleans them."""

    years = np.arange(start, stop + 1, dtype=int)
    paths = [Path(str(path).replace("*", str(year))) for year in years]

    dfs = {}
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)

            # Replace "100+" with "100" in Age column
            df["Age"] = df["Age"].replace("100+", "100")

            # Convert Age, F, M to numeric with coercion
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
            df["F"] = pd.to_numeric(df["F"], errors="coerce")
            df["M"] = pd.to_numeric(df["M"], errors="coerce")

            # Drop rows with missing Age or M (or F if you want)
            df = df.dropna(subset=["Age", "M"])
            df["Age"] = df["Age"].astype(int)
            df["M"] = df["M"].astype(int)

            if not dropF:
                df = df.dropna(subset=["F"])
                df["F"] = df["F"].astype(int)
            else:
                df = df.drop(columns=["F"])

            dfs.update({int(years[0]): df})
            years = years[1:]

    return dfs


def load_irc_data(
    path: Path = PROJECT_ROOT / "data" / "raw" / "irc_fda.csv", override: bool = False
):
    df = pd.read_csv(path)
    logger.info("Dropping null values")
    df["price"] = df["price"].str.replace(",", "", regex=False)
    df["price"] = df["price"].str.replace("-", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="raise")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 200000].reindex()
    # Dropping Factor VII and VON WILLEBRAND
    logger.info("Dropping Factor vii and von willebrand")
    df = df[df["english_name"].str.contains("factor viii", case=False, na=False)]
    df = df[~df["english_name"].str.contains("von willebrand", case=False, na=False)]
    # Saving
    output = PROJECT_ROOT / "data" / "processed" / "irc_fda.xlsx"
    if override:
        logger.info("Saving cleaned dataset")
        save(df, output, sheet_name="results", index=False)
    else:
        if not output.exists():
            logger.info("No preexisting file. saving cleaned dataset")
            save(df, output, sheet_name="results", index=False)
        else:
            logger.info(f"File already exist: {output}")
            logger.info(
                "Set override=True if you are sure too override the existing file"
            )
