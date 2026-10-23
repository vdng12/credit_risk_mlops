import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # to download dataset
    import kagglehub

    # god tier dataframe library
    import polars as pl

    # plotting tools
    import altair as alt
    import matplotlib.pyplot as plt
    import seaborn as sns

    return alt, kagglehub, mo, pl, plt, sns


@app.cell
def _(kagglehub):
    path = kagglehub.competition_download("home-credit-default-risk")
    return (path,)


@app.cell
def _(path, pl):
    df = pl.read_csv(f"{path}/application_train.csv")
    df.describe()
    return (df,)


@app.cell
def _(df, path, pl):
    import csv

    _desc_rows = []
    with open(
        file=f"{path}/HomeCredit_columns_description.csv", encoding="latin-1"
    ) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                _desc_rows.append(
                    {
                        "Table": row[1],
                        "Column": row[2],
                        "Description": row[3],
                        "Special": row[4] if len(row) > 4 else "",
                    }
                )

        col_desc = pl.DataFrame(_desc_rows)

        # Filter for application_train columns only
        app_desc = col_desc.filter(
            pl.col("Table") == "application_{train|test}.csv"
        ).select(["Column", "Description", "Special"])

        cols_in_df = len(df.columns)
        cols_not_in_desc = len([c for c in df.columns if c not in app_desc['Column'].to_list()])
        desc_not_in_df = len([c for c in app_desc['Column'].to_list() if c not in df.columns])

    print(f"Application columns described: {len(app_desc)}")
    print(f"Columns in df: {cols_in_df}")
    print(f"df columns NOT in description: {cols_not_in_desc}")
    print(f"Description columns NOT in df: {desc_not_in_df}")

    app_desc
    return


@app.cell
def _(df, pl):
    numeric_cols = [
        c for c, d in zip(df.columns, df.dtypes) if d in (pl.Int64, pl.Float64)
    ]
    string_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.String]

    print(f"Total columns: {len(df.columns)}")
    print(f"Numerics: {len(numeric_cols)}")
    print(f"Strings: {len(string_cols)}")

    _null_counts = df.null_count()

    null_summary = (
        pl.DataFrame(
            {
                "column": df.columns,
                "null_count": _null_counts.row(0),
                "null_pct": [
                    round(n / len(df) * 100, 2) for n in _null_counts.row(0)
                ],
            }
        )
        .filter(pl.col("null_count") > 0)
        .sort("null_count", descending=True)
    )

    null_summary
    return (null_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There're a lot of missing entries, but that's ok. Missingness is also information. GBT can handle it relatively well. Doing imputation here will only hurts because it introduces massive skews in the dataset.

    You can actually plot it out  and see that the top columns with missing values are housing information.
    """)
    return


@app.cell
def _(alt, null_summary):
    top_nulls = null_summary.head(25)

    alt.Chart(top_nulls.to_pandas()).mark_bar().encode(
        x=alt.X("null_pct:Q", title="Missing %"),
        y=alt.Y("column:N", sort="-x", title=""),
        color=alt.Color(
            "null_pct:Q", legend=None, scale=alt.Scale(scheme="reds")
        ),
    ).properties(
        title="Top 25 Columns by Missing Rate",
        height=400,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look into the TARGET column a little bit
    """)
    return


@app.cell
def _(alt, df, pl):
    default_rate = df["TARGET"].mean()

    class_cnt = (
        df
        .group_by("TARGET")
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.col("TARGET").cast(pl.Utf8).replace({ "0" : "No Default", "1" : "Default"}).alias("label")
        ) 
        .sort("TARGET")
    )

    chart = (
        alt.Chart(class_cnt)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title=None, sort=None),
            y=alt.Y("count:Q", title="Number of Loans"),
            color=alt.Color("label:N", legend=None, scale=alt.Scale(range= ["#4CAF50", "#F44336"])),
        )
        .properties(width=500, height=400, title="Class Distribution - Home Credit Default Risk")
    )

    print(f"Overall Default Rate: {default_rate * 100: .4f}%")
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Class imbalance is pretty common in PD if I recall correctly. Nothing special.

    Btw, Default Rate is a parameter you can tweak in the data generator.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I'll split the application dataframe into 13 column groups (exclude `SK_ID_CURR` and `TARGET`),
    each group corresponds to a "latent factor".

    This grouping is very important for designing the data generator later on.
    """)
    return


@app.cell
def _(df):
    ALL_COLUMN_GROUPS = {
        "client_demographics": [
            "CODE_GENDER",
            "DAYS_BIRTH",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
            "NAME_FAMILY_STATUS",
            "NAME_EDUCATION_TYPE",
            "NAME_TYPE_SUITE",
        ],
        "loan_terms": [
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "NAME_CONTRACT_TYPE",
        ],
        "assets_housing": [
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "OWN_CAR_AGE",
            "NAME_HOUSING_TYPE",
        ],
        "income_employment": [
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
            "DAYS_EMPLOYED",
        ],
        "external_credits": [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ],
        "application_logistics": [
            "WEEKDAY_APPR_PROCESS_START",
            "HOUR_APPR_PROCESS_START",
            "REGION_RATING_CLIENT",
            "REGION_RATING_CLIENT_W_CITY",
            "REGION_POPULATION_RELATIVE",
        ],

        "building_quality" : [
            c for c in df.columns
            if any(
                s in c for s in 
                ['_AVG', '_MODE', '_MEDI', 'HOUSETYPE', 'TOTALAREA', 'WALLSMATERIAL', 'EMERGENCYSTATE', 'FONDKAPREMONT']
            )
        ],


         "credit_bureau_enquiries": [
             c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")
         ],
         "social_circle_risk": [
             "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
             "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"
         ],
         "documentations": [
             c for c in df.columns if c.startswith("FLAG_DOCUMENT_")
         ],
         "contact_reliability": [
             "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
             "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"
         ],
         "address_consistency": [
             "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
             "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
             "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"
         ],
         "application_logistics": [
             "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
             "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
             "REGION_POPULATION_RELATIVE"
         ],
         "registration_stability": [
             "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"
         ],
    }
    return (ALL_COLUMN_GROUPS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I filtered out 5 essential column groups out of 13 groups for the development of the data generator:
    - client_demographics
    - loan_terms
    - assets_housing
    - income_employment
    - external_credits
    - application_logistics

    Let's do some basic analysis on these columns
    """)
    return


@app.cell
def _(ALL_COLUMN_GROUPS):
    L0_GROUPS = [
        "client_demographics",
        "loan_terms",
        "assets_housing",
        "income_employment",
        "application_logistics",
        "external_credits"
    ]

    L0_COLUMNS = [c for g in L0_GROUPS for c in ALL_COLUMN_GROUPS[g]]

    L0_COLUMNS
    return (L0_COLUMNS,)


@app.cell
def _(L0_COLUMNS, df, pl):

    cat_cols = [c for c, t in zip(df.columns, df.dtypes) if c in L0_COLUMNS and t in (pl.Utf8, pl.Categorical, pl.Object)]
    print(f"Categorical columns: {len(cat_cols)} / {len(L0_COLUMNS)}")

    rows = []

    for col in cat_cols:
        vc  = df[col].value_counts().sort("count", descending=True)
        rows.append({
            "column": col,
            "unique": df[col].n_unique(),
            "null_count": df[col].is_null().sum(),
            "null_pct": round(df[col].is_null().sum() / df.height * 100, 2),
            "top_value": vc[0, col] if vc.height > 0 else None,
            "top_count": vc[0, "count"] if vc.height > 0 else 0,
        })

    pl.DataFrame(rows).sort("unique", descending=True)
    return (cat_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ORGANIZATION_TYPE` and `OCCUPATION_TYPE` have 58 and 19 unique values respectively,
    too high in cardinality for one-hot encoding.

    Maybe target encoding is a good choice.
    """)
    return


@app.cell
def _(L0_COLUMNS, cat_cols, df):
    num_cols = [c for c in L0_COLUMNS if c not in cat_cols]
    print(f"Numerical columns: {len(num_cols)} / {len(L0_COLUMNS)}")

    df[num_cols].describe()
    return (num_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - DAYS_BIRTH is by convention, the offset in days from birth to application date and is negative.
    - DAYS_EMPLOYED is also by convention, the offsets in days from employment day to application day. With a sentinel value 365_243 which is a total invalid value by convention, i think it's used to mark unknown entries of employed date.

    By far, no numerical flags are mixed up in here,
    and... I don't know how I should treat EXT_SOURCE_* yet
    """)
    return


@app.cell
def _(df, num_cols, pl):
    target_corr = (
        df.select(
            [pl.corr(c, "TARGET").alias(c) for c in num_cols]
        )
        .unpivot(variable_name="feature", value_name="correlation")
        .sort("correlation")
    )
    return (target_corr,)


@app.cell
def _(target_corr):
    target_corr.tail(5).reverse()
    return


@app.cell
def _(target_corr):
    target_corr.head(5)
    return


@app.cell
def _(ALL_COLUMN_GROUPS, df, plt, sns):
    _cols = df.select(["TARGET", *ALL_COLUMN_GROUPS["external_credits"]])
    _corr = _cols.to_pandas().corr()

    _fig, _ax = plt.subplots(figsize=(5,4))

    sns.heatmap(_corr, annot=True, center=0, vmin=-1, vmax=1, fmt=".3f", ax=_ax)
    _ax.set_title("Correlation: External Credit Scores vs Target")
    _fig
    return


@app.cell
def _(df, pl, plt, sns):
    _ratio_long = (
        df.select(
            credit_income_ratio=pl.col("AMT_CREDIT")
            / pl.col("AMT_INCOME_TOTAL"),
            annuity_income_ratio=pl.col("AMT_ANNUITY")
            / pl.col("AMT_INCOME_TOTAL"),
            credit_term_months=pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY"),
            days_employed_ratio=(
                pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")
            ).clip(0, 1),
            target=pl.col("TARGET"),
        )
        .unpivot(index="target", variable_name="feature", value_name="value")
        .with_columns(
            label=pl.col("target")
            .cast(pl.Utf8)
            .replace({"0": "No Default", "1": "Default"})
        )
        .to_pandas()
    )

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8))

    for _ax, _f in zip(
        _axes.ravel(),
        [
            "credit_income_ratio",
            "annuity_income_ratio",
            "credit_term_months",
            "days_employed_ratio",
        ],
    ):
        _sub = _ratio_long[_ratio_long["feature"] == _f]
        sns.histplot(
            data=_sub,
            x="value",
            hue="label",
            bins=60,
            alpha=0.4,
            palette={"No Default": "#4CAF50", "Default": "#F44336"},
            ax=_ax,
        )
        _ax.set_title(_f)
        _ax.set_xlabel(None)

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I don't really know how to interpret this, btw. But they're feature!

    I remembered reading something about weakly correlated features can still contribute greatly in GBT.
    """)
    return


if __name__ == "__main__":
    app.run()
