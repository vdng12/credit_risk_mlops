import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # to download dataset
    import kagglehub

    # god tier dataframe and numerical computing libraries
    # (I love numpy <3)
    import polars as pl
    import numpy as np
    import pandas as pd

    # plotting tools
    import altair as alt
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ML toolkit
    from sklearn.preprocessing import TargetEncoder, StandardScaler
    import lightgbm as lgb

    return (
        StandardScaler,
        TargetEncoder,
        kagglehub,
        lgb,
        mo,
        np,
        pd,
        pl,
        plt,
        sns,
    )


@app.cell
def _(kagglehub, pl):
    path = kagglehub.competition_download('home-credit-default-risk')
    df = pl.read_csv(f'{path}/application_train.csv')
    df.describe()
    return df, path


@app.cell
def _():
    features = [
      "CODE_GENDER",
      "DAYS_BIRTH",
      "CNT_CHILDREN",
      "CNT_FAM_MEMBERS",
      "NAME_FAMILY_STATUS",
      "NAME_EDUCATION_TYPE",
      "NAME_TYPE_SUITE",
      "AMT_CREDIT",
      "AMT_ANNUITY",
      "AMT_GOODS_PRICE",
      "NAME_CONTRACT_TYPE",
      "FLAG_OWN_CAR",
      "FLAG_OWN_REALTY",
      "OWN_CAR_AGE",
      "NAME_HOUSING_TYPE",
      "AMT_INCOME_TOTAL",
      "NAME_INCOME_TYPE",
      "OCCUPATION_TYPE",
      "ORGANIZATION_TYPE",
      "DAYS_EMPLOYED",
      "WEEKDAY_APPR_PROCESS_START",
      "HOUR_APPR_PROCESS_START",
      "REGION_RATING_CLIENT",
      "REGION_RATING_CLIENT_W_CITY",
      "REGION_POPULATION_RELATIVE",
      "EXT_SOURCE_1",
      "EXT_SOURCE_2",
      "EXT_SOURCE_3"
    ]
    return (features,)


@app.cell
def _(df, features, pl):
    # this cell sorts out the categorical vs. numerical columns
    cat_cols = [c for c, t in zip(df.columns, df.dtypes) if c in features and t in (pl.Utf8, pl.Categorical, pl.Object)]
    num_cols = [c for c in features if c not in cat_cols]
    return cat_cols, num_cols


@app.cell
def _(cat_cols):
    # this cell sorts out high cardinality vs low cardinality columns
    high_card_cols = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
    low_card_cols = [c for c in cat_cols if c not in high_card_cols]
    return high_card_cols, low_card_cols


@app.cell
def _(df, features, pl):
    anomaly_cnt = df.filter(pl.col("DAYS_EMPLOYED") == 365243).height

    print("Numbers of row with abnomal DAYS_EMPLOYED :", anomaly_cnt)

    clean_df = (
        df
        .select(features)
        .with_columns(
           pl.when(pl.col("DAYS_EMPLOYED") == 365243).then(None).otherwise(pl.col("DAYS_EMPLOYED")).alias("DAYS_EMPLOYED")
        )
    )

    clean_df.describe()
    return (clean_df,)


@app.cell
def _(clean_df, pl):
    augmented_df = clean_df.with_columns(
            (pl.col("AMT_CREDIT")/ pl.col("AMT_INCOME_TOTAL")).alias("CREDIT_INCOME_RATIO"),
            (pl.col("AMT_ANNUITY")/ pl.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_RATIO"),
            (pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY")).alias("CREDIT_TERM_MONTHS"),
            (pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")).clip(0, 1).alias("DAYS_EMPLOYED_RATIO"),
    )

    augmented_df
    return


@app.cell
def _(
    StandardScaler,
    TargetEncoder,
    features,
    high_card_cols,
    low_card_cols,
    np,
    num_cols,
    pl,
):
    def build_dataset(raw_df, target=None, enc=None, scaler=None, dummy_cols=None):
        """Turn a raw applications df into the modelling matrix X.

        Clean -> augment -> encode -> scale, and return X alongside the fitted
        transformers and the one-hot column layout.

        * train: call with `target=y_train` -> fits enc/scaler and records the
          dummy column template.
        * test: pass the fitted `enc`, `scaler` and train `dummy_cols` -> the
          output lines up column-for-column with the train matrix.
        """
        cleaned = (
            raw_df.select(features)
            .with_columns(
                pl.when(pl.col("DAYS_EMPLOYED") == 365243)
                .then(None)
                .otherwise(pl.col("DAYS_EMPLOYED"))
                .alias("DAYS_EMPLOYED")
            )
        )

        augmented = cleaned.with_columns(
            (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("CREDIT_INCOME_RATIO"),
            (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_RATIO"),
            (pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY")).alias("CREDIT_TERM_MONTHS"),
            (pl.col("DAYS_EMPLOYED") / pl.col("DAYS_BIRTH")).clip(0, 1).alias("DAYS_EMPLOYED_RATIO"),
        )

        model_features = [
            *num_cols,
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "CREDIT_TERM_MONTHS",
            "DAYS_EMPLOYED_RATIO",
        ]

        # high-cardinality cats -> target encoding
        X_high = augmented.select(high_card_cols).to_numpy()
        if enc is None:
            enc = TargetEncoder(smooth="auto")
            X_high = enc.fit_transform(X_high, target)
        else:
            X_high = enc.transform(X_high)

        # low-cardinality cats -> one-hot, aligned to the train layout
        dummies = augmented.select(low_card_cols).to_dummies(columns=low_card_cols)
        if dummy_cols is None:
            dummy_cols = dummies.columns
        else:
            for col in dummy_cols:
                if col not in dummies.columns:
                    dummies = dummies.with_columns(pl.lit(0).cast(pl.UInt8).alias(col))
            dummies = dummies.select(dummy_cols)

        # numeric columns -> standard scaling
        X_num = augmented.select(model_features).to_numpy()
        if scaler is None:
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X_num)
        else:
            X_num = scaler.transform(X_num)

        X = np.hstack([X_high, dummies.to_numpy(), X_num])
        return X, enc, scaler, dummy_cols

    return (build_dataset,)


@app.cell
def _(build_dataset, df, path, pl):
    # ---- build train + test matrices through the same pipeline ----
    test_df = pl.read_csv(f"{path}/application_test.csv")

    # train: fit the transformers and capture the one-hot column layout
    y_train = df["TARGET"].to_numpy()
    X_train, enc, scaler, dummy_cols = build_dataset(df, target=y_train)

    # test: reuse everything fit on train so the columns line up
    X_test = build_dataset(test_df, enc=enc, scaler=scaler, dummy_cols=dummy_cols)[0]

    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    assert X_train.shape[1] == X_test.shape[1], "train/test feature counts differ"
    return X_test, X_train, y_train


@app.cell
def _(X_test, X_train, lgb, mo, pd, plt, sns, y_train):
    # ---- validation: 5-fold CV on the train set ----
    train_data = lgb.Dataset(X_train, label=y_train)

    param = {"num_leaves": 31, "objective": "binary", "metric": "auc"}
    num_rounds = 100

    cv_results = lgb.cv(
        param,
        train_data,
        num_boost_round=num_rounds,
        nfold=5,
        metrics="auc",
        seed=42,
        eval_train_metric=True,
    )

    cv_curve = pd.DataFrame(
        {
            "round": list(range(num_rounds)),
            "train_auc": cv_results["train auc-mean"],
            "valid_auc": cv_results["valid auc-mean"],
        }
    ).melt("round", var_name="split", value_name="auc")

    cv_fig, cv_ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=cv_curve, x="round", y="auc", hue="split", ax=cv_ax)
    cv_ax.set_xlabel("boosting round")
    cv_ax.set_ylabel("AUC")
    cv_ax.set_title("LightGBM 5-fold CV AUC")
    cv_ax.legend(title=None)

    # ---- final model on full train, then score the test set ----
    bst = lgb.train(param, train_data, num_rounds)
    test_preds = bst.predict(X_test)

    hist_fig, hist_ax = plt.subplots(figsize=(8, 4))
    sns.histplot(test_preds, bins=50, kde=True, ax=hist_ax)
    hist_ax.set_xlabel("predicted default probability")
    hist_ax.set_title("Test set predictions")

    print(f"CV valid AUC @ round {num_rounds}: {cv_results['valid auc-mean'][-1]:.4f}")
    print(f"mean test prediction: {test_preds.mean():.4f}")

    mo.vstack([mo.mpl.interactive(cv_fig), mo.mpl.interactive(hist_fig)])
    return


if __name__ == "__main__":
    app.run()
