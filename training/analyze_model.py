from pathlib import Path

import joblib
import pandas as pd
import numpy as np


MODEL_PATH = Path(
    "models/champion_model.joblib"
)

OUTPUT_PATH = Path(
    "models/feature_importance.csv"
)


def main():

    print("\n============================================")
    print(" MODEL FEATURE IMPORTANCE ANALYSIS")
    print("============================================")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"\nModel type: "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------
    # Extract the actual estimator from pipeline
    # --------------------------------------------------

    estimator = model.named_steps[
        "model"
    ]

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        estimator.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df[
        "importance_pct"
    ] = (
        importance_df["importance"]
        / importance_df["importance"].sum()
        * 100
    )

    print(
        "\nTop 20 features:"
    )

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        "\n============================================"
    )


if __name__ == "__main__":
    main()