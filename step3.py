"""STEP 3 - Train + compare classifiers, log everything to MLflow, save the best.
Run:  python -m src.train_classifier
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.common import (BASE_DIR, CAT_COLS, MODEL_DIR, NUM_COLS, RANDOM_STATE,
                        build_preprocessor, load_clean)

# The label is built from BHK, availability and infrastructure columns, so a model
# that sees them can partly "re-learn the rule". Set True for a stricter test.
DROP_LABEL_INGREDIENTS = False
LABEL_INGREDIENTS = ["BHK", "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score",
                     "Public_Transport_Accessibility", "Availability_Status"]
SAMPLE_SIZE = None   # e.g. 50000 to train faster on a slow laptop


def log_sklearn_model(pipe):
    """Log the pipeline to MLflow using the cloudpickle format (works on MLflow 2 and 3)."""
    try:
        return mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    except TypeError:  # older MLflow versions use artifact_path instead of name
        return mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_clean()
    if SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    num = [c for c in NUM_COLS if not (DROP_LABEL_INGREDIENTS and c in LABEL_INGREDIENTS)]
    cat = [c for c in CAT_COLS if not (DROP_LABEL_INGREDIENTS and c in LABEL_INGREDIENTS)]
    X, y = df[num + cat], df["Good_Investment"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
    }

    mlflow.set_tracking_uri("sqlite:///" + (BASE_DIR / "mlflow.db").as_posix())
    mlflow.set_experiment("RealEstate_Classification")

    results, best = {}, {"f1": -1}
    for name, model in models.items():
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("model", model)])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            proba = pipe.predict_proba(X_te)[:, 1]
            m = {"accuracy": accuracy_score(y_te, pred),
                 "precision": precision_score(y_te, pred),
                 "recall": recall_score(y_te, pred),
                 "f1": f1_score(y_te, pred),
                 "roc_auc": roc_auc_score(y_te, proba)}

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"p_{k}": v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metrics(m)

            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, colorbar=False)
            ax.set_title(f"Confusion matrix - {name}")
            cm_path = MODEL_DIR / f"cm_{name}.png"
            fig.savefig(cm_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            try:
                info = log_sklearn_model(pipe)
                model_uri = info.model_uri
            except Exception as e:   # tracking problems must never stop the training
                print("MLflow model logging skipped:", e)
                model_uri = None

            results[name] = {k: round(float(v), 4) for k, v in m.items()}
            print(name, results[name])
            if m["f1"] > best["f1"]:
                best = {"f1": m["f1"], "name": name, "pipe": pipe, "model_uri": model_uri}

    # ---- save best model + metrics + feature importance ----------------
    joblib.dump(best["pipe"], MODEL_DIR / "classifier.joblib", compress=3)
    with open(MODEL_DIR / "clf_metrics.json", "w") as f:
        json.dump({"best": best["name"], "models": results}, f, indent=2)

    est = best["pipe"].named_steps["model"]
    names = best["pipe"].named_steps["pre"].get_feature_names_out()
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    if imp is None:
        imp = np.zeros(len(names))
    (pd.DataFrame({"feature": names, "importance": imp})
       .sort_values("importance", ascending=False)
       .to_csv(MODEL_DIR / "clf_importance.csv", index=False))

    if best["model_uri"]:
        try:
            mlflow.register_model(best["model_uri"], "good_investment_classifier")
        except Exception as e:      # registry is optional
            print("Model registry skipped:", e)
    print("\nBest classifier:", best["name"])


if __name__ == "__main__":
    main()