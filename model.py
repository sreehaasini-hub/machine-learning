"""Gradient Boosting regression (primary) + supplementary classifier. Trained once, cached."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_processing import CAT, NUM

RANDOM_STATE = 42
TEST_SIZE = 0.2
GB_PARAMS = dict(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8,
                 random_state=RANDOM_STATE)
FEATURES = CAT + NUM


def _pipeline(estimator, num_cols):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                             ("num", "passthrough", num_cols)])
    return Pipeline([("prep", pre), ("gb", estimator)])


def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"mae": mean_absolute_error(y_true, y_pred), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": r2_score(y_true, y_pred)}


def _importances(pipe, num_cols):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["gb"].feature_importances_
    agg, detail = {}, {}
    for n, v in zip(names, imp):
        kind, rest = n.split("__", 1)
        base = next((c for c in CAT if rest.startswith(c + "_")), rest) if kind == "cat" else rest
        agg[base] = agg.get(base, 0) + float(v)
        detail[f"{base}: {rest[len(base) + 1:]}" if kind == "cat" else rest] = float(v)
    top = sorted(detail.items(), key=lambda kv: -kv[1])[:12]
    return sorted(agg.items(), key=lambda kv: -kv[1]), top


def train_all(df):
    X, y = df[FEATURES], df["Yield"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # ---- Primary model: regression on continuous Yield ----
    reg = _pipeline(GradientBoostingRegressor(**GB_PARAMS), NUM).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    agg, detail = _importances(reg, NUM)

    # ---- Transparency check: same model WITHOUT Production (Yield = Production / Area) ----
    num_np = [c for c in NUM if c != "Production"]
    reg_np = _pipeline(GradientBoostingRegressor(**GB_PARAMS), num_np).fit(
        Xtr.drop(columns="Production"), ytr)
    m_np = _metrics(yte, reg_np.predict(Xte.drop(columns="Production")))

    # ---- Supplementary classifier: Low / Medium / High yield (tertiles of Yield) ----
    q1, q2 = df.Yield.quantile([1 / 3, 2 / 3])
    labels = ["Low", "Medium", "High"]
    cls_y = pd.cut(df.Yield, [-np.inf, q1, q2, np.inf], labels=labels)
    Ctr, Cte, ctr, cte = train_test_split(X, cls_y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                          stratify=cls_y)
    clf = _pipeline(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                               subsample=0.8, random_state=RANDOM_STATE), NUM).fit(Ctr, ctr)
    cp = clf.predict(Cte)
    p, r, f, _ = precision_recall_fscore_support(cte, cp, average="macro", zero_division=0)

    return {
        "reg": reg, "n_train": len(Xtr), "n_test": len(Xte), "metrics": _metrics(yte, pred),
        "metrics_no_prod": m_np, "y_test": yte.to_numpy(), "y_pred": pred,
        "imp_agg": agg, "imp_detail": detail,
        "cls": {"labels": labels, "q1": float(q1), "q2": float(q2),
                "cm": confusion_matrix(cte, cp, labels=labels).tolist(),
                "accuracy": accuracy_score(cte, cp), "precision": p, "recall": r, "f1": f,
                "n_test": len(Cte)},
    }


def predict_yield(artifacts, row):
    X = pd.DataFrame([row])[FEATURES]
    return float(artifacts["reg"].predict(X)[0])
