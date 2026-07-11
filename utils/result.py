import json
import os

import numpy as np
import pandas as pd


def print_result(result):

    print(
        f"{result['subject']}"
        f" "
        f"Accuracy={result['accuracy']:.4f}"
        f" "
        f"Precision={result['precision']:.4f}"
        f" "
        f"Recall={result['recall']:.4f}"
        f" "
        f"F1={result['f1']:.4f}"
    )


def _serialize_param(param):

    if param is None:
        return ""

    if isinstance(param, dict):
        # Keep CSV-friendly JSON; None -> null
        return json.dumps(param, ensure_ascii=False, sort_keys=True)

    return str(param)


def _mean_roc(results):

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for result in results:
        if "fpr" not in result or "tpr" not in result:
            continue

        fpr = np.asarray(result["fpr"], dtype=np.float64)
        tpr = np.asarray(result["tpr"], dtype=np.float64)

        if len(fpr) < 2:
            continue

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(float(result.get("auc", 0.0)))

    if not tprs:
        return None

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return {
        "fpr": mean_fpr,
        "tpr": mean_tpr,
        "auc": float(np.mean(aucs))
    }


def save_result(results, model_name):

    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    save_dir = os.path.join(
        root_dir,
        "output",
        model_name
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # LOSO metrics (+ params for each fold)
    loso_rows = []
    param_rows = []
    confusion_matrix_sum = np.zeros((2, 2), dtype=int)
    feature_importances = []

    for result in results:

        loso_rows.append({
            "Fold": result.get("fold"),
            "Subject": result["subject"],
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
            "AUC": result.get("auc"),
            "ValScore": result.get("best_score"),
            "BestParam": _serialize_param(result.get("best_param"))
        })

        param_rows.append({
            "Fold": result.get("fold"),
            "Subject": result["subject"],
            "ValScore": result.get("best_score"),
            "BestParam": _serialize_param(result.get("best_param"))
        })

        confusion_matrix_sum += result["confusion_matrix"]

        if "feature_importance" in result:
            feature_importances.append(
                np.asarray(result["feature_importance"], dtype=np.float64)
            )

    df = pd.DataFrame(loso_rows)
    df.to_csv(
        os.path.join(save_dir, "loso_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(param_rows).to_csv(
        os.path.join(save_dir, "train_params.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    np.save(
        os.path.join(save_dir, "confusion_matrix.npy"),
        confusion_matrix_sum
    )

    roc_data = _mean_roc(results)
    if roc_data is not None:
        np.savez(
            os.path.join(save_dir, "roc_curve.npz"),
            fpr=roc_data["fpr"],
            tpr=roc_data["tpr"],
            auc=roc_data["auc"]
        )

    if feature_importances:
        mean_importance = np.mean(feature_importances, axis=0)
        np.save(
            os.path.join(save_dir, "feature_importance.npy"),
            mean_importance
        )

    summary_row = {
        "Accuracy Mean": df["Accuracy"].mean(),
        "Accuracy Std": df["Accuracy"].std(),
        "Precision Mean": df["Precision"].mean(),
        "Precision Std": df["Precision"].std(),
        "Recall Mean": df["Recall"].mean(),
        "Recall Std": df["Recall"].std(),
        "F1 Mean": df["F1"].mean(),
        "F1 Std": df["F1"].std()
    }

    if "AUC" in df.columns and df["AUC"].notna().any():
        summary_row["AUC Mean"] = df["AUC"].mean()
        summary_row["AUC Std"] = df["AUC"].std()

    summary = pd.DataFrame([summary_row])
    summary.to_csv(
        os.path.join(save_dir, "summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
