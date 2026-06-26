import os
import numpy as np
import pandas as pd

# Print result
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

# Save result
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


    # LOSO results
    rows = []
    confusion_matrix_sum = np.zeros((2, 2), dtype=int)

    for result in results:

        rows.append({
            "Subject": result["subject"],
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"]
        })

        confusion_matrix_sum += result["confusion_matrix"]

    df = pd.DataFrame(rows)
    df.to_csv(

        os.path.join(
            save_dir,
            "loso_results.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )


    # Confusion matrix results
    np.save(

        os.path.join(
            save_dir,
            "confusion_matrix.npy"
        ),

        confusion_matrix_sum

    )


    # Summary
    summary = pd.DataFrame([{

        "Accuracy Mean":
            df["Accuracy"].mean(),

        "Accuracy Std":
            df["Accuracy"].std(),

        "Precision Mean":
            df["Precision"].mean(),

        "Precision Std":
            df["Precision"].std(),

        "Recall Mean":
            df["Recall"].mean(),

        "Recall Std":
            df["Recall"].std(),

        "F1 Mean":
            df["F1"].mean(),

        "F1 Std":
            df["F1"].std()

    }])

    summary.to_csv(

        os.path.join(
            save_dir,
            "summary.csv"
        ),

        index=False,
        encoding="utf-8-sig"

    )