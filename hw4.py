# mlflow server --port 8080 --backend-store-uri sqlite:///mlruns.db
# mlflow server --host 127.0.0.1 --port 8080
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ============================================================
# 1. MLflow configuration
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:8080")


# ============================================================
# 2. Load data
# ============================================================

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. Train multiple models and log them to MLflow
# ============================================================

run_ids = []

for n_estimators in [1, 10, 50]:

    with mlflow.start_run() as run:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)

        f1 = f1_score(y_test, predictions)

        # Log parameters
        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        # Log metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Save Run ID
        run_ids.append(run.info.run_id)

        print(
            f"Run ID: {run.info.run_id}"
            f" | n_estimators: {n_estimators}"
            f" | accuracy: {accuracy:.4f}"
            f" | f1: {f1:.4f}"
        )


# ============================================================
# 4. Compare MLflow Runs
# ============================================================

def select_best_run(
    run_ids,
    primary_metric="accuracy",
    secondary_metric="f1_score",
    tolerance=0.0001
):
    """
    Compare MLflow runs using a primary metric.
    If the primary metric values are tied or very close,
    use the secondary metric as a tie-breaker.
    """

    if not run_ids:
        raise ValueError(
            "No run IDs were provided."
        )

    results = []
    invalid_run_ids = []
    missing_metric_runs = []

    for run_id in run_ids:

        # Retrieve run from MLflow
        try:
            run = mlflow.get_run(run_id)

        except Exception:
            invalid_run_ids.append(run_id)
            continue

        metrics = run.data.metrics

        # Check primary metric
        if primary_metric not in metrics:
            missing_metric_runs.append(run_id)
            continue

        # Check secondary metric
        if secondary_metric not in metrics:
            missing_metric_runs.append(run_id)
            continue

        results.append({
            "run_id": run_id,
            "primary_metric": metrics[primary_metric],
            "secondary_metric": metrics[secondary_metric]
        })

    if not results:
        raise ValueError(
            f"No valid runs contain both "
            f"'{primary_metric}' and "
            f"'{secondary_metric}'."
        )

    # --------------------------------------------------------
    # Find the highest primary metric
    # --------------------------------------------------------

    best_primary = max(
        result["primary_metric"]
        for result in results
    )

    # --------------------------------------------------------
    # Find models whose primary metric is tied
    # or extremely close to the best
    # --------------------------------------------------------

    candidates = [
        result
        for result in results
        if abs(
            result["primary_metric"] - best_primary
        ) <= tolerance
    ]

    # --------------------------------------------------------
    # Tie-break using secondary metric
    # --------------------------------------------------------

    if len(candidates) > 1:

        print(
            f"\nPrimary metric '{primary_metric}' "
            f"is tied or very close."
        )

        print(
            f"Using '{secondary_metric}' "
            f"as the tie-breaker."
        )

        best = max(
            candidates,
            key=lambda x: x["secondary_metric"]
        )

    else:
        best = candidates[0]

    return {
        "best_run_id": best["run_id"],
        "best_primary_metric": best["primary_metric"],
        "best_secondary_metric": best["secondary_metric"],
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "invalid_run_ids": invalid_run_ids,
        "missing_metric_runs": missing_metric_runs,
        "all_results": results
    }


# ============================================================
# 5. Select the best model
# ============================================================

result = select_best_run(
    run_ids,
    primary_metric="accuracy",
    secondary_metric="f1_score"
)


# ============================================================
# 6. Display comparison results
# ============================================================

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for item in result["all_results"]:

    print(
        f"Run ID: {item['run_id']}"
        f" | accuracy: {item['primary_metric']:.4f}"
        f" | f1: {item['secondary_metric']:.4f}"
    )


print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best Run ID: "
    f"{result['best_run_id']}"
)

print(
    f"Best Accuracy: "
    f"{result['best_primary_metric']:.4f}"
)

print(
    f"Best F1-score: "
    f"{result['best_secondary_metric']:.4f}"
)


# ============================================================
# 7. Display warnings if necessary
# ============================================================

if result["invalid_run_ids"]:

    print()
    print("Invalid Run IDs:")

    for run_id in result["invalid_run_ids"]:
        print(f"  - {run_id}")


if result["missing_metric_runs"]:

    print()
    print("Runs missing required metrics:")

    for run_id in result["missing_metric_runs"]:
        print(f"  - {run_id}")


# ============================================================
# 8. Load the best model
# ============================================================

best_run_id = result["best_run_id"]

model_uri = f"runs:/{best_run_id}/model"

best_model = mlflow.sklearn.load_model(
    model_uri
)

print()
print("=" * 70)
print("BEST MODEL LOADED")
print("=" * 70)

print(
    f"Model URI: {model_uri}"
)


# ============================================================
# 9. Verify the selected model
# ============================================================

best_predictions = best_model.predict(X_test)

verified_accuracy = accuracy_score(
    y_test,
    best_predictions
)

verified_f1 = f1_score(
    y_test,
    best_predictions
)

print(
    f"Verified Accuracy: {verified_accuracy:.4f}"
)

print(
    f"Verified F1-score: {verified_f1:.4f}"
)