def run_model(
    model_name,
    train_fn,
    evaluate_fn,
    loso_dataset,
    save_fn,
    param_grid
):
    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    all_results = []

    for fold, split in enumerate(loso_dataset, start=1):

        train_set = split["train_set"]
        val_set = split["val_set"]
        test_set = split["test_set"]
        test_subject = split["test_subject"]

        print(f"Fold {fold}")
        print(f"Test Subject: {test_subject}")

        train = train_fn(
            train_set,
            val_set,
            param_grid
        )
        model = train["best_model"]

        metrics = evaluate_fn(
            model,
            test_set
        )

        result = {
            "subject": test_subject,
            **metrics
        }

        all_results.append(result)

    save_fn(all_results, model_name)