"""Download the dataset, train the fake-news classifier and print its accuracy.

Usage:  python train_model.py
The app also trains automatically on first launch if models/ is empty.
"""

from components.ml_predictor import MODEL_PATH, train_model


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    bundle = train_model(progress=print)
    m = bundle["metrics"]
    print()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Dataset:              {m['dataset']}")
    print(f"Test-set accuracy:    {m['accuracy']:.2f}%  (n={m['n_test']:,})")
    print(f"Fake precision/recall/F1: {m['precision_fake']:.1f}% / {m['recall_fake']:.1f}% / {m['f1_fake']:.1f}%")
    print(f"Confusion [[fake→fake, fake→real], [real→fake, real→real]]: {m['confusion']}")
    for prefix, name in (("realworld", "Real-world headlines"), ("hard", "Style-swapped hard cases")):
        acc = m.get(f"{prefix}_accuracy")
        print(
            f"{name + ':':26}{acc:.1f}% of confident calls correct "
            f"({m[prefix + '_correct']} right, {m[prefix + '_wrong']} wrong, {m[prefix + '_uncertain']} uncertain of {m[prefix + '_n']})"
            if acc is not None else f"{name}: no confident predictions"
        )


if __name__ == "__main__":
    main()
