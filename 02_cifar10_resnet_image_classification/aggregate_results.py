"""Summarize native/readout accuracies and baseline-to-hardened gains."""
import argparse
import csv
import json
import sys


def summarize(results, metric="best"):
    paired = {}
    for result in results:
        config = result["config"]
        key = (config["dataset"], config["arch"], result["algo"], config["seed"],
               config["epochs"], config["batch_size"], config["readout_mode"])
        recipes = paired.setdefault(key, {})
        recipe = config["recipe_name"]
        if recipe in recipes:
            raise ValueError(f"Duplicate result for {key}, {recipe}")
        recipes[recipe] = result

    for key, recipes in sorted(paired.items()):
        gain = ""
        if "baseline" in recipes and "hardened" in recipes:
            gain = 100 * (recipes["hardened"][f"readout_test_acc_{metric}"]
                          - recipes["baseline"][f"readout_test_acc_{metric}"])
        for recipe, result in sorted(recipes.items()):
            native = 100 * result[f"native_test_acc_{metric}"]
            readout = 100 * result[f"readout_test_acc_{metric}"]
            yield dict(
                dataset=key[0], arch=key[1], algo=key[2], seed=key[3],
                epochs=key[4], batch_size=key[5], readout_mode=key[6],
                recipe=recipe, metric=metric, native_pct=round(native, 4),
                readout_pct=round(readout, 4),
                readout_minus_native_pp=round(readout - native, 4),
                hardening_gain_pp=round(gain, 4) if gain != "" else "",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="JSON files produced by train.py")
    parser.add_argument("--metric", choices=("best", "final"), default="best")
    args = parser.parse_args()
    results = []
    for filename in args.results:
        with open(filename) as handle:
            results.append(json.load(handle))
    rows = list(summarize(results, args.metric))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
