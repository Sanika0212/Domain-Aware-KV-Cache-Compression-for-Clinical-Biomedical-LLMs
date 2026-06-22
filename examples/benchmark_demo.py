"""Example script that runs the benchmark runner and prints results."""
from benchmarks.runner import ExperimentRunner


def main():
    runner = ExperimentRunner(total_budget=8, quantize='float16')
    results = runner.run_dataset()
    for r in results:
        print(f"Note {r['note_id']}: orig={r['orig_counts']}, retained={r['retained']}, budgets={r['budgets']}")


if __name__ == '__main__':
    main()
