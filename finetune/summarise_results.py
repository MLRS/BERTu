import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read(path_pattern: str) -> Dict[Path, Dict[str, float]]:
    results = {}
    for path in glob.iglob(path_pattern):
        file_path = Path(path)
        if not file_path.is_file():
            raise ValueError("Path pattern should refer to a file!")

        with open(file_path, "r", encoding="utf-8") as file:
            results[file_path] = json.load(file)

    return results


def extract(files: Dict[Path, Dict[str, float]], keys: List[str]) -> Dict[str, Dict[str, List[float]]]:
    regex = re.compile(r"^(.*)\.\d+$")
    values = defaultdict(lambda: defaultdict(list))
    for path, file in files.items():
        name = regex.findall(str(path.parent.name))[0]
        for key in keys:
            values[name][key].append(float(file[key]))
    return values


def summarise(grouped_values: Dict[str, Dict[str, List[float]]]):
    print("name", end="")
    for key in list(next(iter(grouped_values.values())).keys()):
        print(f"\t{key} mean\t{key} standard deviation", end="")
    print()

    for name, all_values in grouped_values.items():
        print(name, end="")
        for key, values in all_values.items():
            mean = statistics.mean(values)
            standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"\t{mean:.4f}\t{standard_deviation:.4f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metric",
                        type=str,
                        nargs="+",
                        help="The JSON to use to extract the result.")
    parser.add_argument("--file_pattern",
                        type=str,
                        required=True,
                        help="The file pattern where the data is located.")
    args = parser.parse_args()

    files = read(args.file_pattern)
    values = extract(files, args.metric)
    summarise(values)


if __name__ == '__main__':
    main()
