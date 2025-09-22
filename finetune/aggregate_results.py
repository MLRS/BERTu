import os
import sys
import json
import statistics
from collections import defaultdict
from pathlib import Path


base_directory = Path(sys.argv[1])
task = sys.argv[2]
metric = sys.argv[3]

values = []
for directory_path in base_directory.rglob(f"{task}.[0-9]"):
    with open(os.path.join(directory_path, "all_results.json"), "r") as file:
        data = json.load(file)
        values.append(data[f"test_{metric}"] * 100)

print("name mean stdev seeds")
print(base_directory.name, statistics.mean(values), statistics.stdev(values), len(values))
