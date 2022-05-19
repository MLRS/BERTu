import argparse
import os
import os.path
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from dataset_convertors import CONVERTORS, Instance
from utils import read


def _count(instances: List[Instance], path: str):
    path_prefix = os.path.commonprefix(list(set(instance.file_path for instance in instances)))
    counts = defaultdict(int)
    for instance in instances:
        file_path = Path(instance.file_path[len(path_prefix):])  # remove user-specific location
        group = file_path.name  # group by file by default
        if len(Path(file_path).parts) > 1:  # group by domain instead
            group = file_path.parts[0]
        counts[group] += 1

    counts = list(zip(*sorted(([name, count] for name, count in counts.items()), key=itemgetter(1))))
    index = list(range(len(counts[0])))
    figure, axis = plt.subplots(figsize=(12, round(len(index) / 4)))
    axis.barh(index, counts[1])
    axis.set_yticks(index)
    axis.set_yticklabels(counts[0])
    axis.set_xlabel("number of instances")
    axis.set_ylabel("category")
    axis.bar_label(axis.containers[0])
    plt.savefig(os.path.join(path or "", "counts.png"), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(
        description="Analyses datasets, which may be split into different files across sub-directories. "
                    "The number of instances in each domain/sub-directory are analysed & visualised as a graph."
    )

    parser.add_argument("convertor",
                        type=str,
                        choices=CONVERTORS.keys(),
                        help="The type of data to analyse.")

    parser.add_argument("--path",
                        type=str,
                        required=True,
                        help="The path where the data is located. "
                             "If the path references a file, that file will be analysed. Otherwise, if it references "
                             "a directory, all relevant data files within that directory will be analysed.")

    parser.add_argument("--analysis_path",
                        type=str,
                        help="The path where any analysis data is dumped. "
                             "If unspecified, the working directory is used.")

    args = parser.parse_args()

    convertor = CONVERTORS[args.convertor]
    instances = read(args.path, convertor)

    _count(instances, args.analysis_path)


if __name__ == "__main__":
    main()
