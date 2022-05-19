import argparse
import math
import multiprocessing
import operator
import os
import random
from collections import defaultdict
from functools import partial
from itertools import groupby
from math import floor
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from dataset_convertors import CONVERTORS, DatasetConvertor, Instance
from utils import read


def _positive_integer(value):
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value!")
    return integer


def _format_text_for_log(text: str = "") -> str:
    return f"{text} " if text else ""


def _split(instances: List[Instance],
           split_ratios: Dict[str, float]) -> Dict[str, List[Instance]]:
    if len(split_ratios) > 1:
        random.shuffle(instances)

    split_instances = {data_portion: defaultdict(list) for data_portion in split_ratios.keys()}
    instance_start_index, instance_end_index = 0, 0
    number_of_instances = len(instances)
    for data_portion, ratio in split_ratios.items():
        print(f"Calculating {_format_text_for_log(data_portion)}split")
        instance_end_index = instance_start_index + floor(number_of_instances * ratio)
        split_instances[data_portion] = instances[instance_start_index:instance_end_index]

        instance_start_index = instance_end_index
    if instance_end_index < number_of_instances:  # rounding errors
        split_instances[next(iter(split_instances.keys()))] += instances[instance_end_index:]
        instance_end_index = number_of_instances

    assert number_of_instances == instance_end_index, "Something went wrong while splitting!"

    summary_grouper = attrgetter("file_path")
    summary = {data_portion: {file_path: len(list(instances))
                              for file_path, instances in groupby(sorted(instances, key=summary_grouper),
                                                                  key=summary_grouper)}
               for data_portion, instances in split_instances.items()}
    print("Summary: ", summary)

    return split_instances


def _get_target_file_path(source_path: str,
                          target_path: Optional[str],
                          data_portion: str = "",
                          file_extension: str = "") -> str:
    source_path = Path(source_path)
    target_path = target_path or str(source_path.parent)
    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, "data")

    file_path = Path(target_path)
    if data_portion:
        file_path = file_path.with_name(file_path.stem + f"-{data_portion}")
    file_path = file_path.with_suffix(file_extension)
    return str(file_path)


def _convert(instances: List[Instance],
             convertor: DatasetConvertor,
             target_path: str,
             data_portion: str):
    source_file_paths = set(map(operator.attrgetter("file_path"), instances))

    data = {}
    for file_path in source_file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            data[file_path] = convertor.parse(file)

    converted_data = defaultdict(list)
    rank = multiprocessing.current_process()._identity[0] - 1
    for instance in tqdm(instances,
                         desc=f"#{rank} Converting data",
                         position=rank):
        path = Path(instance.file_path)
        key = f"{path.parent.name}_{path.stem}"
        target_file_path = _get_target_file_path(instance.file_path,
                                                 target_path,
                                                 data_portion,
                                                 convertor.target_file_extension)
        converted_data[target_file_path].append(convertor.convert(data[instance.file_path],
                                                                  instance.reference,
                                                                  key))

    return converted_data


def _write(target_path: Optional[str],
           convertor: DatasetConvertor,
           all_instances: Dict[str, List[Instance]],
           should_merge_files: bool,
           workers: int):
    with multiprocessing.Pool(workers) as pool:
        print(f"Created a multiprocessing pool with {workers} workers")
        for data_portion, instances in all_instances.items():
            batch_size = math.ceil(len(instances) / workers)
            data = pool.map(partial(_convert,
                                    convertor=convertor,
                                    target_path=target_path,
                                    data_portion=data_portion),
                            (instances[i:i + batch_size] for i in range(0, len(instances), batch_size)))

            for data_batch in tqdm(data,
                                   desc=f"Writing {_format_text_for_log(data_portion)}data"):
                for file_path, converted_data in data_batch.items():
                    with open(file_path, "a" if should_merge_files else "w", encoding="utf-8") as file:
                        try:
                            file.writelines(converted_data)
                        except Exception as e:
                            print("Encountered an error:", repr(e))


def main():
    parser = argparse.ArgumentParser(
        description="Converts datasets, which may be split into different files across sub-directories. "
                    "Instances from different files are merged & shuffled if necessary, "
                    "applying any specified training/validation/testing data splits."
    )

    parser.add_argument("convertor",
                        type=str,
                        choices=CONVERTORS.keys(),
                        help="The type of data conversion to perform.")

    parser.add_argument("--path",
                        type=str,
                        required=True,
                        help="The path where the data is located. "
                             "If the path references a file, that file will be converted. Otherwise, if it references "
                             "a directory, all relevant data files within that directory will be converted.")

    parser.add_argument("--generated_path",
                        type=str,
                        help="The path where the data should be saved. "
                             "If unspecified, the generated file is saved in the same directory as the source file. "
                             "Otherwise, the specified path is used to write to a single merged, shuffled, "
                             "& split (if any of the development/testing ratios are specified) file.")

    parser.add_argument("--dev_proportion",
                        type=float,
                        default=0.2,
                        help="The proportion of instances to use for the development set. "
                             "Ignored if generated_path is unspecified.")

    parser.add_argument("--test_proportion",
                        type=float,
                        default=0.2,
                        help="The proportion of instances to use for the testing set. "
                             "Ignored if generated_path is unspecified.")

    parser.add_argument("--workers",
                        type=_positive_integer,
                        default=os.cpu_count(),
                        help="The number of multiprocessing workers to use while processing the data.")

    args = parser.parse_args()

    convertor = CONVERTORS[args.convertor]
    instances = read(args.path, convertor)

    should_merge_files = args.generated_path
    split_ratios = {"": 1}  # no split
    if should_merge_files:
        assert 0 <= args.dev_proportion <= 1
        assert 0 <= args.test_proportion <= 1
        if args.dev_proportion > 0 or args.test_proportion > 0:
            split_ratios = {
                "train": 1 - args.dev_proportion - args.test_proportion,
            }
            if args.dev_proportion > 0:
                split_ratios["dev"] = args.dev_proportion
            if args.test_proportion > 0:
                split_ratios["test"] = args.test_proportion
    assert sum(split_ratios.values()) == 1
    split_instances = _split(instances, split_ratios)

    _write(args.generated_path, convertor, split_instances, should_merge_files, args.workers)


if __name__ == "__main__":
    main()
