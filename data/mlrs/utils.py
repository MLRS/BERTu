from pathlib import Path
from typing import List

from tqdm import tqdm

from dataset_convertors import DatasetConvertor, Instance


def read(path: str, convertor: DatasetConvertor) -> List[Instance]:
    path = Path(path)
    if path.is_dir():
        instances = []
        for file_path in tqdm(path.rglob(convertor.source_file_regex),
                              desc="Indexing"):
            instances += read(str(file_path), convertor)
        return instances
    elif path.is_file():
        with open(path, "r", encoding="utf-8") as file:
            instance_references = convertor.index(convertor.parse(file))
            return [Instance(str(path), reference) for reference in instance_references]
    else:
        raise FileNotFoundError(f"Invalid path \"{path}\"")
