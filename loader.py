import csv

from dataset import DataFrame


def load_from_csv(path):
    with open(path) as file:
        reader = csv.DictReader(file, delimiter=",")
        return DataFrame(list(reader), list(reader.fieldnames))


def load_config(path):
    with open(path) as file:
        configs = [line.rstrip("\n").split("=") for line in file.readlines()]
        return {config[0]: config[1] for config in configs}

