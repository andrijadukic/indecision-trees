import argparse
from dataset import DataFrame

from decisiontree import ID3
from helper import ModelFitnessAnalysis
from randomforest import RandomForest
from loader import load_config, load_from_csv


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train",
                        type=str,
                        help="Path to the train dataset")
    parser.add_argument("--test",
                        type=str,
                        help="Path to the test dataset")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        help="Path to the test dataset")

    args = parser.parse_args()

    config = load_config(args.config)

    max_depth = None
    if "max_depth" in config:
        max_depth = int(config["max_depth"])
        if max_depth == -1:
            max_depth = None

    train_dataframe = load_from_csv(args.train)
    test_dataframe = load_from_csv(args.test)

    model_name = config["model"]
    if model_name == "ID3":
        run_id3(ID3(max_depth),
                train_dataframe,
                test_dataframe)
    elif model_name == "RF":
        num_trees = 1 if "num_trees" not in config else int(config["num_trees"])
        feature_ratio = 1. if "feature_ratio" not in config else float(config["feature_ratio"])
        example_ratio = 1. if "example_ratio" not in config else float(config["example_ratio"])
        run_random_forest(RandomForest(num_trees,
                                       max_depth,
                                       feature_ratio,
                                       example_ratio),
                          train_dataframe,
                          test_dataframe)


def run_model_fitness_analysis(model, test_df: DataFrame, predictions):
    fit_analysis = ModelFitnessAnalysis(model, test_df, predictions)
    print(str(round(fit_analysis.accuracy(), 5)))
    fit_analysis.print_cm()


def run_id3(model: ID3, train_df: DataFrame, test_df: DataFrame):
    model.fit(train_df)
    print(model.print())

    predictions = model.predict(test_df)
    print(" ".join(predictions))

    run_model_fitness_analysis(model, test_df, predictions)


def run_random_forest(model: RandomForest, train_df: DataFrame, test_df: DataFrame):
    model.fit(train_df)

    for sample in model.print_samples():
        print(sample)

    predictions = model.predict(test_df)
    print(" ".join(predictions))

    run_model_fitness_analysis(model, test_df, predictions)


if __name__ == '__main__':
    main()
