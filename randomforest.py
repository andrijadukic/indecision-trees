import random

from dataset import DataFrame
from decisiontree import ID3


class RandomForest:

    def __init__(self, num_trees, max_depth, feature_ratio, example_ratio):
        self.features = None
        self.class_label = None
        self.class_label_domain = None
        self.models = None
        self.samples = None
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.feature_ratio = feature_ratio
        self.example_ratio = example_ratio

    def fit(self, df: DataFrame):
        self.class_label = df.header[-1]
        self.class_label_domain = df.unique(self.class_label)
        self.features = set(df.header[:-1])

        self.models = [ID3() for i in range(0, self.num_trees)]

        instance_subset = round(self.example_ratio * len(df))
        feature_subset = round(self.feature_ratio * (len(df.header) - 1))

        self.samples = list()
        for model in self.models:
            bagged_df, sample_columns, sample_indices = self._bagging(df, instance_subset, feature_subset)
            self.samples.append((sample_columns, sample_indices))
            model.fit(bagged_df)

    def predict(self, df: DataFrame):
        model_predictions = [model.predict(df) for model in self.models]

        predictions = list()
        for i in range(0, len(df)):
            votes = {value: 0 for value in self.class_label_domain}
            for prediction in model_predictions:
                votes[prediction[i]] += 1
            predictions.append(max(sorted(votes.keys()), key=lambda x: votes[x]))

        return predictions

    def _bagging(self, df: DataFrame, instance_subset: int, feature_subset: int) -> tuple:
        indices = random.sample(range(0, len(df)), instance_subset)
        sample = df.sample(instance_subset, indices)

        columns = set(random.sample(self.features, feature_subset))

        return sample.keep_columns(columns.union({self.class_label})), columns, indices

    def print_samples(self):
        log = list()
        for sample in self.samples:
            log.append(" ".join(sample[0]))
            log.append(" ".join([str(index) for index in sample[1]]))

        return log
