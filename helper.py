from collections import OrderedDict

from dataset import DataFrame


class ModelFitnessAnalysis:

    def __init__(self, model, df: DataFrame, results: list):
        self.model = model
        self.df = df
        self.results = results

    def accuracy(self) -> float:
        correct = 0
        for index, result in enumerate(self.results):
            if self.df.row(index)[self.model.class_label] == result:
                correct += 1

        return correct / len(self.df)

    def cm(self):
        cm = OrderedDict()
        sorted_domain = sorted(self.model.class_label_domain)
        for val1 in sorted_domain:
            for val2 in sorted_domain:
                cm[(val1, val2)] = 0

        for index, result in enumerate(self.results):
            cm[(self.df.row(index)[self.model.class_label], result)] += 1

        return cm

    def print_cm(self):
        confusionmatrix = self.cm()
        for index, resultpair in enumerate(confusionmatrix.items()):
            if (index - 1) % len(self.model.class_label_domain) == 0:
                print(str(resultpair[1]))
            else:
                print(str(resultpair[1]), end=" ")
