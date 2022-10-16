import random


class DataFrame:

    def __init__(self, rows: list, header: list):
        self.header = header
        self._table = rows

    def __len__(self):
        return len(self._table)

    def insert(self, row: dict):
        self._table.append(row)

    def insert_columns(self, column: str, values: list, position=0):
        if len(self) != len(values):
            raise ValueError("Column length does not match dataframe length")

        for index, row in enumerate(self._table):
            row[column] = values[index]
        self.header.insert(position, column)

    def column(self, column) -> list:
        return [row[column] for row in self._table]

    def rows(self) -> list:
        return self._table

    def row(self, index: int) -> dict:
        return self._table[index]

    def remove(self, column: str):
        return DataFrame([dict(filter(lambda x: x[0] != column, row.items())) for row in self._table],
                         list(filter(lambda x: x != column, self.header)))

    def keep_columns(self, columns: set):
        return DataFrame([dict(filter(lambda x: x[0] in columns, row.items())) for row in self._table],
                         list(filter(lambda x: x in columns, self.header)))

    def unique(self, column: str) -> set:
        return set(self.column(column))

    def value_freq(self, column: str, valueset: set = None) -> dict:
        frequencies = {val: 0 for val in self.unique(column)} if valueset is None else {val: 0 for val in valueset}
        for val in self.column(column):
            frequencies[val] += 1
        return frequencies

    def value_dist(self, column: str, valueset: set = None) -> dict:
        return {val: frequency / len(self) for val, frequency in self.value_freq(column, valueset).items()}

    def group_by(self, column: str) -> dict:
        groups = {value: list() for value in self.unique(column)}
        for row in self.rows():
            groups[row[column]].append(row)

        return {value: DataFrame(rows, self.header) for value, rows in groups.items()}

    def sample(self, k: int, indices=None):
        if indices is None:
            return DataFrame(random.sample(self._table, k), self.header)
        else:
            return DataFrame([self._table[index] for index in indices], self.header)
