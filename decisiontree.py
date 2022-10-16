import math

from dataset import DataFrame


class ID3:

    def __init__(self, maxdepth=None):
        self.features = None
        self.class_label = None
        self.class_label_domain = None
        self.maxdepth = maxdepth
        self.tree = None

    def fit(self, df: DataFrame):
        def _fit(d, features, depth):
            values_dist = d.value_dist(self.class_label, self.class_label_domain)
            for value, ratio in values_dist.items():
                if ratio == 1:
                    return LeafNode(value)

            if not features or (self.maxdepth is not None and depth == self.maxdepth):
                return LeafNode(self._most_common_result(d, values_dist))

            node = FeatureNode(self._max_gain(d, features), d)

            for value, subset in d.group_by(node.feature).items():
                vnode = ValueNode(value)
                node.add_child(vnode)
                vnode.add_child(_fit(subset.remove(node.feature), features.difference({node.feature}), depth + 1))

            return node

        self.class_label = df.header[-1]
        self.class_label_domain = df.unique(self.class_label)
        self.features = set(df.header[:-1])
        self.tree = _fit(df, self.features, depth=0)

    def predict(self, df: DataFrame) -> list:
        def _predict(data, node):
            if isinstance(node, LeafNode):
                return node.result
            elif isinstance(node, ValueNode):
                return _predict(data, node.get_child())
            else:
                child = node.get_child(data[node.feature])
                if child is None:
                    return self._most_common_result(node.df)
                else:
                    return _predict(data, node.get_child(data[node.feature]))

        return [_predict(row, self.tree) for row in df.rows()]

    def _entropy(self, df: DataFrame) -> float:
        entropy = 0
        for value, ratio in df.value_dist(self.class_label, self.class_label_domain).items():
            if ratio == 0:
                continue
            entropy += ratio * math.log2(ratio)

        return -entropy

    def _max_gain(self, df: DataFrame, features: set) -> str:
        gains = {feature: self._gain(df, feature) for feature in features}
        return max(sorted(gains.keys()), key=lambda x: gains[x])

    def _gain(self, df: DataFrame, column: str) -> float:
        count = len(df)
        subset_entropy = self._entropy(df)
        for group in df.group_by(column).values():
            subset_entropy -= len(group) / count * self._entropy(group)
        return subset_entropy

    def _most_common_result(self, df: DataFrame, values_dist=None) -> str:
        if values_dist is None:
            values_dist = df.value_dist(self.class_label)
        return max(sorted(values_dist.keys()), key=lambda x: values_dist[x])

    def print(self) -> str:
        def _traverse(node, _log, depth):
            if isinstance(node, FeatureNode):
                _log.append(str(depth) + ":" + str(node))

            for child in node.children:
                if isinstance(node, FeatureNode):
                    _traverse(child, _log, depth + 1)
                else:
                    _traverse(child, _log, depth)

        log = list()
        _traverse(self.tree, log, depth=0)
        return ", ".join(log)

    def prettyprint(self) -> str:
        def _traverse(node, _log, depth):
            _log.append("-" * depth + str(node))
            for child in node.children:
                _traverse(child, _log, depth + 1)

        log = list()
        _traverse(self.tree, log, depth=0)
        return "\n".join(log)


class Node:

    def __init__(self):
        self.parent = None
        self.children = []

    def add_child(self, node):
        node.parent = self
        self.children.append(node)


class FeatureNode(Node):

    def __init__(self, feature, df: DataFrame):
        super().__init__()
        self.feature = feature
        self.df = df

    def add_child(self, node):
        if not isinstance(node, ValueNode):
            raise ValueError("Malformed tree")
        super().add_child(node)

    def get_child(self, val):
        for child in self.children:
            if child.value == val:
                return child
        return None

    def __str__(self):
        return self.feature


class ValueNode(Node):

    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def add_child(self, node):
        if len(self.children) != 0:
            raise ValueError("Maximum child count exceeded")
        if not isinstance(node, FeatureNode) and not isinstance(node, LeafNode):
            raise ValueError("Malformed tree")
        super().add_child(node)

    def get_child(self):
        return self.children[0]

    def __str__(self):
        return self.value


class LeafNode(Node):

    def __init__(self, result: str):
        super().__init__()
        self.result = result

    def __str__(self):
        return self.result
