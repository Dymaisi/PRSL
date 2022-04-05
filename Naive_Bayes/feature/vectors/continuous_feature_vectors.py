from collections import defaultdict

from Naive_Bayes.feature.vectors.feature_vectors import FeatureVectors
from Naive_Bayes.statistics import gaussian_pdf
from Naive_Bayes.statistics import mean
from Naive_Bayes.statistics import variance


class ContinuousFeatureVectors(FeatureVectors):
    """Collection of continuous feature vectors."""

    def __init__(self):
        self.continuous_features = defaultdict(lambda: defaultdict(lambda: []))
        self.mean_variance = defaultdict(dict)

    def add(self, label, index, feature):
        self.continuous_features[label][index].append(feature.value)

    def get(self, label, index):
        return self.continuous_features[label][index]

    def set_mean_variance(self, label):
        if self.continuous_features:
            for j in self.continuous_features[label]:
                features = self.continuous_features[label][j]
                self.mean_variance[label][j] = mean(features), variance(features)

    def probability(self, label, index):
        mean_variance = self.mean_variance[label][index]
        feature = self.get(label, index)
        return gaussian_pdf(feature[index], *mean_variance)
