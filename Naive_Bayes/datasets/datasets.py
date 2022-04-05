import os

from Naive_Bayes.config import ROOT_DIR
from Naive_Bayes.util import load_csv


def load_male_female():
    """
    Homework data
    """
    return _load_dataset('male-female.csv')


def load_pima_indians():
    return _load_dataset('pima-indians.csv')


def _load_dataset(filename):
    return load_csv(_get_path(filename))


def _get_path(filename):
    path = os.path.join(ROOT_DIR, 'datasets', 'data', filename)
    return os.path.abspath(path)
