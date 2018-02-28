from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
from .my_kappa_calculator import quadratic_weighted_kappa as qwk
from .my_kappa_calculator import linear_weighted_kappa as lwk

logger = logging.getLogger(__name__)

class Evaluator(object):

    def __init__(self, dataset, prompt_id, out_dir, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org):
        pass
    def dump_ref_scores(self):
        pass
    def dump_predictions(self, dev_pred, test_pred, epoch):
        pass
    def calc_correl(self, dev_pred, test_pred):
        pass
    def calc_qwk(self, dev_pred, test_pred):
        pass
    def evaluate(self, model, epoch, print_info=False):
        pass
    def print_info(self):
        pass
    def print_final_info(self):
        pass