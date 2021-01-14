import numpy as np
import pandas as pd
import sys
import unittest
import os
from pathlib import Path 
sys.path.append(str(Path.cwd() / 'src'))
from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import get_metrics
import math


class ComputeMetricsTest(unittest.TestCase):
    def test_metrics(self):
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        logits =[0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1]
        acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
        self.assertTrue(acc == 0)
        self.assertTrue(sensitivity == 0)
        self.assertTrue(math.isnan(specificity))
        self.assertTrue(math.isnan(auc))


        labels = np.array([1, 0, 1, 0, 1, 1, 1, 1])
        predictions = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        logits =[0.9, 0.1, 1, 0.1, 0.9, 0.87, 0.9, 0.89]
        acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
        self.assertTrue(acc == 0.25)
        self.assertAlmostEqual(sensitivity, 0.3333333333333333)
        self.assertTrue(auc == 1.0)



        labels =      np.array([1, 0, 1, 0, 1, 1, 1, 0])
        predictions = np.array([1, 0, 0, 1, 0, 1, 0, 0])
        logits =[0.1, 0.8, 0.5, 0.5, 0.5, 0.87, 0.9, 0.89]
        acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
        self.assertTrue(acc == 0.5)
        self.assertAlmostEqual(sensitivity, 0.4)
        self.assertAlmostEqual(specificity, 0.6666666666666666)
        self.assertTrue(auc == 0.4)


        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
        self.assertTrue(acc == 1)
        self.assertTrue(sensitivity == 1)
        self.assertTrue(math.isnan(specificity))
        self.assertTrue(math.isnan(auc))
  

if __name__ == "__main__":
    unittest.main()
    sys.exit()