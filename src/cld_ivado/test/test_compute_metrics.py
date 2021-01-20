import numpy as np
import pandas as pd
import sys
import unittest
from sklearn.metrics import roc_auc_score
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
    
    def test_auc(self):
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
        probability = np.array([0.032822073, 0.03272833, 0.03777234, 0.033536337, 
                                0.032843955, 0.03501387, 0.03554027, 0.03625841, 
                                0.029225836, 0.03108733, 0.66981703, 0.6634466, 
                                0.64072037, 0.63033247, 0.6085378, 0.6085378, 
                                0.60322386, 0.5710638, 0.58921945, 0.5392657, 
                                0.807792, 0.8031272, 0.8006161, 0.8009957, 
                                0.80026704, 0.7939487, 0.78526366, 0.81253105, 
                                0.8053538, 0.8120024, 0.96580505, 0.97146916,
                                 0.9679805, 0.9671573, 0.97068554, 0.9705009, 
                                 0.96889997, 0.9747322, 0.9728423, 0.9687441,
                                  0.8294275, 0.845363, 0.86396444, 0.873605, 
                                  0.8751063, 0.8656943, 0.8512587, 0.8575548, 
                                  0.8726473, 0.88296974])
        auc = roc_auc_score(labels,  probability)
        self.assertTrue(auc == 1)

  

if __name__ == "__main__":
    unittest.main()
    sys.exit()