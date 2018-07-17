'''
This is a standalone, importable SCScorer model. It does not have tensorflow as a
dependency and is a more attractive option for deployment. The calculations are 
fast enough that there is no real reason to use GPUs (via tf) instead of CPUs (via np)
'''

import math, sys, random, os
import numpy as np
import time
import rdkit.Chem as Chem 
import rdkit.Chem.AllChem as AllChem

import os 
project_root = os.path.dirname(os.path.dirname(__file__))

FP_len = 2048
FP_rad = 2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class RetroTempPrioritizer():
    def __init__(self, FP_len=FP_len):
        self.vars = []
        self.FP_len = FP_len
        self._restored = False

    def restore(self, weight_path=os.path.join(project_root, 'models', '6d3M_Reaxys_10_5', 'model.ckpt-92820.as_numpy.pickle')):
        import cPickle as pickle
        with open(weight_path, 'rb') as fid:
            self.vars = pickle.load(fid)
        print('Restored variables from {}'.format(weight_path))
        self._restored = True
        return self

    def apply(self, x):
        if not self._restored:
            raise ValueError('Must restore model weights!')
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars)-2)
            W = self.vars[i] 
            b = self.vars[i+1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0) # ReLU
        return x


    def mol_to_fp(self, mol, radius=FP_rad):
        if mol is None:
            return np.zeros((nBits,), dtype=np.float32)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=self.FP_len, 
            useChirality=True), dtype=np.bool)

    def smi_to_fp(self, smi, radius=FP_rad):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)

    def get_topk_from_smi(self, smi='', k=100):
        if not smi:
            return []
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return []
        return self.get_topk_from_mol(mol, k=k)
        
    def get_topk_from_mol(self, mol, k=100):
        fp = self.mol_to_fp(mol).astype(np.float32)
        cur_scores = self.apply(fp)
        indices = list(cur_scores.argsort()[-k:][::-1])
        cur_scores.sort()
        probs = softmax(cur_scores)
        return probs[-k:][::-1], indices


if __name__ == '__main__':
    model = RetroTempPrioritizer(FP_len=2048)    
    model.restore(os.path.join(project_root, 'models', '6d3M_Reaxys_10_5', 'model.ckpt-92820.as_numpy.pickle'))

    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        lst = model.get_topk_from_smi(smi)
        print('{} -> {}'.format(smi, lst))