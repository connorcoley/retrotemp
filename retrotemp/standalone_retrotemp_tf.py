'''
This is the code for a standalone importable SCScorer model. It relies on tensorflow 
and simply reinitializes from a save file. 

One method dumps the trainable variables as numpy arrays, which then enables the 
standalone_model_numpy version of this class.
'''

import tensorflow as tf
from utils.nn import linearND
import math, sys, random, os
import numpy as np
import time
import rdkit.Chem as Chem 
import rdkit.Chem.AllChem as AllChem

import os 
project_root = os.path.dirname(os.path.dirname(__file__))

FP_len = 2048
FP_rad = 2
batch_size = 1
NK=100


def mol_to_fp(mol, radius=FP_rad, nBits=FP_len):
    if mol is None:
        return np.zeros((nBits,), dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, 
        useChirality=True), dtype=np.bool)

def smi_to_fp(smi, radius=FP_rad, nBits=FP_len):
    if not smi:
        return np.zeros((nBits,), dtype=np.float32)
    return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)

class RetroTempPrioritizer():
    def __init__(self):
        self.session = tf.Session()

    def build(self, depth=5, hidden_size=300, output_size=163723):
        self.input_mol = tf.placeholder(tf.float32, [batch_size, FP_len])
        self.mol_hiddens = tf.nn.relu(linearND(self.input_mol, hidden_size, scope="encoder0"))
        for d in xrange(1, depth):
            self.mol_hiddens = tf.nn.relu(linearND(self.mol_hiddens, hidden_size, scope="encoder%i"%d))

        self.score = linearND(self.mol_hiddens, output_size, scope="output")
        _, self.topk = tf.nn.top_k(self.score, k=NK)

        tf.global_variables_initializer().run(session=self.session)
        size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
        n = sum(size_func(v) for v in tf.trainable_variables())
        print "Model size: %dK" % (n/1000,)

        self.coord = tf.train.Coordinator()
        return self

    def restore(self, model_path, checkpoint='final'):
        self.saver = tf.train.Saver(max_to_keep=None)
        restore_path = os.path.join(model_path, 'model.%s' % checkpoint)
        self.saver.restore(self.session, restore_path)
        print('Restored values from latest saved file ({})'.format(restore_path))
        return self 

    def get_topk_from_smi(self, smi=''):
        if not smi:
            return []
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return []
        return self.get_topk_from_mol(mol)
        
    def get_topk_from_mol(self, mol):
        fp = mol_to_fp(mol).astype(np.float32).reshape((1, FP_len))
        cur_topk, = self.session.run([self.topk], feed_dict={
            self.input_mol: fp,
        })
        return list(cur_topk[0])

    def dump_to_numpy_arrays(self, dump_path):
        import cPickle as pickle
        with open(dump_path, 'wb') as fid:
            pickle.dump([v.eval(session=self.session) for v in tf.trainable_variables()], fid, -1)

    def load_from_numpy_arrays(self, dump_path):
        import cPickle as pickle 
        with open(dump_path, 'rb') as fid:
            variables = pickle.load(fid)
        for i, v in enumerate(tf.trainable_variables()):
            assign_op = tf.assign(v, variables[i])
            self.session.run(assign_op)
        print('Loaded from numpy arrays')


if __name__ == '__main__':
    model = RetroTempPrioritizer()
    model.build()
    
    model.restore(os.path.join(project_root, 'models', '6d3M_Reaxys_10_5'), 'ckpt-92820')
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        lst = model.get_topk_from_smi(smi)
        print('{} -> {}'.format(smi, lst))
    model.dump_to_numpy_arrays(os.path.join(project_root, 'models', '6d3M_Reaxys_10_5', 'model.ckpt-92820.as_numpy.pickle'))
    
    model.load_from_numpy_arrays(os.path.join(project_root, 'models', '6d3M_Reaxys_10_5', 'model.ckpt-92820.as_numpy.pickle'))
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        lst = model.get_topk_from_smi(smi)
        print('{} -> {}'.format(smi, lst))