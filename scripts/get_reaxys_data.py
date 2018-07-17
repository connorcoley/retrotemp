import os
from pymongo import MongoClient
import rdkit.Chem as Chem
import cPickle as pickle 
import sys

'''
For a specific template set, look through all of the precedent reactions and
pull the products. This is so we can learn a mapping from product molecules to
likely templates
'''

limit = 1e9

db_client = MongoClient('mongodb://username:password@server/authenticationdb', 27017)
reaction_db = db_client['reaxys_v2']['reactions']

RETRO_TRANSFORMS_CHIRAL = {
    'database': 'reaxys_v2',
    'collection': 'transforms_retro_v9',
    'mincount': 10,
    'mincount_chiral': 5
}

project_root = os.path.dirname(os.path.dirname(__file__))
template_name = '{}_{}_{}_{}'.format(
    RETRO_TRANSFORMS_CHIRAL['database'],
    RETRO_TRANSFORMS_CHIRAL['collection'],
    RETRO_TRANSFORMS_CHIRAL['mincount'],
    RETRO_TRANSFORMS_CHIRAL['mincount_chiral'],
)


# Get templates and their refs
database = db_client[RETRO_TRANSFORMS_CHIRAL['database']]
RETRO_DB = database[RETRO_TRANSFORMS_CHIRAL['collection']]
import makeit.retrosynthetic.transformer as transformer 
RetroTransformerChiral = transformer.RetroTransformer(
    mincount=RETRO_TRANSFORMS_CHIRAL['mincount'],
    mincount_chiral=RETRO_TRANSFORMS_CHIRAL['mincount_chiral'],
)
RetroTransformerChiral.load(chiral=True, refs=True, rxns=False) 
RetroTransformerChiral.reorder()
RETRO_CHIRAL_FOOTNOTE = 'Using {} chiral retrosynthesis templates (mincount {} if achiral, mincount {} if chiral) from {}/{}'.format(len(RetroTransformerChiral.templates),
    RETRO_TRANSFORMS_CHIRAL['mincount'], 
    RETRO_TRANSFORMS_CHIRAL['mincount_chiral'], 
    RETRO_TRANSFORMS_CHIRAL['database'], 
    RETRO_TRANSFORMS_CHIRAL['collection'])

print('Loaded {} templates'.format(len(RetroTransformerChiral.templates)))

## Create map from reaction ID to template ID
reaction_id_to_template_num = {}
template_num_to_template_id = {}
for tmp_num, template in enumerate(RetroTransformerChiral.templates):
    template_num_to_template_id[tmp_num] = template['_id']
    for ref in template['references']:
        rxn_id = int(ref.split('-')[0])
        reaction_id_to_template_num[rxn_id] = tmp_num

print('{} total reaction refs'.format(len(reaction_id_to_template_num)))

## Look through reactions now
with open(os.path.join(project_root, 'data', 'reaxys_limit%i_%s.txt' % (limit, template_name)), 'w') as f:
    i = 0
    for rx_doc in reaction_db.find({'RXN_SMILES': {'$exists': True}}, ['_id', 'RXN_SMILES']).sort('_id', 1):
        try:
            # Only look at reactions that made the template cut
            if rx_doc['_id'] not in reaction_id_to_template_num:
                continue

            r, p = rx_doc['RXN_SMILES'].split('>>')
            if (not r) or (not p) or ('.' in p):
                continue
            r_mol = Chem.MolFromSmiles(str(r))
            p_mol = Chem.MolFromSmiles(str(p))
            if (not r_mol) or (not p_mol): 
                continue
            [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
            [a.ClearProp('molAtomMapNumber') for a in p_mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
            n = max(r_mol.GetNumAtoms(), p_mol.GetNumAtoms())
            f.write('%s>>%s %i %i %i %s\n' % (Chem.MolToSmiles(r_mol,True), 
                Chem.MolToSmiles(p_mol,True), n, rx_doc['_id'], 
                reaction_id_to_template_num[rx_doc['_id']], 
                template_num_to_template_id[reaction_id_to_template_num[rx_doc['_id']]]))
            i += 1
            if i % 1000 == 0:
                print('Wrote %i' % i)
            if i >= limit:
                break
        except Exception as e:
            print(e)
