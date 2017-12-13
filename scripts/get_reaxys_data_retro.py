import os
from pymongo import MongoClient
import rdkit.Chem as Chem
import cPickle as pickle 

'''
Get examples from Reaxys where...
(a) we can parse the reactants and products
(b) there is a single product (product salts == multiple products)
(c) there is a corresponding template with sufficient popularity

This is meant to work with data hosted in a MongoDB

While we can't include the actual data, this shows our preprocessing pipeline. The saved file 
consists of the reaction smiles string, the maximum number of atoms in the reactants or products, 
and the document ID for traceability.
'''

limit = 1e9 

db_client = MongoClient('mongodb://guest:guest@askcos2.mit.edu/admin', 27017)
reaction_db = db_client['reaxys_v2']['reactions']

RETRO_TRANSFORMS_CHIRAL = {
    'database': 'reaxys_v2',
    'collection': 'transforms_retro_v9',
    'mincount': 25,
    'mincount_chiral': 10
}

project_root = os.path.dirname(os.path.dirname(__file__))




## Get templates and their refs
def get_retrotransformer_chiral_path(dbname, collname, mincount_retro, mincount_retro_chiral):
    return os.path.join(project_root, 'data', 
        'retrotransformer_chiral_using_%s-%s_mincount%i_mincountchiral%i.pkl' % (dbname, collname, mincount_retro, mincount_retro_chiral))
database = db_client[RETRO_TRANSFORMS_CHIRAL['database']]
RETRO_DB = database[RETRO_TRANSFORMS_CHIRAL['collection']]
import makeit.webapp.transformer_v3 as transformer_v3
RetroTransformerChiral = transformer_v3.Transformer()
save_path = get_retrotransformer_chiral_path(
    RETRO_TRANSFORMS_CHIRAL['database'],
    RETRO_TRANSFORMS_CHIRAL['collection'],
    RETRO_TRANSFORMS_CHIRAL['mincount'],
    RETRO_TRANSFORMS_CHIRAL['mincount_chiral'],
)
if os.path.isfile(save_path):
    with open(save_path, 'rb') as fid:
        RetroTransformerChiral.templates = pickle.load(fid)
        reaction_id_to_template_num = pickle.load(fid)
        template_num_to_template_id = pickle.load(fid)
else:
    ## Load
    mincount_retro = RETRO_TRANSFORMS_CHIRAL['mincount']
    mincount_retro_chiral = RETRO_TRANSFORMS_CHIRAL['mincount_chiral']
    RetroTransformerChiral.load(RETRO_DB, mincount=mincount_retro, get_retro=False, 
        get_synth=False, refs=True, mincount_chiral=mincount_retro_chiral)
    ## Create map from reaction ID to template ID
    reaction_id_to_template_num = {}
    template_num_to_template_id = {}
    for tmp_num, template in enumerate(RetroTransformerChiral.templates):
        template_num_to_template_id[tmp_num] = template['_id']
        for ref in template['references']:
            rxn_id = int(ref.split('-')[0])
            reaction_id_to_template_num[rxn_id] = tmp_num
    ## Save
    print('Saving chiral retro transformer for the (only?) first time')
    with open(save_path, 'wb') as fid:
        pickle.dump(RetroTransformerChiral.templates, fid, -1)
        pickle.dump(reaction_id_to_template_num, fid, -1)
        pickle.dump(template_num_to_template_id, fid, -1)
print('Loaded {} templates'.format(len(RetroTransformerChiral.templates)))


## Look through reactions now
with open(os.path.join(project_root, 'data', 'reaxys_limit%i.txt' % limit), 'w') as f:
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
