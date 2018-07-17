import os
from pymongo import MongoClient
import rdkit.Chem as Chem
import cPickle as pickle 
from collections import defaultdict 

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
#limit = 100

db_client = MongoClient('mongodb://guest:guest@askcos2.mit.edu/admin', 27017)
reaction_db = db_client['reaxys_v2']['reactions']
instance_db = db_client['reaxys_v2']['instances']
chemical_db = db_client['reaxys_v2']['chemicals']

TRANSFORM_SETTINGS = {
    'database': 'reaxys_v2',
    'collection': 'transforms_forward_v2',
    'mincount': 25,
}

project_root = os.path.dirname(os.path.dirname(__file__))


def string_or_range_to_float(text):
    try:
        return float(text)
    except Exception as e:
        x = [z for z in text.strip().split('-') if z not in [u'', u' ']]
        if text.count('-') == 1: # 20 - 30
            try:
                return (float(x[0]) + float(x[1])) / 2.0
            except Exception as e:
                print('Could not convert {}, {}'.format(text, x))
                #print(e)
        elif text.count('-') == 2: # -20 - 0
            try:
                return (-float(x[0]) + float(x[1])) / 2.0
            except Exception as e:
                print('Could not convert {}, {}'.format(text, x))
                #print(e)
        elif text.count('-') == 3: # -20 - -10
            try:
                return (-float(x[0]) - float(x[1])) / 2.0
            except Exception as e:
                print('Could not convert {}, {}'.format(text, x))
                #print(e)
        else:
            print('Could not convert {}'.format(text))
            print(e)
    return -1 # default T

## Get templates and their refs
def get_transformer_path(dbname, collname, mincount_retro):
    return os.path.join(project_root, 'data', 
        'transformer_using_%s-%s_mincount%i.pkl' % (dbname, collname, mincount_retro))
database = db_client[TRANSFORM_SETTINGS['database']]
SYNTH_DB = database[TRANSFORM_SETTINGS['collection']]
import makeit.webapp.transformer_v3 as transformer_v3
Transformer = transformer_v3.Transformer()
save_path = get_transformer_path(
    TRANSFORM_SETTINGS['database'],
    TRANSFORM_SETTINGS['collection'],
    TRANSFORM_SETTINGS['mincount'],
)
if os.path.isfile(save_path):
    with open(save_path, 'rb') as fid:
        Transformer.templates = pickle.load(fid)
        reaction_id_to_template_num = pickle.load(fid)
        template_num_to_template_id = pickle.load(fid)
        reaction_id_to_instance_list = pickle.load(fid)
else:
    ## Load
    mincount = TRANSFORM_SETTINGS['mincount']
    Transformer.load(SYNTH_DB, mincount=mincount, get_retro=False, 
        get_synth=False, refs=True)
    ## Create map from reaction ID to template ID
    reaction_id_to_template_num = {}
    template_num_to_template_id = {}
    reaction_id_to_instance_list = defaultdict(list)
    for tmp_num, template in enumerate(Transformer.templates):
        template_num_to_template_id[tmp_num] = template['_id']
        for ref in template['references']:
            rxn_id = int(ref.split('-')[0])
            reaction_id_to_template_num[rxn_id] = tmp_num
            reaction_id_to_instance_list[rxn_id].append(ref)
    ## Save
    print('Saving transformer for the (only?) first time')
    with open(save_path, 'wb') as fid:
        pickle.dump(Transformer.templates, fid, -1)
        pickle.dump(reaction_id_to_template_num, fid, -1)
        pickle.dump(template_num_to_template_id, fid, -1)
        pickle.dump(reaction_id_to_instance_list, fid, -1)
print('Loaded {} templates'.format(len(Transformer.templates)))

## Define helper function
def xrn_list_to_smiles(xrn_list):
    if not xrn_list: return ''
    smiles_list = []
    for xrn in xrn_list:
        chem_doc = chemical_db.find_one({'_id': xrn})
        if not chem_doc: continue 
        if 'SMILES' not in chem_doc: continue 
        if not chem_doc['SMILES']: continue
        smiles_list.append(chem_doc['SMILES'])
    if not smiles_list:
        raise ValueError('Could not get SMILES for any XRNs in provided list')
    return '.'.join(sorted(smiles_list))


## Look through reactions now
done = False
with open(os.path.join(project_root, 'data', 'reaxys_forward_limit%i.txt' % limit), 'w') as f:
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

            # Get instance information
            for rxd_id in reaction_id_to_instance_list[rx_doc['_id']]:
                rxd_doc = instance_db.find_one({'_id': rxd_id})
                if not rxd_doc: 
                    raise ValueError('Instance ID {} not found'.format(rxd_id))
                rgts = xrn_list_to_smiles(rxd_doc['RXD_RGTXRN']) # reagents - necessary! some contribute atoms
                cats = xrn_list_to_smiles(rxd_doc['RXD_CATXRN']) # catalysts
                slvs = xrn_list_to_smiles(rxd_doc['RXD_SOLXRN']) # solvents
                T = rxd_doc['RXD_T'] # temp [C]
                NYD = rxd_doc['RXD_NYD'] # yield


                f.write('%s>>%s %s %s %s %f %f %s %i %s\n' % (Chem.MolToSmiles(r_mol,True), 
                    Chem.MolToSmiles(p_mol,True), rgts, cats, slvs, string_or_range_to_float(T), NYD, str(rxd_doc['_id']), 
                    reaction_id_to_template_num[rx_doc['_id']], 
                    template_num_to_template_id[reaction_id_to_template_num[rx_doc['_id']]]))
                i += 1
                if i % 1000 == 0:
                    print('Wrote %i' % i)
                if i >= limit:
                    done = True 
                    break
            if done:
                break
        except Exception as e:
            print(e)
