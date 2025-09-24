import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

def get_chembl_targets(protein_name):
    # 支持模糊搜索所有物种
    targets = new_client.target.filter(target_synonym__icontains=protein_name)
    return targets

def get_activities_for_target(target_chembl_id):
    # 获取所有化合物及活性数据
    acts = new_client.activity.filter(target_chembl_id=target_chembl_id)
    return acts

def get_molecule_properties(chembl_id):
    # 拉取分子结构、性质
    mol = new_client.molecule.get(chembl_id)
    try:
        smi = mol['molecule_structures']['canonical_smiles']
        m = Chem.MolFromSmiles(smi)
        mw = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        hba = Descriptors.NumHAcceptors(m)
        hbd = Descriptors.NumHDonors(m)
        return smi, mw, logp, hba, hbd
    except Exception:
        return None, None, None, None, None

def get_mechanism(chembl_id, target_chembl_id):
    # 获取激动剂/拮抗剂等作用类型
    mechs = new_client.mechanism.filter(molecule_chembl_id=chembl_id, target_chembl_id=target_chembl_id)
    for m in mechs:
        return m.get('action_type')
    return None

def process_protein(protein_name, out_path):
    # 主流程：获取所有靶点、分子、标注及性质
    all_data = []
    targets = get_chembl_targets(protein_name)
    for t in targets:
        target_chembl_id = t['target_chembl_id']
        acts = get_activities_for_target(target_chembl_id)
        for a in acts:
            chembl_id = a['molecule_chembl_id']
            smi, mw, logp, hba, hbd = get_molecule_properties(chembl_id)
            if smi is None:
                continue
            action_type = get_mechanism(chembl_id, target_chembl_id)
            record = {
                'protein_name': protein_name,
                'target_chembl_id': target_chembl_id,
                'molecule_chembl_id': chembl_id,
                'smiles': smi,
                'mol_weight': mw,
                'logP': logp,
                'HBA': hba,
                'HBD': hbd,
                'action_type': action_type,
                'standard_type': a.get('standard_type'),
                'standard_value': a.get('standard_value'),
                'standard_units': a.get('standard_units')
            }
            all_data.append(record)
    pd.DataFrame(all_data).to_csv(out_path, sep='\t', index=False)

# ==== 入口：批量处理你的蛋白家族TSV ====
import glob

for tsv_path in glob.glob("*.tsv"):
    df = pd.read_csv(tsv_path, sep='\t')
    for pname in tqdm(df['Name'].unique()):
        pname_clean = pname.strip()
        if not pname_clean:
            continue
        out_file = f"{tsv_path[:-4]}_{pname_clean.replace(' ', '_')}_chembl.tsv"
        process_protein(pname_clean, out_file)
