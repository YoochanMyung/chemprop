import sys
import subprocess
import os
import sys

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import rdmolops
import igraph
import numpy as np
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from mordred import Calculator, descriptors
from rdkit.Chem.SaltRemover import SaltRemover

from rdkit.Chem import rdDepictor  # to generate 2D depictions of molecules
from rdkit.Chem.Draw import rdMolDraw2D # to draw 2D molecules using vectors

import pandas as pd
import random
import json


def standardize_mol(mol, verbose=False):
    """Standardize the RDKit molecule, select its parent molecule, uncharge it, 
    then enumerate all the tautomers.
    If verbose is true, an explanation of the steps and structures of the molecule
    as it is standardized will be output."""
    # Follows the steps from:
    #  https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg Landrum) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ -- thanks JP!
    
    from rdkit.Chem.MolStandardize import rdMolStandardize
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol) 

    # if many fragments, get the "parent" (the actual mol we are interested in) 
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    
    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    
    # Note: no attempt is made at reionization at this step
    # nor ionization at some pH (RDKit has no pKa caculator);
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogues, etc.
    te = rdMolStandardize.TautomerEnumerator() # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    
    assert taut_uncharged_parent_clean_mol != None
    
    if verbose: print(Chem.MolToSmiles(taut_uncharged_parent_clean_mol))
    return taut_uncharged_parent_clean_mol

def standardize_smiles(smiles, verbose=False):
  """Standardize the SMILES string, select its parent molecule, uncharge it, 
    then enumerate all the tautomers."""
  if verbose: print(smiles)
  std_mol = standardize_mol(Chem.MolFromSmiles(smiles), verbose=verbose)
  return Chem.MolToSmiles(std_mol)


def getAllShortestPaths(pajek_file):
    script = f"""
    library("igraph")
    g = read.graph(file = ("{pajek_file}"), format=c("pajek"))
    x <- shortest.paths(g, mode=c("all"), weights=NULL, algorithm=c("johnson"))
    sink("{pajek_file}.matrix.tmp")
    for(i in seq(1,dim(x)[1],1)){{
        for(j in seq(1,dim(x)[2],1)){{
            if(i <= j){{
                cat(i, j, x[i,j], "\\n")
            }}
        }}
    }}
    sink()
    """
    with open(f"{pajek_file}.gen_dist.R", "w") as script_file:
        script_file.write(script)

    # subprocess.call(f"{obabel_path}/R CMD BATCH --no-save --no-restore-data --quiet {pajek_file}.gen_dist.R", shell=True)
    subprocess.call(f"R CMD BATCH --no-save --no-restore-data --quiet {pajek_file}.gen_dist.R", shell=True)
    os.rename(f"{pajek_file}.matrix.tmp", f"{pajek_file}.dist")
    # os.remove(f"{pajek_file}.gen_dist.R")

    return f"{pajek_file}.dist"

def smiles2ShortestPaths(smiles):
    mol = Chem.MolFromSmiles(smiles)
    admatrix = rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adlist = np.ndarray.tolist(admatrix)
    graph = igraph.Graph()
    g = graph.Adjacency(adlist).as_undirected()
    for idx in g.vs.indices:
        g.vs[idx]["AtomicNum"] = mol.GetAtomWithIdx(idx).GetAtomicNum()
        g.vs[idx]["AtomicSymbole"] = mol.GetAtomWithIdx(idx).GetSymbol()
    for bd in bondidxs:
        btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
        g.es[g.get_eid(bd[0], bd[1])]["BondType"] = btype
        # print(bd, mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble())
    # x = g.distances(source=None, target=None, weights=None, mode="all")
    x = g.shortest_paths(source=None, target=None, weights=None, mode="all")
    # print("shortest_paths")
    output_list = list()
    for i in range(g.vcount()):
        for j in range(g.vcount()):
            if i <= j:
                # print(f"{i+1} {j+1} {x[i][j]}")
                output_list.append(f"{i+1} {j+1} {x[i][j]}")

    return output_list

def get_pharmacophores(smiles):
    AtomPharm = dict()
    output = list()
    
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    molecule = Chem.MolFromSmiles(smiles)
    
    if molecule:
        feature_set = factory.GetFeaturesForMol(molecule)
        for feature in feature_set:
            output.append(feature.GetFamily() + "\t" + str(feature.GetAtomIds()))

    output = [line.replace("(", "").replace(")", "").replace(",", "").replace("Lumped", "") for line in output]

    for each in output:
        class_tokens = each.split("\t")
        atom_ind_list = class_tokens[1].split()

        if "Hydrophobe" in class_tokens[0]:
            for ind in atom_ind_list:
                AtomPharm.setdefault(ind, {}).setdefault("Hydrophobe", 1)
        if "Aromatic" in class_tokens[0]:
            for ind in atom_ind_list:
                AtomPharm.setdefault(ind, {}).setdefault("Aromatic", 1)
        if "PosIonizable" in class_tokens[0]:
            for ind in atom_ind_list:
                AtomPharm.setdefault(ind, {}).setdefault("PosIonizable", 1)
        if "NegIonizable" in class_tokens[0]:
            for ind in atom_ind_list:
                AtomPharm.setdefault(ind, {}).setdefault("NegIonizable", 1)
        if "Acceptor" in class_tokens[0]:
            for ind in atom_ind_list:
                AtomPharm.setdefault(ind, {}).setdefault("Acceptor", 1)
        if "Donor" in class_tokens[0]:
            for ind in atom_ind_list:
                AtomPharm.setdefault(ind, {}).setdefault("Donor", 1)
    return AtomPharm

def get_descriptors(smiles):
    header = ["HeavyAtomCount","MolLogP","NumHeteroatoms","NumRotatableBonds","RingCount","TPSA","LabuteASA", "MolWt"]
    props = header
    smiles = smiles.strip()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        prop_output = []
        for prop in props:
            prop_output.append(getattr(Descriptors, prop)(molecule))

    return dict(zip(header, prop_output))

def get_Fcount(smiles):
    return {"FCount":list(smiles).count("F")}

def get_toxicophores(smiles):
    smiles = smiles.rstrip('\n')
    molecule = Chem.MolFromSmiles(smiles)
    fingerprint = list()
    toxicophores = list()
    result_dict = dict()
    smarts_list = ["O=N(~O)a","a[NH2]","a[N;X2]=O","CO[N;X2]=O","N[N;X2]=O","O1[c,C]-[c,C]1","C1NC1","N=[N+]=[N-]","C=[N+]=[N-]","N=N-N","c[N;X2]!@;=[N;X2]c","[OH,NH2][N,O]","[OH]Na","[Cl,Br,I]C","[Cl,Br,I]C=O","[N,S]!@[C;X4]!@[CH2][Cl,Br,I]","[cH]1[cH]ccc2c1c3c(cc2)cc[cH][cH]3","[cH]1cccc2c1[cH][cH]c3c2ccc[cH]3","[$([C,c]OS(=O)(=O)O!@[c,C]),$([c,C]S(=O)(=O)O!@[c,C])]","O=N(~O)N","[$(O=[CH]C=C),$(O=[CH]C=O)]","[N;v4]#N","O=C1CCO1","[CH]=[CH]O","[NH;!R][NH;R]a","[CH3][NH]a","aN([$([OH]),$(O*=O)])[$([#1]),$(C(=O)[CH3]),$([CH3]),$([OH]),$(O*=O)]","a13~a~a~a~a2~a1~a(~a~a~a~3)~a~a~a~2","a1~a~a~a2~a~1~a~a3~a(~a~2)~a~a~a~3","a1~a~a~a2~a~1~a~a~a3~a~2~a~a~a~3","a1~a~a~a~a2~a~1~a3~a(~a~2)~a~a~a~a~3","a1~a~a~a~a2~a~1~a~a3~a(~a~2)~a~a~a~3","a1~a~a~a~a2~a~1~a~a3~a(~a~2)~a~a~a~a~3","a1~a~a~a~a2~a~1~a~a~a3~a~2~a~a~a~3","a1~a~a~a~a2~a~1~a~a~a3~a~2~a~a~a~a~3","a13~a~a~a~a2~a1~a(~a~a~a~3)~a~a~2"]
    tox_label_list = list()
    for x in range(1,37):
        tox_label_list.append(f'Tox_{x}')

    if molecule:
        for smarts in smarts_list:
            smarts = smarts.rstrip('\n')
            toxicophores.append(Chem.MolFromSmarts(smarts))

        # Check for toxicophore occurrence (fingerprint calculation)
        for toxic in toxicophores:
            if molecule.HasSubstructMatch(toxic):
                # fingerprint.append(True)
                fingerprint.append(1)
            else:
                # fingerprint.append(False)
                fingerprint.append(0)

        return dict(zip(tox_label_list,fingerprint))


def get_complementary_descriptors(smiles):
    # prop_output = list()
    prop_output = dict()
    
    header = ["BalabanJ","BertzCT","Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v","HallKierAlpha","Kappa1","Kappa2","Kappa3","NHOHCount","NOCount","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA14","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","VSA_EState1","VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5","VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9","fr_Al_COO","fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH","fr_Ar_OH","fr_COO","fr_COO2","fr_C_O","fr_C_O_noCOO","fr_C_S","fr_HOCCN","fr_Imine","fr_NH0","fr_NH1","fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole","fr_SH","fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid","fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide","fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic","fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether","fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imidazole","fr_imide","fr_isocyan","fr_isothiocyan","fr_ketone","fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine","fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitro_arom_nonortho","fr_nitroso","fr_oxazole","fr_oxime","fr_para_hydroxylation","fr_phenol","fr_phenol_noOrthoHbond","fr_phos_acid","fr_phos_ester","fr_piperdine","fr_piperzine","fr_priamide","fr_prisulfonamd","fr_pyridine","fr_quatN","fr_sulfide","fr_sulfonamd","fr_sulfone","fr_term_acetylene","fr_tetrazole","fr_thiazole","fr_thiocyan","fr_thiophene","fr_unbrch_alkane","fr_urea"]
    smiles = smiles.strip()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        for prop in header:
            # prop_output.append(getattr(Descriptors, prop)(molecule))
            prop_output[prop] = getattr(Descriptors, prop)(molecule)
    
    return prop_output

def graph_signatures(smiles, cutoff_step=1, cutoff_limit=10):
    ind = int(random.random() * 100000)

    pkcsm_pharmacophores = {"Hydrophobe": 1, "Aromatic": 1, "Acceptor": 1, "Donor": 1, "PosIonizable": 1, "NegIonizable": 1}
    pharm_keys = sorted(pkcsm_pharmacophores.keys())

    # Generate graph representation (pajek file format)
    smiles_file = f"tmp_pkCSM.{ind}.smi"
    mol2_file = f"tmp_pkCSM.{ind}.mol2"
    pajek_file = f"tmp_pkCSM.{ind}.net"

    with open(smiles_file, "w") as file:
        file.write(smiles)

    subprocess.run([f"obabel", "-p7.4", "-ismi", smiles_file, "-omol2", "-O", mol2_file], stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)

    with open(pajek_file, "w") as PAJEK:
        with open(mol2_file, "r") as MOL2:
            read_node = 0
            read_edge = 0
            num_nodes = 0
            nodes = []
            edges = []

            for info in MOL2:
                if "<TRIPOS>ATOM" in info:
                    read_node = 1
                    read_edge = 0
                    continue
                elif "<TRIPOS>BOND" in info:
                    read_edge = 1
                    read_node = 0
                    continue

                if len(info) > 75:
                    if read_node == 1 and info[8] != "H":
                        num_nodes = int(info[:7].strip())
                        lab = info[47:52].strip()
                        node_pajek = f"      {num_nodes} \"{lab}\"     0.0000     0.0000     0.0000 ic       Orange bc       Orange x_fact 2.000 y_fact 2.000\n"
                        nodes.append(node_pajek)

                if read_edge == 1:
                    info = info.strip()
                    tokens = info.split()
                    i = int(tokens[1])
                    j = int(tokens[2])

                    if i <= int(num_nodes) and j <= int(num_nodes):
                        edge_pajek = f"{i:10d}{j:10d}     1.000\n"
                        edges.append(edge_pajek)

            PAJEK.write(f"*Vertices {num_nodes}\n")
            PAJEK.write("".join(nodes))
            PAJEK.write("*Edges\n")
            PAJEK.write("".join(edges))

    MOL2.close()
    PAJEK.close()

    AtomPharm = get_pharmacophores(smiles)
    descriptors = get_descriptors(smiles)
    toxicophore_fingerprint = get_toxicophores(smiles)
    HeavyAtomCount = descriptors['HeavyAtomCount']

    # Calculate shortest paths matrix
    dist_file = getAllShortestPaths(pajek_file)
    
    # Calculate pharmacophore count
    PharmCount = {key: 0 for key in pharm_keys}

    for i in range(HeavyAtomCount):
        for key in pharm_keys:
            if AtomPharm.get(str(i)) and key in AtomPharm[str(i)]:
                PharmCount[key] += 1

    edgeCount = {}
    for key1 in pharm_keys:
        for key2 in pharm_keys:
            edgeCount[key1 + ":" + key2] = 0
            if key1 != key2:
                edgeCount[key2 + ":" + key1] = 0

    with open(dist_file, "r") as DIST:
        dist_data = DIST.readlines()

    dist = [[None] * (int(num_nodes) + 1) for _ in range(int(num_nodes)+ 1)]

    for distance_info in dist_data:
        dist_tokens = distance_info.split()
        a = int(dist_tokens[0])
        b = int(dist_tokens[1])
        distance = int(dist_tokens[2])
        dist[a][b] = distance
        dist[b][a] = distance

    # Generate signature
    EdgeCount = dict()
    cutoff_1 = 1.000
    cutoff_temp = cutoff_limit
    while cutoff_temp >= 1:
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes):
                if cutoff_1 <= dist[i][j] <= cutoff_temp:
                    key_set1 = list(AtomPharm.get(str(i), {}))
                    key_set2 = list(AtomPharm.get(str(j), {}))

                    pairs = []
                    for x in range(len(key_set1)):
                        for y in range(len(key_set2)):
                            my_key_set1 = key_set1[x]
                            my_key_set2 = key_set2[y]
                            pairs.append(f"{my_key_set1}:{my_key_set2}")

                    uniq_pairs = list(set(pairs))

                    for pair in uniq_pairs:
                        edgeCount[pair] += 1
                        toks = pair.split(":")

        for a in range(len(pharm_keys)):
            for b in range(a, len(pharm_keys)):
                EdgeCount['{}:{}-{}.00'.format(pharm_keys[a],pharm_keys[b],cutoff_temp)] = edgeCount[pharm_keys[a] + ':' + pharm_keys[b]]

                
        edgeCount = {key: 0 for key in edgeCount}
        cutoff_temp -= cutoff_step
    
    # Remove temporary files
    # os.system(f"rm tmp_pkCSM.{ind}.*")

    result = dict()
    
    for d in [toxicophore_fingerprint, PharmCount, EdgeCount]:
        result.update(d)
    
    return result

def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v
    
def get_deeppk_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # mol = standardize_mol(mol)
    # smiles = Chem.MolToSmiles(mol)
    final_result = dict()
    generator = rdDescriptors.RDKit2D()
    rdkit_columns = list()
    for each in generator.columns:
        rdkit_columns.append(each[0])

    rdkit_features = generator.process(smiles)[1:]
    rdkit_result = dict(zip(rdkit_columns, rdkit_features))
    pdCSM_features = pdCSM_fast(smiles)

    # 3. MODRED
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_features = calc(mol).drop_missing().asdict()

    final_result.update(pdCSM_features)
    final_result.update(rdkit_result)
    final_result.update(mordred_features)
    # final_result = np.array([[k,v] for k,v in final_result.items()])

    # features = final_result[:,1:].squeeze()
    # features = np.vectorize(boolstr_to_floatstr)(features).astype(float).tolist()

    return final_result

def deeppk_features(mol, target):
    def boolstr_to_floatstr(v):
        if v == 'True':
            return '1'
        elif v == 'False':
            return '0'
        else:
            return v

    features_json = json.load(open('/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/only_smiles/classification/features.json','r'))
    feature_list = features_json[target]
    final_result = dict()

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    generator = rdDescriptors.RDKit2D()
    rdkit_columns = list()
    for each in generator.columns:
        rdkit_columns.append(each[0])

    rdkit_features = generator.process(smiles)[1:]
    rdkit_result = dict(zip(rdkit_columns, rdkit_features))
    pdCSM_features = pdCSM_fast(smiles)

    # 3. MODRED
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_features = calc(mol).drop_missing().asdict()
    selected_mordred_features  = dict()
    for k,v in mordred_features.items():
        if k in feature_list:
            selected_mordred_features[k] = v    

    final_result.update(pdCSM_features)
    final_result.update(selected_mordred_features)
    final_result.update(rdkit_result)
    final_result = np.array([[k,v] for k,v in final_result.items()])

    features = final_result[:,1:].squeeze()
    features = np.vectorize(boolstr_to_floatstr)(features).astype(float).tolist()

    return features


def pdCSM_fast(smiles, cutoff_step=1, cutoff_limit=10):
    pkcsm_pharmacophores = {"Hydrophobe": 1, "Aromatic": 1, "Acceptor": 1, "Donor": 1, "PosIonizable": 1, "NegIonizable": 1}
    pharm_keys = sorted(pkcsm_pharmacophores.keys())

    AtomPharm = get_pharmacophores(smiles)
    descriptors = get_descriptors(smiles)
    Fcount = get_Fcount(smiles)
    toxicophore_fingerprint = get_toxicophores(smiles)
    complementary_desc = get_complementary_descriptors(smiles)
    HeavyAtomCount = descriptors['HeavyAtomCount']
    
    # Calculate pharmacophore count
    PharmCount = {key: 0 for key in pharm_keys}

    for i in range(HeavyAtomCount):
        for key in pharm_keys:
            if AtomPharm.get(str(i)) and key in AtomPharm[str(i)]:
                PharmCount[key] += 1

    edgeCount = {}
    for key1 in pharm_keys:
        for key2 in pharm_keys:
            edgeCount[key1 + ":" + key2] = 0
            if key1 != key2:
                edgeCount[key2 + ":" + key1] = 0

    dist_data = smiles2ShortestPaths(smiles)
    num_nodes = int(dist_data[-1].split()[0])
    dist = [[None] * (int(num_nodes) + 1) for _ in range(int(num_nodes)+ 1)]

    for distance_info in dist_data:
        dist_tokens = distance_info.split()
        a = int(dist_tokens[0])
        b = int(dist_tokens[1])
        distance = int(dist_tokens[2])
        dist[a][b] = distance
        dist[b][a] = distance

    # Generate signature
    EdgeCount = dict()
    cutoff_1 = 1.000
    cutoff_temp = cutoff_limit
    while cutoff_temp >= 1:
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes):
                if cutoff_1 <= dist[i][j] <= cutoff_temp:
                    key_set1 = list(AtomPharm.get(str(i), {}))
                    key_set2 = list(AtomPharm.get(str(j), {}))

                    pairs = []
                    for x in range(len(key_set1)):
                        for y in range(len(key_set2)):
                            my_key_set1 = key_set1[x]
                            my_key_set2 = key_set2[y]
                            pairs.append(f"{my_key_set1}:{my_key_set2}")

                    uniq_pairs = list(set(pairs))

                    for pair in uniq_pairs:
                        edgeCount[pair] += 1
                        toks = pair.split(":")

        for a in range(len(pharm_keys)):
            for b in range(a, len(pharm_keys)):
                EdgeCount['{}:{}-{}.00'.format(pharm_keys[a],pharm_keys[b],cutoff_temp)] = edgeCount[pharm_keys[a] + ':' + pharm_keys[b]]

                
        edgeCount = {key: 0 for key in edgeCount}
        cutoff_temp -= cutoff_step
    
    result = dict()
    
    for d in [descriptors, Fcount, toxicophore_fingerprint, complementary_desc, PharmCount, EdgeCount]:
        result.update(d)
    
    return result

def pdCSM(smiles, ind, cutoff_step=1, cutoff_limit=10):
    pkcsm_pharmacophores = {"Hydrophobe": 1, "Aromatic": 1, "Acceptor": 1, "Donor": 1, "PosIonizable": 1, "NegIonizable": 1}
    pharm_keys = sorted(pkcsm_pharmacophores.keys())

    # Generate graph representation (pajek file format)
    # smiles_file = f"tmp_pkCSM.{ind}.smi"
    # mol2_file = f"tmp_pkCSM.{ind}.mol2"
    # pajek_file = f"tmp_pkCSM.{ind}.net"

    # with open(smiles_file, "w") as file:
    #     file.write(smiles)

    # subprocess.run([f"obabel", "-p7.4", "-ismi", str(smiles_file), "-omol2", "-O", str(mol2_file)])

    # with open(pajek_file, "w") as PAJEK:
    #     with open(mol2_file, "r") as MOL2:
    #         read_node = 0
    #         read_edge = 0
    #         num_nodes = 0
    #         nodes = []
    #         edges = []

    #         for info in MOL2:
    #             if "<TRIPOS>ATOM" in info:
    #                 read_node = 1
    #                 read_edge = 0
    #                 continue
    #             elif "<TRIPOS>BOND" in info:
    #                 read_edge = 1
    #                 read_node = 0
    #                 continue

    #             if len(info) > 75:
    #                 if read_node == 1 and info[8] != "H":
    #                     num_nodes = int(info[:7].strip())
    #                     lab = info[47:52].strip()
    #                     node_pajek = f"      {num_nodes} \"{lab}\"     0.0000     0.0000     0.0000 ic       Orange bc       Orange x_fact 2.000 y_fact 2.000\n"
    #                     nodes.append(node_pajek)

    #             if read_edge == 1:
    #                 info = info.strip()
    #                 tokens = info.split()
    #                 i = int(tokens[1])
    #                 j = int(tokens[2])

    #                 if i <= int(num_nodes) and j <= int(num_nodes):
    #                     edge_pajek = f"{i:10d}{j:10d}     1.000\n"
    #                     edges.append(edge_pajek)

    #         PAJEK.write(f"*Vertices {num_nodes}\n")
    #         PAJEK.write("".join(nodes))
    #         PAJEK.write("*Edges\n")
    #         PAJEK.write("".join(edges))

    # MOL2.close()
    # PAJEK.close()

    AtomPharm = get_pharmacophores(smiles)
    descriptors = get_descriptors(smiles)
    Fcount = get_Fcount(smiles)
    toxicophore_fingerprint = get_toxicophores(smiles)
    complementary_desc = get_complementary_descriptors(smiles)
    HeavyAtomCount = descriptors['HeavyAtomCount']

    # Calculate shortest paths matrix
    # dist_file = getAllShortestPaths(pajek_file)
    
    # Calculate pharmacophore count
    PharmCount = {key: 0 for key in pharm_keys}

    for i in range(HeavyAtomCount):
        for key in pharm_keys:
            if AtomPharm.get(str(i)) and key in AtomPharm[str(i)]:
                PharmCount[key] += 1

    edgeCount = {}
    for key1 in pharm_keys:
        for key2 in pharm_keys:
            edgeCount[key1 + ":" + key2] = 0
            if key1 != key2:
                edgeCount[key2 + ":" + key1] = 0

    # with open(dist_file, "r") as DIST:
    #     dist_data = DIST.readlines()

    dist_data = smiles2ShortestPaths(smiles)
    num_nodes = int(dist_data[-1].split()[0])
    # print(num_nodes)
    dist = [[None] * (int(num_nodes) + 1) for _ in range(int(num_nodes)+ 1)]

    for distance_info in dist_data:
        dist_tokens = distance_info.split()
        a = int(dist_tokens[0])
        b = int(dist_tokens[1])
        distance = int(dist_tokens[2])
        dist[a][b] = distance
        dist[b][a] = distance

    # Generate signature
    EdgeCount = dict()
    cutoff_1 = 1.000
    cutoff_temp = cutoff_limit
    while cutoff_temp >= 1:
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes):
                if cutoff_1 <= dist[i][j] <= cutoff_temp:
                    key_set1 = list(AtomPharm.get(str(i), {}))
                    key_set2 = list(AtomPharm.get(str(j), {}))

                    pairs = []
                    for x in range(len(key_set1)):
                        for y in range(len(key_set2)):
                            my_key_set1 = key_set1[x]
                            my_key_set2 = key_set2[y]
                            pairs.append(f"{my_key_set1}:{my_key_set2}")

                    uniq_pairs = list(set(pairs))

                    for pair in uniq_pairs:
                        edgeCount[pair] += 1
                        toks = pair.split(":")

        for a in range(len(pharm_keys)):
            for b in range(a, len(pharm_keys)):
                EdgeCount['{}:{}-{}.00'.format(pharm_keys[a],pharm_keys[b],cutoff_temp)] = edgeCount[pharm_keys[a] + ':' + pharm_keys[b]]

                
        edgeCount = {key: 0 for key in edgeCount}
        cutoff_temp -= cutoff_step
    
    # Remove temporary files
    # os.system(f"rm tmp_pkCSM.{ind}.*")

    result = dict()
    
    for d in [descriptors, Fcount, toxicophore_fingerprint, complementary_desc, PharmCount, EdgeCount]:
        result.update(d)
    
    return result


def pdCSM_csv(smiles_csv, cutoff_step=1, cutoff_limit=10):
    input_pd = pd.read_csv(smiles_csv)

    pkcsm_pharmacophores = {"Hydrophobe": 1, "Aromatic": 1, "Acceptor": 1, "Donor": 1, "PosIonizable": 1, "NegIonizable": 1}
    pharm_keys = sorted(pkcsm_pharmacophores.keys())

    for ind, row in input_pd.iterrows():
        smiles = row['SMILES']

        # Generate graph representation (pajek file format)
        smiles_file = f"tmp_pkCSM.{ind}.smi"
        mol2_file = f"tmp_pkCSM.{ind}.mol2"
        pajek_file = f"tmp_pkCSM.{ind}.net"

        with open(smiles_file, "w") as file:
            file.write(smiles)

        # subprocess.run([f"{obabel_path}/obabel", "-p7.4", "-ismi", smiles_file, "-omol2", "-O", mol2_file])
        subprocess.run([f"obabel", "-p7.4", "-ismi", smiles_file, "-omol2", "-O", mol2_file])

        with open(pajek_file, "w") as PAJEK:
            with open(mol2_file, "r") as MOL2:
                read_node = 0
                read_edge = 0
                num_nodes = 0
                nodes = []
                edges = []

                for info in MOL2:
                    if "<TRIPOS>ATOM" in info:
                        read_node = 1
                        read_edge = 0
                        continue
                    elif "<TRIPOS>BOND" in info:
                        read_edge = 1
                        read_node = 0
                        continue

                    if len(info) > 75:
                        if read_node == 1 and info[8] != "H":
                            num_nodes = int(info[:7].strip())
                            lab = info[47:52].strip()
                            node_pajek = f"      {num_nodes} \"{lab}\"     0.0000     0.0000     0.0000 ic       Orange bc       Orange x_fact 2.000 y_fact 2.000\n"
                            nodes.append(node_pajek)

                    if read_edge == 1:
                        info = info.strip()
                        tokens = info.split()
                        i = int(tokens[1])
                        j = int(tokens[2])

                        if i <= int(num_nodes) and j <= int(num_nodes):
                            edge_pajek = f"{i:10d}{j:10d}     1.000\n"
                            edges.append(edge_pajek)

                PAJEK.write(f"*Vertices {num_nodes}\n")
                PAJEK.write("".join(nodes))
                PAJEK.write("*Edges\n")
                PAJEK.write("".join(edges))

        MOL2.close()
        PAJEK.close()

        AtomPharm = get_pharmacophores(smiles)
        descriptors = get_descriptors(smiles)
        Fcount = get_Fcount(smiles)
        toxicophore_fingerprint = get_toxicophores(smiles)
        complementary_desc = get_complementary_descriptors(smiles)
        HeavyAtomCount = descriptors['HeavyAtomCount']

        # Calculate shortest paths matrix
        dist_file = getAllShortestPaths(pajek_file)
        
        # Calculate pharmacophore count
        PharmCount = {key: 0 for key in pharm_keys}

        for i in range(HeavyAtomCount):
            for key in pharm_keys:
                if AtomPharm.get(str(i)) and key in AtomPharm[str(i)]:
                    PharmCount[key] += 1

        edgeCount = {}
        for key1 in pharm_keys:
            for key2 in pharm_keys:
                edgeCount[key1 + ":" + key2] = 0
                if key1 != key2:
                    edgeCount[key2 + ":" + key1] = 0

        with open(dist_file, "r") as DIST:
            dist_data = DIST.readlines()

        dist = [[None] * (int(num_nodes) + 1) for _ in range(int(num_nodes)+ 1)]

        for distance_info in dist_data:
            dist_tokens = distance_info.split()
            a = int(dist_tokens[0])
            b = int(dist_tokens[1])
            distance = int(dist_tokens[2])
            dist[a][b] = distance
            dist[b][a] = distance

        # Generate signature
        EdgeCount = dict()
        cutoff_1 = 1.000
        cutoff_temp = cutoff_limit
        while cutoff_temp >= 1:
            for i in range(1, num_nodes):
                for j in range(i + 1, num_nodes):
                    if cutoff_1 <= dist[i][j] <= cutoff_temp:
                        key_set1 = list(AtomPharm.get(str(i), {}))
                        key_set2 = list(AtomPharm.get(str(j), {}))

                        pairs = []
                        for x in range(len(key_set1)):
                            for y in range(len(key_set2)):
                                my_key_set1 = key_set1[x]
                                my_key_set2 = key_set2[y]
                                pairs.append(f"{my_key_set1}:{my_key_set2}")

                        uniq_pairs = list(set(pairs))

                        for pair in uniq_pairs:
                            edgeCount[pair] += 1
                            toks = pair.split(":")
                            # outfile.write(f"{edgeCount[toks[0] + ':' + toks[1]]},")
                            # print(toks[0],toks[1],cutoff_temp,edgeCount[toks[0]+":"+toks[1]]) # TODO: check whether we need this.

            for a in range(len(pharm_keys)):
                for b in range(a, len(pharm_keys)):
                    EdgeCount['{}:{}-{}.00'.format(pharm_keys[a],pharm_keys[b],cutoff_temp)] = edgeCount[pharm_keys[a] + ':' + pharm_keys[b]]

                    
            edgeCount = {key: 0 for key in edgeCount}
            cutoff_temp -= cutoff_step
        
        # Remove temporary files
        os.system(f"rm tmp_pkCSM.{ind}.*")

    # print(descriptors)
    # print(Fcount)
    # print(toxicophore_fingerprint)
    # print(complementary_desc)
    # print(PharmCount)
    # print(EdgeCount)
    result = dict()
    
    for d in [descriptors, Fcount, toxicophore_fingerprint, complementary_desc, PharmCount, EdgeCount]:
        result.update(d)
    
    output_fname = f"{os.path.basename(smiles_csv).split('.')[0]}.pdCSM"
    result_pd = pd.DataFrame.from_records(result, index=[0])
    result_pd.to_csv(output_fname, index=False)


if __name__ == "__main__":
    smiles = sys.argv[1]
    ind = sys.argv[2]
    graph_signatures(smiles, ind)

# get_toxicophores("C[C@H](Cl)CN(C)C")
# get_Fcount("C[C@H](Cl)CN(C)C")
# get_descriptors("C[C@H](Cl)CN(C)C")
# get_pharmacophores("CCOC(=O)Oc1ccc(CCNC(=O)C(CCSC)NC(C)=O)cc1OC(=O)OCC")
# get_complementary_descriptors("CCOC(=O)Oc1ccc(CCNC(=O)C(CCSC)NC(C)=O)cc1OC(=O)OCC")
# get_complementary_descriptors("CCOC(=O)Oc1ccc(CCNC(=O)C(CCSC)NC(C)=O)cc1OC(=O)OCC")
# get_complementary_descriptors('CC(C)c1nc(CN(C)C(=O)NC(C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1')
