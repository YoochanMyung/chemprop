from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from .pdCSM import deeppk_features

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def  get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())

@register_features_generator('ames')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'ames')

@register_features_generator('avian_tox')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'avian_tox')

@register_features_generator('bbb_logbb')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bbb_logbb')

@register_features_generator('bbb_logbb_new')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bbb_logbb_new')

@register_features_generator('bcrp')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bcrp')

@register_features_generator('bee_tox')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bee_tox')

@register_features_generator('biodegradation')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'biodegradation')

@register_features_generator('carcinogenicity')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'carcinogenicity')

@register_features_generator('crustacean')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'crustacean')

@register_features_generator('cyp1a2_inhibitor')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp1a2_inhibitor')

@register_features_generator('cyp1a2_substrate')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp1a2_substrate')

@register_features_generator('cyp2c19_inhibitor')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp2c19_inhibitor')

@register_features_generator('cyp2c19_substrate')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp2c19_substrate')

@register_features_generator('cyp2c9_inhibitor')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp2c9_inhibitor')

@register_features_generator('cyp2c9_substrate')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp2c9_substrate')

@register_features_generator('cyp2d6_inhibitor')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp2d6_inhibitor')

@register_features_generator('cyp2d6_substrate')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp2d6_substrate')

@register_features_generator('cyp3a4_inhibitor')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp3a4_inhibitor')

@register_features_generator('cyp3a4_substrate')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cyp3a4_substrate')

@register_features_generator('dili')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'dili')

@register_features_generator('eye_corrosion')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'eye_corrosion')

@register_features_generator('eye_irritation')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'eye_irritation')

@register_features_generator('f20')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'f20')

@register_features_generator('f30')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'f30')

@register_features_generator('genotoxicity')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'genotoxicity')

@register_features_generator('h_ht')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'h_ht')

@register_features_generator('herg')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'herg')

@register_features_generator('herg1')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'herg1')

@register_features_generator('herg2')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'herg2')

@register_features_generator('hia')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'hia')

@register_features_generator('micronucleus_tox')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'micronucleus_tox')

@register_features_generator('nr_ahr')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_ahr')

@register_features_generator('nr_ar')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_ar')

@register_features_generator('nr_ar_lbd')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_ar_lbd')

@register_features_generator('nr_aromatase')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_aromatase')

@register_features_generator('nr_er')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_er')

@register_features_generator('nr_er_lbd')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_er_lbd')

@register_features_generator('nr_gr')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_gr')

@register_features_generator('nr_ppar_gamma')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_ppar_gamma')

@register_features_generator('nr_tr')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'nr_tr')

@register_features_generator('oatp1b1')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'oatp1b1')

@register_features_generator('oatp1b3')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'oatp1b3')

@register_features_generator('ob')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'ob')

@register_features_generator('oct2')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'oct2')

@register_features_generator('pgp_inhibitor')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'pgp_inhibitor')

@register_features_generator('pgp_substrate')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'pgp_substrate')

@register_features_generator('respiratory_tox')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'respiratory_tox')

@register_features_generator('skin_sens')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'skin_sens')

@register_features_generator('sr_are')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'sr_are')

@register_features_generator('sr_atad5')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'sr_atad5')

@register_features_generator('sr_hse')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'sr_hse')

@register_features_generator('sr_mmp')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'sr_mmp')    

@register_features_generator('sr_p53')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'sr_p53')    

@register_features_generator('t0.5')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'t0.5')            
# regression

@register_features_generator('bbb_cns')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bbb_cns')    

@register_features_generator('bioconcF')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bioconcF')    

@register_features_generator('bp')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'bp')    

@register_features_generator('caco2')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'caco2')    

@register_features_generator('caco2_logPaap')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'caco2_logPaap')    

@register_features_generator('cl')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'cl')    

@register_features_generator('fdamdd_reg')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'fdamdd_reg')    

@register_features_generator('fm_reg')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'fm_reg')    

@register_features_generator('fu')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'fu')    

@register_features_generator('hydrationE')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'hydrationE')    

@register_features_generator('lc50')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'lc50')    

@register_features_generator('lc50dm')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'lc50dm')    

@register_features_generator('ld50')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'ld50')    

@register_features_generator('logbcf')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'logbcf')    

@register_features_generator('logd')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'logd')    

@register_features_generator('logp')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'logp')    

@register_features_generator('logs')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'logs')    

@register_features_generator('logvp')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'logvp')    

@register_features_generator('mdck')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'mdck') 

@register_features_generator('mp')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'mp')

@register_features_generator('pka')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'pka')

@register_features_generator('pkb')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'pkb')

@register_features_generator('ppb')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'ppb')

@register_features_generator('pyriformis_reg')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'pyriformis_reg')

@register_features_generator('rat_acute_reg')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'rat_acute_reg')

@register_features_generator('rat_chronic')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'rat_chronic')

@register_features_generator('skin_permeability')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'skin_permeability')

@register_features_generator('vd')
def deeppk_features_generator(mol: Molecule) -> np.ndarray:
    return deeppk_features(mol,'vd')

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
except ImportError:
    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')


"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
