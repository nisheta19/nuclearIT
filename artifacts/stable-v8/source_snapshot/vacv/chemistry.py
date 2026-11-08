from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

POLICY = {
    "canonical": "RDKit canonical isomeric SMILES; atom map numbers removed",
    "preserved": ["stereochemistry", "isotopes", "charges", "all fragments", "tautomer form"],
    "salt_stripping": False, "neutralization": False, "tautomer_normalization": False,
    "scaffold": "Bemis-Murcko without chirality; all acyclic molecules share ACYCLIC group",
    "invalid_train": "fail with audit; no silent exclusions",
    "invalid_test": "mean of training labels, with explicit fallback flag",
}

DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount", "NumAromaticRings", "NumAliphaticRings"]


def key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def featurize(smiles_values, config: dict):
    if "feature_sets" in config:
        values = list(smiles_values)
        parts = [featurize(values, spec) for spec in config["feature_sets"]]
        if not parts:
            raise ValueError("At least one structural representation is required")
        return parts[0][0], np.concatenate([part[1] for part in parts], axis=1)
    invariants = config.get("atom_invariants", "connectivity")
    if invariants not in ("connectivity", "features"):
        raise ValueError("Unknown atom invariants")
    kwargs = {"atomInvariantsGenerator": rdFingerprintGenerator.GetMorganFeatureAtomInvGen()} if invariants == "features" else {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=config["radius"], fpSize=config["bits"], includeChirality=config["include_chirality"], **kwargs)
    records, vectors = [], []
    descriptor_names = config.get("descriptors", [])
    if any(name not in DESCRIPTOR_NAMES for name in descriptor_names):
        raise ValueError("Descriptor is not in the approved structure-only allowlist")
    width = config["bits"] + len(descriptor_names)
    for raw in smiles_values:
        raw = str(raw)
        mol = Chem.MolFromSmiles(raw.strip()) if raw.strip() else None
        if mol is None or mol.GetNumAtoms() == 0:
            records.append({"SMILES": raw, "valid": False, "canonical_smiles": "", "structure_id": "INVALID_" + key(raw), "scaffold_smiles": "", "scaffold_id": "INVALID", "error": "empty_or_unparseable_smiles"})
            # NaN sentinel is never passed into an estimator.
            vectors.append(np.full(width, np.nan, dtype=np.float64))
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        records.append({"SMILES": raw, "valid": True, "canonical_smiles": canonical, "structure_id": key(canonical), "scaffold_smiles": scaffold, "scaffold_id": key(scaffold) if scaffold else "ACYCLIC", "error": ""})
        fingerprint = generator.GetCountFingerprintAsNumPy(mol) if config.get("kind", "binary") == "count" else generator.GetFingerprintAsNumPy(mol)
        vector = np.concatenate([fingerprint, [getattr(Descriptors, name)(mol) for name in descriptor_names]])
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite molecular descriptor; feature definition must be reviewed")
        vectors.append(vector)
    # Keep legacy binary-only precision for exact baseline compatibility.
    dtype = np.float64 if descriptor_names or config.get("kind") == "count" else np.float32
    return pd.DataFrame(records), np.asarray(vectors, dtype=dtype).reshape(-1, width)
