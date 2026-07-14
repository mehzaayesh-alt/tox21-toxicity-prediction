# prep for the tox21 project
# target is SR-p53 - p53 stress-response pathway, basically a proxy for
# whether a compound triggers DNA-damage response. picked this one over
# the other 11 assays in tox21 because it's the most directly tied to
# carcinogenicity risk, which felt like the most "pharma relevant" framing

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # rdkit is loud about invalid SMILES, don't need every warning

TARGET = "SR-p53"
N_BITS = 2048
RADIUS = 2  # ECFP4 equivalent

df = pd.read_csv("tox21.csv")
df = df[["smiles", TARGET]].dropna(subset=[TARGET])
print(f"labeled compounds for {TARGET}: {len(df)}")
print(f"positive (toxic) rate: {df[TARGET].mean():.3f}")

fps = []
labels = []
smiles_kept = []
n_failed = 0

for smi, label in zip(df["smiles"], df[TARGET]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_failed += 1
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    fps.append(np.array(fp))
    labels.append(int(label))
    smiles_kept.append(smi)

print(f"failed to parse: {n_failed}")
print(f"final dataset: {len(fps)} molecules")

X = np.array(fps)
y = np.array(labels)

np.save("X.npy", X)
np.save("y.npy", y)
with open("smiles_kept.txt", "w") as f:
    f.write("\n".join(smiles_kept))

print("saved X.npy", X.shape, "y.npy", y.shape)
