"""
predicting p53 stress-response toxicity (SR-p53, tox21) directly from
molecular structure - ECFP4 fingerprints in, random forest out.

class imbalance is real here (6% positive), so accuracy alone is useless -
a model that predicts "not toxic" every time gets 94% and tells you
nothing. using ROC-AUC and PR-AUC instead, which is what the tox21/
moleculenet papers report too so I can sanity check against published
numbers.
"""

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SEED = 42
N_BITS = 2048
RADIUS = 2

X = np.load("X.npy")
y = np.load("y.npy")
with open("smiles_kept.txt") as f:
    smiles = f.read().splitlines()

X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
    X, y, smiles, test_size=0.2, stratify=y, random_state=SEED
)
print(f"train={len(X_train)}  test={len(X_test)}  positive rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

# class_weight="balanced" instead of oversampling - simpler and works fine
# here given how bit-sparse the fingerprints already are
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)

logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
logreg.fit(X_train, y_train)
lr_probs = logreg.predict_proba(X_test)[:, 1]

results = {
    "rf_roc_auc": float(roc_auc_score(y_test, rf_probs)),
    "rf_pr_auc": float(average_precision_score(y_test, rf_probs)),
    "rf_balanced_acc": float(balanced_accuracy_score(y_test, rf_preds)),
    "logreg_roc_auc": float(roc_auc_score(y_test, lr_probs)),
    "logreg_pr_auc": float(average_precision_score(y_test, lr_probs)),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "positive_rate": float(y.mean()),
}
print("\nRandom Forest  | ROC-AUC={rf_roc_auc:.3f}  PR-AUC={rf_pr_auc:.3f}  bal.acc={rf_balanced_acc:.3f}".format(**results))
print("LogReg         | ROC-AUC={logreg_roc_auc:.3f}  PR-AUC={logreg_pr_auc:.3f}".format(**results))

# ---------- interpretability: which substructures actually drove toxicity predictions ----------
# each of the 2048 fingerprint bits corresponds to a specific local
# substructure (a fragment centered on some atom, radius <=2 bonds).
# pulling the bitInfo out for the training molecules lets me map the
# random forest's top feature-importance bits back to real chemical
# fragments instead of leaving them as opaque bit indices
top_bit_idx = np.argsort(rf.feature_importances_)[::-1][:8]

bit_to_example = {}
for smi in smi_train:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS, bitInfo=info)
    for bit in top_bit_idx:
        bit = int(bit)
        if bit in info and bit not in bit_to_example:
            atom_idx, radius = info[bit][0]
            if radius == 0:
                frag_smarts = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                frag_smarts = Chem.MolFragmentToSmiles(mol, atomsToUse=list(
                    {a for b in env for a in (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx())}
                ))
            bit_to_example[bit] = frag_smarts
    if len(bit_to_example) >= len(top_bit_idx):
        break

print("\ntop fingerprint bits by RF importance -> example substructure:")
importance_report = []
for bit in top_bit_idx:
    bit = int(bit)
    frag = bit_to_example.get(bit, "(not found in sample)")
    imp = float(rf.feature_importances_[bit])
    print(f"  bit {bit:5d}  importance={imp:.4f}  example fragment: {frag}")
    importance_report.append({"bit": bit, "importance": imp, "example_fragment": frag})

results["top_fragments"] = importance_report

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\ndone, results in results.json")
