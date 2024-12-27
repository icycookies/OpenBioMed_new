import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Molecule, Text

if __name__ == "__main__":
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./logs/text_based_molecule_editing/molt5-fs_mol_edit/train/checkpoints/last.ckpt",
        output_prompt="Edited molecule: {output}"
    )
    input_smiles = "Nc1[nH]c(C(=O)c2ccccc2)c(-c2ccccn2)c1C(=O)c1c[nH]c2ccc(Br)cc12"
    input_text = "This molecule can bind with recombinant human 15-LOX-1"
    outputs = pipeline.run(
        molecule=Molecule.from_smiles(input_smiles),
        text=Text.from_str(input_text),
    )
    print(f"Input SMILES: {input_smiles}")
    print(f"Input Text: {input_text}")
    print(outputs[0])