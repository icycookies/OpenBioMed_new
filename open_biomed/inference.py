import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from open_biomed.core.pipeline import InferencePipeline
from open_biomed.core.visualize import MoleculeVisualizer, ProteinVisualizer, ComplexVisualizer
from open_biomed.data import Molecule, Text, Protein, Pocket

def test_text_based_molecule_editing():
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./logs/text_based_molecule_editing/molt5-fs_mol_edit/train/checkpoints/last.ckpt",
        output_prompt="Edited molecule: {output}",
        device="cuda:0"
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

def test_structure_based_drug_design():
    os.system("rm ./tmp/*.pkl")
    pipeline = InferencePipeline(
        task="structure_based_drug_design",
        model="pharmolix_fm",
        model_ckpt="./checkpoints/demo/pharmolix_fm.ckpt",
        output_prompt="Designed molecule: {output}",
        device="cuda:0"
    )
    protein = Protein.from_pdb_file("./tmp/sbdd/4xli_B.pdb")
    ligand = Molecule.from_sdf_file("./tmp/sbdd/4xli_B_ref.sdf")
    pocket = Pocket.from_protein_ref_ligand(protein, ligand)
    outputs = pipeline.run(
        pocket=pocket
    )
    print(outputs)

def visualize_molecule():
    os.system("rm ./tmp/*.png")
    os.system("rm ./tmp/*.gif")
    pipeline = MoleculeVisualizer()
    ligand = Molecule.from_binary_file("./tmp/mol_1739178847441_0.pkl")
    outputs = pipeline.run(
        ligand, mode="2D", rotate=False
    )
    print(outputs)

def visualize_complex():
    os.system("rm ./tmp/*.png")
    os.system("rm ./tmp/*.gif")
    pipeline = ComplexVisualizer()
    ligand = Molecule.from_binary_file("./tmp/mol_1739180786951_0.pkl")
    protein = Protein.from_pdb_file("./tmp/sbdd/4xli_B.pdb")
    outputs = pipeline.run(molecule=ligand, protein=protein, rotate=True)
    print(outputs)

INFERENCE_UNIT_TESTS = {
    "text_based_molecule_editing": test_text_based_molecule_editing,
    "structure_based_drug_design": test_structure_based_drug_design,
    "visualize_molecule": visualize_molecule,
    "visualize_complex": visualize_complex,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="text_based_molecule_editing")
    args = parser.parse_args()

    if args.task not in INFERENCE_UNIT_TESTS:
        raise NotImplementedError(f"{args.task} is not currently supported!")
    else:
        INFERENCE_UNIT_TESTS[args.task]()
    