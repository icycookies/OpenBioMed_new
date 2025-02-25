import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse

from open_biomed.core.pipeline import InferencePipeline
from open_biomed.core.oss_warpper import oss_warpper
from open_biomed.core.visualize import MoleculeVisualizer, ProteinVisualizer, ComplexVisualizer
from open_biomed.data import Molecule, Text, Protein, Pocket
from open_biomed.tasks.aidd_tasks.protein_molecule_docking import VinaDockTask

os.environ["dataset_name"] = "BBBP"

def test_text_based_molecule_editing():
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/logs/text_based_molecule_editing/molt5-fs_mol_edit/train/checkpoints/last.ckpt",
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
    return pipeline

def test_pocket_molecule_docking():
    pipeline = InferencePipeline(
        task="pocket_molecule_docking",
        model="pharmolix_fm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/pharmolix_fm.ckpt",
        output_prompt="Designed molecule: {output}",
        device="cuda:0"
    )
    protein = Protein.from_pdb_file("tmp/test/tmp/sbdd/4xli_B.pdb")
    ligand = Molecule.from_sdf_file("tmp/test/tmp/sbdd/4xli_B_ref.sdf")
    pocket = Pocket.from_protein_ref_ligand(protein, ligand)
    outputs = pipeline.run(
        molecule=ligand,
        pocket=pocket,
    )
    print(outputs)

def test_structure_based_drug_design():
    os.system("rm ./tmp/*.pkl")
    pipeline = InferencePipeline(
        task="structure_based_drug_design",
        model="pharmolix_fm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/pharmolix_fm.ckpt",
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
    predicted_molecule = Molecule.from_binary_file(outputs[1][0])
    vina = VinaDockTask(mode="dock")
    print(vina.run(ligand, protein)[0])
    print(vina.run(predicted_molecule, protein)[0])
    return pipeline


def test_molecule_question_answering():
    pipeline = InferencePipeline(
        task="molecule_question_answering",
        model="biot5",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/logs/molecule_question_answering/biot5-MQA/train/checkpoints/last.ckpt",
        output_prompt="MQA: {output}",
        device="cuda:0"
    )
    molecule = Molecule.from_smiles("C[C@@H]1CCCCO[C@@H](CN(C)C(=O)Cc2ccccc2)[C@@H](C)CN([C@@H](C)CO)C(=O)c2cc(NS(C)(=O)=O)ccc2O1")
    question = Text.from_str("Could you provide the systematic name of this compound according to IUPAC nomenclature?")
    outputs = pipeline.run(
        molecule=molecule,
        text=question,
    )
    print(outputs)
    return pipeline

def test_protein_question_answering():
    pipeline = InferencePipeline(
        task="protein_question_answering",
        model="biot5",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/logs/protein_question_answering/biot5-PQA/train/checkpoints/last.ckpt",
        output_prompt="PQA: {output}",
        device="cuda:0"
    )
    protein = Protein.from_fasta("MRVGVIRFPGSNCDRDVHHVLELAGAEPEYVWWNQRNLDHLDAVVIPGGFSYGDYLRAGAIAAITPVMDAVRELVREEKPVLGICNGAQILAEVGLVPGVFTVNEHPKFNCQWTELRVKTTRTPFTGLFKKDEVIRMPVAHAEGRYYHDNISEVWENDQVVLQFHGENPNGSLDGITGVCDESGLVCAVMPHPERASEEILGSVDGFKFFRGILKFRG")
    question = Text.from_str("Inspect the protein sequence and offer a concise description of its properties.")
    outputs = pipeline.run(
        protein=protein,
        text=question,
    )
    print(outputs)
    return pipeline

def test_mutation_explanation():
    pipeline = InferencePipeline(
        task="mutation_explanation",
        model="mutaplm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/mutaplm.pth",
        output_prompt="Mutation effect: {output}",
        device="cuda:0"
    )
    # protein = Protein.from_fasta("MQPWHGKAMQRASEAGATAPKASARNARGAPMDPTESPAAPEAALPKAGKFGPARKSGSRQKKSAPDTQERPPVRATGARAKKAPQRAQDTQPSDATSAPGAEGLEPPAAREPALSRAGSCRQRGARCSTKPRPPPGPWDVPSPGLPVSAPILVRRDAAPGASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDIKDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAKEGNGFQEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFDEF")
    # mutation = "D95A"
    protein = Protein.from_fasta("MTLENVLEAARHLHQTLPALSEFGNWPTDLTATGLQPRAIPATPLVQALDQPGSPRTTGLVQAIRSAAHLAHWKRTYTEAEVGADFRNRYGYFELFGPTGHFHSTQLRGYVAYWGAGLDYDWHSHQAEELYLTLAGGAVFKVDGERAFVGAEGTRLHASWQSAAMSTGDQPILTFVLWRGEGLNALPRMDAA")
    mutation = "H163A"
    function = Text.from_str("Able to cleave dimethylsulfonioproprionate (DMSP) in vitro, releasing dimethyl sulfide (DMS). DMS is the principal form by which sulfur is transported from oceans to the atmosphere. The real activity of the protein is however subject to debate and it is unclear whether it constitutes a real dimethylsulfonioproprionate lyase in vivo: the very low activity with DMSP as substrate suggests that DMSP is not its native substrate.")
    outputs = pipeline.run(
        wild_type=protein,
        mutation=mutation,
        #function=function,
    )
    print(outputs)
    return pipeline

def test_mutation_engineering():
    pipeline = InferencePipeline(
        task="mutation_engineering",
        model="mutaplm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/mutaplm.pth",
        output_prompt="Designed mutation: {output}",
        device="cuda:1"
    )
    protein = Protein.from_fasta("MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK")
    prompt = Text.from_str("Strongly enhanced InsP6 kinase activity. The mutation in the ITPK protein causes a change in its catalytic activity.")
    outputs = pipeline.run(
        wild_type=protein,
        prompt=prompt
    )
    print(outputs)

def test_protein_generation():
    from open_biomed.models.functional_model_registry import PROTEIN_DECODER_REGISTRY
    from open_biomed.utils.config import Config
    config = Config(config_file="./configs/model/progen.yaml").model
    model = PROTEIN_DECODER_REGISTRY["progen"](config)
    model = model.to("cuda:1")
    print(model.generate_protein()[0])
    protein = Protein.from_fasta('GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM')
    protein = model.collator([model.featurizer(protein)])
    protein = {
        "input_ids": protein["input_ids"].to("cuda:1"),
        "attention_mask": protein["attention_mask"].to("cuda:1"),
    }
    print(model.generate_loss(protein))

def visualize_molecule():
    #os.system("rm ./tmp/*.png")
    #os.system("rm ./tmp/*.gif")
    pipeline = MoleculeVisualizer()
    ligand = Molecule.from_binary_file("tmp/test/tmp/pubchem_240.pkl")
    outputs = pipeline.run(
        ligand, config="ball_and_stick", rotate=False
    )
    oss_file_path = oss_warpper.generate_file_name(outputs)
    url = oss_warpper.upload(oss_file_path, outputs)
    print(url)
    return pipeline

def visualize_protein():
    os.system("rm ./tmp/*.png")
    os.system("rm ./tmp/*.gif")
    pipeline = ProteinVisualizer()
    protein = Protein.from_pdb_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/sbdd/4xli_B.pdb")
    outputs = pipeline.run(protein=protein, config="cartoon", rotate=False)
    print(outputs)
    return pipeline

def visualize_complex():
    #os.system("rm ./tmp/*.png")
    #os.system("rm ./tmp/*.gif")
    pipeline = ComplexVisualizer()
    ligand = Molecule.from_binary_file("tmp/test/tmp/pubchem_240.pkl")
    protein = Protein.from_pdb_file("tmp/test/tmp/pdb_6LVN.pdb")
    try:
        outputs = pipeline.run(molecule=ligand, protein=protein, rotate=True)
        oss_file_path = oss_warpper.generate_file_name(outputs)
        url = oss_warpper.upload(oss_file_path, outputs)
        print(url)
    except:
        print("visualize_complex failed")

    return pipeline



def test_molecule_property_prediction():
    pipeline = InferencePipeline(
        task="molecule_property_prediction",
        model="graphmvp",
        model_ckpt="/AIRvePFS/dair/yk-data/projects/OpenBioMed_new/logs/molecule_property_prediction/graphmvp-BBBP/train/checkpoints/last.ckpt",
        device="cuda:0"
    )
    input_smiles = "Nc1[nH]c(C(=O)c2ccccc2)c(-c2ccccn2)c1C(=O)c1c[nH]c2ccc(Br)cc12"
    outputs = pipeline.run(
        molecule=Molecule.from_smiles(input_smiles)
    )
    return pipeline


INFERENCE_UNIT_TESTS = {
    "text_based_molecule_editing": test_text_based_molecule_editing,
    "pocket_molecule_docking": test_pocket_molecule_docking,
    "structure_based_drug_design": test_structure_based_drug_design,
    "visualize_molecule": visualize_molecule,
    "visualize_protein": visualize_protein,
    "visualize_complex": visualize_complex,
    "molecule_question_answering": test_molecule_question_answering,
    "protein_question_answering": test_protein_question_answering,
    "mutation_explanation": test_mutation_explanation,
    "mutation_engineering": test_mutation_engineering,
    "protein_generation": test_protein_generation,
    "molecule_property_prediction": test_molecule_property_prediction,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="text_based_molecule_editing")
    args = parser.parse_args()

    if args.task not in INFERENCE_UNIT_TESTS:
        raise NotImplementedError(f"{args.task} is not currently supported!")
    else:
        INFERENCE_UNIT_TESTS[args.task]()
    