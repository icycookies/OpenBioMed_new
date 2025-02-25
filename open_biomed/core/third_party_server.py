import os
import subprocess
import glob
import logging
import random
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_biomed.data import Molecule, Protein, Pocket


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Protein_Binding_Site_Prediction():
    def __init__(self, output_path: str = "./temp/p2pocket") -> None:
        self.output_path = output_path
    

    def run(self, pdb_file: str, threads: int=8) -> Pocket:
        

        pdb_filename = os.path.basename(pdb_file)
        pdb_name = os.path.splitext(pdb_filename)[0]
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)   
        output_path = os.path.join(self.output_path, f"p2pocket_{pdb_name}")

        try:
            # Construct the command
            command = [
                "./third_party/p2rank_2.5/prank",
                "predict",
                "-f", pdb_file,
                "-threads", str(threads),
                "-o", output_path
            ]

            # Execute the command
            logging.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0:
                logging.info(f"Successfully processed {pdb_file}")
            else:
                logging.error(f"Failed to process {pdb_file}")
                logging.error(f"Error: {result.stderr}")

            
            file = output_path + "/" + pdb_name + ".pdb_predictions.csv"
            pocket_df = pd.read_csv(file)
            
            pocket_residues = []
            for index, row in pocket_df.iterrows():
                pocket_name = row['name     ']
                residue_ids = row[' residue_ids'].split()  # Split a string into a list by space
                pocket_residues.append([int(i.split("_")[1]) for i in residue_ids])

            
            protein = Protein.from_pdb_file(pdb_file)

            pocket_residue = random.choice(pocket_residues)
            
            pocket = Pocket.from_protein_subseq(protein, pocket_residue)

            return pocket
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return None



if __name__ == "__main__":
    pdb_file = "third_party/p2rank_2.5/test_data/1fbl.pdb"
    pocket_predictor = Protein_Binding_Site_Prediction()
    pocket = pocket_predictor.run(pdb_file)
    print(pocket)