from open_biomed.core.web_request import *
from open_biomed.core.visualize import *
from open_biomed.scripts.inference import *
from open_biomed.core.third_party_server import *

# TODO: Add pocket prediction as a tool
class LazyDictForTool(dict):
    def __missing__(self, key):
        if key == "text_based_molecule_editing":
            self[key] = test_text_based_molecule_editing()
        elif key == "molecule_property_prediction":
            self[key] = test_molecule_property_prediction()
        elif key == "structure_based_drug_design":
            self[key] = test_structure_based_drug_design()
        elif key == "molecule_question_answering":
            self[key] = test_molecule_question_answering()
        elif key == "protein_question_answering":
            self[key] = test_protein_question_answering()
        elif key == "protein_binding_site_prediction":
            self[key] = ProteinBindingSitePrediction()
        elif key == "visualize_molecule":
            self[key] = MoleculeVisualizer()
        elif key == "visualize_protein":
            self[key] = ProteinVisualizer()
        elif key == "visualize_complex":
            self[key] = ComplexVisualizer()
        elif key == "molecule_name_request":
            self[key] = PubChemRequester(db_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{accession}/SDF")
        elif key == "molecule_structure_request":
            self[key] = PubChemStructureRequester()
        elif key == "protein_pdb_request":
            self[key] = PDBRequester()
        elif key == "web_search":
            self[key] = WebSearchRequester()
        else:
            raise NotImplementedError(f"{key} is currently not supported!")
        return self[key]

TOOLS = LazyDictForTool()