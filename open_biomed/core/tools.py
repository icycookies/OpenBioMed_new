from open_biomed.core.web_request import *
from open_biomed.core.visualize import *
from open_biomed.scripts.inference import *

text_based_molecule_editing_pipeline = test_text_based_molecule_editing()
structure_based_drug_design_pipeline = test_structure_based_drug_design()
molecule_question_answering_pipeline = test_molecule_question_answering()
visualize_complex_pipeline = visualize_complex()
visualize_molecule_pipeline = visualize_molecule()
pubchemrequester = PubChemRequester(db_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{accession}/SDF")
websearchrequester =  WebSearchRequester()
molecule_property_prediction_pipeline = test_molecule_property_prediction()


TOOLS = {
    "text_based_molecule_editing": text_based_molecule_editing_pipeline,
    "structure_based_drug_design": structure_based_drug_design_pipeline,
    "molecule_question_answering": molecule_question_answering_pipeline,
    "visualize_molecule": visualize_molecule_pipeline,
    "visualize_complex": visualize_complex_pipeline,
    "pubchemrequest": pubchemrequester,
    "websearchrequest": websearchrequester,
    "molecule_property_prediction": molecule_property_prediction_pipeline
}