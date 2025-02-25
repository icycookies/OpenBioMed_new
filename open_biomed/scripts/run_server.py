from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn.functional as F
import uvicorn
import asyncio
from typing import Optional

# import function
from open_biomed.scripts.inference import test_text_based_molecule_editing, test_structure_based_drug_design, test_molecule_question_answering, visualize_molecule, visualize_complex, test_molecule_property_prediction
from open_biomed.data import Molecule, Text, Protein, Pocket
from open_biomed.core.oss_warpper import oss_warpper
from open_biomed.core.web_request import PubChemRequester, MSARequester, PDBRequester, UniProtRequester, WebSearchRequester



app = FastAPI()


# Define the request body model
class TaskRequest(BaseModel):
    task: str
    model: str
    molecule: Optional[str] = None  # 使用 Optional[str]
    protein: Optional[str] = None
    pocket: Optional[str] = None
    text: Optional[str] = None
    dataset: Optional[str] = None
    query: Optional[str] = None

class SearchRequest(BaseModel):
    database: str
    query: str

molecule_property_prediction_prompt = {
    "BBBP": "The blood-brain barrier prediction result is {output}. "
            "This result indicates the model's prediction of whether the compound can effectively penetrate the blood-brain barrier. "
            "A positive result suggests that the compound may have potential for central nervous system targeting, "
            "while a negative result implies limited permeability.",
    
    "ClinTox": "The clinical toxicity prediction result is {output}. "
                "This result reflects the model's assessment of the likelihood that the compound will fail clinical trials due to toxicity concerns. "
                "A positive result indicates a higher risk of toxicity, while a negative result suggests the compound is less likely to exhibit significant toxicity in clinical settings.",
    
    "Tox21": "The Tox21 toxicity assessment result is {output}. "
             "This result provides an evaluation of the compound's potential toxicity, focusing on nuclear receptors and stress response pathways. "
             "A positive result indicates the presence of toxic effects, while a negative result suggests the compound is less likely to exhibit these toxicities.",
    
    "ToxCast": "The ToxCast toxicity screening result is {output}. "
               "This result is based on high-throughput in vitro screening and indicates the compound's potential toxicity profile. "
               "A positive result suggests significant toxicity, while a negative result implies lower toxicity risk.",
    
    "SIDER": "The SIDER adverse drug reaction analysis result is {output}. "
             "This result provides insights into the potential adverse drug reactions (ADRs) associated with the compound. "
             "A positive result indicates a higher likelihood of adverse reactions, while a negative result suggests fewer potential ADRs.",
    
    "HIV": "The HIV inhibition prediction result is {output}. "
           "This result indicates the model's prediction of the compound's ability to inhibit HIV replication. "
           "A positive result suggests strong inhibitory activity, while a negative result implies limited effectiveness against HIV.",
    
    "BACE": "The BACE-1 activity prediction result is {output}. "
            "This result provides a prediction of the compound's binding affinity to human β-secretase 1 (BACE-1). "
            "A positive result indicates strong binding activity, suggesting potential as a BACE-1 inhibitor, while a negative result implies weaker binding.",
    
    "MUV": "The MUV virtual screening validation result is {output}. "
           "This result indicates the model's assessment of the compound's potential as a hit in virtual screening. "
           "A positive result suggests the compound is likely to be active against the target, while a negative result implies lower activity."
}

# 定义一个类，通过字符串判断文件后缀来选择不同的数据读取方式
class IO_Reader:
    def __init__(self):
        pass
    @staticmethod
    def get_molecule(string):
        if string.endswith(".sdf"):
            return Molecule.from_sdf_file(string)
        elif string.endswith(".pkl"):
            return Molecule.from_binary_file(string)
        else:
            return Molecule.from_smiles(string)
    
    @staticmethod
    def get_protein(string):
        if string.endswith(".pdb"):
            return Protein.from_pdb_file(string)
        elif string.endswith(".pkl"):
            return Protein.from_binary_file(string)
        else:
            return Protein.from_fasta(string)

    @staticmethod
    def get_text(string):
        return Text.from_str(string)


# Create a dictionary to store the pipeline instance

text_based_molecule_editing_pipeline = test_text_based_molecule_editing()
structure_based_drug_design_pipeline = test_structure_based_drug_design()
molecule_question_answering_pipeline = test_molecule_question_answering()
visualize_complex_pipeline = visualize_complex()
visualize_molecule_pipeline = visualize_molecule()
pubchemrequester = PubChemRequester(db_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{accession}/SDF")
websearchrequester =  WebSearchRequester()
molecule_property_prediction_pipeline = test_molecule_property_prediction()


pipelines = {
    "text_based_molecule_editing": text_based_molecule_editing_pipeline,
    "structure_based_drug_design": structure_based_drug_design_pipeline,
    "molecule_question_answering": molecule_question_answering_pipeline,
    "visualize_molecule": visualize_molecule_pipeline,
    "visualize_complex": visualize_complex_pipeline,
    "pubchemrequest": pubchemrequester,
    "websearchrequest": websearchrequester,
    "molecule_property_prediction": molecule_property_prediction_pipeline
}


@app.post("/run_pipeline/")
async def run_pipeline(request: TaskRequest):
    request = request.model_dump()
    task_name = request["task"].lower()
    model = request["model"].lower()

    try:
        # Call the corresponding function based on the task name and get the pipeline instance
        if task_name == "text_based_molecule_editing":
            pipeline = pipelines["text_based_molecule_editing"]
            required_inputs = ["molecule", "text"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="molecule and text are required for text_based_molecule_editing task")
            molecule = IO_Reader.get_molecule(request["molecule"])
            text = IO_Reader.get_text(request["text"])
            outputs = pipeline.run(
                molecule=molecule,
                text=text)
            print(outputs)
            smiles = outputs[0][0].split(pipeline.output_prompt.split("{output}")[0])[1]
            path = outputs[1][0]
            output = {"task": task_name, "model": model, "molecule": path, "molecule_preview": smiles}
        elif task_name == "structure_based_drug_design":
            pipeline = pipelines["structure_based_drug_design"]
            required_inputs = ["protein", "molecule"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="protein and molecule are required for structure_based_drug_design task")
            protein = IO_Reader.get_protein(request["protein"])
            ligand = IO_Reader.get_molecule(request["molecule"])
            pocket = Pocket.from_protein_ref_ligand(protein, ligand)
            outputs = pipeline.run(
                pocket=pocket
            )
            smiles = outputs[0][0].split(pipeline.output_prompt.split("{output}")[0])[1]
            path = outputs[1][0]
            output =  {"task": task_name, "model": model, "molecule": path, "molecule_preview": smiles}
        elif task_name == "molecule_question_answering":
            pipeline = pipelines["molecule_question_answering"]
            required_inputs = ["text"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="text is required for molecule_question_answering task")
            text = IO_Reader.get_text(request["text"])
            outputs = pipeline.run(
                text=text
            )
            text = outputs[0][0].split(pipeline.output_prompt.split("{output}")[0])[1]
            output =  {"task": task_name, "model": model, "text": text}
        elif task_name == "visualize_molecule":
            pipeline = pipelines["visualize_molecule"]
            required_inputs = ["molecule"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="molecule are required for visualize_molecule task")
            ligand = Molecule.from_binary_file(request["molecule"])
            outputs = pipeline.run(ligand, config="ball_and_stick", rotate=False)
            oss_file_path = oss_warpper.generate_file_name(outputs)
            outputs = oss_warpper.upload(oss_file_path, outputs)
            output = {"task": task_name, "image": outputs}
        elif task_name == "visualize_complex":
            pipeline = pipelines["visualize_complex"]
            required_inputs = ["protein", "molecule"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="protein and molecule are required for visualize_complex task")
            ligand = Molecule.from_binary_file(request["molecule"])
            protein = Protein.from_pdb_file(request["protein"])
            outputs = pipeline.run(molecule=ligand, protein=protein, rotate=True)
            oss_file_path = oss_warpper.generate_file_name(outputs)
            outputs = oss_warpper.upload(oss_file_path, outputs)
            output = {"task": task_name, "image": outputs}
        elif task_name == "molecule_property_prediction":
            pipeline = pipelines["molecule_property_prediction"]
            required_inputs = ["molecule", "dataset"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="molecule and dataset are required for molecule_property_prediction task")
            molecule = IO_Reader.get_molecule(request["molecule"])
            dataset = IO_Reader.get_text(request["dataset"])
            outputs = pipeline.run(
                molecule=molecule,
                #dataset=dataset
            )
            output = outputs[0][0].cpu()
            # 应用softmax函数
            output = F.softmax(output, dim=0).tolist()
            #output = [round(x, 4) for x in output]
            output = {"task": task_name, "model": model, "score": str(output[0])}

            #output = {"score": format(float(outputs[0][0]), ".4f")}


        else:
            raise HTTPException(status_code=400, detail="Invalid task_name")
    
        return output
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web_search/")
async def web_search(request: SearchRequest):
    request = request.model_dump()
    database = request["database"].lower()
    # Call the corresponding function based on the task name and get the pipeline instance
    try:
        if database == "pubchem":
            requester = pipelines["pubchemrequest"]
            required_inputs = ["query"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="query are required for pubchemrequest task")
            outputs = await requester.run(request["query"])
            smiles = Molecule.from_binary_file(outputs).smiles
            return {"database": database, "molecule": outputs, "molecule_preview": smiles}
        if database == "web":
            requester = pipelines["websearchrequest"]
            required_inputs = ["query"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="query are required for websearch task")
            outputs = requester.run(request["query"])
            return {"database": database, "text": outputs}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
