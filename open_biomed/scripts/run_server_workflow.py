import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio

# import function
from open_biomed.core.llm_request import ReportGeneratorSBDD, ReportGeneratorGeneral

app = FastAPI()

# Define the request body model
class ReportRequest(BaseModel):
    task: str
    config_file: str
    num_repeats: int

@app.post("/run_workflow/")
async def run_workflow(request: ReportRequest):
    request = request.model_dump()
    config_file = request["config_file"]
    task = request["task"].lower()
    try:
        if task == "sbdd":
            requester = ReportGeneratorSBDD()
            required_inputs = ["config_file", "num_repeats"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="workflow config file is required for report generation task")

            resp = await requester.run(config_file=request["config_file"], num_repeats=request["num_repeats"])

            report_content = resp['final_resp']
            report_thinking = resp['reasoning']
            return {"type": task+"_report", "content": report_content, "thinking": report_thinking}
        else:
            requester = ReportGeneratorGeneral()
            required_inputs = ["config_file", "num_repeats"]
            if not all(key in request for key in required_inputs):
                raise HTTPException(
                    status_code=400, detail="workflow config file is required for report generation task")

            resp = await requester.run(config_file=request["config_file"], num_repeats=request["num_repeats"])

            report_content = resp['final_resp']
            report_thinking = resp['reasoning']
            return {"type": task+"_report", "content": report_content, "thinking": report_thinking}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
