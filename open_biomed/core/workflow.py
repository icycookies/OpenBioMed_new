from typing import Any, Dict, TextIO

import asyncio
import copy
import json
import numpy as np
import os
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from open_biomed.core.pipeline import Pipeline
from open_biomed.core.tool_registry import TOOLS
from open_biomed.core.visualize import Visualizer
from open_biomed.utils.config import Config
from open_biomed.utils.misc import wrap_and_select_outputs, create_tool_input

def parse_frontend(input: str) -> str:
    pass

def get_str_from_dict(input: Dict[str, Any]) -> str:
    input_str = copy.deepcopy(input)
    for key in input_str:
        input_str[key] = str(input_str[key])
    return json.dumps(input_str)

class WorkflowNode():
    def __init__(self,
        name: str,
        executable: Any,
        inputs: Dict[str, Any]={},
    ) -> None:
        self.name = name
        self.executable = executable
        self.inputs = {}
        for key, value in inputs.items():
            self.inputs[key] = create_tool_input(key, value)
        self.orig_inputs = copy.deepcopy(self.inputs)

class Workflow():
    def __init__(self, config: Config) -> None:
        # create a list of tools as nodes and a DAG as edges for the workflow to run
        self.nodes = []
        self.edges = []
        for tool in config.tools:
            self.nodes.append(WorkflowNode(
                name=tool["name"],
                executable=TOOLS[tool["name"]],
                inputs=tool.get("inputs", {})
            ))
        for edge in config.edges:
            self.edges.append((edge["start"], edge["end"]))

    @classmethod
    def from_forntend_export(cls, file: str) -> None:
        yml_file = parse_frontend(file)
        return cls(config=Config(config_file=yml_file))

    async def run(self, num_repeats=10, context: TextIO=sys.stdout) -> Any:
        context.write("Now we have a workflow with the following tools: \n")
        for i, node in enumerate(self.nodes):
            context.write(f"Tool No.{i + 1}: {node.executable.print_usage()}\n\n\n")
        for repeat in range(num_repeats):
            context.write(f"Repeating workflow execution No.{repeat + 1}:\n")
            edge_list = [[] for i in range(len(self.nodes))]
            in_deg = [0 for i in range(len(self.nodes))]
            for edge in self.edges:
                edge_list[edge[0]].append(edge[1])
                in_deg[edge[1]] += 1
            queue = []
            for i in range(len(self.nodes)):
                self.nodes[i].inputs = copy.deepcopy(self.nodes[i].orig_inputs)
                if in_deg[i] == 0:
                    queue.append(i)
            while len(queue) > 0:
                u = queue.pop(0)
                context.write(f"Next we execute Tool No.{u + 1}.\n The inputs of this tool are: {get_str_from_dict(self.nodes[u].inputs)}\n")
                if getattr(self.nodes[u].executable, "requires_async", False):
                    outputs = await asyncio.create_task(self.nodes[u].executable.run(**self.nodes[u].inputs))
                elif isinstance(self.nodes[u].executable, Visualizer):
                    # Execute visualization in a subprocess to avoid internal errors of PyMol when an InferencePipeline object is created
                    vis_process = [
                        "python3", "./open_biomed/core/visualize.py", 
                        "--task", self.nodes[u].name,
                        "--save_output_filename", "./tmp/visualization_file.txt",
                    ]
                    for key, value in self.nodes[u].inputs.items():
                        if key in ["molecule", "protein", "pocket"]:
                            vis_process.append(f"--{key}")
                            vis_process.append(value.save_binary())
                    subprocess.Popen(vis_process).communicate()
                    outputs = open("./tmp/visualization_file.txt", "r").read()
                    outputs = [outputs], [outputs]
                else:
                    outputs = self.nodes[u].executable.run(**self.nodes[u].inputs)
                outputs = wrap_and_select_outputs(outputs, context)
                context.write(f"The outputs of this tool are: {get_str_from_dict(outputs)}\n\n\n")
                for out_node in edge_list[u]:
                    in_deg[out_node] -= 1
                    if in_deg[out_node] == 0:
                        queue.append(out_node)
                    for key, value in outputs.items():
                        self.nodes[out_node].inputs[key] = copy.deepcopy(value)

if __name__ == "__main__":
    config = Config(config_file="./configs/workflow/drug_design.yaml")
    workflow = Workflow(config)
    workflow.run(num_repeats=10, context=open("./logs/workflow_outputs.txt", "w"))
    # workflow.run(num_repeats=1)