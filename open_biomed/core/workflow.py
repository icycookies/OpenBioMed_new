from typing import Any, Dict, TextIO

import argparse
import asyncio
import copy
import json
import numpy as np
import os
import yaml
import re
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_biomed.core.tool_registry import TOOLS
from open_biomed.core.visualize import Visualizer
from open_biomed.utils.config import Config
from open_biomed.data import Molecule, Protein, Text
from open_biomed.utils.misc import wrap_and_select_outputs, create_tool_input

def parse_frontend(filepath: str, output_floder: str = "tmp/workflow") -> str:

    def get_createdata_node(node):
        node_dict = {}
        id = node["id"]
        node_dict["value"] = []
        description = node["data"]["node"]["description"]
        data_context = node["data"]["node"]["template"]
        keys = list(data_context)
        pattern = re.compile(r'^field_\d+_key$')  # get param
        filtered_keys = [key for key in keys if pattern.match(key)]
        for key in filtered_keys:
            node_dict["value"].append(data_context[key]["value"])
        return node_dict

    # just get id and description
    def get_tool_node(node):
        node_dict = {}
        id = node["id"]
        description = node["data"]["node"]["description"]
        node_dict["description"] = [description]
        return node_dict


    def remove_duplicate_path(nodes):
        # node handled
        seen = set()
        new_nodes = []
        for head, tail in nodes:
            if (head, tail) not in seen:
                new_nodes.append([head, tail])
                seen.add((head, tail))
            else:
                print(f"The duplicate path has been removed: {head} -> {tail}")

        return new_nodes

    def check_strings(str_list, nested_list):
        for sublist in nested_list:
            for str in str_list:
                if str in sublist:
                    return False
        return True

    def merge_nodes(nodes):
        keywords = ["MergeDataComponent", "ParseData"]
        node_list_copy = copy.deepcopy(nodes)
        new_nodes_list = []
        for i, (head, tail) in enumerate(node_list_copy):
            if any(keyword in head for keyword in keywords):
                for j, (next_head, next_tail) in enumerate(node_list_copy):
                    if i != j and head == next_tail:
                        # merge path
                        new_nodes_list.append([next_head, tail])
            if any(keyword in tail for keyword in keywords):
                for j, (next_head, next_tail) in enumerate(node_list_copy):
                    if i != j and tail == next_head:
                        # merge path
                        new_nodes_list.append([head, next_tail])
        return nodes + new_nodes_list

    with open(file_path, 'r') as file:
        node_edge_data = json.load(file)
    tool_node = {}
    createdata_node = {}
    for node in node_edge_data["data"]["nodes"]:
        if "ChatInput" in node["id"] or "ChatOutput" in node["id"] or "ParseData" in node["id"]:
            continue
        if "PharmolixCreateData" in node["id"]:
            data = get_createdata_node(node)
            createdata_node[node["id"]] = data
        else:
            data = get_tool_node(node)
            tool_node[node["id"]] = data

    edge_list = []
    for node in node_edge_data["data"]["edges"]:
        source = node["source"]
        target = node["target"]
        if "PharmolixCreateData" in source:
            tool_node[target]["value"] = createdata_node[source]["value"]
            edge_list.append([source, target])
        else:
            edge_list.append([source, target])

    path_nodes = copy.deepcopy(edge_list)
    merge_nodes_list = merge_nodes(path_nodes)

    # remove paths containing path_keywords
    path_keywords = ["MergeDataComponent", "ParseData", "ChatOutput", "ChatInput", "Image Output", "PharmolixCreateData"]
    merge_nodes_list_copy = copy.deepcopy(merge_nodes_list)
    for index in range(len(merge_nodes_list_copy) - 1, -1, -1):
        # print(index)
        value = merge_nodes_list_copy[index]
        if any(keyword in value[0] for keyword in path_keywords) or any(keyword in value[1] for keyword in path_keywords):
            merge_nodes_list.pop(index)

    # get tool nodes info
    tool_node_filted = {}
    key_list = []
    for key, value in tool_node.items():
        if not any(keyword in key for keyword in path_keywords):
            tool_node_filted[key] = value
            key_list.append(key)

    # get the list of node in paths
    path_nodes_list = []
    for i in merge_nodes_list:
        path_nodes_list.append(i[0])
        path_nodes_list.append(i[1])
    path_nodes_list = list(set(path_nodes_list))

    assert len(path_nodes_list) == len(key_list), "please check the frontend json file"

    yaml_dict = {}
    yaml_dict["tools"] = []
    yaml_dict["edges"] = []
    for node in path_nodes_list:
        node_info = tool_node_filted[node]
        if 'value' in node_info.keys():
            yaml_dict["tools"].append({"name": node, "inputs": node_info["value"]})
        else:
            yaml_dict["tools"].append({"name": node})

    for path in merge_nodes_list:
        yaml_dict["edges"].append({"start": path_nodes_list.index(path[0]), "end": path_nodes_list.index(path[1])})
    
    if not os.path.exists(output_floder):
        os.makedirs(output_floder)

    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    output_path = os.path.join(output_floder, f"{file_name_without_extension}.yaml")

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml_dict, file, allow_unicode=True, sort_keys=False)

    print(f"yaml file has been saved to {output_path}")
    return output_path


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
            self.edges.append((edge["start"], edge["end"], edge.get("name_mapping", None)))

    @classmethod
    def from_forntend_export(cls, file: str) -> None:
        yml_file = parse_frontend(file)
        return cls(config=Config(config_file=yml_file))

    async def run(self, num_repeats=10, context: TextIO=sys.stdout, tool_outputs: TextIO=sys.stdout) -> Any:
        context.write("Now we have a workflow with the following tools: \n")
        for i, node in enumerate(self.nodes):
            context.write(f"Tool No.{i + 1}: {node.executable.print_usage()}\n\n\n")
        for repeat in range(num_repeats):
            try:
                context.write(f"Repeating workflow execution No.{repeat + 1}:\n")
                edge_list = [[] for i in range(len(self.nodes))]
                in_deg = [0 for i in range(len(self.nodes))]
                for edge in self.edges:
                    edge_list[edge[0]].append((edge[1], edge[2]))
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

                    tool_name = self.nodes[u].executable.print_usage().split(".\n")[0].lower().split()
                    dir_name, file_name = os.path.dirname(outputs[1][0]), os.path.basename(outputs[1][0])
                    file_name = "_".join(tool_name) + "_" + file_name
                    if isinstance(outputs[0][0], Molecule) and hasattr(outputs[0][0], "conformer"):
                        file_name = file_name.replace("pkl", "sdf")
                        file_path = os.path.join(dir_name, file_name)
                        outputs[0][0].save_sdf(file_path)
                        tool_outputs.write(file_path+"\n")
                    elif isinstance(outputs[0][0], Protein) and hasattr(outputs[0][0], "conformer"):
                        file_name = file_name.replace("pkl", "pdb")
                        file_path = os.path.join(dir_name, file_name)
                        outputs[0][0].save_pdb(file_path)
                        tool_outputs.write(file_path+"\n")
                    elif outputs[1][0][-4:] in [".png"]:
                        file_path = os.path.join(dir_name, file_name)
                        os.rename(outputs[1][0], file_path)
                        tool_outputs.write(file_path+"\n")
                    
                    outputs = wrap_and_select_outputs(outputs, context)
                    context.write(f"The outputs of this tool are: {get_str_from_dict(outputs)}\n\n\n")
                    for out_node, out_name_mapping in edge_list[u]:
                        in_deg[out_node] -= 1
                        if in_deg[out_node] == 0:
                            queue.append(out_node)
                        for key, value in outputs.items():
                            if out_name_mapping is not None and key in out_name_mapping:
                                key = out_name_mapping[key]
                            context.write(f'Tool No.{u + 1} passes its "{key}" output to Tool No.{out_node + 1}\n')
                            self.nodes[out_node].inputs[key] = copy.deepcopy(value)
            except:
                context.write("The workflow execution failed")

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=str, default="stable_drug_design")
    parser.add_argument("--num_repeats", type=int, default=1)
    args = parser.parse_args()

    config = Config(config_file=f"./configs/workflow/{args.workflow}.yaml")
    workflow = Workflow(config)
    asyncio.run(workflow.run(num_repeats=args.num_repeats, context=open("./logs/workflow_outputs.txt", "w"), tool_outputs=open("./logs/workflow_tool_outputs.txt", "w")))
    # workflow.run(num_repeats=1)
    """
    file_path = "/AIRvePFS/dair/yk-data/projects/OpenBioMed_new/configs/workflow/Stable_molecule_design.json"
    fronted_file = parse_frontend(file_path)