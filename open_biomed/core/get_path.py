
import json
import re
import os
import copy
import json
import yaml


def get_create_data(node):
    node_dict = {}
    id = node["id"]
    node_dict["value"] = []
    description = node["data"]["node"]["description"]
    data_context = node["data"]["node"]["template"]
    keys = list(data_context)
    pattern = re.compile(r'^field_\d+_key$')
    filtered_keys = [key for key in keys if pattern.match(key)]
    for key in filtered_keys:
        node_dict["value"].append(data_context[key]["value"])
    return node_dict

# 只需要拿到id和description
def get_tool_data(node):
    node_dict = {}
    id = node["id"]
    description = node["data"]["node"]["description"]
    node_dict["description"] = [description]
    return node_dict


def remove_duplicate_nodes(nodes):
    # 创建一个集合来存储已经处理过的节点
    seen = set()
    # 创建一个新列表来存储去重后的节点
    new_nodes = []

    for head, tail in nodes:
        # 检查当前节点是否已经处理过
        if (head, tail) not in seen:
            # 如果没有处理过，添加到新列表中
            new_nodes.append([head, tail])
            # 将当前节点标记为已处理
            seen.add((head, tail))
        else:
            print(f"重复节点已移除: {head} -> {tail}")

    return new_nodes

def check_strings(str_list, nested_list):
    for sublist in nested_list:
        for str in str_list:
            if str in sublist:
                return False  # 如果任意一个字符串在子列表中，返回 False
    return True  # 如果两个字符串都不在任何子列表中，返回 True

def merge_nodes(nodes):
    keywords = ["MergeDataComponent", "ParseData"]
    node_list_copy = copy.deepcopy(nodes)  # 创建副本
    new_list = []
    for i, (head, tail) in enumerate(node_list_copy):
        if any(keyword in head for keyword in keywords):
            for j, (next_head, next_tail) in enumerate(node_list_copy):
                if i != j and head == next_tail:
                    # 合并节点
                    new_list.append([next_head, tail])
        if any(keyword in tail for keyword in keywords):
            for j, (next_head, next_tail) in enumerate(node_list_copy):
                if i != j and tail == next_head:
                    # 合并节点
                    new_list.append([head, next_tail])
    return nodes + new_list



def get_path(file_path):
    
    with open(file_path, 'r') as file:
        node_edge_data = json.load(file)
    tool_node = {}
    data_node = {}
    for node in node_edge_data["data"]["nodes"]:
        if "ChatInput" in node["id"] or "ChatOutput" in node["id"] or "ParseData" in node["id"]:
            continue
        if "PharmolixCreateData" in node["id"]:
            data = get_create_data(node)
            data_node[node["id"]] = data
        else:
            data = get_tool_data(node)
            tool_node[node["id"]] = data

    edge_list = []
    for node in node_edge_data["data"]["edges"]:
        source = node["source"]
        target = node["target"]
        if "PharmolixCreateData" in source:
            tool_node[target]["value"] = data_node[source]["value"]
            edge_list.append([source, target])
        else:
            edge_list.append([source, target])

    nodes = copy.deepcopy(edge_list)
    merge_nodes_list = merge_nodes(nodes)
    
    keywords = ["MergeDataComponent", "ParseData", "ChatOutput", "ChatInput", "Image Output"]
    merge_nodes_list_copy = copy.deepcopy(merge_nodes_list)
    for index in range(len(merge_nodes_list_copy) - 1, -1, -1):
        print(index)
        value = merge_nodes_list_copy[index]
        if any(keyword in value[0] for keyword in keywords) or any(keyword in value[1] for keyword in keywords):
            merge_nodes_list.pop(index)
            print("len: ", len(merge_nodes_list), value)


    ## 获得fileter_tool_node
    # 遍历关键词列表，检查并删除字典中符合条件的键
    tool_node_filted = {}
    key_list = []
    for key, value in tool_node.items():
        if not any(keyword in key for keyword in keywords):
            tool_node_filted[key] = value
            print(key)
            key_list.append(key)

    keywords_path = ['MergeDataComponent', 'ParseData', 'ChatOutput', 'ChatInput', 'Image Output', "PharmolixCreateData"]
    merge_nodes_list_filter = []
    for i in merge_nodes_list:
        if not any([keyword in i[0] or keyword in i[1] for keyword in keywords_path]):
            merge_nodes_list_filter.append(i)

    node_list = []
    for i in merge_nodes_list_filter:
        node_list.append(i[0])
        node_list.append(i[1])
    node_list = set(node_list)

    yaml_dict = {}
    yaml_dict["tools"] = []
    yaml_dict["edges"] = []
    node_list = list(node_list)
    for node in node_list:
        node_info = tool_node_filted[node]
        if 'value' in node_info.keys():
            yaml_dict["tools"].append({"name": node, "inputs": node_info["value"]})
        else:
            yaml_dict["tools"].append({"name": node})

    for path in merge_nodes_list_filter:
        yaml_dict["edges"].append({"start": node_list.index(path[0]), "end": node_list.index(path[1])})


    with open("output.yaml", "w", encoding="utf-8") as file:
        yaml.dump(yaml_dict, file, allow_unicode=True, sort_keys=False)

    print("YAML 文件已保存为 output.yaml")


if __name__ == "__main__":
    file_path = "/AIRvePFS/dair/yk-data/projects/OpenBioMed_new/configs/workflow/Stable_molecule_design.json"
    get_path(file_path)
    