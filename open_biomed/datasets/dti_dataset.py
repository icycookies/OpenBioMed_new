from abc import ABC, abstractmethod
from typing import Tuple, Union, Any, Dict, Optional, List
import json
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
import pickle
import pdb
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#如上的代码，添加了open_biomed到系统路径中！
print(sys.path)
import copy
#pdb.set_trace()
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

from open_biomed.data import Molecule, Protein, Text
from open_biomed.utils.dti_utils.mol_utils import can_smiles
from open_biomed.datasets.base_dataset import BaseDataset,assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized,ProteinOneHotFeaturizer,MoleculeOneHotFeaturizer
#import open_biomed.utils.featurizer
from open_biomed.utils.split_utils import kfold_split, cold_drug_split, cold_protein_split, cold_cluster_split
class DTIDataset(BaseDataset,ABC):
    #注意，在其中的基类中已经定义了get_subset这个函数，到时候直接调用就好！
    #这个get_subset函数的作用是，返回一个新的dataset，这个dataset中只包含了indexes中的索引对应的数据
    #相当于老板代码中的index_select函数
    #具体子集如何调用，见如下示例：
    #subset_dataset = dataset.get_subset(indexes=[0, 2, 4], attrs=['data', 'labels'])
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        #这部分中，把需要的三个变量初始化！
        self.smiles, self.proteins, self.labels = [], [], []
        #注意，这里就是smiles！就是
        super(BaseDataset, self).__init__()
       
        #这部分中，把path，split_strategy，in_memory这三个参数保存到了datasets中的config.yaml中，之后读入的时候方便操作！
        #我严重怀疑其实没必要这样写！
        self.path = cfg["path"]
        #其中的参数为：./datasets/dti/Yamanishi08
        #path中存储的是：OpenBioMed/open_biomed
        self.split_strategy = cfg["split_strategy"]
        #其实这个in_memory参数，大可不必添加，因为已经在内存中了！
        #self.in_memory = cfg["in_memory"]
        self.featurizer = featurizer
        self._load_data()

    def _load_data(self) -> None:
        # 具体数据加载逻辑
        pass
    
    #如下这部分代码是实现自动的数据特征化处理的函数！而且需要注意，只要利用索引去访问数据集中的内容，就会调用这个函数！
    #所以，在这个函数中，需要把数据集中的内容，全部特征化处理！
    #这里如何选取特征数据提取器，还是需要去config中去指定！
    #这里的访问是这个意思，就是你要是用dataset[0][0]去访问这个数据集中的元素，那就是可以调用这个getitem函数！
    @featurize
    def __getitem__(self, index)-> Dict[str, Featurized[Any]]:
        return {
            #请注意，这里不能用drug作为索引，否则的话没有办法对其进行特征化处理！
            "drug": self.smiles[index],
            "protein": self.proteins[index],
            "label": self.labels[index],
        }
    
    
    def __len__(self) -> int:
        return len(self.pair_index)
            
class DTIClassificationDataset(DTIDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(DTIClassificationDataset, self).__init__(cfg, featurizer)
        self.kfold_split()
    #    @assign_split这里没办法用这个装饰器，因为这个里面只有训练集和测试集！
    def kfold_split(self):
        #请注意，返回的事self.folds，这个是一个字典，包含了train和test的索引
        #这部分代码是kfold，所以在主函数的调用过程中，需要for循环，每次取出一个fold
        #如下这部分是生成索引：
        self.split_indexes = {}
        if self.split_strategy in ["warm", "cold_drug", "cold_protein"]:
            self.nfolds = 5
            if self.split_strategy == "warm":
                folds = kfold_split(len(self), 5)
            elif self.split_strategy == "cold_drug":
                folds = cold_drug_split(self, 5)
            else:
                folds = cold_protein_split(self, 5)
            self.folds = []
            for i in range(5):
                self.folds.append({
                    "train": np.concatenate(folds[:i] + folds[i + 1:], axis=0).tolist(), 
                    "test": folds[i]
                })
        elif self.split_strategy == "cold_cluster":
            self.nfolds = 9
            self.folds = cold_cluster_split(self, 3)

class DTIRegressionDataset(DTIDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(DTIRegressionDataset, self).__init__(cfg, featurizer)
        # self.kfold_split()

class Yamanishi08(DTIClassificationDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(Yamanishi08, self).__init__(cfg, featurizer)
    
    def _load_data(self) :
        data = json.load(open(os.path.join(self.path, "drug.json")))
        self.smiles = [Molecule.from_smiles(data[item]["SMILES"]) for item in data]
        temp_smiles=[data[item]["SMILES"] for item in data]
        drugsmi2index = dict(zip(temp_smiles, range(len(self.smiles))))
        data = json.load(open(os.path.join(self.path, "protein.json")))
        self.proteins = [Protein.from_sequence(data[item]["sequence"]) for item in data]
        temp_proteins = [data[item]["sequence"] for item in data]
        proteinseq2index = dict(zip(temp_proteins, range(len(self.proteins))))

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []
        i = 0 
        for row in df.iterrows():
            row = row[1]
            self.pair_index.append((drugsmi2index[row["compound_iso_smiles"]], proteinseq2index[row["target_sequence"]]))
            self.labels.append(int(row["affinity"]))
        logger.info("Yamanishi08's dataset, total %d samples" % (len(self)))
    
    def split(self,index=0,split_cfg = None):
        #NOTE:
        #这个函数的操作和实现是完全返回一个可以用的数据集了！
        #对于kfold来说，返回的数据集应该是根据nfolds来定的！
        self.split_indexes = {}
        attr = ["pair_index", "labels"]
        self.split_indexes["train"] = self.folds[index]["train"]
        self.split_indexes["test"]=self.folds[index]["test"]
        train_dataset=self.get_subset(self.split_indexes["train"], attr)
        valid_dataset=None
        test_dataset=self.get_subset(self.split_indexes["test"], attr)       
        ret=(train_dataset,valid_dataset,test_dataset)
        return ret

class BMKG_DTI(DTIClassificationDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(BMKG_DTI, self).__init__(cfg, featurizer)

    def _load_data(self) :
        data = json.load(open(os.path.join(self.path, "drug.json")))
        self.smiles = [Molecule.from_smiles(data[item]["SMILES"]) for item in data]
        drugid2index = dict(zip(data.keys(), range(len(self.smiles))))
        data = json.load(open(os.path.join(self.path, "protein.json")))
        self.proteins = [Protein.from_sequence(data[item]["sequence"]) for item in data]
        proteinid2index = dict(zip(data.keys(), range(len(self.proteins))))

        df = pd.read_csv(os.path.join(self.path, "data.csv"))
        self.pair_index, self.labels = [], []
        #这块会有一个drugid缺失的情况：所以写如下的代码：
        for row in df.iterrows():
            row = row[1]
            try:
                drug_index=drugid2index[row["drug_id"]]
            except KeyError:
                print(f"Drug ID {row['drug_id']} not found in drug.json")
                continue
            try:
                protein_index=proteinid2index[str(row["protein_id"])]
            except KeyError:
                print(f"Protein ID {row['protein_id']} not found in protein.json")
                continue
            self.pair_index.append((drug_index, protein_index))
            #self.pair_index.append((drugid2index[row["drug_id"]], proteinid2index[str(row["protein_id"])]))
            self.labels.append(int(row["affinity"]))
    
    @assign_split
    def split(self,index=0,split_cfg = None):
        #NOTE:
        #这个函数的操作和实现是完全返回一个可以用的数据集了！
        #对于kfold来说，返回的数据集应该是根据nfolds来定的！
        attr = ["pair_index", "labels"]
        self.split_indexes = {}
        self.split_indexes["train"] = self.folds[index]["train"]
        self.split_indexes["test"]=self.folds[index]["test"]
        train_dataset=self.get_subset(self.split_indexes["train"], attr)
        #train_dataset.split="train"
        valid_dataset=None
        test_dataset=self.get_subset(self.split_indexes["test"], attr)
       # test_dataset.split="test"
        ret=(train_dataset,valid_dataset,test_dataset)
        del self
        return ret

class KIBA(DTIRegressionDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(KIBA, self).__init__(cfg, featurizer)

    def _load_data(self) :
        Y = pickle.load(open(os.path.join(self.path, "Y"), "rb"), encoding='latin1')
        label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)
        
        can_smis_dict = json.load(open(os.path.join(self.path, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        can_smis = list(can_smis_dict.values())
        self.smiles = [can_smiles(smi) for smi in can_smis]
        self.smiles = [Molecule.from_smiles(smi) for smi in self.smiles]
       
        proteins_dic = json.load(open(os.path.join(self.path, "proteins.txt")), object_pairs_hook=OrderedDict)
        self.proteins = list(proteins_dic.values())
        self.proteins = [Protein.from_sequence(seq) for seq in self.proteins]

        # data:
        self.pair_index = []
        self.labels = []
        train_folds = json.load(open(os.path.join(self.path, "folds/train_fold_setting1.txt")))
        for fold in train_folds:
            for i in fold:
                self.pair_index.append((label_row_inds[i], label_col_inds[i]))
                self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.train_index = list(range(len(self.labels)))
        test_fold = json.load(open(os.path.join(self.path, "folds/test_fold_setting1.txt")))
        for i in test_fold:
            self.pair_index.append((label_row_inds[i], label_col_inds[i]))
            self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.test_index = list(range(len(self.train_index), len(self.labels)))
        # if self.is_davis:
        #     self.labels = [-float(np.log10(y / 1e9)) for y in self.labels]
        #新增添的部分：为了能够和那个新版的模板代码保持一致！
        self.split_indexes={}
        self.split_indexes["train"]=self.train_index
        self.split_indexes["test"]=self.test_index
        #

        logger.info("%s dataset, %d samples" % ("kiba", len(self)))

    @assign_split
    def split(self, split_cfg = None):
        #attr=["drug","protein","label"]
        attr=["pair_index","labels"]
        valid_dataset=None
        ret=(
            self.get_subset(self.split_indexes["train"],attr),
            valid_dataset,
            self.get_subset(self.split_indexes["test"],attr)
        )
        del self
        return ret

class Davis(DTIRegressionDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(Davis, self).__init__(cfg, featurizer)

    def _load_data(self) :
        Y = pickle.load(open(os.path.join(self.path, "Y"), "rb"), encoding='latin1')
        label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)
        
        can_smis_dict = json.load(open(os.path.join(self.path, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        can_smis = list(can_smis_dict.values())
        self.smiles = [can_smiles(smi) for smi in can_smis]
        self.smiles = [Molecule.from_smiles(smi) for smi in self.smiles]
       
        proteins_dic = json.load(open(os.path.join(self.path, "proteins.txt")), object_pairs_hook=OrderedDict)
        self.proteins = list(proteins_dic.values())
        self.proteins = [Protein.from_sequence(seq) for seq in self.proteins]

        # data:
        self.pair_index = []
        self.labels = []
        train_folds = json.load(open(os.path.join(self.path, "folds/train_fold_setting1.txt")))
        for fold in train_folds:
            for i in fold:
                self.pair_index.append((label_row_inds[i], label_col_inds[i]))
                self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.train_index = list(range(len(self.labels)))
        test_fold = json.load(open(os.path.join(self.path, "folds/test_fold_setting1.txt")))
        for i in test_fold:
            self.pair_index.append((label_row_inds[i], label_col_inds[i]))
            self.labels.append(Y[label_row_inds[i], label_col_inds[i]])
        self.test_index = list(range(len(self.train_index), len(self.labels)))
        self.labels = [-float(np.log10(y / 1e9)) for y in self.labels]
        
        #新增添的部分：为了能够和那个新版的模板代码保持一致！
        self.split_indexes={}
        self.split_indexes["train"]=self.train_index
        self.split_indexes["test"]=self.test_index
        #

        logger.info("%s dataset, %d samples" % ("davis", len(self)))
    
    @assign_split
    def split(self, split_cfg = None):
        attr=["pair_index","labels"]
        valid_dataset=None
        ret=(
            self.get_subset(self.split_indexes["train"],attr),
            valid_dataset,
            self.get_subset(self.split_indexes["test"],attr)
        )
        del self
        return ret