import pdb
import os
import sys
add_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(add_path)
sys.path.append(add_path)
import logging
import unittest
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MoleculeOneHotFeaturizer, ProteinOneHotFeaturizer,EnsembleFeaturizer,DummyFeaturizer
from open_biomed.datasets.dti_dataset import Yamanishi08, BMKG_DTI, Davis,KIBA

from open_biomed.data import Molecule, Protein, Text
# 配置日志
logging.basicConfig(level=logging.INFO)

class TestDTIDataset(unittest.TestCase):

    def setUp(self):
        # 配置数据集路径
        # self.cfg = Config(config_file=None, config_dict={
        #     "path": "datasets/dti/Yamanishi08",
        #     "split_strategy": "warm",
        #     # "in_memory": True
        # })
        MoleculeOneHotFeaturizer_config={
            "name": "OneHot",
            "max_len": 357}
        ProteinOneHotFeaturizer_config={
            "name": "OneHot",
            "max_length": 1024}
        # 定义特征器
        self.mol_featurizer = MoleculeOneHotFeaturizer(MoleculeOneHotFeaturizer_config)
        self.prot_featurizer = ProteinOneHotFeaturizer(ProteinOneHotFeaturizer_config)
        self.featurizer = {
            "drug": self.mol_featurizer,
            "protein": self.prot_featurizer
        }

    def test_Yamanishi08(self):
        # 配置Yamanishi08数据集
        yamanishi08_dict ={
            "path": "datasets/dti/Yamanishi08",
            "split_strategy": "warm"}
        yamanishi08_cfg=Config.from_dict(**yamanishi08_dict)
        dataset = Yamanishi08(yamanishi08_cfg, self.featurizer)
        dataset._load_data()
        self.assertGreater(len(dataset.smiles), 0)
        self.assertGreater(len(dataset.proteins), 0)
        self.assertGreater(len(dataset.labels), 0)
        #检测出来如上的代码中，这个smile和protein之间数量不匹配，不过这个本身就是多对多的关系
        self.assertEqual(len(dataset.smiles), len(dataset.proteins))
        self.assertEqual(len(dataset.smiles), len(dataset.labels))
        # 分割数据集
        splits = dataset.split()
        self.assertEqual(len(splits), 5)
        for train_set, test_set in splits:
            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(test_set), 0)
            self.assertEqual(len(set(train_set.split_indexes["train"]) & set(test_set.split_indexes["test"])), 0)

    def test_BMKG_DTI(self):
        # 配置BMKG数据集
        bmkg_dict={
            "path": "datasets/dti/BMKG_DTI",
            "split_strategy": "warm",
        }
        # bmkg_cfg = Config(config_file=None, config_dict={
        #     "path": "datasets/dti/BMKG_DTI",
        #     "split_strategy": "warm",
        # })
        bmkg_cfg=Config.from_dict(**bmkg_dict)
        dataset = BMKG_DTI(bmkg_cfg, self.featurizer)
        dataset._load_data()
        self.assertGreater(len(dataset.smiles), 0)
        self.assertGreater(len(dataset.proteins), 0)
        self.assertGreater(len(dataset.labels), 0)
        self.assertEqual(len(dataset.smiles), len(dataset.proteins))
        self.assertEqual(len(dataset.smiles), len(dataset.labels))
        # 分割数据集
        splits = dataset.split()
        self.assertEqual(len(splits), 5)
        for train_set, test_set in splits:
            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(test_set), 0)
            self.assertEqual(len(set(train_set.split_indexes["train"]) & set(test_set.split_indexes["test"])), 0)

    def test_Davis_KIBA(self):
        # 配置Davis数据集
        davis_dict ={ 
            "path": "datasets/dti/davis",
            "split_strategy": "warm"} 
        davis_cfg=Config.from_dict(**davis_dict)
        dataset = Davis_KIBA(davis_cfg, self.featurizer)
        dataset._load_data()
        self.assertGreater(len(dataset.smiles), 0)
        self.assertGreater(len(dataset.proteins), 0)
        self.assertGreater(len(dataset.labels), 0)
        self.assertEqual(len(dataset.smiles), len(dataset.proteins))
        self.assertEqual(len(dataset.smiles), len(dataset.labels))
        # 分割数据集
        splits = dataset.split()
        self.assertEqual(len(splits), 2)
        train_set, test_set = splits
        self.assertGreater(len(train_set), 0)
        self.assertGreater(len(test_set), 0)
        self.assertEqual(len(set(train_set.split_indexes["train"]) & set(test_set.split_indexes["test"])), 0)

if __name__ == '__main__':
    yamanishi08_dict ={
            "path": "datasets/dti/Yamanishi08",
            "split_strategy": "warm"}
    
    bmkg_dict={
            "path": "datasets/dti/BMKG_DTI",
            "split_strategy": "warm",
        }
    davis_dict ={
            "path": "datasets/dti/davis",
            "split_strategy": "warm"
    }
    kiba_dict ={
            "path": "datasets/dti/kiba",
            "split_strategy": "warm"
    }
    yamanishi08_cfg=Config.from_dict(**yamanishi08_dict)
    print(yamanishi08_cfg.path)
    pdb.set_trace()
    bmkg_cfg=Config.from_dict(**bmkg_dict)
    davis_cfg=Config.from_dict(**davis_dict)
    kiba_cfg=Config.from_dict(**kiba_dict)
    MoleculeOneHotFeaturizer_config={
            "name": "OneHot",
            "max_len": 357}
    ProteinOneHotFeaturizer_config={
            "name": "OneHot",
            "max_length": 1024}
        # 定义特征器
    mol_featurizer = MoleculeOneHotFeaturizer(MoleculeOneHotFeaturizer_config)
    prot_featurizer = ProteinOneHotFeaturizer(ProteinOneHotFeaturizer_config)
    label_featurizer = DummyFeaturizer()
    davis_featurizer = {
        "drug": mol_featurizer,
        "protein": prot_featurizer,
        "label": label_featurizer
    }
    #注意吗，这里传入的一定要加两个新号！
 #   ensemble_featurizer = EnsembleFeaturizer(**davis_featurizer)
  #deepseek给出来的如下的修改方式：
    ensemble_featurizer = EnsembleFeaturizer(to_ensemble=davis_featurizer)
    pdb.set_trace()
    yamanishi_dataset=Yamanishi08(yamanishi08_cfg, ensemble_featurizer)
    bmkg_dataset = BMKG_DTI(bmkg_cfg, ensemble_featurizer)
    davis_dataset = Davis(davis_cfg, ensemble_featurizer)
    kiba_dataset = KIBA(kiba_cfg, ensemble_featurizer)
    #result=bmkg_dataset.split(index=0,split_cfg=None)
    yamanishi08_result=yamanishi_dataset.split(split_cfg=None)
    bmkg_result=bmkg_dataset.split(split_cfg=None)
    davis_result=davis_dataset.split(split_cfg=None)
    kiba_result=kiba_dataset.split(split_cfg=None)
    pdb.set_trace()
    print("Split Successfully!")


