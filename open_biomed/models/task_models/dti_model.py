from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import torch
import pytorch_lightning as pl
import os
import sys
import pdb
add_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(add_path)
sys.path.append(add_path)
#import pytorch_lightning as pl
from open_biomed.data import Molecule, Text,Protein
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.featurizer import MoleculeOneHotFeaturizer,ProteinOneHotFeaturizer
from open_biomed.utils.collator import DTICollator,EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer
from open_biomed.utils.misc import sub_dict
from open_biomed.utils.dti_utils.metrics import *
from open_biomed.models.task_models.dti_modelsConfig import SUPPORTED_MOL_ENCODER, SUPPORTED_PROTEIN_ENCODER
from open_biomed.models.task_models.dti_modelsConfig.molecule.cnn import MolCNN
from open_biomed.models.task_models.dti_modelsConfig.protein.cnn import ProtCNN
from open_biomed.models.task_models.dti_modelsConfig.predictor import MLP
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error


class DTIModel(pl.LightningModule):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__()
        #如下的这行我不确定是否正确！
        self.save_hyperparameters(model_cfg.todict())  # 保存所有配置参数
        #加载编码工具：
        self.drug_encoder = MolCNN(model_cfg["mol_encoder_config"])
        """
        "vocab_size": 63,
        "max_length": 100,
        "in_ch": [63, 32, 64, 96],
        "kernels": [4, 8, 12],
        "output_dim": 256
        """
        self.protein_encoder = ProtCNN(model_cfg["protein_encoder_config"])
        """
            "vocab_size": 26,
            "max_length": 1024,
            "in_ch": [26, 32, 64, 96],
            "kernels": [4, 8, 12],
            "output_dim": 256
        """
        self.featurizers = {
            "drug": MoleculeOneHotFeaturizer(
                {
                    "max_len":357,
                }
            ),
            "protein": ProteinOneHotFeaturizer({
                "max_length":1024
            }
            )
        }
        self.collator=DTICollator({
            "mol": {
            "modality": ["structure"],
            "featurizer": {
                "structure": {
                    "name": "OneHot",
                    "max_len": 357
                }
            }
            },
            "protein": {
                "modality": ["structure"],
                "featurizer": {
                    "structure": {
                        "name": "OneHot",
                        "max_length": 1024
                    }
                }
            }
        })
        
        self.task=model_cfg.task
        if self.task == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
            self.pred_dim=2
        elif self.task == "regression":
            self.loss_fn = nn.MSELoss()
            self.pred_dim=1
        self.pred_head=MLP(model_cfg["pred_head"], self.drug_encoder.output_dim + self.protein_encoder.output_dim, self.pred_dim,self.task)
        """
        "pred_head": {
            "hidden_size": [512, 256],
            "activation": "relu",
            "batch_norm": false,
            "dropout": 0.1
        }
        """
        """
        
        根据任务的不同,pred_dim也不同！
        if args.dataset in ['yamanishi08', 'bmkg-dti']:
            args.task = "classification"
            pred_dim = 2
        else:
            args.task = "regression"
            pred_dim = 1
        """
        # 损失函数和评估指标
        #旧版代码中是输入了label和predict！
        #但是新版代码中没有体现啊！
       
    def _add_task(self) -> None:
        self.supported_tasks["dti"] = {
        "forward_fn": self.forward_dti,
        "predict_fn": self.predict_dti,
        "featurizer": EnsembleFeaturizer(
            sub_dict(self.featurizers, ["drug","protein"]),
        ),
        "collator": EnsembleCollator({
            **sub_dict(self.collators, ["mol","protein"]),
            "label": self.collators["label"]
        })
        }

    def forward(self,drug,protein) :
        h_drug = self.drug_encoder.encode_mol(drug)
        h_protein = self.protein_encoder.encode_protein(protein)
        h = torch.cat((h_drug, h_protein), dim=1)
        pred=self.pred_head(h)
        return pred



    def forward_dti(self,
    drug: List[Molecule],
    protein: List[Protein],
    label: List[float]) -> Dict[str, torch.Tensor]:
        h_drug = self.drug_encoder.encode_mol(drug)
        h_protein = self.protein_encoder.encode_protein(protein)
        h = torch.cat((h_drug, h_protein), dim=1)
        pred=self.pred_head(h)
        loss=self.loss_fn(pred,label)
        return {"loss":loss,"prediction":pred}
    

    def predict_dti(self,
    drug: List[Molecule],
    protein: List[Protein]) -> List[float]:
        with torch.no_grad():
           output = self.forward_dti(drug, protein)
           predictions=output["prediction"]
           return predictions

    def training_step(self, batch, batch_idx):
        drug, protein, label = batch
        pred = self(drug, protein)
        loss = self.loss_fn(pred, label)
        self.log("train_loss", loss)
        # if self.hparams.config["task"] == "classification":
        #     acc = self.accuracy(pred, label)
        #     self.log("train_acc", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        drug, protein, label = batch
        pred = self(drug, protein)
        loss = self.loss_fn(pred, label)
        self.log("val_loss", loss)
        if self.task == "classification":
            roc_auc_result=roc_auc(label, pred)
            pr_auc_result=pr_auc(label, pred)
            f1_result=f1_score(label, pred)
            Precision_result=precision_score(label, pred)
            Recall_result=recall_score(label, pred)
            self.log("val_roc_auc", roc_auc_result)
            self.log("val_pr_auc", pr_auc_result)
            self.log("val_f1", f1_result)
            self.log("val_Precision", Precision_result)
            self.log("val_Recall", Recall_result)
        elif self.task == "regression":
            mse_result = mean_squared_error(label, pred)
            pearson_result = pearsonr(label, pred)[0]
            spearman_result = spearmanr(label, pred)[0]
            ci_result = concordance_index(label, pred)
            rm2_result =rm2_index(label, pred)
            self.log("val_mse", mse_result)
            self.log("val_pearson", pearson_result)
            self.log("val_spearman", spearman_result)
            self.log("val_ci", ci_result)
            self.log("val_rm2", rm2_result)
        return loss
           
    def test_step(self, batch, batch_idx):
        drug, protein, label = batch
        pred = self(drug, protein)
        loss = self.loss_fn(pred, label)
        self.log("test_loss", loss)
        if self.task == "classification":
            roc_auc_result=roc_auc(label, pred)
            pr_auc_result=pr_auc(label, pred)
            f1_result=f1_score(label, pred)
            Precision_result=precision_score(label, pred)
            Recall_result=recall_score(label, pred)
            self.log("val_roc_auc", roc_auc_result)
            self.log("val_pr_auc", pr_auc_result)
            self.log("val_f1", f1_result)
            self.log("val_Precision", Precision_result)
            self.log("val_Recall", Recall_result)
        elif self.task == "regression":
            mse_result = mean_squared_error(label, pred)
            pearson_result = pearsonr(label, pred)[0]
            spearman_result = spearmanr(label, pred)[0]
            ci_result = concordance_index(label, pred)
            rm2_result =rm2_index(label, pred)
            self.log("val_mse", mse_result)
            self.log("val_pearson", pearson_result)
            self.log("val_spearman", spearman_result)
            self.log("val_ci", ci_result)
            self.log("val_rm2", rm2_result)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config["lr"])
        return optimizer

    def predict_step(self,batch,batch_idx:int,dataloader_index:int=0):
        drug, protein, _ = batch
        pred = self(drug, protein)
        return{
            "logits": self(drug, protein),
            "probs": torch.softmax(self(drug, protein), dim=-1)
        }

if __name__ == "__main__":
    model_cfg = Config.from_dict(**{
       "task": "classification",
       "mol_encoder_config":
       {
           "vocab_size": 63,
            "max_length": 100,
            "in_ch": [63, 32, 64, 96],
            "kernels": [4, 8, 12],
            "output_dim": 256
       },
       "protein_encoder_config":
        {
            "vocab_size": 26,
            "max_length": 1024,
            "in_ch": [26, 32, 64, 96],
            "kernels": [4, 8, 12],
            "output_dim": 256
       },
       "pred_head": {
            "hidden_size": [512, 256],
            "activation": "relu",
            "batch_norm": False,
            "dropout": 0.1
       },
       
       "pred_dim": 2,
       "lr": 0.001,
       "weight_decay": 0.0001
    })
    pdb.set_trace()
    model=DTIModel(model_cfg)
    drug=[Molecule.from_smiles("MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQAMELGLEAIFDHIVRRNRGGWCLQVNQLLYWALTTIGFQTTMLGGYFYIPPVNKYSTGMVHLLLQVTIDGRNYIVDAGSGSSSQMWQPLELISGKDQPQVPCIFCLTEERGIWYLDQIRREQYITNKEFLNSHLLPKKKHQKIYLFTLEPRTIEDFESMNTYLQTSPTSSFITTSFCSLQTPEGVYCLVGFILTYRKFNYKDNTDLVEFKTLTEEEVEEVLRNIFKISLGRNLVPKPGDGSLTI")]
    protein=[Protein.from_sequence("MDIEAYFERIGYKNSRNKLDLETLTDILEHQIRAVPFENLNMHCGQAMELGLEAIFDHIVRRNRGGWCLQVNQLLYWALTTIGFQTTMLGGYFYIPPVNKYSTGMVHLLLQVTIDGRNYIVDAGSGSSSQMWQPLELISGKDQPQVPCIFCLTEERGIWYLDQIRREQYITNKEFLNSHLLPKKKHQKIYLFTLEPRTIEDFESMNTYLQTSPTSSFITTSFCSLQTPEGVYCLVGFILTYRKFNYKDNTDLVEFKTLTEEEVEEVLRNIFKISLGRNLVPKPGDGSLTI")]
    label=[0]
    MoleculeOneHotFeaturizer_config={
            "name": "OneHot",
            "max_len": 357}
    ProteinOneHotFeaturizer_config={
            "name": "OneHot",
            "max_length": 1024}
        # 定义特征器
    mol_featurizer = MoleculeOneHotFeaturizer(MoleculeOneHotFeaturizer_config)
    prot_featurizer = ProteinOneHotFeaturizer(ProteinOneHotFeaturizer_config)
    #这里需要注意了，这个特征提取器，只能够特征提取每一个分子的信息！而整个的列表它不会提取！
    drug=[mol_featurizer(element) for element in drug]
    protein=[prot_featurizer(element) for element in protein]
    label=[0]
    #你看一下定义中的内容，几乎全部只是传入了一个Molecule或者是一个Protein的列表！

    #这里实现的步骤是把二维的向量沿着第0维度堆叠成为三维的向量！，需要注意，在后面dim=0，才可以实现把一个列表转化为一个三维度的向量，否则还是shape不对！
    drug=torch.stack([x.squeeze() for x in drug],dim=0)
    protein=torch.stack([x.squeeze() for x in protein],dim=0)
    label=torch.tensor(label)
    pdb.set_trace()
    #这里，我明白了，需要
    output=model.forward_dti(drug,protein,label)
    print("test Successfully!!")

