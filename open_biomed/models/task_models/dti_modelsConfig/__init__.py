# from models.molecule import *
# from models.protein import *
# from models.cell import *
# from models.knowledge import *
# from models.text import *
# from models.multimodal import *
from open_biomed.models.task_models.dti_modelsConfig.molecule.cnn import *
#from open_biomed.models.task_models.dti_modelsConfig.molecule.mgnn import *
from open_biomed.models.task_models.dti_modelsConfig.protein.cnn import *
#from open_biomed.models.task_models.dti_modelsConfig.protein.mcnn import *
#dti_model 相关配置：
SUPPORTED_MOL_ENCODER = {
    "cnn": MolCNN,
    # "tgsa": GINTGSA,
    # "graphcl": GraphCL,
    # "graphmvp": GraphMVP,
    # "molclr": MolCLR,
   # "mgnn": MGNN,
    # "molt5": MolT5,
    # "bert": MolBERT,
    # "biomedgpt-1.6b": BioMedGPTCLIP,
    # "biomedgpt-10b": BioMedGPTV,
    # "kv-plm": KVPLM,
    # "momu": MoMu,
    # "molfm": MolFM
}
SUPPORTED_PROTEIN_ENCODER = {
    "cnn": ProtCNN,
    "cnn_gru": CNNGRU,
    #"mcnn": MCNN,
    "pipr": CNNPIPR,
    # "prottrans": ProtTrans
}
