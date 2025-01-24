import torch.nn as nn
#import pytorch_lightning as pl

activation = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
}
#利用其继承于pl.LightningModule类，实现了一个MLP模型
class MLP(nn.Module):
# class MLP(pl.LightningModule):
    def __init__(self, config, input_dim, output_dim,task):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        hidden_dims = [input_dim] + config["hidden_size"] + [output_dim]
        for i in range(len(hidden_dims) - 1):
            self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                self.model.append(nn.Dropout(config["dropout"]))
                if config["activation"] != "none":
                    self.model.append(activation[config["activation"]])
                if config["batch_norm"]:
                    self.model.append(nn.BatchNorm1d())

    def forward(self, h):
        return self.model(h).squeeze()
    
