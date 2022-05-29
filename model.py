from torch import nn
import torch
import transformers
from config import Config
class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = transformers.DistilBertModel.from_pretrained(Config.MODEL_NAME)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        output = self.bert(ids, mask)[0]
        output = self.drop(output)
        output = self.fc(output[:,0,:])
        output = torch.sigmoid(output)
        return output