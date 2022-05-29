
from torch.utils.data import Dataset, DataLoader
from config import Config
import torch


class MarkdownDataset(Dataset):
    
    def __init__(self, df,dict_cellid_source, mode='train'):
        super().__init__()
        self.df = df
        self.dict_cellid_source = dict_cellid_source
        self.mode=mode

    def __getitem__(self, index):
        row = self.df[index]

        label = row[-1]

        txt = self.dict_cellid_source[row[0]] + '[SEP]' +self.dict_cellid_source[row[1]]
        inputs = Config.TOKENIZER.encode_plus(
                    txt,
                    None,
                    add_special_tokens=True,
                    max_length=Config.MAX_LEN,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True
                )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([label])




    def __len__(self):
        return len(self.df)