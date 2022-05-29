
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

        txt = self.dict_cellid_source[row[0]]
        code = self.dict_cellid_source[row[1]]
        inputs = Config.TOKENIZER.encode_plus(
                    txt,
                    None,
                    add_special_tokens=True,
                    max_length=Config.MAX_LEN,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True
                )
        code_tokens_ids = Config.CODE_TOKENIZER.encode_plus(
                    code,
                    None,
                    add_special_tokens=True,
                    max_length=Config.MAX_LEN,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True
        )
        # code_tokens = Config.CODE_TOKENIZER.tokenize(code,)
       
        #code_tokens_ids = Config.CODE_TOKENIZER.convert_tokens_to_ids(code_tokens)


        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        id_codes = torch.LongTensor(code_tokens_ids['input_ids'])

        return ids, mask, id_codes




    def __len__(self):
        return len(self.df)