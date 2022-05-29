from pathlib import Path
import transformers
from torch.cuda.amp import GradScaler
class Config:
    NB_EPOCHS = 2
    LR = 3e-4
    T_0 = 20
    η_min = 1e-4
    MAX_LEN = 120
    TRAIN_BS = 16
    VALID_BS = 16
    #MODEL_NAME = 'bert-large-uncased'
    MODEL_NAME = "distilbert-base-uncased"
    CODE_MODEL_NAME = "microsoft/codebert-base"
    data_dir = Path('D:\\workspace\\kaggle\\data\\AI4Code')
    #TOKENIZER = transformers.BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    TOKENIZER = transformers.DistilBertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    CODE_TOKENIZER = transformers.AutoTokenizer.from_pretrained("microsoft/codebert-base")
    scaler = GradScaler()
    DEVICE="cpu"
