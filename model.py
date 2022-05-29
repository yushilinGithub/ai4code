from torch import nn
import torch
import torch.nn.functional as F
import transformers
from config import Config
import torch.distributed as dist
class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.labels = torch.arange(Config.TRAIN_BS,device=Config.DEVICE)
        self.bert = transformers.DistilBertModel.from_pretrained(Config.MODEL_NAME)
        self.codebert = transformers.AutoModel.from_pretrained(Config.CODE_MODEL_NAME)
    def forward(self, ids, mask, ids_code):
        output = self.bert(ids, mask)[0]
        output_code = self.codebert(ids_code)[0]
        return output[:,0,:],output_code[:,0,:]

    def loss(self, outputs_text,output_codes):

        local_batch_size = outputs_text.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = torch.arange(
                local_batch_size, device=outputs_text.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        outputs_text = F.normalize(outputs_text, dim=-1, p=2)
        output_codes = F.normalize(output_codes, dim=-1, p=2)


        # cosine similarity as logits
        logits_per_text =  outputs_text @ output_codes.t()
        logits_per_code =   output_codes @ outputs_text.t()

        loss = (F.cross_entropy(logits_per_text, self.labels) + \
            F.cross_entropy(logits_per_code, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'acc': acc}

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()