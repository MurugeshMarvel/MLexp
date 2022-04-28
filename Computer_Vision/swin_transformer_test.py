import torch as T
import os
from torch import nn
import sys
import logging
sys.path.append('../')
from src.models import SwinTransformer
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 50
NUM_EPOCH = 100
CUDA_FLAG = False
VALIDATE_GAP = 5

checkpoint_save_dir = 'save_dir'
# model_kwargs = dict(
#     img_size=24, in_chans=1,
#     patch_size=2, window_size=2, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))

model_kwargs = dict(
    img_size=28, in_chans=1,num_classes=10,
    patch_size=2, window_size=2, embed_dim=96, depths=(2,), num_heads=(3,))

model = SwinTransformer(**model_kwargs)

# inp = T.rand((1,1,28, 28))

# out = model.forward(inp)

# logging.info(out.shape)

trainset = datasets.MNIST(root='./', train=True, download=True, transform=ToTensor())

testset = datasets.MNIST(root='./', train=False, download=True, transform=ToTensor())


train_loader = DataLoader(trainset, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

test_loader = DataLoader(testset, shuffle=False, batch_size=TEST_BATCH_SIZE)

loss_fn = T.nn.CrossEntropyLoss()
optimizer_fn = T.optim.Adadelta(params=model.parameters(), lr=0.01, rho=0.95, eps=1e-08)

if T.cuda.is_available():
    model.cuda()
    CUDA_FLAG = True
log_text = ''
min_valid_loss = None 
for e in range(NUM_EPOCH):
    train_loss = 0.
    logging.info(f"Training on epoch - {e}")
    for img, out in tqdm(train_loader):
        img = img.cuda() if CUDA_FLAG else img
        out = out.cuda() if CUDA_FLAG else out
        logging.debug(img.shape)
        pred_out = model(img)
        logging.debug(pred_out.shape)
        #pred_out = T.argmax(pred_out, dim=1)
        #preds = pred_out.view(pred_out.size(0))
        logging.debug(f" Pred oout - {pred_out.shape}")
        logging.debug(f"Actual out - {out.shape}")
        loss = loss_fn(pred_out, out)
        loss.backward()
        optimizer_fn.step()
        train_loss += loss.item()
    total_train_loss = train_loss / len(train_loader)
    training_log = f'Epoch {e+1} \t\t Training Loss: {total_train_loss}'
    log_text += training_log + "\n" 
    logging.info(training_log)
    
    if e % VALIDATE_GAP  == 0:
        valid_loss = 0.0
        total_valid_outs = []
        total_valid_preds = []
        for val_data, val_out in tqdm(test_loader):
            total_valid_outs += val_out.tolist()
            if T.cuda.is_available():
                val_data, val_out = val_data.cuda(), val_out.cuda()
            val_pred = model(val_data)
            # val_pred = val_pred.view(val_pred.size(0))
            total_valid_preds += val_pred.tolist()
            loss = loss_fn(val_pred, val_out)
            valid_loss += loss.item()
        total_valid_loss = valid_loss / len(test_loader)
        min_valid_loss = total_valid_loss if min_valid_loss == None else min_valid_loss

        validation_log = f'Epoch {e+1} \t\t Validation Loss: {total_valid_loss}'
        log_text += "######\n" + validation_log + "\n" + "######"
        logging.info(validation_log)
        with open(os.path.join(checkpoint_save_dir, 'training_log.txt'), 'w') as f:
            f.write(log_text)
            
        # if min_valid_loss > total_valid_loss:
        #     validation_log_best = f'Validation Loss Decreased({min_valid_loss:.6f}--->{total_train_loss:.6f}) \t Saving The Model'
        #     log_text += "######\n" + validation_log_best + "\n" + "######"
        #     print(validation_log_best)
        #     min_valid_loss = total_train_loss
        #     T.save({
        #                 'epoch': e,
        #                 'model_state_dict': self.model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'loss': min_valid_loss}, os.path.join(self.checkpoint_save_dir, f'model_at_best.ckpt'))
        #     with open(os.path.join(self.checkpoint_save_dir, 'training_log.txt'), 'w') as f:
        #         f.write(log_text)




