from model.multimodel import MultiResNetAttentionModel
from tools.train import train
from utils.loader import get_loader


train_labels = 'train.csv'
valid_labels = 'valid.csv'

(train_loader, weights), valid_loader = get_loader(train_labels), get_loader(valid_labels, train=False)

model = MultiResNetAttentionModel(unfreeze_layers=('layer4', ))
train(model, (train_loader, valid_loader), binary_weights=weights, log_name='20241021')

