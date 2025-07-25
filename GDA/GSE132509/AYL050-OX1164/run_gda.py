import argparse
import os.path as osp
import numpy as np

from pygda.datasets import CitationDataset
from pygda.models import KBL
from pygda.metrics import eval_micro_f1, eval_macro_f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------
# Argument Parser
# ------------------------------
parser = argparse.ArgumentParser()

# General training parameters
parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--num_layers', type=int, default=2, help='number of GNN layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden layer size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use: e.g., cuda:0 or cpu')
parser.add_argument('--source', type=str, default='AYL050', help='source domain dataset')
parser.add_argument('--target', type=str, default='OX1164', help='target domain dataset')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of training epochs')
parser.add_argument('--filename', type=str, default='results.txt', help='file to store results')

# Model-specific parameters
parser.add_argument('--disc', type=str, default='JS', help='discriminator type')
parser.add_argument('--weight', type=float, default=0.01, help='loss trade-off parameter')

args = parser.parse_args()

# ------------------------------
# Load Source and Target Datasets
# ------------------------------
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'Citation', args.source)
source_dataset = CitationDataset(path, args.source)

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'Citation', args.target)
target_dataset = CitationDataset(path, args.target)

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))

# ------------------------------
# Initialize the KBL Model
# ------------------------------
model = KBL(
    in_dim=num_features,
    hid_dim=args.nhid,
    num_classes=num_classes,
    num_layers=args.num_layers,
    weight_decay=args.weight_decay,
    lr=args.lr,
    dropout=args.dropout_ratio,
    epoch=args.epochs,
    device=args.device,
    k_cross=args.k_cross,
    k_within=args.k_within
)

# ------------------------------
# Train & Evaluate
# ------------------------------
print("Training the KBL model...")
model.fit(source_data, target_data)

print("Evaluating...")
logits, labels = model.predict(target_data)
preds = logits.argmax(dim=1)

# Compute metrics
mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='macro')
recall = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')
conf_matrix = confusion_matrix(labels, preds)

# ------------------------------
# Save Results
# ------------------------------
with open(args.filename, 'a') as f:
    f.write("Selected Model: KBL\n")
    f.write("Model Parameters:\n")
    f.write(f"Input Dimension: {num_features}\n")
    f.write(f"Hidden Dimension: {args.nhid}\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write(f"Device: {args.device}\n")
    f.write(f"Epochs: {model.epoch}\n")

    f.write("\nDataset Information:\n")
    f.write(f"Source Domain Data: {args.source}\n")
    f.write(f"Target Domain Data: {args.target}\n\n")

    f.write("Evaluation Results:\n")
    f.write(f"Accuracy: {accuracy:.7f}\n")
    f.write(f"Precision: {precision:.7f}\n")
    f.write(f"Recall: {recall:.7f}\n")
    f.write(f"F1 Score: {f1:.7f}\n")
    f.write(f"Micro-F1: {mi_f1:.7f}\n")
    f.write(f"Macro-F1: {ma_f1:.7f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Labels:\n{labels}\n")
    f.write(f"Predictions:\n{preds}\n")

print("Results have been saved to", args.filename)
