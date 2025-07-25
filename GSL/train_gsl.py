import time
import argparse
import numpy as np
import torch
from deeprobust.graph.defense import GCN, ProGNN
from deeprobust.graph.utils import preprocess
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true', default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=200, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='graph_data',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='no', choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0.1, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False, help='whether use symmetric matrix')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"


def batch_process_file_groups(file_groups):
    """Process multiple file groups for GSL training."""

    for group in file_groups:
        # Load adjacency matrix
        adj_df_path = group['adj_matrix']
        adj_df = pd.read_csv(adj_df_path, index_col=0, header=0)
        adj = csr_matrix(adj_df.values)

        # Load features
        features_path = group['features']
        features_df = pd.read_csv(features_path, index_col=0, header=0).T
        print('Features shape:', features_df.shape)
        features = csr_matrix(features_df.values)

        # Split train/val/test
        idx_train, idx_temp = train_test_split(
            np.arange(adj.shape[0]),
            test_size=0.3,
            random_state=args.seed
        )
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=0.5,
            random_state=args.seed
        )

        # Load labels
        labels_path = group['labels']
        labels_df = pd.read_csv(labels_path, index_col=0, header=0).T
        print('Labels shape:', labels_df.shape)
        labels = csr_matrix(labels_df.values)

        # Attack setting
        if args.attack == 'no':
            perturbed_adj = adj
        elif args.attack == 'random':
            from deeprobust.graph.global_attack import Random
            attacker = Random()
            n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
            attacker.attack(adj, n_perturbations, type='add')
            perturbed_adj = attacker.modified_adj
        else:
            perturbed_adj = adj

        # Initialize GCN model
        model = GCN(
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=14,
            dropout=args.dropout,
            device=device
        )

        # Train model
        if args.only_gcn:
            print('Training only GCN...')
            perturbed_adj, features, labels = preprocess(
                perturbed_adj, features, labels,
                preprocess_adj=False,
                sparse=True,
                device=device
            )
            model.fit(features, perturbed_adj, labels, idx_train, idx_val, verbose=True, train_iters=args.epochs)
            model.test(idx_test)
        else:
            print("Training ProGNN...")
            perturbed_adj, features, labels = preprocess(
                perturbed_adj, features, labels,
                preprocess_adj=False,
                device=device
            )
            prognn = ProGNN(model, args, device)
            prognn.fit(features, perturbed_adj, labels, idx_train, idx_val)
            adjfin, acc_test = prognn.test(features, labels, idx_test)
            print(f'Accuracy on test set: {acc_test:.4f}')
            print(f'Perturbed adjacency matrix: {adjfin}')


if __name__ == '__main__':
    file_groups = [
        {
            'adj_matrix': r'data/GSE132509/OX1164/OX1164_matrix_zhikong_200_3_knn_graph.csv_shared_3000_hvg_knn_graph.csv',
            'features': r'data/GSE132509/OX1164/OX1164_matrix_zhikong_200_3.csv',
            'labels': r'data/GSE132509/OX1164/OX1164_matrix_zhikong_200_3_cell_state_auc_matrix.csv',
            'output': r'data/GSE132509/OX1164/OX1164_matrix_zhikong_200_3_knn_graph.csv_shared_3000_hvg_knn_graph_adjfin_graph.csv'
        },
        {
            'adj_matrix': r'data/GSE132509/AYL050/AYL050_matrix_zhikong_200_3_knn_graph.csv_shared_3000_hvg_knn_graph.csv',
            'features': r'data/GSE132509/AYL050/AYL050_matrix_zhikong_200_3.csv',
            'labels': r'data/GSE132509/AYL050/AYL050_matrix_zhikong_200_3_cell_state_auc_matrix.csv',
            'output': r'data/GSE132509/AYL050/AYL050_matrix_zhikong_200_3_knn_graph.csv_shared_3000_hvg_knn_graph_adjfin_graph.csv'
        }
    ]
    batch_process_file_groups(file_groups)
