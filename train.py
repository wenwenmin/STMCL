import argparse
import torch
import os
import torch.nn.functional as F
from dataset import SKIN, HERDataset, DATA_BRAIN, TenxDataset
from model import STMCL
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=90, help='')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--temperature', type=float, default=1., help='temperature')
parser.add_argument('--dim', type=int, default=50, help='spot_embedding dimension (# HVGs)')  # 171, 785, 58, 50
parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
parser.add_argument('--dataset', type=str, default='Alex', help='dataset')  # Mouse_spleen


def load_data(args):
    if args.dataset == 'her2st':
        print(f'load dataset: {args.dataset}')
        train_dataset = HERDataset(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = HERDataset(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == 'cscc':
        print(f'load dataset: {args.dataset}')
        train_dataset = SKIN(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = SKIN(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == 'Brain':
        print(f'load dataset: {args.dataset}')
        train_dataset = DATA_BRAIN(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = DATA_BRAIN(train=False, fold=args.fold)
        return train_dataLoader, test_dataset

    elif args.dataset == 'Mouse_spleen':
        print(f'load dataset: {args.dataset}')
        train_dataset = Mouse_Spleen(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = Mouse_Spleen(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == 'Alex':
        examples = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
        datasets = [
            TenxDataset(image_path=f"D:\dataset\Alex_NatGen/{example}/image.tif",
                        spatial_pos_path=f"D:\dataset\Alex_NatGen/{example}/spatial/tissue_positions_list.csv",
                        reduced_mtx_path=f"./data/preprocessed_expression_matrices/Alex/{example}/preprocessed_matrix.npy",
                        barcode_path=f"D:\dataset\Alex_NatGen/{example}/filtered_count_matrix/barcodes.tsv.gz")
            for example in examples
        ]

        datasets.pop(args.fold)
        print("Test name: ", examples[args.fold], "Test fold: ", args.fold)

        train_dataset = torch.utils.data.ConcatDataset(datasets)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        return train_loader, examples


def train(model, train_dataLoader, optimizer, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def save_model(args, model, test_dataset=None, examples=0):
    os.makedirs(f"./model_result_STMCL/{args.dataset}/{examples}", exist_ok=True)
    torch.save(model.state_dict(),
               f"./model_result_STMCL/{args.dataset}/{examples}/best_{args.fold}.pt")


def main():
    args = parser.parse_args()
    for i in range(6):
        args.fold = i
        print("当前fold:", args.fold)
        train_dataLoader, examples = load_data(args)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = STMCL(spot_embedding=args.dim, temperature=args.temperature,
                     image_embedding=args.image_embedding_dim, projection_dim=args.projection_dim).cuda()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        for epoch in range(args.max_epochs):
            model.train()
            train(model, train_dataLoader, optimizer, epoch)

        save_model(args, model, test_dataset=examples[i])
        print("Saved Model")


main()
