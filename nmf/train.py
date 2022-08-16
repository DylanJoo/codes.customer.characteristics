import argparse
import numpy as np
import pandas as pd
import collections
import torch
from torch.utils.data import DataLoader
from models import AutoEncoder
from datasets import InvoiceDataset


def load_matrix(path):
    df = pd.read_csv(path)
    df.rename(columns={'Unnamed: 0': 'month', 'Unnamed: 1': 'user_id'}, inplace=True)
    return df

def mapping(id_list):
    id_mapping = collections.OrderedDict()
    for x_id in id_list:
        id_mapping.setdefault(x_id, len(id_mapping))
    return id_mapping


def main(args):
    df = load_matrix(args.path_user_item_monthly_matrix)

    # preprocessing columns


    # convert to numbers
    user_mapping = mapping(df.user_id.unique())
    item_rating_mapping = mapping(df.drop(columns=['user_id', 'month']).columns.unique())

    df.rename(columns=item_rating_mapping, inplace=True)
    df.replace({'user_id': user_mapping}, inplace=True)

    # model
    model = AutoEncoder(
            dim=len(item_rating_mapping),
            latent_dim=args.latent_dim,  # define latent embeddings and user's embeddings
            entity_size=len(user_mapping), 
            hidden_dims=(512, 256, 128), 
            dropout_rate=0.1
    )

    # data
    dataset = InvoiceDataset(
            record_arr=df.drop(columns=['user_id', 'month']).values,
            user_id_arr=df.user_id.values,
            month_arr=df.month.values
    )

    # dataloader
    dataloader = DataLoader(
            dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True
    )

    # Training setup
    # torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    total_loss = 0.0
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Training 
    model.to(device)
    model.train()
    for epoch in range(args.train_epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            for k in batch:
                batch[k] = batch[k].to(device)
            output = model(**batch)
            loss = output['loss']

            loss.backward()
            optimizer.step()

            cur_loss = loss.detach().cpu().numpy()
            total_loss += loss

            if i % 100 == 0:
                print(f'Training: {i} batches ...')

        print(f'| epoch {epoch:3d} | '
              f'{i:5d}/{len(dataset)//args.train_batch_size} batches | '
              f'training loss {cur_loss}')
    

    # saving
    torch.save(model.state_dict(), args.model_path.replace('detail', f'EPS{args.train_epochs}_H{args.latent_dim}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments 
    parser.add_argument('-matrix', '--path_user_item_monthly_matrix', type=str, default='input_matrix_2020.csv')
    # model arguments
    parser.add_argument('-latent', '--latent_dim', type=int, default=100)
    parser.add_argument('-max_hidden', '--max_hidden_dim', type=int, default=512)
    parser.add_argument('-model_path', '--model_path', type=str, default='./models/ae.detail.ckpt')
    # training arguments
    parser.add_argument('-bs','--train_batch_size', type=int, default=8)
    parser.add_argument('-epochs','--train_epochs', type=int, default=10)
    parser.add_argument('-lr','--learning_rate', type=int, default=0.01)
    # evaluation arguments
    parser.add_argument('--dummy', action='store_true', default=False)


    args = parser.parse_args()

    main(args)
