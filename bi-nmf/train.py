import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from bias_model import BiasedModel, CustomDataset
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

def data_processing():
    #input_item_amount.pkl
    data_item = pd.read_pickle("../data/input_item_amount.pkl")
    data_cate = pd.read_pickle("../data/input_amount.pkl")
    item_cat = pd.read_pickle("../data/item_cate_map.pkl").values
    data_item = data_item.abs()
    data_cate = data_cate.abs()
    train_item = np.apply_along_axis(lambda x : x/x.sum(), 1, data_item.values) 
    train_cate = np.apply_along_axis(lambda x : x/x.sum(), 1, data_cate.values) 
    train_item[np.isnan(train_item)] = 0
    train_cate[np.isnan(train_cate)] = 0
    return train_item, train_cate, item_cat, data_item.columns, data_cate.columns


def square_loss_func(y_pred, y_true):
        # zero_loss = 0
        # if torch.sum(torch.sum(v, dim = 1)) > 0:
        #     zero_loss = 1000000
        # return torch.sum(torch.square(y_pred - y_true)) + 1 * torch.sum(torch.abs(v)) + zero_loss
    return (torch.sum(torch.square(y_pred - y_true)))


def loss_constaint(item_cat, v_i, v_c, alpha):
    wb = torch.matmul(v_i, torch.transpose(item_cat, 0, 1).to(dtype=torch.float32))
    return alpha * (torch.sum(torch.square(wb - v_c)))

def plot_training(item_loss_list, cate_loss_list, const_loss_list):
    plt.figure()
    plt.plot(item_loss_list, label = "item loss")
    plt.plot(cate_loss_list, label = "category loss")
    plt.plot(const_loss_list, label = "constraint loss")
    plt.legend()
    plt.savefig(f"./{EXP}/figures/training_loss.png")
    np.save(f"./{EXP}/results/item_loss_list.npy", item_loss_list)
    np.save(f"./{EXP}/results/cate_loss_list.npy", cate_loss_list)
    np.save(f"./{EXP}/results/const_loss_list.npy", const_loss_list)
    # np.save(f"./{EXP}/results/u_full.npy", u_full.data.numpy())

def save_result(u_indivd,u_common, u_full, v_c, v_i, item_names, cate_names):
    np.save(f"./{EXP}/results/u_indivd.npy", u_indivd.data.numpy())
    np.save(f"./{EXP}/results/u_full.npy", u_full.data.numpy())
    pd.DataFrame(v_c.data.numpy(), columns=cate_names).to_pickle(f"./{EXP}/results/v_category.pkl")
    pd.DataFrame(v_i.data.numpy(), columns=item_names).to_pickle(f"./{EXP}/results/v_item.pkl")
    
def evaluate(model_cpu, input_cate, input_item, item_names, cate_names):
    u_indivd,u_common, u_full, v_c, v_i, y_pred_cate, y_pred_item = model_cpu(torch.Tensor(input_cate),torch.Tensor(input_item))
    save_result(u_indivd,u_common, u_full, v_c, v_i, item_names, cate_names)

def main(args):
    train_item, train_cate, item_cat, item_names, cate_names = data_processing()

    batch_size = args.batch_size
    n_iters = args.n_iters
    train = CustomDataset(torch.Tensor(train_cate),torch.Tensor(train_item))
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_cat = torch.tensor(item_cat).to(device)
    
    k = args.n_k
    n_cates = train_cate.shape[1]
    n_items = train_item.shape[1]
    n_users = train_cate.shape[0]
    # T = train_data.shape[0]
    factorize_model = BiasedModel(n_cates, n_items, K = k, bias=True)
    factorize_model.to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(factorize_model.parameters(), lr=learning_rate)
    loss_list = []
    item_loss_list = []
    cate_loss_list = []
    const_loss_list = []
    for iter in range(n_iters):
        # Forward pass: Compute predicted y by passing x to the model
        batch_loss = 0
        batch_loss_item = 0
        batch_loss_cate = 0
        batch_loss_const = 0
        for idx, (x_c, x_i) in enumerate(train_loader):
            x_c = x_c.to(device)
            x_i = x_i.to(device)
            # x, common_uk, u, v_c_, v_i_, torch.matmul(x, v_c_), torch.matmul(x, v_i_)
            u_indivd,u_common, u_full, v_c, v_i, y_pred_cate, y_pred_item = factorize_model(x_c, x_i)
                # l1_reg_fc = factorize_model.fc.weight.norm(1)
            loss_item = square_loss_func(y_pred_cate, x_c)
            loss_cate = square_loss_func(y_pred_item, x_i)
            loss_const = loss_constaint(item_cat, v_i, v_c, args.alpha)
            # if not torch.equal(torch.matmul(v_i, torch.transpose(torch.tensor(item_cat), 0, 1).to(dtype=torch.float32)), v_c):
                # loss = loss + 10000 
            loss = loss_item + loss_cate + loss_const
            # loss += 0.001 * torch.sum(torch.abs(v))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.detach().cpu().numpy()
            batch_loss_item += loss_item.detach().cpu().numpy()
            batch_loss_cate += loss_cate.detach().cpu().numpy()
            batch_loss_const += loss_const.detach().cpu().numpy()
        loss_list.append(batch_loss)
        item_loss_list.append(batch_loss_item)
        cate_loss_list.append(batch_loss_cate)
        const_loss_list.append(batch_loss_const)
        print(iter, f'total:{batch_loss}')
        print(iter, f'item:{batch_loss_item}')
        print(iter, f'cate:{batch_loss_cate}')
        print(iter, f'const:{batch_loss_const}')
    
    torch.save(factorize_model.state_dict(), f"{EXP}/checkpoints/model_K{k}_batch{batch_size}.pt")
    plot_training(item_loss_list, cate_loss_list, const_loss_list)
    evaluate(factorize_model.to("cpu"), train_cate, train_item, item_names, cate_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments 
    # parser.add_argument('-matrix', '--path_user_item_monthly_matrix', type=str, default='input_matrix_2020.csv')
    # parser.add_argument('-matrix', '--path_user_item_monthly_matrix', type=str, default='input_matrix_2020.csv')
    # parser.add_argument('-matrix', '--path_user_item_monthly_matrix', type=str, default='input_matrix_2020.csv')

    # training arguments
    parser.add_argument('-alpha', '--alpha', type = float, default = 1)
    parser.add_argument('-k','--n_k', type=int, default=25)
    parser.add_argument('-bs','--batch_size', type=int, default=8)
    parser.add_argument('-iters','--n_iters', type=int, default=10000)
    parser.add_argument('-lr','--lr', type=float, default=0.01)
 
    args = parser.parse_args()

    # makedir
    EXP=f"K-{args.n_k}.BS-{args.batch_size}-{args.n_iters}"
    os.makedirs(f'./{EXP}/figures', exist_ok=True)
    os.makedirs(f'./{EXP}/results', exist_ok=True)
    os.makedirs(f'./{EXP}/checkpoints', exist_ok=True)
    main(args)
