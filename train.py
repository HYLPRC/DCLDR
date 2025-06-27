import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from dataload import *
from model.DCLDR import DCLDR
from utils import *
import torch
torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=3, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    parser.add_argument('--num_hidden', type=int, default=200)
    parser.add_argument('--num_proj_hidden1', type=int, default=100)
    parser.add_argument('--num_proj_hidden2', type=int, default=150)
    parser.add_argument('--device', default='cuda:0', 
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--intra', type=float, default=0.2)
    parser.add_argument('--inter', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.1)

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/DCLDR/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(args.device)
    didi_graph = didi_graph.to(args.device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(args.device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(args.device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(args.device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()


    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs, ACCs, precs, recalls, f1s, mccs = [], [], [], [], [], [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = DCLDR(args)
        model = model.to(args.device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(args.device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(args.device)
        X_test = torch.LongTensor(data['X_test'][i]).to(args.device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(args.device)

        for epoch in range(args.epochs):
            model.train()
            dr_hgt, di_hgt, dr_sim, di_sim, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            
            rel_loss = nn.BCEWithLogitsLoss()
            loss_drug = LOSS(args, dr_hgt, dr_sim, batch_size=0, flag=0)
            loss_dis = LOSS(args, di_hgt, di_sim, batch_size=0, flag=1)
            loss =args.beta * ( loss_drug +  loss_dis )/2 + rel_loss(train_score[:, 1], Y_train.squeeze().float())
            # loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_hgt, di_hgt, dr_sim, di_sim, test_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        ACCs.append(best_accuracy)
        precs.append(best_precision)
        recalls.append(best_recall)
        f1s.append(best_f1)
        mccs.append(best_mcc)


    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print('ACCs:', ACCs)
    acc_mean = np.mean(ACCs)
    acc_std = np.std(ACCs)
    print('Mean AUPR:', acc_mean, '(', acc_std, ')')

    print('precs:', precs)
    precs_mean = np.mean(precs)
    precs_std = np.std(precs)
    print('Mean AUPR:', precs_mean, '(', precs_std, ')')
    
    print('recalls:', recalls)
    recalls_mean = np.mean(recalls)
    recalls_std = np.std(recalls)
    print('Mean AUPR:', recalls_mean, '(', recalls_std, ')')

    print('AUPR:', f1s)
    f1s_mean = np.mean(f1s)
    f1s_std = np.std(f1s)
    print('Mean AUPR:', f1s_mean, '(', f1s_std, ')')

    print('AUPR:', mccs)
    mccs_mean = np.mean(mccs)
    mccs_std = np.std(mccs)
    print('Mean AUPR:', mccs_mean, '(', mccs_std, ')')
    
