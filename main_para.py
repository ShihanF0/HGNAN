import argparse
from models import *
import trainer
import datasets
import numpy as np
import torch
import os
import copy
import os.path as osp
import concurrent.futures
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
import time
from datasets import get_data

class EarlyStopping:
    def __init__(self, patience=25, min_is_better=True):
        self.patience = patience
        self.min_is_better = min_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def reset(self):
        self.counter = 0

    def __call__(self, score):
        if self.min_is_better:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + 1e-5:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def run_single_run(seed, run_index, n_layers, early_stop_flag, dropout, model_name, num_epochs, wd, hidden_channels, lr, bias, patience, loss_thresh, data_name, one_m, normalize_m, weight, aggregation, tuning, args, mode):
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()

    if mode == 'train':
        train_loader, val_loader, test_loader, num_features, num_classes = get_data(
            data_name=args.data_name,
            model_name=args.model_name,
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=args.batch_size, seed=seed,
            s_values=args.s_values
        )
    else:
        train_loader, val_loader, test_loader, num_features, num_classes = get_data(
            data_name=args.data_name,
            model_name=args.model_name,
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=args.batch_size, seed=seed,
            s_values=args.s_values
        )
        
    if num_classes == 2:
        loss_type = torch.nn.BCEWithLogitsLoss
        out_dim = 1
    else:
        loss_type = torch.nn.CrossEntropyLoss
        out_dim = num_classes

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"num_gpu: {num_gpus}")
        device_id = run_index % num_gpus
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
        
    if tuning == False:
        log_file = f'logs/{data_name}_{model_name}_training_log_run{run_index}.txt'
    else:
        log_file = f'logs/{data_name}_{model_name}_tuning_log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log = open(log_file, 'w')

    if model_name in ['HGNAN-node', 'HGNAN-edge']:
        model = HGNAN(in_channels=num_features,
                      hidden_channels=hidden_channels,
                      num_layers=n_layers,
                      out_channels=out_dim,
                      dropout=dropout,
                      bias=bias,
                      device=device,
                      limited_m=one_m,
                      normalize_m=normalize_m,
                      weight=weight,
                      aggregation=aggregation,
                      num_s_values=len(args.s_values))
    if tuning == False:
        param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('model size: {:.3f}MB'.format(size_all_mb))
        log.write('model size: {:.3f}MB\n'.format(size_all_mb))
    
    model.to(device)
    
    config = {
        'lr': lr,
        'hidden_channels': hidden_channels,
        'n_conv_layers': n_layers,
        'wd': wd,
        'bias': bias,
        'dropout': dropout,
        'output_dim': out_dim,
        'num_epochs': num_epochs,
        'model': model.__class__.__name__,
        'device': device.type,
        'loss_thresh': loss_thresh,
        'seed': seed,
        'data_name': data_name,
        'early_stop_flag': early_stop_flag,
        'num_features': num_features,
        'limited_m': one_m,
        'normalize_m': normalize_m,
        'run_index': run_index,
        'num_classes': num_classes,
    }

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    if tuning == False:
        for name, val in config.items():
            print(f'{name}: {val}')
            log.write(f'{name}: {val}\n')
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = loss_type()
    early_stop = EarlyStopping(patience=patience, min_is_better=True)

    best_val_loss = 10000000
    best_model_state = None
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.95,
        patience=8,
        min_lr=1e-7,
    )

    for epoch in range(num_epochs):
        if tuning == False:
            print(f"Run {run_index}, Training epoch {epoch}:")
            log.write(f"Run {run_index}, Training epoch {epoch}:\n")
            
        train_loss, train_acc, train_auroc, train_auprc, train_recall, train_precision, train_f1, train_time = \
            trainer.train_epoch(model, dloader=train_loader, optimizer=optimizer, device=device, loss_fn=loss_fn)
        
        
        if tuning == False:
            print("Validating on Validation Set:")
            log.write("Validating on Validation Set:\n")
            
        val_loss, val_acc, val_auroc, val_auprc, val_rec, val_prec, val_f1, val_time = \
            trainer.test_epoch(model, dloader=val_loader, device=device, val_mask=True, loss_fn=loss_fn)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            if tuning == False:
                torch.save(model.state_dict(),
                           f'models/{data_name}_{model_name}_{seed}_best_val_acc.pt')
            if tuning == False:
                    test_loss, test_acc, test_auroc, test_auprc, test_rec, test_prec, test_f1, test_time = \
                        trainer.test_epoch(model, dloader=val_loader, device=device, val_mask=False, loss_fn=loss_fn)
                    best_info = (f"[Best Val Updated] Epoch: {epoch:03d} "
                                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                                 f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")
                    print(best_info)
                    log.write(best_info)
        
        if tuning == False:
            epoch_info = (f"Epoch: {epoch:03d}, "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
            print(epoch_info)
            log.write(epoch_info)
        
        early_stop(val_loss)
        if train_loss < loss_thresh:
            print(f'Loss under {loss_thresh} at epoch: {epoch}')
            log.write(f'Loss under {loss_thresh} at epoch: {epoch}\n')
            break
        if early_stop_flag and early_stop.early_stop:
            print(f'Early stop at epoch: {epoch}')
            log.write(f'Early stop at epoch: {epoch}\n')
            break
    model.load_state_dict(best_model_state)
    if mode == 'train':
        final_val_loss, final_val_acc, final_val_auroc, final_val_auprc, final_val_recall, final_val_prec, final_val_f1, final_val_time = \
            trainer.test_epoch(model, dloader=test_loader, device=device, val_mask=True, loss_fn=loss_fn)
        final_test_loss, final_test_acc, final_test_auroc, final_test_auprc, final_test_recall, final_test_prec, final_test_f1, final_test_time = \
            trainer.test_epoch(model, dloader=test_loader, device=device, val_mask=False, loss_fn=loss_fn)
    elif mode == 'evaluation':
        final_test_loss, final_test_acc, final_test_auroc, final_test_auprc, final_test_recall, final_test_prec, final_test_f1, final_test_time = \
            trainer.test_epoch(model, dloader=test_loader, device=device, val_mask=False, loss_fn=loss_fn)
        final_val_loss, final_val_acc = None, None
        
    final_test_info = (f"Final Test Loss: {final_test_loss:.4f}, Final Test Acc: {final_test_acc:.4f}, "
                       f"Final Test AUROC: {final_test_auroc:.4f}, Final Test AUPRC: {final_test_auprc:.4f}, Fianl Test Recall: {final_test_recall:.4f} Final Test Precision: {final_test_prec:.4f}, Final Test F1: {final_test_f1:.4f}, final_test_time: {final_test_time:.4f}\n")
    print(final_test_info)
    log.write(final_test_info)
    log.close()

    if device.type == 'cuda':
        memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        memory_usage = 0.0

    # Record the memory usage and time taken
    end_time = time.time()
    run_duration = end_time - start_time

    return final_val_loss, final_val_acc, final_test_loss, final_test_acc, final_test_auroc, final_test_auprc,  final_test_prec, final_test_recall, final_test_f1, memory_usage, run_duration

def run_exp_parallel(runs, n_layers, early_stop_flag, dropout, model_name, num_epochs, wd, hidden_channels, lr, bias, patience, loss_thresh, data_name, one_m, normalize_m, weight, aggregation, tuning, args, mode, s_max):
    np.random.seed(0)
    seeds = np.random.randint(low=0, high=10000, size=runs)
    
    final_val_losses, final_val_accs = [], []
    final_test_losses, final_test_accs = [], []
    final_test_aurocs, final_test_auprcs = [], []
    final_test_precs, final_test_recalls, final_test_f1s = [], [], []
    memory_usages, run_durations = [], []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, seed in enumerate(seeds):
            future = executor.submit(run_single_run, int(seed), i, n_layers, early_stop_flag, dropout, model_name,
                                     num_epochs, wd, hidden_channels, lr, bias, patience, loss_thresh,
                                     data_name, one_m, normalize_m, weight, aggregation, tuning, args, mode)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            val_loss, val_acc, test_loss, test_acc, test_auroc, test_auprc, test_prec, test_recall, test_f1, memory_usage, duration = future.result()
            
            final_val_losses.append(val_loss)
            final_val_accs.append(val_acc)
            final_test_losses.append(test_loss)
            final_test_accs.append(test_acc)
            final_test_aurocs.append(test_auroc)
            final_test_auprcs.append(test_auprc)
            final_test_precs.append(test_prec)
            final_test_recalls.append(test_recall)
            final_test_f1s.append(test_f1)
            memory_usages.append(memory_usage)
            run_durations.append(duration)

    final_val_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in final_val_losses]
    final_val_accs = [acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in final_val_accs]
    final_test_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in final_test_losses]
    final_test_accs = [acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in final_test_accs]
    final_test_aurocs = [auc.cpu().item() if isinstance(auc, torch.Tensor) else auc for auc in final_test_aurocs]
    final_test_auprcs = [ap.cpu().item() if isinstance(ap, torch.Tensor) else ap for ap in final_test_auprcs]
    final_test_precs = [p.cpu().item() if isinstance(p, torch.Tensor) else p for p in final_test_precs]
    final_test_recalls = [r.cpu().item() if isinstance(r, torch.Tensor) else r for r in final_test_recalls]
    final_test_f1s = [f1.cpu().item() if isinstance(f1, torch.Tensor) else f1 for f1 in final_test_f1s]
    
    memory_usages = [mem.cpu().item() if isinstance(mem, torch.Tensor) else mem for mem in memory_usages]
    run_durations = [dur.cpu().item() if isinstance(dur, torch.Tensor) else dur for dur in run_durations]

    final_val_loss_mean, final_val_loss_std = np.mean(final_val_losses), np.std(final_val_losses)
    final_val_acc_mean, final_val_acc_std = np.mean(final_val_accs), np.std(final_val_accs)
    
    final_loss_mean, final_loss_std = np.mean(final_test_losses), np.std(final_test_losses)
    final_acc_mean, final_acc_std = np.mean(final_test_accs), np.std(final_test_accs)
    final_auroc_mean, final_auroc_std = np.mean(final_test_aurocs), np.std(final_test_aurocs)
    final_auprc_mean, final_auprc_std = np.mean(final_test_auprcs), np.std(final_test_auprcs)
    final_prec_mean, final_prec_std = np.mean(final_test_precs), np.std(final_test_precs)
    final_recall_mean, final_recall_std = np.mean(final_test_recalls), np.std(final_test_recalls)
    final_f1_mean, final_f1_std = np.mean(final_test_f1s), np.std(final_test_f1s)

    memory_usage = np.mean(memory_usages)
    run_duration = np.mean(run_durations)

    summary_info = (
        f"Final Val Loss:  {final_val_loss_mean:.4f} ± {final_val_loss_std:.4f}, "
        f"Final Val Acc:   {final_val_acc_mean*100:.2f} ± {final_val_acc_std*100:.2f}\n"
        f"Final Test Loss: {final_loss_mean:.4f} ± {final_loss_std:.4f}, "
        f"Final Test Acc:  {final_acc_mean*100:.2f} ± {final_acc_std*100:.2f}\n"
        f"Test AUROC:      {final_auroc_mean*100:.2f} ± {final_auroc_std*100:.2f}, "
        f"Test AUPRC:      {final_auprc_mean*100:.2f} ± {final_auprc_std*100:.2f}\n"
        f"Test Precision:  {final_prec_mean*100:.2f} ± {final_prec_std*100:.2f}, "
        f"Test Recall:     {final_recall_mean*100:.2f} ± {final_recall_std*100:.2f}\n"
        f"Test F1:         {final_f1_mean*100:.2f} ± {final_f1_std*100:.2f}\n"
        f"Avg Memory:      {memory_usage:.2f} MB, "
        f"Avg Duration:    {run_duration:.4f} sec"
    )
    print(summary_info)

    if tuning:
        res_root = 'hyperparameter_tuning'
        if not osp.isdir(res_root):
            os.makedirs(res_root)
        
        csv_filename = f'{res_root}/{data_name}_tuning.csv'
        print(f"Save tuning results to {csv_filename}")
        
        try:
            file_is_empty = not osp.exists(csv_filename) or os.stat(csv_filename).st_size == 0
        except OSError:
            file_is_empty = True

        if file_is_empty:
            header = (
                "s_max,wd,lr,dropout,n_layers,hidden_channels,batch_size,"
                "val_loss_mean,val_loss_std,val_acc_mean,val_acc_std,"
                "test_loss_mean,test_loss_std,test_acc_mean,test_acc_std,"
                "test_auroc_mean,test_auroc_std,test_auprc_mean,test_auprc_std,"
                "test_prec_mean,test_prec_std,test_recall_mean,test_recall_std,"
                "test_f1_mean,test_f1_std,memory_mb,duration_s\n"
            )
        else:
            header = ""

        with open(csv_filename, 'a+', encoding='utf-8') as write_obj:
            if header:
                write_obj.write(header)
            cur_line = (
                f'{s_max},{wd},{lr},{dropout},{n_layers},{hidden_channels},{args.batch_size},'
                f'{final_val_loss_mean:.4f},{final_val_loss_std:.4f},'
                f'{final_val_acc_mean:.4f},{final_val_acc_std:.4f},'
                f'{final_loss_mean:.4f},{final_loss_std:.4f},'
                f'{final_acc_mean:.4f},{final_acc_std:.4f},'
                f'{final_auroc_mean:.4f},{final_auroc_std:.4f},'
                f'{final_auprc_mean:.4f},{final_auprc_std:.4f},'
                f'{final_prec_mean:.4f},{final_prec_std:.4f},'
                f'{final_recall_mean:.4f},{final_recall_std:.4f},'
                f'{final_f1_mean:.4f},{final_f1_std:.4f},'
                f'{memory_usage:.2f},{run_duration:.4f}\n'
            )
            write_obj.write(cur_line)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=3)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=1000)
    parser.add_argument('--bias', dest='bias', type=int, default=1)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0)
    parser.add_argument('--early_stop', dest='early_stop', type=int, default=1)
    parser.add_argument('--wd', dest='wd', type=float, default=0.001)
    parser.add_argument('--data_name', dest='data_name', type=str, default='cora_ca',
                        choices=['Mushroom','zoo','NTU2012','iAF1260b','iJR904','iYO844','iSB619','actor','pokec'])
    parser.add_argument('--model_name', dest='model_name', type=str, default='HGNAN-node', choices=['HGNAN-node','HGNAN-edge'])
    parser.add_argument('--runs', dest='runs', type=int, default=10)
    parser.add_argument('--one_m', dest='one_m', type=int, default=1)
    parser.add_argument('--normalize_m', dest='normalize_m', type=int, default=1)
    parser.add_argument('--patience', dest='patience', type=int, default=200)
    parser.add_argument('--train_size', dest='train_size', type=float, default=0.5)
    parser.add_argument('--val_size', dest='val_size', type=float, default=0.25)
    parser.add_argument('--m_per_feature', dest='m_per_feature', type=bool, default=False)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--weight', dest='weight', action='store_true')
    parser.add_argument('--aggregation', dest='aggregation', type=str, default="overall", choices=['overall','neighbor'])
    parser.add_argument('--tuning', dest='tuning', action='store_true')
    parser.add_argument('--s_max', dest='s_max', type=int, default=1, help='The maximum value of s. The model will aggregate embeddings for s=1, 2, ..., s_max.')
    parser.add_argument('--mode', dest='mode', type=str, default='train', choices=['train','evaluation'])

    args = parser.parse_args()
    
    s_values_list = list(range(1, args.s_max + 1))
    print(f"Running experiment with s values: {s_values_list}")
    args.s_values = s_values_list
    
    loss_thresh = 0.0001

    if args.tuning == False:
        print(args)
    
    train_loader, val_loader, test_loader, num_features, num_classes = datasets.get_data(
        data_name=args.data_name,
        model_name=args.model_name,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        s_values=args.s_values
    )

    print(f"num_classes: {num_classes}")
    run_exp_parallel(runs=args.runs, n_layers=args.n_layers, early_stop_flag=args.early_stop, dropout=args.dropout,
                     model_name=args.model_name, num_epochs=args.num_epochs, wd=args.wd,
                     hidden_channels=args.hidden_channels, lr=args.lr, bias=args.bias, patience=args.patience,
                     loss_thresh=loss_thresh, data_name=args.data_name, one_m=args.one_m, normalize_m=args.normalize_m, 
                     weight=args.weight, aggregation=args.aggregation, tuning=args.tuning, args=args, mode =args.mode, s_max=args.s_max)