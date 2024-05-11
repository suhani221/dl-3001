import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import json
from collections import OrderedDict
import numpy as np
import sys
import torch.nn.functional as F

from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from dependecy.dataloader_classification import load_dataset_classification
import logging
from dependecy.args import get_args
import os  # Added missing import for os module
from dependecy.dataloader_classification import utils

# Model Definition

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*48*7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, seq_lengths):
        batch, max_seq_len, num_ch, in_dim = x.shape
        x = x.reshape(-1, num_ch, in_dim).unsqueeze(1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)
        out = self.dropout(out)

        out = out.reshape(batch*max_seq_len, -1)
        out = self.fc1(out)
        out = F.relu(out)

        # Instead of LSTM, directly use the output from the fully connected layer
        logits = self.fc2(out)

        # Reshape logits to match the expected output shape
        logits = logits.reshape(batch, max_seq_len, -1)
        # Select the last relevant output based on seq_lengths
        logits = utils.last_relevant_pytorch(logits, seq_lengths, batch_first=True)

        return logits

def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    return last_output


def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """

    # Define loss function
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch !=30) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for x, y, seq_lengths, supports, _, _ in train_loader:
                batch_size = x.shape[0]

                # input seqs
                x = x.to(device)
                y = y.view(-1).to(device)  # (batch_size,)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, num_classes)
               
                logits = model(x, seq_lengths)
              
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)  # (batch_size,)                
                loss = loss_fn(logits, y)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
               
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    # To evaluate mode
    model.eval()

    # Define loss function
   
    loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, seq_lengths, supports, _, file_name in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.view(-1).to(device)  # (batch_size,)
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            # (batch_size, num_classes)
          
            logits = model(x, seq_lengths)
            

            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)

            # Log info
            progress_bar.update(batch_size)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

 
    best_thresh = best_thresh

    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if args.task == "detection" else "weighted")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh)]
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results

# Main Execution Setup
def main(args):
    dataloaders, _, scaler =load_dataset_classification(
        input_dir=args.input_dir,
        raw_data_dir=args.raw_data_dir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        time_step_size=args.time_step_size,
        max_seq_len=args.max_seq_len,
        standardize=True,
        num_workers=args.num_workers,
        padding_val=0.,
        augmentation=args.data_augment,
        adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
        graph_type=args.graph_type,
        top_k=args.top_k,
        filter_type=args.filter_type,
        use_fft=args.use_fft,
        preproc_dir=args.preproc_dir)
    
    
    
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))


    model = CNN(args.num_classes)
    
    if args.do_train:

  
        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

# Train model
        model = model.to(device)
        train(model, dataloaders, args, device, args.save_dir, log, tbx)



        # Load best model
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

# Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    dev_results = evaluate(model,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                 for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))

if __name__ == '__main__':
    main(get_args())


