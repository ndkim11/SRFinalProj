#-*- coding: utf-8 -*-
import os
import json
import pdb
import argparse
import time
import torch
import torch.nn as nn
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from pyctcdecode import build_ctcdecoder
import soundfile
import numpy as np
import editdistance
import pickle
from tqdm import tqdm
from datetime import datetime


## ===================================================================
## Load labels
## ===================================================================

def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char
            
        return char2index, index2char

## ===================================================================
## Data loader
## trainset  = SpeechDataset(args.train_list, args.train_path, args.max_length, char2index)
## ===================================================================

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, max_length, char2index):
        super(SpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list,'r') as f:
            data = json.load(f) 

        # convert seconds to frames
        max_length *= 16000

        # sort data in length order and filter data less than max_length
        data = sorted(data, key=lambda d: d['len'], reverse=True)
        self.data = [x for x in data if x['len'] <= max_length]

        self.dataset_path   = data_path
        self.char2index     = char2index

    def __getitem__(self, index):

        # read audio using soundfile.read
        
        audio = self.data[index]['file']
        # print(audio)
        audio, _ = soundfile.read(os.path.join(self.dataset_path, audio))
        # read transcript and convert to indices
        transcript = self.data[index]['text']
        transcript = self.parse_transcript(transcript)

        return torch.FloatTensor(audio), torch.LongTensor(transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return len(self.data)


## ===================================================================
## Define collate function
## ===================================================================

def pad_collate(batch):
    (xx, yy) = zip(*batch)

    ## compute lengths of each item in xx and yy
    x_lens = [len(item) for item in xx]
    y_lens = [len(item) for item in yy]

    ## zero-pad to the longest length
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0.0)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0.0)

    return xx_pad, yy_pad, x_lens, y_lens

## ===================================================================
## Define sampler 
## ===================================================================

class BucketingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):

        # Shuffle bins in random order
        np.random.shuffle(self.bins)

        # For each bin
        for ids in self.bins:
            # Shuffle indices in random order
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

## ===================================================================
## Baseline speech recognition model
## ===================================================================

class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_classes=11, conf_layers=16, mfcc = 40, lstm_layers=2,dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()

        #Conformer input : (batch_size, sequence_length, channels)
        self.conformer = torchaudio.models.Conformer(
            # input_dim = 80,
            input_dim= mfcc,
            num_heads = 4,
            ffn_dim = 128,
            num_layers = conf_layers,
            depthwise_conv_kernel_size = 31,
            dropout = dropout
        )

        ## define RNN layers as self.lstm - use a 1-layer bidirectional LSTM with 256 output size and 0.1 dropout
        
        # self.lstm = nn.LSTM(40,256,dropout=0.1,bidirectional=True,num_layers=1,batch_first=True)
        self.lstm = nn.LSTM(mfcc,256,bidirectional=True,num_layers=lstm_layers,batch_first=True)

        ## define the fully connected layer
        self.classifier = nn.Linear(512,n_classes)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc= mfcc) # out : (batch_size, mfcc components, sequence_length)
        self.instancenorm = nn.InstanceNorm1d(mfcc)

    def forward(self, x):

        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
          x = self.preprocess(x)+1e-6
          x = self.instancenorm(x).detach()

        x = x.transpose(1,2) # (batch, sequence, mfcc)
        lengths = torch.empty(len(x)).cuda()
        for i, length in enumerate(x):
            lengths[i] = len(length)
        
        # print(lengths)
        x, _  = self.conformer(input = x, lengths = lengths)


        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM() in:(batch,time,n_class)
        # LSTM decoder for Conformer
        x = self.lstm(x)[0]
        ## pass the network through the classifier
        # x = (batch, time, feature)
        x = self.classifier(x)

        return x
    

## ===================================================================
## Train an epoch on GPU
## ===================================================================

def process_epoch(model,loader,criterion,optimizer,trainmode=True):
    # Set the model to training or eval mode
    if trainmode:
        model.train()

    else:
        model.eval()

    ep_loss = 0
    ep_cnt  = 0

    with tqdm(loader, unit="batch") as tepoch:

        for data in tepoch:

            ## Load x and y
            x = data[0].cuda()
            y = data[1].cuda()
            y_len = torch.LongTensor(data[3])

            # print("shape: ",y.shape)
            # Add some noise to x            
            x = x + torch.normal(mean=0, std=torch.std(x)*1e-3, size=x.shape).cuda()

            # Forward pass
            output = model(x)
            # print(output.shape)
            # Take the log softmax - the output must be in (time, batch, n_class) order
            output = nn.functional.log_softmax(output,dim=2)
            output = output.transpose(0,1)
            
            ## compute the loss using the CTC objective
            x_len = torch.LongTensor([output.size(0)]).repeat(output.size(1))
            loss = criterion(output, y, x_len, y_len)

            if trainmode:
              loss.backward()
              optimizer.step()
              optimizer.zero_grad()

            # keep running average of loss
            ep_loss += loss.item() * len(x)
            ep_cnt  += len(x)

            # print value to TQDM
            tepoch.set_postfix(loss=ep_loss/ep_cnt)

    return ep_loss/ep_cnt


## ===================================================================
## Greedy CTC Decoder
## ===================================================================

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """
        Given a sequence emission over labels, get the best path.
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript        
        """
        
        
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = indices.numpy()

        indices = [i for i in indices if i != self.blank]
        return indices


## ===================================================================
## Evaluation script
## ===================================================================

def process_eval(model,data_path,data_list,index2char,save_path=None):

    # set model to evaluation mode
    model.eval()

    # initialise the greedy decoder
    # greedy_decoder = GreedyCTCDecoder(blank=len(index2char))

    with open('./data/label.json', encoding="utf-8") as label_file:
        labels = json.load(label_file)
        labels.append("|")
        labels[1]="^"

    decoder = build_ctcdecoder(
    labels,
    kenlm_model_path= "srkenlm.arpa",  # either .arpa or .bin file
    alpha=0.5,  # tuned on a val set
    beta=1.0,  # tuned on a val set
    )

    # load data from JSON
    with open(data_list,'r') as f:
        data = json.load(f)

    results = []

    for file in tqdm(data):

        # read the wav file and convert to PyTorch format
        audio, _ = soundfile.read(os.path.join(data_path, file['file']))
        
        x = torch.FloatTensor(audio).cuda()
        x = x.unsqueeze(dim=0)
        # print('x :', x.shape)

        # forward pass through the model
        model.eval()
        with torch.no_grad():
            output = model(x)
            output = nn.functional.log_softmax(output, dim=2)
            output = output.transpose(0,1)

        # decode using the greedy decoder Out : [batch,time,labels]
        # pred = greedy_decoder(output.cpu().detach().squeeze())        # convert to text
        # text = ''.join([index2char[x] for x in pred])

        text = decoder.decode(logits=output.cpu().detach().squeeze().numpy())
        text = text.replace(' ', '')
        text = text.replace('^',' ')

        file['pred'] = text
        if 'text' in file:
            file['edit_dist']   = editdistance.eval(text.replace(' ',''),file['text'].replace(' ',''))
            file['gt_len']      = len(file['text'].replace(' ',''))
        results.append(file)
    
    # save results to json file
    with open(os.path.join(save_path,'results.json'), 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    # print CER if there is ground truth
    if 'text' in file:
        cer = sum([x['edit_dist'] for x in results]) / sum([x['gt_len'] for x in results])
        print('Character Error Rate is {:.2f}%'.format(cer*100))


## ===================================================================
## Main execution script
## ===================================================================

def main():

    parser = argparse.ArgumentParser(description='EE738 Exercise')

    ## related to data loading
    parser.add_argument('--max_length', type=int, default=10,   help='maximum length of audio file in seconds')
    parser.add_argument('--train_list', type=str, default='data/ks_train.json')
    parser.add_argument('--val_list',   type=str, default='data/ks_val.json')
    parser.add_argument('--labels_path',type=str, default='data/label.json')
    parser.add_argument('--train_path', type=str, default='data/kspon_train')
    parser.add_argument('--val_path',   type=str, default='data/kspon_eval')
    parser.add_argument('--date', default=datetime.now().strftime('%d_%H_%M'))


    ## related to training
    parser.add_argument('--max_epoch',  type=int, default=30,       help='number of epochs during training')
    parser.add_argument('--batch_size', type=int, default=20,      help='batch size')
    parser.add_argument('--lr',         type=float, default=1e-4,     help='learning rate')
    parser.add_argument('--seed',       type=int, default=2222,     help='random seed initialisation')
    parser.add_argument('--weight_decay',type=float, default=1e-6, help='weight decay for Adam')
    parser.add_argument('--conf_layers', type=int, default=16)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--mfcc',       type=int, default=40)
    parser.add_argument('--dropout',    type=float,default=0.1)
    
    ## relating to loading and saving
    parser.add_argument('--initial_model',  type=str, default='',   help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path',      type=str, default='./checkpoints',   help='location to save checkpoints')
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='Use tensorboard to display train log')
    parser.add_argument('--modelname', type=str, default='model1')

    ## related to inference
    parser.add_argument('--eval',   dest='eval',    action='store_true', help='Evaluation mode')
    parser.add_argument('--gpu',    type=int,       default=0,      help='GPU index')

    args = parser.parse_args()

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    model = SpeechRecognitionModel(n_classes=len(char2index)+1, conf_layers = args.conf_layers, mfcc=args.mfcc, lstm_layers=args.lstm_layers, dropout=args.dropout).cuda()
    print('Model loaded. Number of parameters: {:.3f} Million'.format(sum(p.numel() for p in model.parameters())/1000000))

    ## load from initial model 
    if args.initial_model != '':
        model.load_state_dict(torch.load(args.initial_model))

    # make directory for saving models and output
    assert args.save_path != ''
    
    if args.eval:
        export_name = "Conformer_Eval_{}_date{}".format(args.initial_model, datetime.now().strftime('%d_%H_%M'))
        args.save_path = os.path.join(args.save_path, export_name)
    
    if not args.eval: #only in train mode we make a tensorboard
        export_name = "Conformer_Train_epoch{}_batch{}_conf{}_lstm{}_mfcc{}_dropout{}".format(args.max_epoch, args.batch_size, args.conf_layers, args.lstm_layers, args.mfcc, args.dropout)#datetime.now().strftime('%d_%H_%M')
        args.save_path = os.path.join(args.save_path, export_name)
        writer = None
        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(args.save_path)
            print("Using tensorboard! Results will be stored in " + args.save_path)
            
    os.makedirs(args.save_path,exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval(model, args.val_path, args.val_list, index2char, save_path=args.save_path)
        quit()

    # initialise seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define datasets
    trainset  = SpeechDataset(args.train_list, args.train_path, args.max_length, char2index)
    valset    = SpeechDataset(args.val_list,   args.val_path,   args.max_length, char2index)

    # initiate loader for each dataset with 'collate_fn' argument
    # do not use more than 6 workers
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_sampler=BucketingSampler(trainset, args.batch_size), 
        num_workers=4, 
        collate_fn=pad_collate,
        prefetch_factor=4)
    valloader   = torch.utils.data.DataLoader(valset,   
        batch_sampler=BucketingSampler(valset, args.batch_size), 
        num_workers=4, 
        collate_fn=pad_collate,
        prefetch_factor=4)

    ## define the optimizer with args.lr learning rate and appropriate weight decay
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ## set loss function with blank index
    
    ctcloss = nn.CTCLoss(blank=len(char2index)).cuda()

    ## initialise training log file
    f_log = open(os.path.join(args.save_path,'train.log'),'a+')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    bestVloss = 100
    ## Train for args.max_epoch epochs
    for epoch in range(0, args.max_epoch):
        print(f"Processing epoch {epoch}")
        
        tloss = process_epoch(model, trainloader, ctcloss, optimizer, trainmode=True)
        vloss = process_epoch(model, valloader, ctcloss, optimizer, trainmode=False)

        if vloss < bestVloss:
            # save checkpoint to file
            save_file = '{}/best_model.pt'.format(args.save_path)
            print('Saving model {}'.format(save_file))
            torch.save(model.state_dict(), save_file)

        # write training progress to log
        f_log.write('Epoch {:03d}, train loss {:.3f}, val loss {:.3f}\n'.format(epoch, tloss, vloss))
        f_log.flush()

        if args.use_tensorboard: 
            writer.add_scalar("Training Loss", tloss, epoch+1)
            writer.add_scalar("Validation Loss",vloss, epoch+1)

    if args.use_tensorboard:
        writer.flush()
        writer.close()

    f_log.close()

if __name__ == "__main__":
    main()
