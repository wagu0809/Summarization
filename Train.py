import argparse
import os
import random
import glob
import torch
import time
import logging

import DataLoader
from DataLoader import load_dataset
from Modules import Summarizer, build_optim
from models.trainer_ext import build_trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def train_ext(args, device_id):

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    print('Device ID %d' % device_id)
    print('Device %s' % device)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        print('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return DataLoader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                     shuffle=True, is_test=False)

    model = Summarizer(args, device, checkpoint)
    optim = build_optim(args, model, checkpoint)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)


def validate_ext(args, device_id):

    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    xent_lst = []
    for i, cp in enumerate(cp_files):
        print(cp)
        step = int(cp.split('.')[-2].split('_')[-1])
        xent = validate(args, device_id, cp, step)
        xent_lst.append((xent, cp))

    xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
    print('PPL %s' % str(xent_lst))
    for xent, cp in xent_lst:
        step = int(cp.split('.')[-2].split('_')[-1])
        test_ext(args, device_id, cp, step)
   


def validate(args, device_id, pt, step):
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from

    print('Loading checkpoint from %s' % test_from)
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    model = Summarizer(args, device, checkpoint)
    valid_iter = DataLoader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                       args.batch_size, device,
                                       shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)

    return stats.xent()


def test_ext(args, device_id, pt, step):
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    print('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    model = Summarizer(args, device, checkpoint)
    test_iter = DataLoader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                      args.test_batch_size, device,
                                      shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='./data/cnndm')
    parser.add_argument("-model_path", default='./model/')
    parser.add_argument("-result_path", default='./results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=5000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='1', type=str)
    parser.add_argument('-gpu_ranks', default='1', type=str)
    parser.add_argument('-log_file', default='./logs/logs.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device_id = 0 if device == "cuda" else -1

    # train extraction way
    if args.mode == 'train':
        logging.basicConfig(filename='./logs/train_logs.log', level=logging.DEBUG, format='%(asctime)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        logger = logging.getLogger()
        train_ext(args, device_id)
    elif args.mode == 'validate':
        logging.basicConfig(filename='./logs/validate_logs.log', level=logging.DEBUG, format='%(asctime)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        logger = logging.getLogger()
        validate_ext(args, device_id)



