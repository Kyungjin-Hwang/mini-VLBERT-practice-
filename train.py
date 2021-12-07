from transformers import *
import random
from torch.utils.data import Dataset
import logging
import torch
from tensorboardX import SummaryWriter
import inspect
import shutil
import numpy as np
import torch
import torch.nn
import torch.optim as optim
from resnet_vlbert_for_pretraining import ResNetVLBERTForPretraining
from previous.misc import summary_parameters
from previous import pretrain_metrics
from previous.composite_eval_metric import CompositeEvalMetric
from previous.load import smart_resume
from previous.checkpoint import Checkpoint
from previous.trainer import train
from previous.create_logger import create_logger

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
RNG_SEED = -1
TRAIN_LOSS_LOGGERS = [('mlm_loss', 'MLMLoss'), ('mvrc_loss', 'MVRCLoss')]



def clip_pad_images(tensor, pad_shape, pad=0):
    """
    Clip clip_pad_images of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    H, W = tensor.shape[1:]
    h = pad_shape[1]
    w = pad_shape[2]

    tensor_ret = torch.zeros((tensor.shape[0], h, w), dtype=tensor.dtype) + pad
    tensor_ret[:, :min(h, H), :min(w, W)] = tensor[:, :min(h, H), :min(w, W)]

    return tensor_ret


def clip_pad_boxes(tensor, pad_length, pad=0):
    """
        Clip boxes of the pad area.
        :param tensor: [k, d]
        :param pad_shape: K
        :return: [K, d]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    k = tensor.shape[0]
    d = tensor.shape[1]
    K = pad_length
    tensor_ret = torch.zeros((K, d), dtype=tensor.dtype) + pad
    tensor_ret[:min(k, K), :] = tensor[:min(k, K), :]

    return tensor_ret


def clip_pad_1d(tensor, pad_length, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros((pad_length, ), dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_length)] = tensor[:min(tensor.shape[0], pad_length)]

    return tensor_ret


def clip_pad_2d(tensor, pad_shape, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros(*pad_shape, dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])] \
        = tensor[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])]

    return tensor_ret

class GeneralCorpus(Dataset):
    def __init__(self, ann_file, tokenizer=None, seq_len=64, min_seq_len=64,
                 encoding="utf-8",
                 **kwargs):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.ann_file = ann_file
        self.encoding = encoding
        self.corpus = self.load_corpus()
        self.test_mode = False

    def load_corpus(self):
        corpus = []
        with open(self.ann_file, 'r', encoding=self.encoding) as f:
            corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in f.readlines()])
        corpus = [l.strip() for l in corpus if l.strip() != '']

        return corpus

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        raw = self.corpus[item]
        # tokenize
        tokens = self.tokenizer.basic_tokenizer.tokenize(raw)
        # add more tokens if len(tokens) < min_len
        _cur = (item + 1) % len(self.corpus)
        while len(tokens) < self.min_seq_len:
            _cur_tokens = self.tokenizer.basic_tokenizer.tokenize(self.corpus[_cur])
            tokens.extend(_cur_tokens)
            _cur = (_cur + 1) % len(self.corpus)

        # masked language modeling
        tokens, mlm_labels = self.random_word_wwm(tokens)

        # convert token to its vocab id
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]

        return ids, mlm_labels

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        return output_tokens, output_label

class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        if 'image' in self.data_names:
            if batch[0][self.data_names.index('image')] is not None:
                max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
                image_none = False
            else:
                image_none = True
        if 'boxes' in self.data_names:
            max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        if 'text' in self.data_names:
            max_text_length = max([len(data[self.data_names.index('text')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}

            if 'image' in self.data_names:
                if image_none:
                    out['image'] = None
                else:
                    image = ibatch[self.data_names.index('image')]
                    out['image'] = clip_pad_images(image, max_shape, pad=0)

            if 'boxes' in self.data_names:
                boxes = ibatch[self.data_names.index('boxes')]
                out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            if 'text' in self.data_names:
                text = ibatch[self.data_names.index('text')]
                out['text'] = clip_pad_1d(text, max_text_length, pad=0)

            if 'mlm_labels' in self.data_names:
                mlm_labels = ibatch[self.data_names.index('mlm_labels')]
                out['mlm_labels'] = clip_pad_1d(mlm_labels, max_text_length, pad=-1)

            if 'mvrc_ops' in self.data_names:
                mvrc_ops = ibatch[self.data_names.index('mvrc_ops')]
                out['mvrc_ops'] = clip_pad_1d(mvrc_ops, max_boxes, pad=0)

            if 'mvrc_labels' in self.data_names:
                mvrc_labels = ibatch[self.data_names.index('mvrc_labels')]
                out['mvrc_labels'] = clip_pad_boxes(mvrc_labels, max_boxes, pad=0)

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            else:
                out_tuple += (torch.stack(tuple(items), dim=0), )

        return out_tuple

def make_dataloader(mode='train', shuffle=True, distributed=False, num_replicas=None, rank=None):
    dataset = None
    if mode == 'train':
        ann_file = './bc1g.doc'
        dataset = GeneralCorpus('./bc1g.doc',tokenizer)
        print(len(dataset))
        batch_size = 64
        num_workers = 1
#         image_set = cfg.DATASET.TRAIN_IMAGE_SET
#         aspect_grouping = cfg.TRAIN.ASPECT_GROUPING
#         num_gpu = len(cfg.GPUS.split(','))
#         batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
#         num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
#     elif mode == 'val':
#         ann_file = cfg.DATASET.VAL_ANNOTATION_FILE
#         image_set = cfg.DATASET.VAL_IMAGE_SET
#         aspect_grouping = False
#         num_gpu = len(cfg.GPUS.split(','))
#         batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
#         shuffle = cfg.VAL.SHUFFLE
#         num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
#     else:
#         ann_file = cfg.DATASET.TEST_ANNOTATION_FILE
#         image_set = cfg.DATASET.TEST_IMAGE_SET
#         aspect_grouping = False
#         num_gpu = len(cfg.GPUS.split(','))
#         batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
#         shuffle = cfg.TEST.SHUFFLE
#         num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu

#     transform = build_transforms(cfg, mode)

#     if dataset is None:

#         dataset = build_dataset(dataset_name=cfg.DATASET.DATASET, ann_file=ann_file, image_set=image_set,
#                                 seq_len=cfg.DATASET.SEQ_LEN, min_seq_len=cfg.DATASET.MIN_SEQ_LEN)
#                                 with_precomputed_visual_feat=cfg.NETWORK.IMAGE_FEAT_PRECOMPUTED,
#                                 mask_raw_pixels=cfg.NETWORK.MASK_RAW_PIXELS,
#                                 with_rel_task=cfg.NETWORK.WITH_REL_LOSS,
#                                 with_mlm_task=cfg.NETWORK.WITH_MLM_LOSS,
#                                 with_mvrc_task=cfg.NETWORK.WITH_MVRC_LOSS,
#                                 answer_vocab_file=cfg.DATASET.ANSWER_VOCAB_FILE,
#                                 root_path=cfg.DATASET.ROOT_PATH, data_path=cfg.DATASET.DATASET_PATH,
#                                 test_mode=(mode == 'test'), transform=transform,
#                                 zip_mode=cfg.DATASET.ZIP_MODE, cache_mode=cfg.DATASET.CACHE_MODE,
#                                 cache_db=True if (rank is None or rank == 0) else False,
#                                 ignore_db_cache=cfg.DATASET.IGNORE_DB_CACHE,
#                                 add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
#                                 aspect_grouping=aspect_grouping,
#                                 mask_size=(cfg.DATASET.MASK_SIZE, cfg.DATASET.MASK_SIZE),
#                                 pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME)

    sampler = torch.utils.data.sampler.RandomSampler(dataset)
#     sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False)
#     batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size)
    collator = BatchCollator(dataset=dataset, append_ind=False)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size = batch_size,
                                             shuffle = True,
                                             collate_fn=collator)

    return dataloader


def train_net(args, batch_size, lr):
    # setup logger
    logger, final_output_path = create_logger('./output/pretrain/vlbert',
                                              args.cfg,
                                              'train',
                                              split='train')
    model_prefix = os.path.join(final_output_path, '')
    if args.log_dir is None:
        args.log_dir = os.path.join(final_output_path, 'tensorboard_logs')

    print(args)
    logger.info('training args:{}\n'.format(args))
    # pprint.pprint(config)
    #     logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # manually set random seed
    if RNG_SEED > -1:
        random.seed(RNG_SEED)
        np.random.seed(RNG_SEED)
        torch.random.manual_seed(RNG_SEED)
        torch.cuda.manual_seed_all(RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
    #     model = ResNetVLBERTForPretraining(config)
    model = ResNetVLBERTForPretraining()
    summary_parameters(model, logger)
    shutil.copy(args.cfg, final_output_path)
    shutil.copy(inspect.getfile(eval('ResNetVLBERT')), final_output_path)
    num_gpus = 1

    total_gpus = num_gpus
    rank = None
    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir is not None else None

    # model
    torch.cuda.set_device(0)
    model.cuda()

    #     train_loader = make_dataloader(config, mode='train', distributed=False)
    #     val_loader = make_dataloader(config, mode='val', distributed=False)
    train_loader = make_dataloader(mode='train', distributed=False)
    val_loader = make_dataloader(mode='val', distributed=False)
    train_sampler = None

    batch_size = batch_size

    base_lr = lr * batch_size

    ###### 여기까지

    #     optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if _k in n],
    #                                      'lr': base_lr * _lr_mult}
    #                                     for _k, _lr_mult in config.TRAIN.LR_MULT]
    #     optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()
    #                                                     if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])]})
    optimizer_grouped_parameters = []

    optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()]})

    optimizer = optim.Adam(optimizer_grouped_parameters,
                           lr=lr * batch_size,
                           weight_decay=0.0001)  ##

    # partial load pretrain state dict
    #     if config.NETWORK.PARTIAL_PRETRAIN != "":
    #         pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)['state_dict']
    #         prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
    #         if len(prefix_change) > 0:
    #             pretrain_state_dict_parsed = {}
    #             for k, v in pretrain_state_dict.items():
    #                 no_match = True
    #                 for pretrain_prefix, new_prefix in prefix_change:
    #                     if k.startswith(pretrain_prefix):
    #                         k = new_prefix + k[len(pretrain_prefix):]
    #                         pretrain_state_dict_parsed[k] = v
    #                         no_match = False
    #                         break
    #                 if no_match:
    #                     pretrain_state_dict_parsed[k] = v
    #             pretrain_state_dict = pretrain_state_dict_parsed
    #         smart_partial_load_model_state_dict(model, pretrain_state_dict)

    # metrics
    metric_kwargs = {'allreduce': args.dist,
                     'num_replicas': world_size if args.dist else 1}
    train_metrics_list = []
    val_metrics_list = []
    #     if config.NETWORK.WITH_REL_LOSS:
    #         train_metrics_list.append(pretrain_metrics.RelationshipAccuracy(**metric_kwargs))
    #         val_metrics_list.append(pretrain_metrics.RelationshipAccuracy(**metric_kwargs))
    #     if config.NETWORK.WITH_MLM_LOSS:
    #         if config.MODULE == 'ResNetVLBERTForPretrainingMultitask':
    #             train_metrics_list.append(pretrain_metrics.MLMAccuracyWVC(**metric_kwargs))
    #             train_metrics_list.append(pretrain_metrics.MLMAccuracyAUX(**metric_kwargs))
    #             val_metrics_list.append(pretrain_metrics.MLMAccuracyWVC(**metric_kwargs))
    #             val_metrics_list.append(pretrain_metrics.MLMAccuracyAUX(**metric_kwargs))
    #         else:
    #             train_metrics_list.append(pretrain_metrics.MLMAccuracy(**metric_kwargs))
    #             val_metrics_list.append(pretrain_metrics.MLMAccuracy(**metric_kwargs))
    #     if config.NETWORK.WITH_MVRC_LOSS:
    #         train_metrics_list.append(pretrain_metrics.MVRCAccuracy(**metric_kwargs))
    #         val_metrics_list.append(pretrain_metrics.MVRCAccuracy(**metric_kwargs))
    train_metrics_list.append(pretrain_metrics.MLMAccuracy(**metric_kwargs))
    val_metrics_list.append(pretrain_metrics.MLMAccuracy(**metric_kwargs))
    train_metrics_list.append(pretrain_metrics.MVRCAccuracy(**metric_kwargs))
    val_metrics_list.append(pretrain_metrics.MVRCAccuracy(**metric_kwargs))

    for output_name, display_name in TRAIN_LOSS_LOGGERS:
        train_metrics_list.append(pretrain_metrics.LossLogger(output_name, display_name=display_name, **metric_kwargs))
        val_metrics_list.append(pretrain_metrics.LossLogger(output_name, display_name=display_name, **metric_kwargs))

    train_metrics = CompositeEvalMetric()
    val_metrics = CompositeEvalMetric()
    for child_metric in train_metrics_list:
        train_metrics.add(child_metric)
    for child_metric in val_metrics_list:
        val_metrics.add(child_metric)

    # epoch end callbacks
    epoch_end_callbacks = []
    if (rank is None) or (rank == 0):
        epoch_end_callbacks = [Checkpoint(model_prefix, 1)]
    host_metric_name = 'MLMAccWVC'
    validation_monitor = ValidationMonitor(do_validation, val_loader, val_metrics,
                                           host_metric_name=host_metric_name)

    # optimizer initial lr before
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    # resume/auto-resume
    begin_epoch = 0
    if rank is None or rank == 0:
        begin_epoch = smart_resume(model, optimizer, validation_monitor, model_prefix, logger)

    #     if args.dist:
    #         begin_epoch = torch.tensor(config.TRAIN.BEGIN_EPOCH).cuda()
    #         distributed.broadcast(begin_epoch, src=0)
    #         config.TRAIN.BEGIN_EPOCH = begin_epoch.item()

    # batch end callbacks
    #     batch_size = len(config.GPUS.split(',')) * (sum(config.TRAIN.BATCH_IMAGES)
    #                                                 if isinstance(config.TRAIN.BATCH_IMAGES, list)
    #                                                 else config.TRAIN.BATCH_IMAGES)
    #     batch_end_callbacks = [Speedometer(batch_size, config.LOG_FREQUENT,
    #                                        batches_per_epoch=len(train_loader),
    #                                        epochs=config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH)]

    # setup lr step and lr scheduler
    #     if config.TRAIN.LR_SCHEDULE == 'plateau':
    #         print("Warning: not support resuming on plateau lr schedule!")
    #         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                                   mode='max',
    #                                                                   factor=config.TRAIN.LR_FACTOR,
    #                                                                   patience=1,
    #                                                                   verbose=True,
    #                                                                   threshold=1e-4,
    #                                                                   threshold_mode='rel',
    #                                                                   cooldown=2,
    #                                                                   min_lr=0,
    #                                                                   eps=1e-8)
    #     elif config.TRAIN.LR_SCHEDULE == 'triangle':
    #         lr_scheduler = WarmupLinearSchedule(optimizer,
    #                                             config.TRAIN.WARMUP_STEPS if config.TRAIN.WARMUP else 0,
    #                                             t_total=int(config.TRAIN.END_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS),
    #                                             last_epoch=int(config.TRAIN.BEGIN_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)  - 1)
    #     elif config.TRAIN.LR_SCHEDULE == 'step':
    #         lr_iters = [int(epoch * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS) for epoch in config.TRAIN.LR_STEP]
    #         lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_iters, gamma=config.TRAIN.LR_FACTOR,
    #                                          warmup_factor=config.TRAIN.WARMUP_FACTOR,
    #                                          warmup_iters=config.TRAIN.WARMUP_STEPS if config.TRAIN.WARMUP else 0,
    #                                          warmup_method=config.TRAIN.WARMUP_METHOD,
    #                                          last_epoch=int(config.TRAIN.BEGIN_EPOCH * len(train_loader) / config.TRAIN.GRAD_ACCUMULATE_STEPS)  - 1)
    #     else:
    #         raise ValueError("Not support lr schedule: {}.".format(config.TRAIN.LR_SCHEDULE))
    lr_scheduler = WarmupLinearSchedule(optimizer,
                                        8000,
                                        t_total=int(10 * len(train_loader)),
                                        last_epoch= int(begin_epoch * len(train_loader)) - 1)
    # broadcast parameter and optimizer state from rank 0 before training start
    #     if args.dist:
    #         for v in model.state_dict().values():
    #             distributed.broadcast(v, src=0)
    #         # for v in optimizer.state_dict().values():
    #         #     distributed.broadcast(v, src=0)
    #         best_epoch = torch.tensor(validation_monitor.best_epoch).cuda()
    #         best_val = torch.tensor(validation_monitor.best_val).cuda()
    #         distributed.broadcast(best_epoch, src=0)
    #         distributed.broadcast(best_val, src=0)
    #         validation_monitor.best_epoch = best_epoch.item()
    #         validation_monitor.best_val = best_val.item()

    # apex: amp fp16 mixed-precision training
    #     if config.TRAIN.FP16:
    #         # model.apply(bn_fp16_half_eval)
    #         model, optimizer = amp.initialize(model, optimizer,
    #                                           opt_level='O2',
    #                                           keep_batchnorm_fp32=False,
    #                                           loss_scale=config.TRAIN.FP16_LOSS_SCALE,
    #                                           max_loss_scale=128.0,
    #                                           min_loss_scale=128.0)
    #         if args.dist:
    #             model = Apex_DDP(model, delay_allreduce=True)

    train(model, optimizer, lr_scheduler, train_loader, train_sampler, train_metrics,
          begin_epoch, 10, logger,
          rank=rank, batch_end_callbacks=None, epoch_end_callbacks=epoch_end_callbacks,
          writer=writer, validation_monitor=validation_monitor, fp16=False,
          clip_grad_norm=10,
          gradient_accumulate_steps=1)

    return rank, model