import argparse
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from model.modules.bert_for_ner import BertCrfForNer
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from utils.utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils.data import Data
import torch.autograd as autograd

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertCrfForNer, BertTokenizer),  # BertForTokenClassification
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def batchify_with_label_3(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    gazs = [sent[1] for sent in input_batch_list]
    reverse_gazs = [sent[2] for sent in input_batch_list]
    char_seq_lengths = torch.LongTensor(list(map(len, chars)))
    max_seq_len = char_seq_lengths.max()
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()

    for idx, (seq, seqlen) in enumerate(zip(chars, char_seq_lengths)):
        char_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1] * int(seqlen))

    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    mask = mask[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    gaz_list = [gazs[i] for i in char_perm_idx]
    reverse_gaz_list = [reverse_gazs[i] for i in char_perm_idx]

    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_lengths = char_seq_lengths.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()

    return gaz_list, reverse_gaz_list, char_seq_tensor, char_seq_lengths, char_seq_recover, mask


def train(args, data, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size  # * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_eval_score = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)

    train_Ids = data.train_Ids
    train_num = len(train_Ids)

    for iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        count = 0
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            start = count * args.train_batch_size
            end = (count + 1) * args.train_batch_size
            count += 1

            if end > train_num:
                end = train_num
            instance = train_Ids[start:end]
            if not instance:
                continue
            gaz_list, reverse_gaz_list, batch_char, batch_charlen, batch_charrecover, mask = batchify_with_label_3(instance, data.HP_gpu)

            batch = tuple(t.to(args.device) for t in batch)
            # all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'input_lens': batch[3], "labels": batch[4], "gaz": gaz_list,
                      "reverse_gaz": reverse_gaz_list, "char_seq_lengths": batch_charlen}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Update learning rate schedule
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (iter > 4 or iter == 0) \
                        and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (args.local_rank == -1 and args.evaluate_during_training):
                        results, _ = evaluate(args, data, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        results2, _ = evaluate(args, data, model, tokenizer, labels, pad_token_label_id, mode="test")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    if results['f1'] > best_eval_score:
                        # Save model checkpoint
                        best_eval_score = results['f1']
                        output_dir = os.path.join(args.output_dir, "best_bertCrf_model_{:.0f}_{:.4f}".format(global_step, best_eval_score))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving best model to %s", output_dir)
                        logger.info("\033[35mBest dev f1: %.4fs, test f1: %.4fs\033[0m", results['f1'], results2['f1'])

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, data, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    instances = []
    if mode == "train":
        instances = data.train_Ids
    elif mode == "dev":
        instances = data.dev_Ids
    elif mode == 'test':
        instances = data.test_Ids
    train_num = len(instances)
    for step, batch in enumerate(eval_dataloader):

        start = step * args.eval_batch_size
        end = (step + 1) * args.eval_batch_size
        if end > train_num:
            end = train_num

        instance = instances[start:end]
        if not instance:
            continue
        dev_gaz_list, dev_reverse_gaz_list, dev_batch_char, dev_batch_charlen, _, _ = batchify_with_label_3(instance, data.HP_gpu)

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'input_lens': batch[3], "labels": batch[4], "gaz": dev_gaz_list,
                      "reverse_gaz": dev_reverse_gaz_list, "char_seq_lengths": dev_batch_charlen}

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )
            inputs['mode'] = mode
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed oing process the dataset, and the others will use the cache
        torch.distributed.barrier()

        # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    # args.overwrite_cache = False
    if os.path.exists(cached_features_file) and not args.overwrite_cache and mode != 'train':
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    # (3821,256)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/weibo/",
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-chinese",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default="./new_weibo_bert_v0",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="./TEMP_transformers",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", default=False,
                        help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true", default=True,
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", default=True,
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=2021, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        # args.n_gpu = 1
        # torch.cuda.set_device(2)
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        # filename=os.path.join('./logs', args.output_dir + '.out'),
        # filemode='a',
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    labels = get_labels()
    num_labels = len(labels)
    args.id2label = {i: label for i, label in enumerate(labels)}
    args.label2id = {label: i for i, label in enumerate(labels)}
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    # CrossEntropyLoss().ignore_index
    pad_token_label_id = num_labels - 1

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.output_hidden_states = True

    LK_enable = True
    data = Data()
    if LK_enable:
        data.model_name = 'SLK-NER'
        data.HP_gpu = True
        data.MAX_SENTENCE_LENGTH = args.max_seq_length
        gaz_file = 'data/ctb.50d.vec'
        train_file = os.path.join(args.data_dir, 'train.char.bmes')
        dev_file = os.path.join(args.data_dir, 'dev.char.bmes')
        test_file = os.path.join(args.data_dir, 'test.char.bmes')
        # gaz
        data.build_gaz_file(gaz_file)
        # gaz_alphabet, build gaz_alphabet by Word EnumerateMatchList
        data.build_gaz_alphabet(train_file)
        data.build_gaz_alphabet(dev_file)
        data.build_gaz_alphabet(test_file)
        data.gaz_alphabet.close()
        # instence_texts, instence_Ids
        data.generate_instance_with_gaz(train_file, 'train')
        data.generate_instance_with_gaz(dev_file, 'dev')
        data.generate_instance_with_gaz(test_file, 'test')
        # pretrain_gaz_embedding, gaz_emb_dim=50
        data.build_gaz_pretrain_emb(gaz_file)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        label2id=args.label2id,
        device=args.device,
        data=data
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, data, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [os.path.join(args.output_dir, "best_bertCrf_model_290_0.6724")]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config, label2id=args.label2id, device=args.device, data=data)
            model.to(args.device)
            result, _ = evaluate(args, data, model, tokenizer, labels, pad_token_label_id, mode="test", prefix=global_step)

            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}

            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        model = model_class.from_pretrained(os.path.join(args.output_dir, "best_bertCrf_model_290_0.6724"), config=config, label2id=args.label2id,
                                            device=args.device, data=data)

        model.to(args.device)
        result, predictions = evaluate(args, data, model, tokenizer, labels, pad_token_label_id, mode="train")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results_v0.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        # Save predictions
        output_test_predictions_file = 'data/weibo/w_bert_train.char.bmes'
        content_list = []
        with open(os.path.join(args.data_dir, "train.char.bmes"), 'r', encoding="utf-8") as f:
            chars = []
            for line in f:
                if len(line) > 1:
                    pairs = line.strip().split()
                    char = pairs[0]
                    chars.append(char)
                else:
                    content_list.append(chars)
                    chars = []
        print(len(content_list), len(predictions))

        fout = open(output_test_predictions_file, 'w', encoding="utf-8")
        for idx in range(len(predictions)):
            sent_length = len(predictions[idx])
            for idy in range(sent_length):
                fout.write(content_list[idx][idy] + " " + predictions[idx][idy] + '\n')
            fout.write('\n')
        fout.close()

    return results


if __name__ == "__main__":
    main()
