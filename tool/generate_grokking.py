import os, sys, argparse, logging, re, copy
from datetime import datetime
from pathlib import Path
import torch
import pandas as pd
from itertools import chain

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import util, ml_setup, complete_ml_setup
from py_src.service import record_model_stat

from py_src.ml_setup_base import transformer_for_grokking
from py_src.ml_setup_base.grokking import step
from py_src.ml_setup_base.dataset_modular import ArithmeticDataset, ArithmeticIterator

logger = logging.getLogger("generate_grokking")

def loading_dataset_from(path):
    pattern = r'modulus(\d+)'
    match = re.search(pattern, Path(path).name)
    assert match is not None, f"dataset folder name should start with modulus{{modulus}}...."
    modulus = int(match.group(1))
    dataset_train = ArithmeticDataset.load_from_file(f"{path}/train.txt", modulus, name=path, train=True, tokenizer_path=f"{path}/tokenizer.txt")
    dataset_val = ArithmeticDataset.load_from_file(f"{path}/val.txt", modulus, name=path, train=False, tokenizer_path=f"{path}/tokenizer.txt")
    return dataset_train, dataset_val

def generate_dataset(output_folder_path, train_pct, expression, modulus, train_split_type, operand_length):
    train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=train_pct,
        operator=expression,
        train_split_type=train_split_type,
        modulus=modulus,
        operand_length=operand_length,
    )

    name = f"{train_dataset.name}"
    train_dataset.save_to_file(os.path.join(output_folder_path, name, "train.txt"))
    val_dataset.save_to_file(os.path.join(output_folder_path, name, "val.txt"))
    train_dataset.tokenizer.save_tokens(os.path.join(output_folder_path, name, "tokenizer.txt"))
    return train_dataset, val_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate some high accuracy models')
    parser.add_argument("-n", "--number_of_models", type=int, default=1)
    parser.add_argument("-c", '--core', type=int, default=4, help='specify the number of CPU cores to use')
    parser.add_argument("-w", "--worker", type=int, default=1, help='specify how many models to train in parallel')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')

    parser.add_argument("-m", "--model_type", type=str, default='transformer_for_grokking')
    parser.add_argument("-dpath", "--dataset_path", type=str, default=None)

    parser.add_argument("-dexp", "--dataset_exp", type=str, default=None)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument("-tp", "--train_pct", type=float, default=50)
    parser.add_argument("-st", "--split_type", type=str, default='random', choices=["random", "chessboard", "updown", "leftright", "tl_to_br", "tr_to_bl", "interlace_row", "interlace_col", "chessboard_random"])
    parser.add_argument("-ol", "--operand_length", type=int, default=None)

    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-minlr", "--min_lr", type=float, default=None)
    parser.add_argument("-epoch", "--epoch", type=int, default=None)
    parser.add_argument("-wd", "--weight_decay", type=float, default=None)
    parser.add_argument("-bs", "--batchsize", type=int, default=None)

    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'], help='which format to save the training states')
    parser.add_argument("--save_interval", type=int, default=1, help='save model state per n epoch')

    parser.add_argument("-s","--random_seed", type=int, help='specify the random seed')
    parser.add_argument("-i", "--start_index", type=int, default=0, help='specify the start index for model names')

    parser.add_argument("-t", "--transfer_learn", type=str, default=None, help='specify a model weight file to perform transfer learning from.')
    parser.add_argument("--disable_reinit", action='store_true', help='disable reinitialization')
    parser.add_argument("--inverse_train_val", action='store_true', help='inverse train and validation set')

    parser.add_argument("--m_nlayer", default=None, type=int, help='specify the number of transformer layers, default=2')
    parser.add_argument("--m_n_heads", default=None, type=int, help='specify the number of attention heads, default=4')
    parser.add_argument("--m_d_model", default=None, type=int, help='specify the depth, default=128 ')
    parser.add_argument("--m_context_len", default=None, type=int, help='specify the size of context window, default=50')
    parser.add_argument("--m_pos_encoding", default=None, type=str, choices=["default", "trainable"], help='specify the type of positional encoding')

    args = parser.parse_args()


    arg_lr = args.learning_rate
    arg_min_lr = args.min_lr
    arg_epoch = args.epoch
    arg_wd = args.weight_decay
    arg_bs = args.batchsize

    arg_transfer_learn_model_path = args.transfer_learn
    arg_number_of_models = args.number_of_models
    arg_disable_reinit = args.disable_reinit
    arg_inverse_train_val = args.inverse_train_val

    m_nlayer = args.m_nlayer
    m_n_heads = args.m_n_heads
    m_d_model = args.m_d_model
    m_context_len = args.m_context_len
    m_pos_encoding = args.m_pos_encoding

    # random seed
    random_seed = args.random_seed
    if random_seed is not None:
        util.set_seed(random_seed, logger)

    # logger
    util.set_logging(logger, "main")
    logger.info("logging setup complete")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type
    current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type="arithmetic_exp_unknown", device=device)
    logger.info(f"model name: {current_ml_setup.model_name}")

    # create output folder
    output_folder_name = args.output_folder_name
    if output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.mkdir(output_folder_path)

    # dataset
    if args.dataset_path is not None:
        # load dataset
        train_ds, val_ds = loading_dataset_from(args.dataset_path)
    else:
        # generate dataset
        train_ds, val_ds = generate_dataset(output_folder_path, args.train_pct, args.dataset_exp, args.modulus, args.split_type, args.operand_length)

    # swap dataset?
    if arg_inverse_train_val:
        logger.info(f"inverse training and validation set, setting arg_number_of_models to 2")
        arg_number_of_models = 2

    batch_size = current_ml_setup.training_batch_size if arg_bs is None else arg_bs
    tokenizer = train_ds.tokenizer
    criterion = current_ml_setup.criterion

    # prepare for training
    digit_number_of_models = len(str(arg_number_of_models))
    init_model_for_inverse_train_val = None
    for index in range(arg_number_of_models):
        save_name = str(index).zfill(digit_number_of_models)
        model: transformer_for_grokking.Transformer = copy.deepcopy(current_ml_setup.model)

        if any(x is not None for x in [m_nlayer, m_n_heads, m_d_model, m_context_len, m_pos_encoding]):
            logger.info(f"use non-default model")
            m_nlayer = 2 if m_nlayer is None else m_nlayer
            m_n_heads = 4 if m_n_heads is None else m_n_heads
            m_d_model = 128 if m_d_model is None else m_d_model
            m_context_len = 50 if m_context_len is None else m_context_len
            m_pos_encoding = "default" if m_pos_encoding is None else m_pos_encoding
            trainable_position_encoding = m_pos_encoding == "trainable"
            model = transformer_for_grokking.Transformer(n_layers=m_nlayer, n_heads=m_n_heads, d_model=m_d_model,
                                                         max_context_len=m_context_len, trainable_position_encoding=trainable_position_encoding)

        train_dl = ArithmeticIterator(train_ds, device, batchsize_hint=-1)
        val_dl = ArithmeticIterator(val_ds, device, batchsize_hint=-1)
        if arg_inverse_train_val:
            if index == 0:
                logger.info(f"inverse train val mode: currently train on train partition")
                current_ml_setup.re_initialize_model(model)
                init_model_for_inverse_train_val = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                output_folder_path_current = os.path.join(output_folder_path, "train")
                os.mkdir(output_folder_path_current)
            elif index == 1:
                logger.info(f"inverse train val mode: currently train on val partition")
                model.load_state_dict(init_model_for_inverse_train_val)
                train_dl, val_dl = val_dl, train_dl # swap training / val dataset
                output_folder_path_current = os.path.join(output_folder_path, "val")
                os.mkdir(output_folder_path_current)
            else:
                raise NotImplementedError("index >= 2 is not defined")
            model.to(device)
        else:
            # transfer learning?
            if arg_transfer_learn_model_path is None:
                # not transfer learning, we should reinitialize model weights
                if arg_disable_reinit:
                    logger.info(f"re-initialize model is disabled")
                else:
                    logger.info(f"re-initialize model")
                    current_ml_setup.re_initialize_model(model)
            else:
                existing_model_state, existing_model_name, existing_dataset_name = util.load_model_state_file(arg_transfer_learn_model_path)
                logger.info(f"load model weights for transfer learning, original model type: {existing_model_name}, dataset type: {existing_dataset_name}")
                model.load_state_dict(existing_model_state)
            model.to(device)
            output_folder_path_current = os.path.join(output_folder_path, "val")

        # get optimizer stuff
        wd = 0 if arg_wd is None else arg_wd
        lr = 1e-3 if arg_lr is None else arg_lr
        min_lr = 1e-4 if arg_min_lr is None else arg_min_lr
        total_epoch = 150000 if arg_epoch is None else arg_epoch

        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=wd, lr=lr, betas=(0.9, 0.98), eps=1e-8)
        warmup_epoch = 10
        cosine_epoch = total_epoch - warmup_epoch
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epoch)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epoch, eta_min=min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epoch])
        lr_scheduler = scheduler

        # service
        arg_save_format = args.save_format
        arg_save_interval = args.save_interval
        if arg_save_format != 'none':
            record_model_service = record_model_stat.ModelStatRecorder(1, current_ml_setup.model_name, current_ml_setup.dataset_name)
            model_state_path = f"{output_folder_path_current}/{save_name}"
            os.makedirs(model_state_path)
            record_model_service.initialize_without_runtime_parameters([0], model_state_path, save_format=arg_save_format, lmdb_db_name=f"{save_name}")
        else:
            record_model_service = None

        epoch_loss_lr_log_file = open(os.path.join(output_folder_path_current, f"{save_name}.log.csv"), "w")
        epoch_loss_lr_log_file.write("epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,lrs" + "\n")
        epoch_loss_lr_log_file.flush()

        for epoch in range(total_epoch):
            train_correct = None
            train_loss = 0
            train_count = 0
            if epoch == 0 and record_model_service is not None:
                model_stat = model.state_dict()
                record_model_service.trigger_without_runtime_parameters(-1, [0], [model_stat])
            for batch_idx, batch in enumerate(train_dl):
                output = step(batch_idx, batch, model, optimizer, lr_scheduler, tokenizer,train=True)
                loss = output.loss_value
                train_loss += loss * output.sample_count
                train_count += output.sample_count
                train_correct = 0 if train_correct is None else train_correct
                train_correct += output.correct_count

            val_loss, val_correct, val_count = 0.0, 0.0, 0
            for batch_idx, batch in enumerate(val_dl):
                output = step(batch_idx, batch, model, optimizer, lr_scheduler, tokenizer,train=False)
                val_loss += output.loss_value * output.sample_count
                val_correct += output.correct_count
                val_count += output.sample_count

            # print progress
            lrs = []
            for param_group in optimizer.param_groups:
                lrs.append(param_group['lr'])
            logger.info(f"epoch[{epoch}] loss,accuracy= (train) {train_loss / train_count:.4},{train_correct / train_count:.4} (val) {val_loss / val_count:.4},{val_correct / val_count:.4} lrs={lrs}")
            epoch_loss_lr_log_file.write(f"{epoch},{train_loss / train_count:.4e},{train_correct / train_count:.4e},{val_loss / val_count:.3e},{val_correct / val_count:.4e},{lrs}" + "\n")

            # services
            if record_model_service is not None:
                model_stat = model.state_dict()
                if epoch % arg_save_interval == 0:
                    record_model_service.trigger_without_runtime_parameters(epoch, [0], [model_stat])

        # final record
        ## record final correct position
        final_correct_position = {"lhs": [], "rhs": [], "correct?": []}
        for batch_idx, batch in enumerate(chain(train_dl, val_dl)):
            output = step(batch_idx, batch, model, optimizer, lr_scheduler, tokenizer, train=False)
            x = batch["text"]
            num_0 = [int(tokenizer.itos[val.item()]) for val in x[:, 1]]
            num_1 = [int(tokenizer.itos[val.item()]) for val in x[:, 3]]
            final_correct_position["lhs"].extend(num_0)
            final_correct_position["rhs"].extend(num_1)
            final_correct_position["correct?"].extend(output.correct_location.tolist())
        final_correct_position = pd.DataFrame(final_correct_position)
        final_correct_position = final_correct_position.sort_values(by=["lhs", "rhs"])
        final_correct_position.to_csv(os.path.join(output_folder_path_current, "final_correct_position.csv"), index=False)

        ## save final model state
        util.save_model_state(os.path.join(output_folder_path_current, f"{save_name}.model.pt"),
                              model.state_dict(), current_ml_setup.model_name, current_ml_setup.dataset_name)