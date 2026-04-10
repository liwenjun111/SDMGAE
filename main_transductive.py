
import logging
import numpy as np
from tqdm import tqdm
import torch

from SMGAE_main.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

from SMGAE_main.datasets.data_util import load_dataset
from SMGAE_main.evaluation_5cv import node_classification_evaluation_5cv
# from SMGAE_main.models import build_model



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)




def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    # graph, (num_features, num_classes) = load_GAEdataset()
    graph, (num_features, num_classes) = load_dataset()

    auc_list = []
    auprc_list = []

    for i, seed in enumerate(seeds):
            print(f"####### Run {i} for seed {seed}")
            set_random_seed(seed)

            if logs:
                logger = TBLogger(
                    name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
            else:
                logger = None

            x = graph.ndata["feat"]

            final_auc, final_auprc = node_classification_evaluation_5cv(
                None, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob
            )
            auc_list.append(final_auc)
            auprc_list.append(final_auprc)
            if logger is not None:
                logger.finish()


    final_auc, final_auc_std = np.mean(auc_list), np.std(auc_list)
    final_auprc, final_auprc_std = np.mean(auprc_list), np.std(auprc_list)
    print(f"#final_auc: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"#final_auprc: {final_auprc:.4f}±{final_auprc_std:.4f}")

    return final_auc, final_auprc

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)


