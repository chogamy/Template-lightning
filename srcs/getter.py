import os
import yaml

from lightning import Trainer
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint

from srcs.data.metric import METRIC
from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_args(args):
    with open(args.trainer_args) as f:
        args.trainer_args = yaml.load(f, Loader=yaml.FullLoader)
    # Define your model's name
    args.trainer_args["default_root_dir"] = os.path.join("NEW NAME")
    return args


def get_datamodule(args, tokenizer):
    dm = DataModule(args, tokenizer)
    dm.setup(args.mode)

    return dm


def get_model(args):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    metric = METRIC[args.data]()

    if args.mode == "fit":
        model = LightningWrapper(args, tokenizer, metric)

    if args.mode == "test":
        path = os.path.join(
            args.trainer_args["default_root_dir"],
            "lightning_logs",
            "version_0",
            "checkpoints",
            "last.ckpt",
        )
        model = LightningWrapper.load_from_checkpoint(
            path, args=args, tokenizers=tokenizer, metric=metric
        )

    return model, tokenizer


def get_callbacks(args):
    ckpt_callback = ModelCheckpoint(
        dirpath=args.trainer_args["default_root_dir"], filename="last"
    )

    return [ckpt_callback]


def get_trainer(args):
    args.trainer_args["callbacks"] = get_callbacks(args)
    trainer = Trainer(**args.trainer_args)

    return trainer
