import yaml

from lightning import Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model

from srcs.data.metric import METRIC
from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_args(args):
    with open(args.trainer_args) as f:
        args.trainer_args = yaml.load(f, Loader=yaml.FullLoader)
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

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    # peft(WIP)
    if args.peft == "lora":
        peft_config = LoraConfig()

        model = get_peft_model(model, peft_config)

    metric = METRIC[args.data]()

    if args.mode == "fit":
        model = LightningWrapper(model, config, tokenizer, metric)

    if args.mode == "test":
        path = None
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        """
        load from ckpt
        """

    return model, tokenizer


def get_trainer(args):
    trainer = Trainer(**args.trainer_args)

    return trainer
