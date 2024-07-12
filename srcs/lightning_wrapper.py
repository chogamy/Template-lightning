import torch
from torch.nn import functional as F
import lightning as L
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model


class LightningWrapper(L.LightningModule):
    def __init__(self, args, tokenizer, metric) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.metric = metric

        self.model = AutoModelForCausalLM.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)

        # peft(WIP)
        if args.peft == "lora":
            peft_config = LoraConfig()

            model = get_peft_model(model, peft_config)

        self.save_hyperparameters(args)

    def forward(self, batch):
        loss = self.model(batch)
        return loss

    def training_step(self, batch, batch_id):
        loss = self(batch)

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def predict(self, batch):
        outputs = self.model.predict(batch)

        return outputs

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        outputs = self.predict(batch)

        self.metric.update()

    @torch.no_grad()
    def test_step(self, batch, batch_id):
        outputs = self.predict(batch)

        self.metric.update()

    def on_validation_epoch_end(self):
        self.log_dict(self.metric.compute())
        self.metric.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metric.compute())
        self.metric.reset()

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(steps * self.args.warmup_rate)

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
