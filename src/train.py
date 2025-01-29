"""Full parameters & QLoRA Training."""
import os
import json
import pathlib
import math
import torch
from transformers import Seq2SeqTrainer, EvalPrediction, set_seed
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from utils.others import (
    get_logger,
    safe_save_model_for_hf_trainer,
    SavePeftModelCallback,
    IGNORE_TOKEN_ID
)
from utils.common import (
    prepare_args,
    load_tokenizer_and_model,
)
from utils.data_collator import DataCollatorForSeq2Seq, DataCollatorForDistill
from utils.trainer import DistillTrainer
import pdb

logger = get_logger(__name__)

local_rank = None
import pdb

# pdb.set_trace()

def train():
    global local_rank
    model_args, data_args, training_args, args = prepare_args()
    if args.use_flash_attn and "llama" in args.model_name_or_path.lower():
        from utils.llama_flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )
        replace_llama_attn_with_flash_attn()
    if args.deepspeed is not None and "zero_stage3" in args.deepspeed:
        logger.info("Must use zero_to_fp32.py to save model!")
    local_rank = args.local_rank
    set_seed(args.seed)
    tokenizer, model = load_tokenizer_and_model(args)
    model.to('cuda') #added

    # #################multigpu
    # DEV = torch.device('cuda:0')
    # try:
    #     model = model.to(DEV)
    # except:
    #     print('need to load to multi gpu')
    #     gpu_dist = [2,15,15]
    #     gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    #     print('gpus',gpus)
    #     if len(gpus) > 1:
    #         llama_multigpu(model, gpus, gpu_dist)
    #     else:
    #         model = model.to(DEV)

    # #################################

    dataset_name_list = args.dataset_name.split(",")
    #pdb.set_trace()
    logger.info(f"Loading {len(dataset_name_list)} dataset/datasets.")
    if len(dataset_name_list) == 1:
        raw_dataset = load_from_disk(dataset_name_list[0])
    else:
        raw_dataset = DatasetDict()
        if args.do_train:
            train_datasets = [load_from_disk(path)['train'] for path in dataset_name_list]
            raw_dataset["train"] = concatenate_datasets(train_datasets)
        if args.do_eval:
            validation_datasets = [load_from_disk(path)['validation'] for path in dataset_name_list]
            raw_dataset["validation"] = concatenate_datasets(validation_datasets)
            test_datasets = [load_from_disk(path)['test'] for path in dataset_name_list]
            raw_dataset["test"] = concatenate_datasets(test_datasets)
    dataset = raw_dataset
    if args.do_train:
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    if args.do_eval:
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))


    #calculate total training data num
    # num_train_examples = len(train_dataset)
    # print(f"Total number of training examples: {num_train_examples}")
    # exit()

    if args.do_distill:
        data_collator = DataCollatorForDistill(tokenizer,
                                               padding="max_length",
                                               max_length=args.model_max_length,
                                               label_pad_token_id=IGNORE_TOKEN_ID,
                                               training_args=training_args,
                                               )
        trainer = DistillTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if args.do_eval else None,
            data_collator=data_collator
        )
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer,
                                               padding="max_length",
                                               max_length=args.model_max_length,
                                               label_pad_token_id=IGNORE_TOKEN_ID,
                                               )
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if args.do_eval else None,
            data_collator=data_collator
        )
    if args.training_mode == "qlora":
        trainer.add_callback(SavePeftModelCallback)
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        if args.training_mode == "full" and list(pathlib.Path(args.output_dir).glob("checkpoint-*")):
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
            # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
            train_result = trainer.train()
        model.config.use_cache = True
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if args.training_mode == "full":
            if args.deepspeed is not None and "zero_stage3" in args.deepspeed:
                trainer.save_model()
            else:
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if (args.do_train or args.do_eval):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache = invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            for e in kwargs:
                if kwargs[e] is not None and hasattr(kwargs[e], "device"):
                    if kwargs[e].device != self.dev:
                        kwargs[e] = kwargs[e].to(self.dev)

            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) - 1 else gpus[(i - 1) // pergpu]),
                                   i == 0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0] - 1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers) - len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i == 0)

    model.gpus = gpus


if __name__ == "__main__":
    train()
