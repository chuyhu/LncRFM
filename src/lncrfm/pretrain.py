import argparse
import json
import os
from pathlib import Path
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from .data import get_mlm_tokenizer
from .data.RNAdata import preprocess_and_load_dataset
from .data.collator import PretrainDataCollatorWithStructure
from .model import get_lncrfm
from .model.modeling_lncrfm import DEFAULT_SSM_CFG

def parse_args():
    parser = argparse.ArgumentParser(description='LncRFM pretraining')
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--data_path', default=os.getenv('LNCRFM_CORPUS_ROOT', 'LncCorpus/data'))
    parser.add_argument('--from_pretrained', type=Path)
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--max_length', type=int, default=2050)
    parser.add_argument('--min_seq_len', type=int, default=200)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--motif_kernel_sizes', type=int, nargs='+', default=[3, 7])
    parser.add_argument('--motif_dropout', type=float, default=0.0)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-06)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--max_steps_override', type=int, default=300000)
    parser.add_argument('--step_interval_override', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--lr_scheduler_type', choices=['linear'], default='linear')
    parser.add_argument('--disable_eval', action='store_true')
    return parser.parse_args()

def model_options(args):
    if args.from_pretrained:
        return json.loads((args.from_pretrained / 'model_config.json').read_text())['factory_kwargs']
    return {'dim': args.dim, 'layer': args.layer, 'max_length': args.max_length, 'use_motif_stem': True, 'motif_kernel_sizes': args.motif_kernel_sizes, 'motif_dropout': args.motif_dropout, 'residual_in_fp32': True, 'use_motif_gate': True, 'motif_gate_init': 0.0, 'position_embedding_type': 'none', 'use_bidirectional_mixer': True, 'pair_aware_fusion': True, 'bidirectional_strategy': 'mean', 'bidirectional_weight_tie': False, 'attn_layer_idx': [5, 11], 'attn_cfg': {'num_heads': args.num_attention_heads, 'dropout': 0.0, 'use_masked_attention': True}, 'ssm_cfg': dict(DEFAULT_SSM_CFG)}

class SaveModelConfig(TrainerCallback):

    def __init__(self, options):
        self.options = options

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {'schema': 'lncrfm-weights-v1', 'model_name': 'LncRFM', 'precision': 'float32', 'factory_kwargs': self.options}
        (directory / 'model_config.json').write_text(json.dumps(config, indent=2) + '\n')

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            self.save(Path(args.output_dir) / f'checkpoint-{state.global_step}')

def pretrain(args):
    set_seed(args.seed)
    options = model_options(args)
    tokenizer_file = args.from_pretrained / 'tokenizer.json' if args.from_pretrained else None
    tokenizer = get_mlm_tokenizer(tokenizer_file=str(tokenizer_file) if tokenizer_file else None, max_length=min(args.max_length, options['max_length']))
    model = get_lncrfm(**options, tokenizer=tokenizer, dtype=torch.float32, from_pretrained=str(args.from_pretrained) if args.from_pretrained else None)
    dataset = preprocess_and_load_dataset(args.data_path, tokenizer, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
    if not args.disable_eval and 'test' not in dataset:
        raise ValueError('Evaluation requires the released LncCorpus held-out partition')
    collator = PretrainDataCollatorWithStructure(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    callback = SaveModelConfig(options)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=collator, train_dataset=dataset['train'], eval_dataset=None if args.disable_eval else dataset['test'], callbacks=[callback], args=TrainingArguments(output_dir=args.run_name, learning_rate=args.lr, num_train_epochs=args.epoch, max_steps=args.max_steps_override, weight_decay=0.0, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_steps=args.warmup_steps, lr_scheduler_type=args.lr_scheduler_type, max_grad_norm=0.5, fp16=False, bf16=False, group_by_length=True, length_column_name='length', logging_steps=max(1, args.step_interval_override // 10), save_steps=args.step_interval_override, evaluation_strategy='no' if args.disable_eval else 'steps', eval_steps=None if args.disable_eval else args.step_interval_override, load_best_model_at_end=not args.disable_eval, metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=3, save_safetensors=True, remove_unused_columns=False, ddp_find_unused_parameters=True, report_to='none'))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.run_name)
    if trainer.is_world_process_zero():
        callback.save(args.run_name)

def run_pretrain():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    pretrain(parse_args())
