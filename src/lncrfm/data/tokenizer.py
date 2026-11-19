import os
from transformers import PreTrainedTokenizerFast
SRC_DIR = os.path.abspath(os.path.dirname(__file__))

def get_mlm_tokenizer(tokenizer_file=None, max_length=514, *args, **kargs):
    if tokenizer_file is None:
        tokenizer_file = os.path.join(SRC_DIR, 'tokenizer_mlm.json')
    tokenizer = PreTrainedTokenizerFast(*args, tokenizer_file=tokenizer_file, padding_side='right', truncation_side='right', cls_token='[CLS]', bos_token='[CLS]', sep_token='[SEP]', eos_token='[SEP]', unk_token='[UNK]', mask_token='[MASK]', pad_token='[PAD]', model_max_length=max_length, **kargs)
    return tokenizer
