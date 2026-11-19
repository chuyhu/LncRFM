def process_mlm_input_seq(seq, bos_token='[CLS]', eos_token='[SEP]'):
    seq = seq.upper().replace('T', 'U')
    return f'{bos_token}{seq}{eos_token}'
