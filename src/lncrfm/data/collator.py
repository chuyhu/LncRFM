# Modified from https://github.com/CatIIIIIIII/RNAErnie2

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch

@dataclass
class PretrainDataCollatorWithStructure(DataCollatorForLanguageModeling):
    step_cnt: int = 0
    paired_mlm: bool = True

    def set_ratio(self, step: int, total_steps: int=1677890, final_ratio=0.5):
        if step < total_steps * 0.15:
            ratio = 0.0
        elif step < total_steps * 0.85:
            ratio = final_ratio * (step - total_steps * 0.15) / (total_steps * 0.7)
        else:
            ratio = final_ratio
        return ratio

    def __call__(self, examples: Any) -> Dict[str, Any]:
        self.step_cnt += 1
        pad_examples = [{k: v for k, v in example.items() if k in ['input_ids', 'attention_mask']} for example in examples]
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(self.tokenizer, pad_examples, return_tensors='pt', pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {'input_ids': _torch_collate_batch(pad_examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}
        if 'connects' not in examples[0]:
            raise ValueError('PretrainDataCollatorWithStructure requires `connects` in each example.')
        batch['connects'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(example['connects'], dtype=torch.long) for example in examples], batch_first=True, padding_value=0)
        batch['structure_mask'] = (batch['connects'] != 0).float()
        special_tokens_mask = batch.pop('special_tokens_mask', None)
        if self.paired_mlm:
            ratio = self.set_ratio(self.step_cnt)
            mask_strategy = np.random.choice(['tokens', 'structure'], p=[1 - ratio, ratio])
        else:
            mask_strategy = 'tokens'
        if mask_strategy == 'tokens':
            batch['input_ids'], batch['labels'] = self.torch_mask_tokens(batch['input_ids'], special_tokens_mask=special_tokens_mask)
        elif mask_strategy == 'structure':
            batch['input_ids'], batch['labels'] = self.torch_mask_structure(batch['input_ids'], batch['connects'], special_tokens_mask=special_tokens_mask)
        else:
            raise ValueError('Unknown masking strategy.')
        return batch

    def mask_transform(self, inputs: Any, labels: Any, masked_indices: Any) -> Tuple[Any, Any]:
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return (inputs, labels)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any]=None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        return self.mask_transform(inputs, labels, masked_indices)

    def torch_mask_structure(self, inputs: Any, connects: Any, special_tokens_mask: Optional[Any]=None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        connects_indices = connects != 0
        match_masked_connects_indices = masked_indices & connects_indices
        for i, (row, col) in enumerate(zip(*torch.nonzero(match_masked_connects_indices, as_tuple=True))):
            match_masked_connects_indices[row, connects[row, col]] = True
        final_masked_indices = match_masked_connects_indices | masked_indices & ~connects_indices
        labels[~final_masked_indices] = -100
        return self.mask_transform(inputs, labels, final_masked_indices)
