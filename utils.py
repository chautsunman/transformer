from typing import List
import torch

class SpecialSymbols:
    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SpecialSymbols.BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([SpecialSymbols.EOS_IDX])))

def token_to_sentence(tokens, vocab_transform, replace_pad=False):
    sentence = " ".join(vocab_transform.lookup_tokens(list(tokens)))
    sentence = sentence.replace(SpecialSymbols.special_symbols[SpecialSymbols.BOS_IDX], "")
    sentence = sentence.replace(SpecialSymbols.special_symbols[SpecialSymbols.EOS_IDX], "")
    if replace_pad:
        sentence = sentence.replace(SpecialSymbols.special_symbols[SpecialSymbols.PAD_IDX], "")
    return sentence
