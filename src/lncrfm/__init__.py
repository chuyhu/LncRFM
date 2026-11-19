from importlib import import_module


__version__ = "0.0.7"


def get_lncrfm(*args, **kwargs):
    return import_module(".model", __name__).get_lncrfm(*args, **kwargs)


def get_tokenizer(*args, **kwargs):
    return import_module(".data.tokenizer", __name__).get_mlm_tokenizer(*args, **kwargs)


__all__ = ["get_lncrfm", "get_tokenizer"]
