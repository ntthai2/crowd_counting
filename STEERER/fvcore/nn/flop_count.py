"""Minimal fvcore.nn.flop_count shim. Returns placeholder — training is unaffected."""


def flop_count(model, inputs, **kwargs):
    """Stub: returns empty dict. fvcore unavailable on Python 3.13."""
    return {}
