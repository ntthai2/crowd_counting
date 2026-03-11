"""Minimal fvcore.nn.parameter_count shim."""


def parameter_count_table(model, max_depth=10):
    """Stub: returns a simple parameter count string."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total parameters: {total:,} | Trainable: {trainable:,}"
