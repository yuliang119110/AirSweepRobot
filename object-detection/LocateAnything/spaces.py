# Local stub for the HuggingFace spaces package.
# The real package provides ZeroGPU allocation decorators (@spaces.GPU).
# On a local GPU machine we don't need ZeroGPU, so the decorator is a no-op.

def GPU(*args, **kwargs):
    # Called as @spaces.GPU(duration=120) -> returns a decorator
    def decorator(fn):
        return fn
    # If called without parentheses, args[0] is the function itself
    if args and callable(args[0]):
        return args[0]
    return decorator
