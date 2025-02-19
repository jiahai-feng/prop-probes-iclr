from functools import partial, update_wrapper, wraps
from inspect import signature, Parameter


class FlowFunction:
    is_flow = True

    def __init__(self, func):
        self.func = func
        update_wrapper(self, func)

    def __call__(self, kwargs):
        return lift(self.func)(kwargs)


def lift(func):
    @wraps(func)
    def lifted(kwargs):
        sig = signature(func)
        for name, param in sig.parameters.items():
            if param.default != Parameter.empty:
                if not name in kwargs:
                    kwargs[name] = param.default
        entries = {}
        for name, param in sig.parameters.items():
            if name in kwargs:
                entries[name] = kwargs[name]
            else:
                raise Exception(f"Missing parameter {name} for function {func}")
        out_args = func(**entries)
        for name, value in out_args.items():
            if name in kwargs and name not in entries:
                raise Exception(
                    f"Cannot write into variable {name} that was not read by function {func}"
                )
            kwargs[name] = value
        return kwargs

    return lifted


def lift_start(func):
    @wraps(func)
    def lifted(kwargs):
        sig = signature(func)
        for name, param in sig.parameters.items():
            if param.default != Parameter.empty:
                if not name in kwargs:
                    kwargs[name] = param.default
        entries = {}
        for name, param in sig.parameters.items():
            if name in kwargs:
                entries[name] = kwargs[name]
            else:
                raise Exception(f"Missing parameter {name} for function {func}")
        return func(**entries)

    return lifted


def lower(func, terminal=None):
    @wraps(func)
    def lowered(**kwargs):
        if terminal is None:
            return func(kwargs)
        else:
            return terminal(**func(kwargs))

    return lowered


def chain(*flow_funcs):
    @wraps(flow_funcs[0])
    def wrapper(args):
        for f in flow_funcs:
            args = f(args)
        return args

    return wrapper


def stream(flows, terminal=lambda **kwargs: kwargs):
    return lower(chain(*flows), terminal)


def flow(func):
    return FlowFunction(func)
