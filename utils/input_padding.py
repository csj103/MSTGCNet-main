"""Resolve input-convolution padding without changing legacy model defaults."""


def native_input_padding(experiment_mode):
    return 'circular' if experiment_mode == 'E0' else 'replicate'


def resolve_input_padding(configs):
    requested = getattr(configs, 'input_padding', 'default')
    if requested == 'default':
        return native_input_padding(getattr(configs, 'experiment_mode', 'E0'))
    if requested not in {'circular', 'replicate'}:
        raise ValueError('input_padding must be default, circular, or replicate')
    return requested
