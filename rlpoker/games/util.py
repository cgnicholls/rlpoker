from rlpoker.extensive_game import ExtensiveGame


def parse_params(spec: str):
    """Parse the param dictionary from the specifier.

    Args:
        spec: string of the form 'key1_value1:key2_value2:...:keyn_valuen', i.e. colon separated key-value pairs, with
            each key-value pair separated by an underscore.
    """
    params = dict()
    if len(spec) == 0:
        return params

    parts = spec.split(':')
    for part in parts:
        k, v = part.split('_', 1)
        params[k] = v
    return params


class ExtensiveGameBuilder:

    @staticmethod
    def build(spec: str) -> ExtensiveGame:
        """Build an ExtensiveGame given a specifier."""
        raise NotImplementedError()