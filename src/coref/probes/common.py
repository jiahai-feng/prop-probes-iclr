class DomainProbe:
    def __init__(self, values, type):
        self.values = values
        self.type = type
    def __call__(self, activations):
        '''
        Args:
            activations: torch.FloatTensor[batch_size, layer, pos, dim]
                - these are normalized activations
        Returns:
            classification: torch.LongTensor[batch_size, pos] in [-1, len(self.values)]
        '''
        raise NotImplementedError

class PredicateProbe:
    def __call__(self, activations):
        '''
        Args:
            activations: torch.FloatTensor[batch_size, layer, pos, dim]
                - these are normalized activations
        Returns:
            predicates: List[Tuple[Any, Any, str]]
                - predicates are (name, value, domain.name)
                - typically, values are strings, but for capitals it's a pair of strings
        '''
        raise NotImplementedError