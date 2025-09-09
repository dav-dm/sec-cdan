from torch import nn


class DomainDiscriminator(nn.Sequential):
    """
    Domain discriminator model
    """
    def __init__(self, in_feature, hidden_size, sigmoid=True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
            
        super(DomainDiscriminator, self).__init__(
            nn.Linear(in_feature, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            final_layer
        )

    def get_parameters(self):
        return [{"params": self.parameters(), "lr": 1.}]