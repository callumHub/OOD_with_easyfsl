from torch import nn


def my_mlp(x_dim, z_dim, y_dim):


    encoder = nn.Sequential(
        nn.Linear(x_dim, z_dim),
        nn.ReLU(),
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(z_dim, y_dim),
    )

    '''
    encoder = nn.Sequential(
        nn.Linear(x_dim, y_dim),
        nn.ReLU(),
    )
    '''

    return encoder

def pt_map_mlp(x_dim, z_dim, y_dim):
    encoder = nn.Sequential(
        nn.Linear(x_dim, z_dim),
        nn.ReLU(),
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(z_dim, y_dim),
        nn.ReLU(),
    )
    return encoder

def my_bilstm(x_dim, z_dim, y_dim, device):
    if device == "cpu":
        encoder = nn.LSTM(x_dim, z_dim, num_layers=1, batch_first=True, bidirectional=True)
    else:
        encoder = nn.LSTM(x_dim, z_dim, num_layers=1, batch_first=True, bidirectional=True).cuda()
    return encoder