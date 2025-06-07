import torch
import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, RNVPCouplingBlock

class ActNorm2d(nn.Module):
    """
    Activation Normalization for invertible flows (Glow).
    Initializes scale and bias per channel on the first batch,
    then becomes a standard, learnable affine transform.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__()
        self.initialized = False

        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def initialize_parameters(self, x):
        # x: [B, C, H, W]
        with torch.no_grad():
            # Compute mean and std over batch, height, and width (per channel)
            mean = x.mean(dim=[0, 2, 3], keepdim=True)    # shape: [1, C, 1, 1]
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.bias.data = -mean
            self.log_scale.data = torch.log(1.0 / (std + 1e-6))  # So that after transform, std ≈ 1

    def forward(self, x, logdet=None, reverse=False):
        if not self.initialized:
            self.initialize_parameters(x)
            self.initialized = True

        if reverse:
            x = (x - self.bias) * torch.exp(-self.log_scale)
        else:
            x = x * torch.exp(self.log_scale) + self.bias

        if logdet is not None:
            B, C, H, W = x.shape
            dlogdet = H * W * torch.sum(self.log_scale)
            if reverse:
                logdet = logdet - dlogdet
            else:
                logdet = logdet + dlogdet
            return x, logdet
        else:
            return x

class NormalizingFlow(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.flattened_size = input_shape[0] * input_shape[1] * input_shape[2]

        nodes = [InputNode(self.flattened_size, name='input')]
        for k in range(2):
            nodes.append(Node(nodes[-1], PermuteRandom, {'seed': k}))
            nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                {'subnet_constructor': lambda c_in, c_out: nn.Sequential(
                    nn.Linear(c_in, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, c_out)
                ),
                'clamp': 1.5}
            ))
        nodes.append(OutputNode(nodes[-1], name='output'))
        self.flow = GraphINN(nodes)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, reverse=False):
        x = x.view(x.size(0), -1)
        if not reverse:
            z, log_jac_det = self.flow(x)
            return z, log_jac_det
        else:
            x_recon, _ = self.flow(x, rev=True)
            output = x_recon.view(-1, *self.input_shape)
            return self.sigmoid(output) #self.tanh(x_recon.view(-1, *self.input_shape))


# CNN Subnetwork constructor for RealNVP blocks
def cnn_subnet(in_channels, out_channels):
    h_size = 64
    return nn.Sequential(
        nn.Conv2d(in_channels, h_size, kernel_size=3, padding=1),
        ActNorm2d(h_size),
        nn.GELU(),
        nn.Conv2d(h_size, h_size, kernel_size=1),
        ActNorm2d(h_size),
        nn.GELU(),
        nn.Conv2d(h_size, out_channels, kernel_size=3, padding=1)
    )


class CNNRealNVPFlow(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape  # e.g., (C, H, W)

        nodes = [InputNode(*input_shape, name='input')]
        for k in range(16):
            nodes.append(Node(nodes[-1], PermuteRandom, {'seed': k}))
            nodes.append(Node(
                nodes[-1],
                RNVPCouplingBlock,
                {
                    'subnet_constructor': cnn_subnet,
                    'clamp': 1.5
                }
            ))
        nodes.append(OutputNode(nodes[-1], name='output'))

        self.flow = GraphINN(nodes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, reverse=False):
        if not reverse:
            z, log_jac_det = self.flow(x)
            return z, log_jac_det
        else:
            x_recon, _ = self.flow(x, rev=True)
            return x_recon


class Autoencoder(torch.nn.Module):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.hidden_size = 128 # Hard coded.. why not?
        self.input_shape = input_shape
        self.flattened_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.block = lambda input_dim, output_dim: nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        self.encode = nn.Sequential(
            self.block(self.flattened_size, self.hidden_size),
            nn.ReLU(),
            self.block(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.block(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.block(self.hidden_size, self.hidden_size//2),
        )

        self.decode = nn.Sequential(
            self.block(self.hidden_size//4, self.hidden_size),
            nn.ReLU(),
            self.block(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            self.block(self.hidden_size, self.flattened_size)
        )

    def sample_latent(self, x):
        mu, logvar = x.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, self.flattened_size)
        x = self.encode(x)
        x = self.sample_latent(x)
        x = self.decode(x)
        x = x.view(-1, *self.input_shape)
        return x

    def decode_latent(self, z):
        z = z.view(-1, 32)
        x = self.decode(z)
        x = x.view(-1, *self.input_shape)
        return x