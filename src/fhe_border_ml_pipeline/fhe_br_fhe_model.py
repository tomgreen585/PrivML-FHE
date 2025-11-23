import numpy as np

"""
fhe_model.py

Full FHE model implementation. Uses previous plaintext model as weight and bias inputs.
Currently using CKKS Scheme. Does computations over two convolutions performing image to image.
"""

class FHE_Model:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.detach().cpu().numpy()
        self.conv1_bias = torch_nn.conv1.bias.detach().cpu().numpy()
        self.conv2_weight = torch_nn.conv2.weight.detach().cpu().numpy()
        self.conv2_bias = torch_nn.conv2.bias.detach().cpu().numpy()

    def forward(self, enc_channels, windows_nb):
        assert len(enc_channels) == 1, "Expecting 1 grayscale channel only"
        x = enc_channels[0]

        conv1_out = []
        for i in range(4):
            w = self.conv1_weight[i, 0, :, :].reshape(-1, 1).astype(np.float64)
            b = float(self.conv1_bias[i])
            y = x.conv2d_im2col(w, windows_nb)
            y += b
            y.square_()
            conv1_out.append(y)

        out = None
        for i in range(4):
            w = float(self.conv2_weight[0, i, 0, 0])
            if out is None:
                out = conv1_out[i] * w
            else:
                out += conv1_out[i] * w
        out += float(self.conv2_bias[0])
        
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
