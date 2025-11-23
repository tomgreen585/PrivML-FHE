"""
fhe_model.py

Full FHE model implementation. Uses previous plaintext model as weight and bias inputs.
Currently using CKKS Scheme, .square() are indications of the first and second activation function.
"""

class FHE_Model:
    def __init__(self, torch_nn):
        conv1_weight = torch_nn.conv1.weight.data
        self.conv1_weight = []
        for i in range(3):
            channel_weight = conv1_weight[0, i, :, :].tolist()
            self.conv1_weight.append(channel_weight)
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
    
    def forward(self, enc_x_list, windows_nb):
        y = enc_x_list[0].conv2d_im2col(self.conv1_weight[0], windows_nb)
        y += enc_x_list[1].conv2d_im2col(self.conv1_weight[1], windows_nb)
        y += enc_x_list[2].conv2d_im2col(self.conv1_weight[2], windows_nb)
        y += self.conv1_bias[0]
        y.square_()
        x = y.mm(self.fc1_weight) + self.fc1_bias
        x = x.mm(self.fc2_weight) + self.fc2_bias
        return x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
