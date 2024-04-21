import sklearn.datasets
import sklearn.model_selection
import sklearn.feature_selection
import sklearn.pipeline
import torch
import torch.nn
import torch.nn.functional
import skorch
import numpy as np

class MyConvolultionNeuralNetworkModel(torch.nn.Module):  
    def __init__(
            self,
            n_input_channels: int,     # Features/columns, just data 
            input_image_height: int,         # Assuming the image is square
            n_output_probs: int,
            conv_layer_sizes=(32, 64, ),
            conv_kernel_sizes = (3, 3, ),
            act_func_maxpool =torch.nn.functional.relu,  # Activation function for non linearity
            dense_layer_sizes=(100, 100, ), # Making a tuple by adding the paranthese
            act_func_dense = torch.nn.functional.relu,
            dropout=0.5 # How much information will be lost in percentage (in layers)
    ):
        super().__init__()
        # Initializing the attributes (basically the input parameters)
        self.input_image_height = input_image_height
        self.conv_layer_sizes = conv_layer_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.act_func_maxpool = act_func_maxpool
        self.dense_layer_sizes = dense_layer_sizes
        self.act_func_dense = act_func_dense

        # Convolution Network Initialization (transformation [our case its nn.Linear], activation [our case its nn.ReLU], output (nn.Linear))
        self.conv_network = torch.nn.ModuleList()

        # Setting up the 2D Convolution Neural Network (in channels, out_channels, kernel_sizes)
        self.conv_network.append(
            torch.nn.Conv2d(in_channels=n_input_channels, 
                            out_channels=conv_layer_sizes[0],
                            kernel_size=conv_kernel_sizes
                            ) 
        )

        # Still Initializing ()
        self.conv_network.append(
            torch.nn.MaxPool2d(2)
            )
        
        # The rest of the Convolution Network (We are going to start from 1 "[1:]")
        for index in range(len(self.conv_layer_sizes) - 1):
            # Add the convolution layer [FOR CONVOLUTION]
            self.conv_network.append(
                torch.nn.Conv2d(
                    in_channels=self.conv_layer_sizes[index],
                    out_channels=self.conv_layer_sizes[index + 1],
                    kernel_size=conv_kernel_sizes[index + 1]
                )
            )

            # [FOR DROPOUT]
            self.conv_network.append(
                torch.nn.Dropout(p=dropout)
            )

            # [FOR MAXPOOL]
            self.conv_network.append(
                torch.nn.MaxPool2d(2)
            )


        # Dense network after Convolution Network
        self.dense_network = torch.nn.ModuleList()
        self.dense_network.append(
            torch.nn.Linear(
                self.calc_dense_n_inputs(),
                self.dense_layer_sizes[0]
            )
        )

        # Rest of the Dense Network
        for layer_size in self.dense_layer_sizes:
            self.dense_network.append(
                torch.nn.Linear(
                    layer_size,
                    layer_size
                )
            )
        # The output
        self.output = torch.nn.Linear(self.dense_layer_sizes[0], 1)

    def calc_dense_n_inputs(self):
        """
        Assume no padding, 1 dilation, 1 stride in Conv2D layers
        Assume no padding and kernel size 2 in MaxPool2D layers

        :returns: number of inputs into the dense (FC or linear) network
        """
        final_size = self.input_image_height

        #
        for conv_kernel_size in self.conv_kernel_sizes:
            # Conv2D (an example would be (1, 32, 3))
            final_size = np.floor(final_size - (conv_kernel_size - 1))

            # MaxPool2D
            final_size = np.floor((final_size - (2 - 1) - 1) / 2 + 1)

        return int(self.conv_layer_sizes[-1] * np.square(final_size))

    def forward(self, X, **kwargs):
        for layer in self.conv_network:
            if isinstance(layer, torch.nn.MaxPool2d):
                X = self.act_func_maxpool(layer(X))
            else:
                X = layer(X)
        
        X = X.view(-1, X.size(1) * X.size(2) * X.size(3))
        
        for layer in self.dense_network:
            X = self.act_func_dense(layer(X))
            
        return torch.nn.functional.softmax(X, dim=-1)