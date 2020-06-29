import torch
import torch.utils.data
import torch.nn as nn
import util
import activation_functions as actfuns


#  -------------------- Models

class CombinactNN(nn.Module):
    def __init__(self, net_struct, actfun,
                 num_inputs=784,
                 num_outputs=10,
                 alpha_dist="per_perm",
                 permute_type="shuffle"):
        super(CombinactNN, self).__init__()

        # Validate input
        error = util.test_net_inputs(actfun, net_struct, num_inputs)
        if error is not None:
            raise ValueError(error)

        self.net_struct = net_struct  # Network architecture
        self.actfun = actfun  # Current activation function used by NN
        self.num_inputs = num_inputs  # Number of network inputs
        self.num_hidden_layers = net_struct['num_layers']  # Number of hidden layers
        self.num_outputs = num_outputs  # Number of network outputs
        self.linear_layers = nn.ModuleList()  # Module list of fully connected layers
        self.all_batch_norms = nn.ModuleList()  # Module list of batch norm layers

        # Defines variables used by Combinact
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns())  # Number of actfuns used by combinact
            self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
            self.shuffle_maps = []  # List of shuffle maps used for shuffle permutations
            self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
            self.permute_type = permute_type  # Reference to chosen permutation type

        curr_inputs = self.num_inputs
        for layer in range(self.num_hidden_layers):
            M, k, p, g = self.get_layer_architecture(layer)

            # Splits inputs and pre-acts into groups of size g, creates a linear layer for each
            self.linear_layers.append(
                nn.ModuleList([nn.Linear(int(curr_inputs / g), int(M / g)) for i in range(g)]))

            # Creates a batch-norm for each group
            self.all_batch_norms.append(nn.ModuleList([nn.BatchNorm1d(int(M))]))

            if self.actfun == "combinact":
                if alpha_dist == "per_cluster":
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(M * p / k), self.num_combinact_actfuns)))
                if alpha_dist == "per_perm":
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(p, self.num_combinact_actfuns)))
                if permute_type == "shuffle":
                    self.shuffle_maps.append([])
                    for perm in range(p):
                        self.shuffle_maps[layer].append(torch.randperm(int(M)))

            curr_inputs = int(M * p / k)

        self.linear_layers.append(nn.ModuleList([nn.Linear(curr_inputs, self.num_outputs)]))

    def get_layer_architecture(self, layer):
        M = self.net_struct['M'][layer]
        k = self.net_struct['k'][layer]
        p = self.net_struct['p'][layer]
        g = self.net_struct['g'][layer]
        return M, k, p, g

    def activation(self, x, actfun, layer, M, k, p, g):

        if actfun == 'relu':
            return actfuns.get_actfuns()['relu'](x).unsqueeze(dim=2)
        elif actfun == 'abs':
            return actfuns.get_actfuns()['abs'](x).unsqueeze(dim=2)
        else:
            batch_size = x.shape[0]

            # Create permutations
            x = x.reshape(batch_size, M, 1)
            for i in range(1, p):
                permutation = util.permute(x,
                                           self.permute_type,
                                           offset=i,
                                           num_groups=2,
                                           layer=layer,
                                           shuffle_map=self.shuffle_maps[layer][i]).view(batch_size, M, 1)
                x = torch.cat((x[:, :, :i], permutation), dim=2)

            num_clusters = int(M / k)
            x = x.reshape(batch_size, num_clusters, k, p)  # Split M up into clusters of size k

            # Call activation function
            if actfun == "combinact":
                return actfuns.calc(actfun, x, layer, M, k, p, g,
                                    alpha_primes=self.all_alpha_primes[layer],
                                    alpha_dist=self.alpha_dist)
            else:
                return actfuns.calc(actfun, x, layer, M, k, p, g)

    def forward(self, x):

        # x is initially torch.Size([100, 1, 28, 28]), this step converts to torch.Size([100, 784])
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        curr_inputs = self.num_inputs

        # For each hidden layer
        for layer in range(self.num_hidden_layers):
            M, k, p, g = self.get_layer_architecture(layer)  # Define layer architecture
            x = x.reshape(batch_size, int(curr_inputs / g), g)  # Group layer inputs

            # For each group in the current hidden layer
            outputs = torch.zeros((batch_size, int(M / g), g), device=x.device)
            for i, fc in enumerate(self.linear_layers[layer]):
                curr_group = fc(x[:, :, i])  # Pass through linear layer
                outputs[:, :, i] = curr_group  # Store output groups in single tensor

            x = outputs.reshape(batch_size, M)  # Ungroup outputs
            x = self.all_batch_norms[layer][0](x)  # Apply batch norm
            x = self.activation(x, self.actfun, layer, M, k, p, g)  # Activate

            # We transpose so that when we reshape our outputs, the results from the permutations merge correctly
            x = torch.transpose(x, dim0=1, dim1=2)

            curr_inputs = M * p / k  # Number of inputs for next layer

        x = x.reshape(batch_size, -1)

        # Applies final linear layer to all outputs of final hidden layer
        x = self.linear_layers[-1][0](x)

        return x
