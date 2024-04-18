'''muyGP_torch applied to toy example
'''

import sys
modules_to_remove = [m for m in sys.modules.keys() if m.startswith("Muy")]
for m in modules_to_remove:
    sys.modules.pop(m)

# !!! set environment variables !!!
%env MUYGPYS_BACKEND=torch
%env MUYGPYS_FTYPE=32

import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.datasets import make_friedman1
from math import floor
import matplotlib.pyplot as plt

from MuyGPyS.gp import MuyGPS 
from MuyGPyS.gp.deformation import l2, Isotropy
from MuyGPyS.gp.hyperparameter import Parameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch

# muygpy_torch related functions
from MuyGPyS.torch import MuyGPs_layer
from MuyGPyS.examples.muygps_torch import predict_model


# generate data with famous friedman function
data_dim = 6
X, y = make_friedman1(n_samples = 5000, n_features = data_dim, noise = 0.01, random_state=42)
gp_input_dim = 3

# train test split
train_n = int(floor(0.8 * len(X)))
train_features = torch.tensor(X[:train_n, :], dtype = torch.float32)
train_responses = torch.tensor(y[:train_n].reshape(-1,1), dtype = torch.float32)

test_features = torch.tensor(X[train_n:, :], dtype = torch.float32)
test_responses = torch.tensor(y[train_n:].reshape(-1,1), dtype = torch.float32)

torch.autograd.set_detect_anomaly(True)
test_count, _ = test_features.shape
train_count, _ = train_features.shape

# initialize NN object for training
nn_count = 50
nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="exact")

# a batch is a downsampled training set (a set of representatives, other training data serve as pool for NN selection)
batch_count = 500
batch_indices, batch_nn_indices = sample_batch(
    nbrs_lookup, batch_count, train_count
)

batch_nn_indices = torch.tensor(batch_nn_indices, dtype = torch.int32)
batch_indices = torch.tensor(batch_indices, dtype = torch.int32)

# get the nearest neighbors for each row of test dataset
test_nn_indices, test_nn_dists = nbrs_lookup.get_nns(test_features)

batch_features = train_features[batch_indices,:]
batch_targets = train_responses[batch_indices]
batch_nn_targets = train_responses[batch_nn_indices] # responses of neighbors of the downsampled batch

if torch.cuda.is_available():
    train_features = train_features.cuda()
    train_responses = train_responses.cuda()
    test_features = test_features.cuda()
    test_responses = test_responses.cuda()


class SVDKMuyGPs(nn.Module):
    def __init__(
        self,
        muygps_model, # input an initialized muygps model
        batch_indices,
        batch_nn_indices,
        batch_targets,
        batch_nn_targets,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(data_dim,5),
            nn.ReLU(),
            nn.Linear(5,2),
            nn.Sigmoid() # transform NN output, i.e., GP input to [0,1]
        )
        self.batch_indices = batch_indices
        self.batch_nn_indices = batch_nn_indices
        self.batch_targets = batch_targets
        self.batch_nn_targets = batch_nn_targets
        self.GP_layer = MuyGPs_layer(
            muygps_model,
            batch_indices,
            batch_nn_indices,
            batch_targets,
            batch_nn_targets,
        )
    
    def forward(self,x): 
        predictions = self.embedding(x)
        predictions, variances = self.GP_layer(predictions)
        return predictions, variances
    

# create a svdkmuygps model
muygps_model = MuyGPS(
    kernel=Matern(
        smoothness=Parameter(0.5),
        deformation=Isotropy(
            l2,
            length_scale=Parameter(1.0, (0.1, 2))
        ),
    ),
    noise=HomoscedasticNoise(1e-6),
)

model = SVDKMuyGPs(
    muygps_model = muygps_model,
    batch_indices=batch_indices,
    batch_nn_indices=batch_nn_indices,
    batch_targets=batch_targets,
    batch_nn_targets=batch_nn_targets,
)

# train the model
if torch.cuda.is_available():
    model = model.cuda()
 
training_iterations = 200
optimizer = torch.optim.Adam(
    [{'params': model.parameters()}], lr=1e-2
)
scheduler = ExponentialLR(optimizer, gamma=0.97)
NN_update_steps = 5
loss_func = nn.MSELoss()

def train(nbrs_lookup):

    # keep track of the following items
    weight_history = {name: [] for name, _ in model.embedding.named_parameters()}
    train_nbr_history = []
    pred_history = []
    loss_history = []

    for i in range(training_iterations):
        model.train()
        optimizer.zero_grad()
        predictions, variances = model(train_features)
        loss = loss_func(predictions,batch_targets)
        loss.backward()      
        optimizer.step()
        scheduler.step()

        # record neural network weights
        for name, param in model.embedding.named_parameters():
            weight_history[name].append(param.data.clone().cpu().numpy())

        # record training loss
        print(f"Iter {i + 1}/{training_iterations} - Loss: {loss.item()}")
        loss_history.append(loss.item())
        
        # update nearest neighbors of training data
        if np.mod(i,NN_update_steps) == 0:

            model.eval()
            nbrs_lookup = NN_Wrapper(
                model.embedding(train_features).detach().numpy(), # output of neural net part
                nn_count, nn_method="exact"
            )
            batch_nn_indices,_ = nbrs_lookup._get_nns(
                model.embedding(batch_features).detach().numpy(),
                nn_count=nn_count,
            )
            batch_nn_targets = train_responses[batch_nn_indices, :]  
            model.batch_nn_indices = batch_nn_indices
            model.batch_nn_targets = batch_nn_targets

            # keep track of nbrs change for training data
            train_nbr_history.append(batch_nn_indices)

        # predict on test set
        predictions, variances = predict_model(
            model=model,
            test_features=test_features,
            train_features=train_features,
            train_responses=train_responses,
            nbrs_lookup=nbrs_lookup,
            nn_count=nn_count,
        )
        pred_history.append(torch.nn.functional.mse_loss(predictions, test_responses, reduction='mean').detach())
        print(f"test MSE:{pred_history[-1]}")

        torch.cuda.empty_cache()

    # after training loop, update nbrs and model before return
    nbrs_lookup = NN_Wrapper(
        model.embedding(train_features).detach().numpy(),
        nn_count,
        nn_method="exact",
    )
    batch_nn_indices,_ = nbrs_lookup._get_nns(
        model.embedding(batch_features).detach().numpy(),
        nn_count=nn_count,
    )
    batch_nn_targets = train_responses[batch_nn_indices, :]
    model.batch_nn_indices = batch_nn_indices
    model.batch_nn_targets = batch_nn_targets

    return nbrs_lookup, model, pred_history, loss_history, weight_history, train_nbr_history


nbrs_lookup, model_trained, pred_history, loss_history, weight_history, train_nbr_history = train(nbrs_lookup)


plt.figure()
plt.plot(np.arange(0, training_iterations), loss_history, label = 'Loss')
for i in range(training_iterations):
    if  np.mod(i,NN_update_steps) == 0:
        plt.axvline(x=i+1, color='grey')
plt.xlim(right=training_iterations)
plt.xlabel('iteration')
plt.title('Loss')
plt.show()


plt.figure()
plt.plot(np.arange(0, training_iterations),pred_history, label = 'test MSE')
for i in range(training_iterations):
    if  np.mod(i,NN_update_steps) == 0:
        plt.axvline(x=i+1, color='grey')
plt.xlim(right=training_iterations)
plt.xlabel('iteration')
plt.title('test MSE')
plt.show()


#----------------------------------------------------
# Plotting weights and biases along training process
#----------------------------------------------------

# Function to normalize weights
def normalize(weights):
    return (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

for name, weight_matrices in weight_history.items():
    mat_for_plot = []

    # Flatten weights and biases such that each row corresponds to params from each train iteration
    for weight_matrix in weight_matrices:
        # # Normalize params from same interation
        # all_weights = np.array(weight_matrix)
        # min_val, max_val = np.min(all_weights), np.max(all_weights)
        # normalized_weights = (all_weights - min_val) / (max_val - min_val)
        mat_for_plot.append(weight_matrix.flatten())
    mat_for_plot = np.array(mat_for_plot)

    # Determine the number of plots based on the flattened size
    num_params = mat_for_plot.shape[1]

    # Adjust the number of columns based on the number of parameters
    ncols = min(num_params, 5)  # Limiting to 10 columns for display purposes
    nrows = (num_params + ncols - 1) // ncols  # Calculate the needed rows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    for cl in range(num_params):
        ax = axes.flat[cl] if num_params > 1 else axes
        ax.scatter(list(range(mat_for_plot.shape[0])), mat_for_plot[:, cl])
        ax.set_title(f'{name}, Param {cl + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weights')

    # Tight layout often solves the overlapping issue but can still be problematic with many subplots
    plt.tight_layout()
    plt.show()

# TODO
# compare initial MSE and final MSE to check the impact of initialization
# repeat same exp multiple times to see how final MSE varies