from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dims, noise_type=NoiseType.DIAGONAL, fixed_noise_level=None):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], n_components),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], dim_out * n_components + num_sigma_channels)
        )

        self.init_hyperparameters()
        self.use_pi_network = (self.n_components > 1)

    def init_hyperparameters(self):
        self.learning_rate = 1e-3
        self.batch_size = 100
        self.gradient_clip = 0.01

    def forward(self, x, eps=1e-6):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma


    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)

        # Transform dimensions
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        if self.use_pi_network:
            y = y.unsqueeze(1)

        z_score = (y - mu) / sigma

        #normal_loglik = -0.5 * torch.einsum("ij,ik->i", z_score, z_score)
        test1 = -0.5 * torch.mul(z_score, z_score)
        test2 = torch.log(sigma)
        #print(test1.size())
        #print(test2.size())
        #raise Exception
        normal_loglik = -0.5 * torch.mul(z_score, z_score) - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(math.pi))

        # This logsumexp is to add the weighted probability distributions
        if self.use_pi_network:
            loglik = torch.logsumexp(log_pi + normal_loglik, dim=1) # adding log_pi is for weighting probability distribution, but doesn't work well
        else:
            loglik = normal_loglik

        loglik_minibatch = torch.sum(loglik)

        return -loglik_minibatch

        

    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.gather(rand_normal, index=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples
    
    def train(self, training_data, num_epochs=10, should_reset=False):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0
            avg_prob = 0
            minibatches = train_test.split_df_into_minibatches(training_data, self.batch_size)
            for batch_num, minibatch in enumerate(minibatches):
                [X_train, y_train] = minibatch

                loss = self.loss(X_train, y_train)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

                lik = torch.exp(-loss/self.batch_size)
                avg_prob += lik

            num_samples = len(minibatches) * self.batch_size
            epoch_loss = epoch_loss / num_samples
            avg_prob = avg_prob / len(minibatches)
            print(f"[epoch {epoch}] loss: {epoch_loss:>7f} | prob: {avg_prob:>7f}")

    def test(self, testing_data):

        with torch.no_grad():
            x = testing_data.drop(['podium'], axis = 1)
            y = testing_data['podium'].to_numpy()
            for i in range(len(y)):
                if y[i] > 20:
                    y[i] = 20

            x_tensor = torch.tensor(x.values.astype(np.float32))
            y_tensor = torch.tensor(y.astype(np.float32))

            loss = self.loss(x_tensor, y_tensor)

            log_pi, mu, sigma = self.forward(x_tensor)

            log_pi = log_pi.unsqueeze(2)
            
            dims = mu.size()

            count = torch.arange(start=1, end=21)
            count = count.repeat(dims[0], dims[1], 1)

            z_score = (count - mu) / sigma

            normal_loglik = -0.5 * torch.mul(z_score, z_score) - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(math.pi))
            if self.use_pi_network:
                loglik = torch.logsumexp(log_pi + normal_loglik, dim=1)
            else:
                loglik = torch.logsumexp(normal_loglik, dim=1)

            lik = torch.exp(loglik)

            return loss.item() / len(testing_data), lik.detach().numpy()

    def reset(self):
        pass
        #for layer in self.children():
        #    if hasattr(layer, 'reset_parameters'):
        #        layer.reset_parameters()
    

import data_gather
import train_test

races = data_gather.get_race_data('file')
results = data_gather.get_race_results('file', races)
driver_standings = data_gather.get_driver_standings('file', races)
constructor_standings = data_gather.get_constructor_standings('file', races)
qualifying = data_gather.get_qualifying_results('file', races)
weather = data_gather.get_weather('file', races)


races = data_gather.get_merged_data('scrape', races=races, results=results, qualifying=qualifying, driver_standings=driver_standings, constructor_standings=constructor_standings, weather=weather)
#races = data_gather.get_merged_data('file')
races.to_csv('final_df.csv')
#train_test.split_df_into_minibatches(races, 10)


dim_in = len(races.columns) - 3
dim_out = 1
n_components = 8
hidden_dim = [100, 50]
mdn_net = MixtureDensityNetwork(dim_in, dim_out, n_components, hidden_dim)

train_test.test_year_range(mdn_net, 2020, 2022)

