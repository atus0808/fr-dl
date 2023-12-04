#torch is used throughout the whole project
import torch
#for neural network operations
from torch import nn

from torch.nn.functional import mse_loss
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
#from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
#used for generating random numbers and shuffling matrices
import random as r
import sklearn as sk
#used to normalize
from sklearn import preprocessing
#used for OMP
from sklearn.linear_model import orthogonal_mp as omp
from sklearn import metrics
#to use MNIST
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#used for a few matrix analysis plots
import matplotlib.pyplot as plt
#pandas is used just to create datasets to use in seaborn
import pandas as pd

def compute_sigma_from_SNR(snr:float, power:torch.Tensor) -> torch.Tensor:
    """
    GOAL: 
     employ SNR to generate standard deviation for white noise
    """
    sigma =  power / (10**(snr/20))
    print(f'sigma: {type(sigma)}')
    return sigma

def create_synthetic_data_Xu(sample_size:int, num_classes:int, num_atoms_by_classes:int, y_dim:int, S:int, snr:float = 40.0) -> tuple:
    """
    GOAL: 
     generate synthetic data
    """
    num_atoms = num_classes*num_atoms_by_classes
    D = torch.rand(y_dim, num_atoms)-0.5
    D = preprocessing.normalize(D, norm = "l2", axis = 0, copy = True)
    D = torch.from_numpy(D).float()

    A = torch.zeros((sample_size, num_atoms))

    signals_by_class = int(sample_size/num_classes)
    indexes = [i for i in range(num_atoms_by_classes)]
    row_ndx = 0
    true_labels = torch.zeros(sample_size)
    for i in range(num_classes):
        class_offset = i * num_atoms_by_classes
        for j in range(signals_by_class):
            selected_atoms = np.random.permutation(indexes)[0:S] + class_offset
            coeff = torch.rand(S)
            a = torch.zeros(num_atoms)
            a[selected_atoms] = coeff
            A[row_ndx,:] = a
            row_ndx += 1
            true_labels[j+i*(signals_by_class)] = i

    # create signals
    #X = (D @ A.T).T + sigma*torch.randn((sample_size, y_dim))
    X = (D @ A.T).T
    Xnorms = torch.norm(X, dim=1, p=2)
    sigma = torch.unsqueeze(compute_sigma_from_SNR(snr, Xnorms), dim=-1)
    print(f"sigma: {sigma.shape}")
    print(f"(sample_size, y_dim): {sample_size}, {y_dim}")
    Noise = sigma * torch.randn((sample_size, y_dim))
    Y = X + Noise

    return A, D, Y, true_labels


def data_init(n:int, M:int, C:int, K:int, S:int, snr:int = 40) -> tuple:
  """
  GOAL:
  generate a randomized data matrix with added gaussian noise, return its
  components A and D and the related labels for classification

  INPUT:
  n - number of signals
  M - features per signal
  C - number of classes
  K - number of atoms per dictionary
  S - linear combination dimension
  snr - singal to noise ratio, default set to 40 as per Xu et al

  OUTPUT:
  A - matrix of coefficients
  D - matrix of dictionaries
  Y - data matrix
  true_labels - label associated to each row of Y

  """
  n_c = n//C
  c_tot = K*C

  #u -> (a - b)*u + b where u is U ~ (0, 1) and becomes u ~ [a, b] ; a = 10, b = -10.
  D = (- 10 - 10)*torch.rand(M, c_tot) + 10
  D = preprocessing.normalize(D, norm = "l2", axis = 0, copy = True)
  D = torch.from_numpy(D).float()

  #coefficient matrix
  A = torch.zeros((n, c_tot))
  true_labels = torch.zeros(n)
  for c in range(C):
    offset = c*K
    for i in range(n_c):
      idx = np.random.choice(K, S, False) + offset
      lin_coef = (- 1 - 1)*torch.rand(S) + 1
      A[i+c*(n_c), idx] = lin_coef
      true_labels[i+c*(n_c)] = c
  Y = (D @ A.T).T

  #noise generation
  noise = torch.zeros((n, M))
  for i in range(n):
    temp_magn = np.linalg.norm(Y[i])**2
    sd = temp_magn*(10**-(snr/10))
    temp_noise = torch.empty(40).normal_(mean=0,std=sd)
    noise[i, :] = temp_noise
  Y += noise

  return A, D, Y, true_labels


def get_scatter_terms(C:int, K:int, D: torch.tensor) -> tuple:
    """
   GOAL:
    Give scatter within Sw and scatter between Sb

   INPUT:
    C (int): number of classes
    K (int): number of atoms per sub-dictionary
    D (torch.tensor): learned dictionary

   OUTPUT:
    Sw (float): scatter within
    Sw (float): scatter between
    """
    Sw = 0
    Sb = 0
    m = torch.mean(D,1)

    for i in range(C):
        start_ndx = i * K
        end_ndx = (i+1) * K
        mc = torch.mean(D[:,start_ndx:end_ndx],1)
        Sb += torch.norm(mc - m, p=2)**2
        for j in range(start_ndx, end_ndx):
            Sw += torch.norm(D[:,j] - mc, p=2)**2

    Sb *= K
    return Sw, Sb


def penalty_term(C:int, K:int, D: torch.tensor, pen_term:str, alpha:float)->float:
  """
  GOAL:
  Generate a Fisher discriminant term, if Xu = True it uses the Xu formulation,
  otherwise the Sw/Sb thesis formulation

  INPUT:
    D (torch tensor): the dictionary with the network parameters
    added:
    pen_term (bool) - a boolean value that is True if we want to return the Xu et al version
         of the penalty term

  OUTPUT:
    if pen_term is "fisher_ratio":
      float: the ratio between the scatter_within and the scatter_between
    else:
      float: the difference between scatter_within and scatter_between
  """
  assert pen_term in ("fisher_ratio", "Xu")

  Sw, Sb = get_scatter_terms(C, K, D)
  #added a condition to determine which version to output
  if pen_term is "fisher_ratio":
    #talk about adding alpha, test on test set perhaps
    return Sw/Sb
  if pen_term is "Xu":
    return alpha*Sw - Sb

def cluster(data:torch.tensor, C:int) -> tuple:
  """
  GOAL:
  This function takes a randomly ordered matrix of training signals and re-orders them by cluster.
  It's necessary for the DDL class to compute class specific loss functions (which are then summed).

  INPUT:
  data (torch tensor): a matrix of randomly ordered signal, its last column must be the labels vector.
  C (int): the number of classes

  OUTPUT:
  cluster_list (list): a list of C torch tensors, each being a cluster of signals
  cluster_tensor (torch tensor): a single tensor containing all (training) signals ordered row-wise by label
  idx_num (list): a list of integers, each indicating how many signals are present in each cluster
  """
  cluster_list = [torch.zeros((0, data.shape[1]-1)) for c in range(C)]
  for i in range(data.shape[0]):
    cluster_list[int(data[i,-1])] = torch.cat((cluster_list[int(data[i,-1])], torch.unsqueeze(data[i,:-1], dim = 0)), 0)

  cluster_tensor = torch.zeros((data.shape[0], data.shape[1]-1))
  cluster_num = [0 for c in range(C)]
  for c in range(C):
    cluster_num[c] = cluster_list[c].shape[0]
    cluster_tensor[sum(cluster_num[:c]):sum(cluster_num[:c+1]), :] = cluster_list[c]

  return cluster_list, cluster_tensor, cluster_num

def dict_init(cluster_tensor:torch.tensor, C:int, K:int, rand_gen:bool = True) -> list:
  """
  GOAL:
  This function aims to initialize a dictionary from the empirical distribution of the given data.

  INPUT:
  cluster_tensor (torch tensor): tensor of ordered (by label) data;
  C (int): number of classes
  K (int): number of atoms per subdictionary
  rand_gen (bool): specifies whether to draw randomly from the tensor or if we should take the first K
  iterations

  OUTPUT:
  D_list (list): a list containing C subdictionaries (torch tensors)
  """

  D_list = []
  for c in range(C):
    if rand_gen is True:
      idx = torch.randperm(cluster_tensor.shape[0])[:K]
      D_list.append(cluster_tensor[idx, :])
    else:
      D_list.append(cluster_tensor[:K, :])
    D_list[-1] = D_list[-1].T
    D_list[-1] = preprocessing.normalize(D_list[-1], norm = "l2", axis = 0, copy = True)
    D_list[-1] = torch.from_numpy(D_list[-1]).float()
  return D_list

def matching(C:int, predicted_labels: torch.tensor, true_labels: torch.tensor) -> float:
  #match best labels
  pos = torch.zeros(C, dtype = int)
  #for each class of predicted labels, extract the relevant true labels
  for c in range(C):
    temp_true = []
    for i in range(predicted_labels.shape[0]):
      if int(predicted_labels[i]) == c:
        temp_true.append(int(true_labels[i]))
    #for each class of predicted labels, calculate which of the true classes would minimize cost
    #cost is calculated by counting the clustering mistakes per class of predicted labels
    cost = torch.zeros(C, dtype = int)
    for k in range(C):
      for i in range(len(temp_true)):
        if k != int(temp_true[i]):
          cost[k] += 1
    pos[c] = torch.argmin(cost)

  #create a new label tensor, in it shuffle the labels based on the item pos
  new_labels = torch.zeros((predicted_labels.shape[0]))
  for c in range(C):
    for i in range(predicted_labels.shape[0]):
      if int(predicted_labels[i]) == c:
        new_labels[i] = pos[c]

  #calculate accuracy on shuffled labels
  res = 0
  for j in range(predicted_labels.shape[0]):
    if int(new_labels[j]) == int(true_labels[j]):
      res += 1
  return res/predicted_labels.shape[0]
