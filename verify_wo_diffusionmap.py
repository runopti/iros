import scipy.io as sio
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import pickle
import os
import time
from sklearn.manifold import SpectralEmbedding
from sklearn.gaussian_process import GaussianProcessRegressor
import GPy
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import gp


# Step 0: split the data into train and test
datapath_base = "/data/yutaro/IROS/"
#datapath_big = "/data/yutaro/IROS/sim_data_full_v11_d4_m1.mat"
datapath_big = "/data/yutaro/IROS/sim_data_full_v13_d4_m1.mat"
#datapath_small = "/data/yutaro/IROS/sim_data_partial_v111_d4_m1.mat"
datapath_small = "/data/yutaro/IROS/sim_data_partial_v13_d4_m1.mat"

big = sio.loadmat(datapath_big)
small = sio.loadmat(datapath_small)


# Step 1: Normalize the training data
def compute_normalization_parameters(data):
    """
    Compute normalization parameters (min, max per feature)
    :param data: matrix with data organized by rows [num_samples x num_features]
    :return: min and max per feautre as row matrices of dimension [1 x num_variables]
    """
    min_param = np.min(data, axis=0)
    max_param = np.max(data, axis=0)
    return np.expand_dims(min_param, 0), np.expand_dims(max_param, 0)

def normalize_data_per_row(data, min_param, max_param):
    """
    Normalize a given matrix of data (samples must be organized per row)
    :param data: input data
    :param min_param: min (for each feature) for normalization
    :param max_param: max (for each feature) for normalization
    :return: normalized data, (data - min_param) / max_param - min_param
    """
    # sanity checks!
    assert len(data.shape) == 2, "Expected the input data to be a 2D matrix"
    assert data.shape[1] == min_param.shape[1], "Data - min_param size mismatch ({} vs {})".format(data.shape[1], min_param.shape[1])
    assert data.shape[1] == max_param.shape[1], "Data - max_param size mismatch ({} vs {})".format(data.shape[1], max_param.shape[1])

    # TODO. Complete. Replace the line below with code to whitten the data.
    normalized_data = np.divide(data - min_param, max_param - min_param)

    return normalized_data

def my_train_test_split(data, target, test_size, random_state):
    n_data = data.shape[0]
    n_test = int(n_data * test_size)
    return data[:(n_data-n_test),:], data[(n_data-n_test):,:], target[:(n_data-n_test),:], target[(n_data-n_test):,:]

def get_train_test_normalized(big):
    X_train, X_test, y_train, y_test = my_train_test_split(big['D'][:,:6], big['D'][:,6:], test_size=0.1, random_state=42)
    min_param, max_param = compute_normalization_parameters(X_train)
    min_param_y, max_param_y = compute_normalization_parameters(y_train)
    X_train_normalized = normalize_data_per_row(X_train, min_param, max_param)
    X_test_normalized = normalize_data_per_row(X_test, min_param, max_param)
    y_train_normalized = normalize_data_per_row(y_train, min_param_y, max_param_y)
    y_test_normalized = normalize_data_per_row(y_test, min_param_y, max_param_y)
    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized

def get_train_test_normalized(big):
    X_train, X_test, y_train, y_test = my_train_test_split(big['D'][:,:6], big['D'][:,6:], test_size=0.1, random_state=42)
    min_param, max_param = compute_normalization_parameters(X_train[:,:4])
    min_param_y, max_param_y = compute_normalization_parameters(y_train[:,:4])
    X_train_normalized = np.concatenate([normalize_data_per_row(X_train[:,:4], min_param, max_param), X_train[:,4:6]],axis=1)
    X_test_normalized = np.concatenate([normalize_data_per_row(X_test[:,:4], min_param, max_param), X_test[:,4:6]],axis=1)
    y_train_normalized = np.concatenate([normalize_data_per_row(y_train[:,:4], min_param_y, max_param_y), y_train[:,4:6]],axis=1)
    y_test_normalized = np.concatenate([normalize_data_per_row(y_test[:,:4], min_param_y, max_param_y),y_test[:,4:6]],axis=1)
    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized



def transition_model(x_query, X_train_normalized, y_train_normalized, k1=1000, m=3, k2=100):
    file_path = os.path.join(datapath_base, 'nbrs.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            nbrs = pickle.load(f)
    else:
        nbrs = NearestNeighbors(n_neighbors=k2, algorithm='ball_tree').fit(X_train_normalized)
        print("This is going to take about 15 min...")
        with open('nbrs.pkl', 'wb') as f:
            pickle.dump(nbrs, f)
    
    # Step1: Find K1 (=1000) nearest points in training data. Let the closest pair to $(x,a)$ be $(x_h, a_h)$
    distances, indices = nbrs.kneighbors(x_query.reshape(1,-1))
    #print(distances[0,0])
    #print(X_train_normalized.shape)
    X_selected = X_train_normalized[indices.reshape(-1), :]
    y_selected = y_train_normalized[indices.reshape(-1), :]
    
    # Step2: Diffusin map is created based on these 1000 points, which yields a reduced dimensional data
    #embedding = SpectralEmbedding(n_components=m) 
    #X_spectral = embedding.fit_transform(X_selected) 
    # y_spectral = embedding.fit_transform(y_selected) 
    
    # Step3: K2 (=100) closest points in the reduced dimensional data to the $(x_h, a_h)$ in the diffusion map is found
    #nbrs_diffusion = NearestNeighbors(n_neighbors=k2, algorithm='ball_tree').fit(X_spectral)
    #distances_diffusion, indices_diffusion = nbrs_diffusion.kneighbors(X_spectral[0, :].reshape(1, -1))

    # Step4: Perform GP regression on these 100 points.
    X_GP_input = X_selected
    y_GP_input = y_selected
    #print("X_GP_input.shape: {}".format(X_GP_input.shape))
    #print("y_GP_input.shape: {}".format(y_GP_input.shape))

    """
    #kernel = DotProduct() + WhiteKernel()
    y_GP_input = y_GP_input - X_GP_input[:, :4]
    gpr = GaussianProcessRegressor(kernel=None,random_state=0,n_restarts_optimizer=10).fit(X_GP_input, y_GP_input)
    ds_next = gpr.predict(x_query.reshape(1,-1)) 
    print("ds_next.shape: {}".format(ds_next.shape))
    s_next = x_query[0, :4] + ds_next
    print("s_next.shape: {}".format(s_next.shape))
    """
    ### Avishai's stuff
    
    ds_next = np.zeros((4,))
    std_next = np.zeros((4,))
    y_GP_input = y_GP_input - X_GP_input[:, :4]
    for i in range(4):
        gpr = gp.GaussianProcess(X_GP_input, y_GP_input[:,i], optimize = True, theta = None) 
        mm, vv = gpr.predict(x_query.reshape(-1))
        ds_next[i] = mm
        std_next[i] = np.sqrt(np.diag(vv))
    print("std_next: {}".format(std_next))
    s_next = x_query[0, :4] + ds_next.reshape(1,-1) #np.random.normal(ds_next, std_next).reshape(1,-1)   
    
    
    #kernel = GPy.kern.RBF(input_dim=6)
    #model = GPy.models.GPRegression(X_GP_input, y_GP_input, kernel)
    #model.optimize()
    #y_pred = model.predict(x_query.reshape(1,-1))
    #print(y_pred)
    #print(s_next)
    return s_next
    #return y_pred[0]


def calc_mse(X_test_normalized, y_test_normalized, X_train_normalized, y_train_normalized):
    mse = 0
    for i in range(X_test_normalized.shape[0]):
        y_new = transition_model(X_test_normalized[i,:].reshape(1,-1), X_train_normalized, y_train_normalized)
        #print(y_test_normalized[i, :].reshape(1,-1).shape)
        #print(y_new.shape)
        mse += np.mean(np.linalg.norm(y_test_normalized[i, :].reshape(1,-1) - y_new)**2)
        #print(mse)
    return mse / X_test_normalized.shape[0]

def calc_mse_traj(X_test_normalized, y_test_normalized, X_train_normalized, y_train_normalized):
    mse_list = []
    k = 100
    prev = X_test_normalized[0,:].reshape(1,-1)
    traj_list = [prev]
    for i in range(0, k):
        y_new = transition_model(prev, X_train_normalized, y_train_normalized)
        mse = np.mean(np.linalg.norm(y_test_normalized[i, :].reshape(1,-1) - y_new)**2)
        mse_list.append(mse)
        print("y_new.shape: {}".format(y_new.shape)) # (1,4)
        traj_list.append(y_new)
        print(X_test_normalized[i+1, 4:6].reshape(1,-1))
        print(X_test_normalized[i+1, 4:6].reshape(1,-1).shape)
        prev = np.concatenate([y_new, X_test_normalized[i+1, 4:6].reshape(1,-1)], axis=1)
        #prev = np.concatenate([np.concatenate([y_new, X_test_normalized[i+1, 4].reshape(1,-1)], axis=1), X_test_normalized[i+1, 5].reshape(1,-1)], axis=1)
    return mse_list, traj_list
    
    
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = get_train_test_normalized(big)
tic = time.time()
#mse_big = calc_mse(X_test_normalized[:100,:], y_test_normalized[:100,:], X_train_normalized, y_train_normalized)
#print(mse_big)
mse_list_big, traj_list_big = calc_mse_traj(X_train_normalized[200:400,:], y_train_normalized[200:400,:], X_train_normalized, y_train_normalized)
import pickle
with open('mse_list_big.pkl', 'wb') as f:
    pickle.dump(mse_list_big, f)
with open('traj_list_big.pkl', 'wb') as f:
    pickle.dump(traj_list_big, f)
with open('traj_ground_truth.pkl', 'wb') as f:
    pickle.dump(X_train_normalized[200:400,:], f)
toc = time.time()
print("time: {}".format(tic - toc))

_, X_test_normalized, _, y_test_normalized = get_train_test_normalized(small)
tic = time.time()
#print(calc_mse(X_test_normalized[:100,:], y_test_normalized[:100,:], X_train_normalized, y_train_normalized))
mse_list_small, traj_list_small = calc_mse_traj(X_test_normalized[:200,:], y_test_normalized[:200,:], X_train_normalized, y_train_normalized)
with open('mse_list_small.pkl', 'wb') as f:
    pickle.dump(mse_list_small, f)
with open('traj_list_small.pkl', 'wb') as f:
    pickle.dump(traj_list_small, f)
toc = time.time()
print("time: {}".format(tic - toc))