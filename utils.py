import numbers
import math
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad


import torch
import torch.nn.functional as F
import time
import sys
from torch.autograd import Variable
import networkx as nx

LogField = namedtuple('LogField', ('data', 'plot', 'x_axis', 'divide_by'))

def calculate_max_hops_diagonal(grid_size_x, grid_size_y, base_proximity_threshold):
    """Calculate max_hops based on the diagonal of the grid."""
    diagonal = math.sqrt(grid_size_x**2 + grid_size_y**2)
    return max(1, min(grid_size_x, grid_size_y, math.ceil(diagonal / base_proximity_threshold)))

def calculate_max_hops_average(grid_size_x, grid_size_y, base_proximity_threshold):
    """Calculate max_hops based on the average of grid dimensions."""
    avg_size = (grid_size_x + grid_size_y) / 2
    return max(1, min(grid_size_x, grid_size_y, math.ceil(avg_size / base_proximity_threshold)))

def calculate_max_hops_adaptive(grid_size_x, grid_size_y, nagents):
    """Calculate max_hops adaptively based on grid size and number of agents."""
    grid_area = grid_size_x * grid_size_y
    agent_density = nagents / grid_area
    base_hops = min(grid_size_x, grid_size_y) // 2
    return max(1, min(base_hops, math.ceil(base_hops * (1 + agent_density))))

def generate_noise(signal, noise_type, noise_lvl, mean=0.0):
    """
    Noise generation method. If noise level=0.0 then no noise is applied

    Arguments:
    signal: vector of shape (batch_size, nagents, hid_size)
    noise_type: type of noise to add, such as 'gaussian', 'uniform' etc.
    noise_lvl: noise level, a positive scalar
    mean: mean of the noise distribution, defaults to 0.0
    
    Mean:
    For realistic communication noise, it's often better to avoid biases by setting mean=0.0.
    Observation noise might have a non-zero mean if there's a systematic bias in the observations
    STD:
    The std of the noise should be proportional to the signal-to-noise ratio (SNR), how the signal is
    corrupted by noise. A higher SNR means that the signal is more distinguishable from the noise, 
    and vice versa. 
    
    Gaussian noise: For Gaussian noise, the std parameter controls the spread of the noise distribution. 
    Lower values lead to less variability, while higher values introduce larger deviations.
    Uniform noise: the range of the noise distribution is determined by the low and high parameters.
    
    Gaussian noise, mean = 0.0, std = 0.3162 (for SNR = 10 dB with signal power = 1)
    For uniform distr, low=0.0, high= 1.0954 (for SNR = 10 dB with signal power = 1)
    
    Generate 

    Returns:
    signal: vector of shape (batch_size, nagents, hid_size)
    
    """
    
    # Dictionary of noise functions
    noise_functions = {
        'gaussian': torch.normal(mean=mean, std=noise_lvl, size=signal.shape),
        # Uses lambda function to create uniform noise 
        'uniform' : torch.distributions.Uniform(low=mean, high=noise_lvl).sample(signal.shape)
        # OR 'uniform' : torch.random(size=signal.shape) # returns a sample of size in the [0,1] interval
        
    }
       
    # clipped_noise = torch.clamp(noise_functions[noise_type], min=-1.0, max=1.0)
    
    # Add noise to the comm signal
    noise = noise_functions[noise_type]
    signal += noise
            
    return signal

def emulate_comm_quality(comm, noise_type, mean=0.0): 
    """
    Emulate fluctuating communication quality by adding noise, dropping, or delaying messages.

    Arguments:
    comm: communication vector of shape (batch_size, nagents, hid_size)
    noise_type: type of noise to add, such as 'gaussian', 'uniform' etc.
    
    Returns:
    A noisy, dropped, delayed or jumbled communication vector of the same shape as comm
    """
    
    # batch_size, n, _ = comm.size()
    
    # Generate random value for noise_level and probabilities for adverse comm conditions
    noise_level = np.random.uniform(low=0.0, high=0.5) # Keep noise values low
    drop_prob = np.random.uniform(low=0.0, high=1.0)
    delay_prob = np.random.uniform(low=0.0, high=1.0)
    jumble_prob = np.random.uniform(low=0.0, high=1.0)
    
    ## Add noise
    if noise_level > 0.3:
        comm = generate_noise(comm, noise_type, noise_level, mean)
        
    ## Drop parts of messages
    if drop_prob > 0.1:
        # Generate tensor with probs (uniform [0,1]) of shape comm
        # mask = torch.bernoulli(torch.full((batch_size, n, _), drop_prob) # high drop rate
        mask = torch.bernoulli(torch.rand(comm.shape)) # Mask tensor of the same shape as comm
        comm *= mask # Apply mask
        
    ## Drop all messages to agent/s randomly
    if drop_prob > np.random.rand():
        mask = torch.full((comm.size(1),),drop_prob) # generate probs on which to drop communication
        mask = torch.bernoulli(mask).unsqueeze(1)
        mask = mask.expand_as(comm)
        comm *= mask
         
    ## Jumble messages randomly
    # TODO: jumble messages
    
    ## Delay messages randomly
    # TODO: delay messages bettween comm rounds
        
    ## Recalculate the communication vector after noise and message manipulation?
    
    return comm


def drop_messages(comm, drop_prob_whole, drop_prob_part):
    """
    Dopping whole messages and/or parts of messages based on a probability.
    If both drop_prob_whole, drop_prob_part are non-zero, both types of drops may be applied.
    if either are zero, only one type of drop may be applied.

    Arguments:
    comm: communication vector of shape (batch_size, nagents, hid_size)
    drop_prob_part: fixed probability to drop parts of the message
    apply_both_prob: probability to apply both whole and part message drops
    
    Returns:
    A communication vector with dropped messages, of the same shape as comm
    """
    
    drop_prob = np.random.uniform(low=0.0, high=1.0)
    
    
    ## Drop parts of messages
    if drop_prob_part > 0 and drop_prob < drop_prob_part:
        # Generate tensor with probs (uniform [0,1]) of shape comm
        # mask = torch.bernoulli(torch.full((batch_size, n, _), drop_prob) # high drop rate
        mask = torch.bernoulli(torch.rand(comm.shape)) # Mask tensor of the same shape as comm
        comm *= mask # Apply mask
        
    ## Drop all messages to agent/s randomly
    if drop_prob_whole > 0 and drop_prob < drop_prob_whole:
        mask = torch.full((comm.size(1),),drop_prob) # generate probs on which to drop communication
        mask = torch.bernoulli(mask).unsqueeze(1)
        mask = mask.expand_as(comm)
        comm *= mask
    
    return comm

def jumble_2_messages(comm, jumble_prob):
    """
    Jumbles messages between 2 agents based on a probability.
    
    Arguments:
    comm: communication vector of shape (batch_size, nagents, hid_size)
    jumble_prob: fixed probability to jumble messages between agents
    
    Returns:
    A communication vector with jumbled messages, of the same shape as comm
    """
    batch_size, nagents, hid_size = comm.shape
    
    # Jumble messages between agents
    if np.random.uniform(low=0.0, high=1.0) < jumble_prob:
        # Choose two agents at random to jumble their messages
        agents_to_jumble = np.random.choice(nagents, size=2, replace=False)
        agent1, agent2 = agents_to_jumble
        
        # Store the original messages
        comm_agent1 = comm[:, agent1, :].clone()
        comm_agent2 = comm[:, agent2, :].clone()
        
        # Sum the messages of the chosen agents
        comm[:, agent1, :] += comm[:, agent2, :]
        comm[:, agent2, :] += comm[:, agent1, :]
       
      
        # Optionaly introduce some randomness by adding noise
        # noise = torch.randn_like(comm[:, agent1, :]) * noise_level
        # comm[:, agent1, :] += noise
        # comm[:, agent2, :] += noise
        
        # Set original messages to zero or a small value, if needed
        # comm[:, agent1, :] *= 0
        # comm[:, agent2, :] *= 0
  
    return comm

def jumble_messages_norm(comm, jumble_prob):
    """
    Jumbles messages between a random number of agents based on a probability.
    it then averages the jumbled signal (based on the number of agents)
    and distributes the average, every agent receives the same averaged signal
    
    averaging: 
    1. normalizing the signal in magnitude   
    2. Can also be seen as simulating a scenario where agents share a faulty communication channel, 
    leading to consensus-like behavior among a subset of agents.
    
    Arguments:
    comm: communication vector of shape (batch_size, nagents, hid_size) or (nagents, hid_size)
    jumble_prob: fixed probability to jumble messages between agents
    
    Returns:
    A communication vector with jumbled messages, of the same shape as comm
    """   
    # assuming last two dimensions are (nagents, hid_size) everything else is batch-like
    nagents, hid_size = comm.shape[-2:]
    
    # Jumble messages between agents
    if np.random.uniform(low=0.0, high=1.0) < jumble_prob:
        # Choose a random number of agents to jumble their messages
        num_agents_to_jumble = np.random.randint(2, nagents + 1)
        agents_to_jumble = np.random.choice(nagents, size=num_agents_to_jumble, replace=False)
        
        # Sum the messages of the selected agents
        # summed_messages = comm[agents_to_jumble, :].sum(dim=1, keepdim=True)
        summed_messages = comm[..., agents_to_jumble, :].sum(dim=-2, keepdim=True)
        
        # Distribute the summed messages equally (or with some noise) among the selected agents
        # comm[agents_to_jumble, :] = summed_messages / num_agents_to_jumble
        comm[..., agents_to_jumble, :] = summed_messages / num_agents_to_jumble
        
        # Optionally maybe add some noise to each agents message after jumbling?
        # noise = torch.randn_like(comm[agents_to_jumble, :]) * noise_level
        # comm[agents_to_jumble, :] += noise
        
        #comm.unsqueeze(0)

    return comm

def jumble_messages(comm, jumble_prob):
    """
    Jumbles messages between a random number of agents based on a probability.
    sums the jumbled signals and distributes the sum. the resulting messages are
    incresed in magnitude (no normalisation/averaging)
    Every agent in the selected group receives the same summed signal.
    
    Arguments:
        comm: communication vector of shape (batch_size, nagents, hid_size) or (nagents, hid_size)
        jumble_prob: fixed probability to jumble messages between agents
    
    Returns:
        A communication vector with jumbled messages, of the same shape as comm
    """
    # assuming the last two dimensions are (nagents, hid_size)
    nagents, hid_size = comm.shape[-2:]
    
    if np.random.uniform(low=0.0, high=1.0) < jumble_prob:
        # Choose a random number of agents to jumble their messages
        num_agents_to_jumble = np.random.randint(2, nagents + 1)
        agents_to_jumble = np.random.choice(nagents, size=num_agents_to_jumble, replace=False)
        
        # Sum the messages of the selected agents without averaging
        summed_messages = comm[..., agents_to_jumble, :].sum(dim=-2, keepdim=True)
        
        # Replace the messages for the selected agents with the summed message
        comm[..., agents_to_jumble, :] = summed_messages
        
        # Optionally maybe add some noise to each agents message after jumbling?
        # noise = torch.randn_like(comm[agents_to_jumble, :]) * noise_level
        # comm[agents_to_jumble, :] += noise
        
    return comm

def misroute_messages(comm, misroute_prob):
    """
    Misrouting messages among randomly selected agents by permuting their communication tensors.
    
    Args:
        comm (torch.Tensor): Communication tensor of shape [..., nagents, hid_size]
        missroute_prob (float): Probability of jumbling occurring (0.0 to 1.0)
    
    Returns:
        torch.Tensor: Modified communication tensor
    """
    # Last two dimensions are (nagents, hid_size), rest are batch-like
    nagents, hid_size = comm.shape[-2:]
    
    # Jumble messages with given probability
    if np.random.uniform(low=0.0, high=1.0) < misroute_prob:
        # Choose a random number of agents to jumble (at least 2)
        num_agents_to_missroute = np.random.randint(2, nagents + 1)
        agents_to_missroute = np.random.choice(nagents, size=num_agents_to_missroute, replace=False)
        agents_to_missroute = torch.from_numpy(agents_to_missroute).to(comm.device)
        
        # Generate a random permutation of the selected agents' indices
        perm = torch.randperm(num_agents_to_missroute, device=comm.device)
        
        # Permute messages among the selected agents
        comm[..., agents_to_missroute, :] = comm[..., agents_to_missroute[perm], :]
    
    return comm

def comm_delay_oldest(comm, hop_delay_buffer, delay_prob):
    
    # Apply delay for each communication hop
    if np.random.uniform(low=0.0, high=1.0) < delay_prob and len(hop_delay_buffer) > 0:
        delayed_comm = hop_delay_buffer[0]  # Use the oldest delayed comm tensor
    else:
        delayed_comm = comm  # No delay, use current comm

    # Store the current comm tensor for future hops
    hop_delay_buffer.append(comm)    
    
    return comm

def comm_delay_n_hops(comm, hop_delay_buffer, max_hop_delay, delay_prob):
        """
        Applies message delays using a buffer.
        
        Arguments:
        comm -- the current communication tensor (nagents x hiddensize)
        hop_delay_buffer -- the deque storing previous communication states
        max_hop_delay -- the maximum delay in hops
        delay_prob: fixed probability to delay messages 
        
        Returns:
        delayed_comm -- the delayed communication tensor (nagents x hiddensize)
        """
        
        # Store the current comm for future delayed use - SHOULD THIS BE AT THE BEGGINING?
        hop_delay_buffer.append(comm.clone())
        
        # Check if the buffer has enough previous states to apply the delay
        if np.random.uniform(low=0.0, high=1.0) < delay_prob and len(hop_delay_buffer) >= max_hop_delay:
            # Retrieve the comm from max_hop_delay steps ago
            delayed_comm = hop_delay_buffer[-max_hop_delay]
        else:
            # If the buffer doesn't have enough delayed states, use the current comm
            delayed_comm = comm.clone()

        # # Store the current comm for future delayed use - SHOULD THIS BE AT THE BEGGINING?
        # hop_delay_buffer.append(comm.clone())
        
        return delayed_comm


def calc_comm_entropy(comm):
      """
      Calculates the entropy of a message tensor.
      
      Uses the entropy formula for discrete probability distributions, 
      H(x) = (-\sum p(x) \log p(x) ), where ( p(x) ) is the probability of a message 
      
      Args:
          message_vector: A tensor with shape (batch_size, num_agents, message_dim).
    
      Returns:
          A NumPy array with shape (batch_size,) containing comm entropy.
      """
    
      # Calculate the probabilities for each message tensor dimension
      probs = F.softmax(comm, dim=-1)
      
      # Calculate the log probabilities
      log_probs = torch.log(probs)
    
      # Calculate the entropy of messages for each agent
      entropy = -torch.sum(probs * log_probs, dim=-1)
   
      return entropy.numpy()
  
def calc_comm_entropy_discr(comm, bins):
    """
    Calculates the entropy of discretised continuous-valued messages.

    Args:
        comm: A tensor with shape (batch_size, num_agents, message_dim) representing the continuous-valued messages.
        bins: The number of bins to use for discretisation.

    Returns:
        A NumPy array with shape (batch_size,) containing the communication entropy for each batch.
    """
    
    # Flatten the tensor to a 2D array with shape (num_samples, num_variables)
    samples = comm.view(-1, comm.size(-1)).cpu().numpy()

    min_val = samples.min()
    max_val = samples.max()

    # Create bin edges
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Discretise the continuous values
    digitized_samples = np.digitize(samples, bin_edges) - 1

    # Count the occurrences of each bin
    bin_counts = np.array([np.sum(digitized_samples == i) for i in range(bins)])

    # Normalise the counts to get a probability distribution
    probs = bin_counts / np.sum(bin_counts)

    # Calculate the log probabilities, avoiding log(0) by adding a small constant
    log_probs = np.log(probs + 1e-10)

    # Calculate the entropy of the discretised messages
    entropy = -np.sum(probs * log_probs)

    return entropy
  
def differential_entropy(comm):
    """
    Calculates the differential entropy,  
    Assuming PDF is not known, estimating it from input data using Kenrel Density Estimation (KDE)
    
    Uses the differential entropy formula for continuous distributions, 
    a continuous random variable ( X ) with a probability density function ( p(x) ) over a continuous interval:
    H(X)=−∫p(x)logp(x)dx
    
    Args:
        Comm: tensor of shape batch_size, num_agents, message_dim
    
    Returns:
        The differential entropy
    """
    
    # Flatten the comm tensor to a 1D array for KDE
    comm_flattened = comm.flatten().numpy()
        
    # Estimates the probability density function using Kernel Density Estimation
    pdf = gaussian_kde(comm_flattened) 
   
    a = comm.min() #Lower bound of the integral
    b = comm.max() #Upper bound of the integral
    
    entropy, _ = quad(lambda x: -pdf(x) * np.log(pdf(x)), a, b)
    
    return entropy
  
def calc_diff_entropy_gaussian(comm):
    """
    Calculates the differential entropy of a continuous message tensor 
    Assuming a Gaussian distribution .
    
    The differential entropy formula for continuous distributions,
    h(x) = -∫ p(x) log p(x) dx, where p(x) is the probability density function of the message.
    
    The differential entropy of a Gaussian random n-vector (X)
    h(f) = 0.5 * log((2 * pi * e)^n * det(R))
    
    Args:
        comm: A tensor with shape (batch_size, num_agents, message_dim).
    
    Returns:
        A NumPy array with shape (batch_size,) containing diff entropy.
    """
    
    # n = comm.shape[-1]
    n = comm.shape[0]
    
    # Assuming a Gaussian distribution 
    # Flatten the tensor to a vector with shape (num_samples, num_variables)
    comm = comm.view(-1, comm.size(-1)) #.cpu().numpy()
   
    # Calculate the covariance matrix for each batch
    # The covariance matrix R is the matrix of covariances between each pair of the elements of the vector
    R = torch.cov(comm)
   
    # Calculate the determinant of the covariance matrix
    det_R = torch.det(R)
    
    if det_R < 1e-6:
        return torch.tensor(0.0).numpy()
    else:
        # Calculate the differential entropy using the formula for multivariate Gaussian distribution
        # h(f) = 0.5 * log((2 * pi * e)^n * det(R))
        differential_entropy = 0.5 * torch.log((2 * np.pi * np.e)**n * det_R)
        return differential_entropy.numpy()
        
# def merge_stat(src, dest):
#     for k, v in src.items():
#         if not k in dest:
#             dest[k] = v
#         elif isinstance(v, numbers.Number):
#             dest[k] = dest.get(k, 0) + v
#         elif isinstance(v, np.ndarray): # for rewards in case of multi-agent
#             dest[k] = dest.get(k, 0) + v
#         else:
#             if isinstance(dest[k], list) and isinstance(v, list):
#                 dest[k].extend(v)
#             elif isinstance(dest[k], list):
#                 dest[k].append(v)
#             else:
#                 dest[k] = [dest[k], v]

def merge_stat(src, dest):
    for k, v in src.items():
        if k == 'hidden_s_comm':  # Check for the specific key
            if k not in dest:
                dest[k] = v
            else:
                dest[k] = np.vstack((dest[k], v))
        elif not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray):  # for rewards in case of multi-agent
            dest[k] = dest.get(k, 0) + v
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]
                
def visualize_graph(adj):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j].item() > 0:
                G.add_edge(i, j)
    
    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    plt.title("Graph Visualization based on Adjacency Matrix")
    plt.show()

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    return log_prob

def multinomials_log_densities(actions, log_probs):
    log_prob = [0] * len(log_probs)
    for i in range(len(log_probs)):
        log_prob[i] += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    log_prob = torch.cat(log_prob, dim=-1)
    return log_prob

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

class Timer:
    def __init__(self, msg, sync=False):
        self.msg = msg
        self.sync = sync

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("{}: {} s".format(self.msg, self.interval))

def pca_numpy(X, k=2):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Project the data onto the first k principal components
    X_pca = np.dot(X_centered, U[:, :k])
    
    return X_pca

def pca(X, k=2):
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

def pca_normalized(X, k=2):
    # Normalize the data
    X_min = torch.min(X, 0)[0]
    X_max = torch.max(X, 0)[0]
    X_norm = (X - X_min) / (X_max - X_min)
    
    # Center the data
    X_mean = torch.mean(X_norm, 0)
    X_centered = X_norm - X_mean.expand_as(X_norm)
    
    # Apply SVD
    U, S, V = torch.svd(torch.t(X_centered))
    return torch.mm(X_centered, U[:, :k])

def init_args_for_env(parser):
    env_dict = {
        'levers': 'Levers-v0',
        'number_pairs': 'NumberPairs-v0',
        'predator_prey': 'PredatorPrey-v0',
        'traffic_junction': 'TrafficJunction-v0',
        'starcraft': 'StarCraftWrapper-v0'
    }

    args = sys.argv
    env_name = None
    for index, item in enumerate(args):
        if item == '--env_name':
            env_name = args[index + 1]

    if not env_name or env_name not in env_dict:
        return

    import gym
    import ic3net_envs

    if env_name == 'starcraft':
        import gym_starcraft

    env = gym.make(env_dict[env_name])
    env.init_args(parser)

def display_models(list_models):
    print('='*100)
    print('Model log:\n')
    for model in list_models:
        print(model)
    print('='*100 + '\n')
