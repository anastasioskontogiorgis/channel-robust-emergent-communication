import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from utils import emulate_comm_quality, comm_delay_n_hops, jumble_messages, jumble_messages_norm, generate_noise, drop_messages
from gnn_layers import GraphAttention
from collections import deque

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args, num_inputs):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
            num_inputs {number} -- Environment observation dimension for agents
        """

        self.args = args
        self.name = 'magic'
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        # self.args.max_hop_delay = 2 # original implementation of model has 2 comm rounds
        # Persistent buffer for episode-level delay
        self.hop_delay_buffer = deque(maxlen=self.args.delay_step)  # e.g., maxlen=2
        # ... other initializations ...
        
        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
        
        # initialize the gat encoder for the --- Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.num_inputs, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the --- Sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)
        # self.message_decoder_delay = nn.Linear(args.gat_hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)
        
    def reset_delay_buffer(self):
        """Reset the delay buffer at the start of each episode."""
        self.hop_delay_buffer.clear()

    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * num_inputs]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents
        obs, extras = x

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)  # ------------ Encoder
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)
        
        # ----- LSTM obs temporality 
        # hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state.squeeze(), cell_state.squeeze())) # added squeeze in the tuple
        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))
        
        # comm: [n * hid_size]
        comm = hidden_state # - COMMUNICATION IN
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
            
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # ----- Sub-scheduler 1 (consists of a gat layer and hard attention (MLP)
        # if args.first_graph_complete == True, sub-scheduler 1 will be disabled
        if not self.args.first_graph_complete: # not fixed communication all to all but need to learn the whom to commuicate
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask) # returns the agnents (adj) that are alive 
                encoded_state1 = self.gat_encoder(comm, adj_complete) # gat layer (input messages and adj of alive agents)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        # sub-processor 1 - COMMUNICATION OUT
        comm = F.elu(self.sub_processor1(comm, adj1))  # this layer outputs comm of 32 dims
        delayed_comm = comm
        
        # ----- Sub-scheduler 2
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    encoded_state2 = encoded_state1
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)
            
        # sub-processor 2 - COMMUNICATION OUT
        comm = self.sub_processor2(comm, adj2)
        
        # mask communication to dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        
        ############# APPLY Communication contraints ################################
        if self.args.comm_constraints == 'simple' and self.args.noise_level > 0.0:
            comm = generate_noise(comm,
                        self.args.noise_type, # Default Gaussian
                        self.args.noise_level,
                        self.args.noise_mean)
        elif self.args.comm_constraints == 'complex':
            comm = emulate_comm_quality(comm, 
                                  self.args.noise_type, 
                                  self.args.noise_mean)
        elif self.args.comm_constraints == 'drops':
            # Apply message drop
            comm = drop_messages(comm.unsqueeze(0),
                              self.args.drop_prob_whole, 
                              self.args.drop_prob_part)
    
            # Reshape back to [nagents, hid_size]
            comm = comm.squeeze(0)
        elif self.args.comm_constraints == 'jumble':
            comm = jumble_messages(comm, 
                                   self.args.jumble_prob)
        elif self.args.comm_constraints == 'delay':
            comm = comm_delay_n_hops(comm, 
                                      self.hop_delay_buffer, 
                                      self.args.delay_step, 
                                      self.args.delay_prob)
        #############################################################################
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)
        
        value_head = self.value_head(torch.cat((hidden_state, comm.squeeze()), dim=-1))  # added comm.squeeze()
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    # generate final G that defines communication graph between agents 
    #TODO: Make dynamic
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:  # direcrted communication A->B then not B->A
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else: # could no targeting in the direction of the flow (A->B and B->A ) we take the average, symetric
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    # get alive agents
    #TODO: Add here the functionality for teams and extended neighborhood comm/situational awareness through hops?
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose  # get the currect adj of agents that are alive by mult with mask
        
        return adj
