import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
from models import *
from commNet import CommNetMLP
from ga_comm import GACommNetMLP
from tar_comm import TarCommNetMLP
# from magic_original import MAGIC # the original magic without noise functionality added
from magic import MAGIC # orginal magic WITH noise funcitonality added
# from magic_delay_v1 import MAGIC
from trainer import Trainer

sys.path.append("..") 
import data
from utils import *
from action_utils import parse_action_args
from multi_processing import MultiProcessTrainer
# import gym

# gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')

# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=1000, type=int,  # ------------------------ num of Epochs
                    help='number of training epochs')       
parser.add_argument('--epoch_size', type=int, default=10, # ------------------------ num of Episodes(10)nagent
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1, # ------------------------ num of Processes (1)
                    help='How many processes to run, 1 for testing')

# general for models
parser.add_argument('--hid_size', default=128, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', action='store_true', default=True,
                    help='make the model recurrent in time')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')

# environment
parser.add_argument('--env_name', default="traffic_junction",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int, # 20 for easy
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')

# env args - added for debuging from main inside IDE
parser.add_argument('--dim', default=6,   # 6 for easy
                    help='Dimension of box (i.e length of road)')
parser.add_argument('--vision', type=int, default=1, # ------------------------------------- Vision
                 help="Vision of car")
parser.add_argument('--add_rate_min', type=float, default=0.1, # --------- easy (ic3net) 0.1 | med 0.05 | hard  0.02 
                 help="rate at which to add car (till curr. start)")
parser.add_argument('--add_rate_max', type=float, default=0.3, # --------- easy  0.3 | med 0.02 | hard 0.05
                 help=" max rate at which to add car")
parser.add_argument('--curr_start', type=float, default=125, # ---------------------- Harder after epoch: (testing) 125 | training (easy/med) 250, hard 375
                 help="start making harder after this many epochs [0]")
parser.add_argument('--curr_end', type=float, default=625,  # ---------------------- Hardest after epoch: (tesing) 625 | training (easy/med) 1250, hard 1875
                 help="when to make the game hardest [0]")
parser.add_argument('--difficulty', type=str, default='easy',
                 help="Difficulty level, easy|medium|hard")
parser.add_argument('--vocab_type', type=str, default='bool',
                 help="Type of location vector to use, bool|scalar")

# other
parser.add_argument('--mode',  default="test", type=str,  # ---------------------- Mode [Test/Train]
                    help='Choose mode test or train')
parser.add_argument('--plot', action='store_true', default=False, # ---------------  PLOT
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', action="store_true", default=False, # --------------- Save Model [on/off]
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='./saved/traffic_junction/ic3net/run1/model.pt', type=str, # ------------ Load (./saved/traffic_junction/magic/run17/model.pt)
                    help='load the model')
parser.add_argument('--export', action='store_true', default=True, # ------------------ Export stats file [on/off]
                    
                    help='Export model training stats')
parser.add_argument('--display', action="store_true", default=False, # ------------ Display env
                    help='Display environment state') 
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# Model specific args - ------------------------------------------------------- Baseline models
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,  # RELEASE CHANGE: was default=True (could not be disabled from CLI); scripts now pass --ic3net explicitly
                    help="enable ic3net model")
parser.add_argument('--tarcomm', action='store_true', default=False,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False,
                    help="enable gacomm model")
parser.add_argument('--magic', action='store_true', default=False,
                    help="enable magic model")
parser.add_argument('--nagents', type=int, default=5,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_mask_zero', action='store_true', default=False, # -------------- Comm [On/Off]
                    help="Whether communication should be there")
parser.add_argument('--comm_passes', type=int, default=2, # -------------------------------- Comm Passes
                    help="Total (and Max) number of comm passes per step over the model")
parser.add_argument('--mean_ratio', default=1.0, type=float, # --------------------------------- Coop level (1.0 for comnet)
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='LSTM', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',  # ------------ Hard attention [on/off]
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')
parser.add_argument('--directed', action='store_true', default=True,
                    help='whether the communication graph is directed')
parser.add_argument('--self_loop_type1', default=1, type=int,  # was 2
                    help='self loop type in the first gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--self_loop_type2', default=1, type=int,   # was 2
                    help='self loop type in the second gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gat_num_heads', default=4, type=int, #medium 4
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=32, type=int,    # was 64
                    help='hidden size of one head in gat')
parser.add_argument('--ge_num_heads', default=4, type=int,
                    help='number of heads in the gat encoder')
parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                    help='whether normalize the coefficients in the first gat layer of the message processor')
parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the second gat layer of the message proccessor')
parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)')
parser.add_argument('--use_gat_encoder', action='store_true', default=True,
                    help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gat encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=True,   # was false
                    help='whether the first communication graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=True,
                    help='whether the second communication graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False, # was false
                    help='whether learn a new communication graph at the second round of communication')
parser.add_argument('--message_encoder', action='store_true', default=False,
                    help='whether use the message encoder')
parser.add_argument('--message_decoder', action='store_true', default=True,  # was false
                    help='whether use the message decoder')

# Comm contraints args
parser.add_argument('--noise_type', default='',
                    help='The type of noise to use: gaussian, uniform')
parser.add_argument('--noise_level', default=0.0, type=float, # ----------------------- Noise level
                    help='Scaling factor to adjust noise level, 0 no noise')
parser.add_argument('--noise_mean', default=0.0, type=float,
                    help='Mean for generating a distribution to sample noise, 0 for no bias')
parser.add_argument('--noise_cliping', default=False, action='store_true',
                    help='Whether to clip inside the [-1,1] range')
parser.add_argument('--comm_constraints', default='', type=str, # ------------------ Comm Constraint Type 
                    help='Which comm constraint to activate: simple (only noise), drops, complex, jumble, delay')
parser.add_argument('--drop_prob_whole', default=0.0, type=float,
                    help='Probability for a whole message to be droped')
parser.add_argument('--drop_prob_part', default=0.0, type=float,
                    help='Probability for part of a message to be droped')
parser.add_argument('--jumble_prob', default=0.4, type=float, 
                    help='Probability for messages to be jumbled')
parser.add_argument('--delay_prob', default=0.0, type=float, 
                    help='Probability for messages being delayed at each comm round, or env step for Ga-comm')
parser.add_argument('--delay_step', default=2, type=int, 
                    help='Number of steps for message delay at each set of comm rounds')


init_args_for_env(parser)
args = parser.parse_args()

# Make sure no noise is enabled during training
if args.mode == 'train':
    args.comm_constraints = ''
    args.noise_level = 0.0

# Set processes to 1 for testing mode
if args.mode == 'test':
    args.nprocesses = 1
    
if args.tarcomm:
    args.ic3net = 1 # use tarcom with IC3Net (hard attn)

# uses commnet base model with hard attention and indivisualised rewards
if args.ic3net: 
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0
    
if args.gacomm:
    args.commnet = 1
    args.mean_ratio = 0
    
if args.magic:
    args.mean_ratio = 0
    
# Hard attention
# if args.hard_attn and args.commnet:
#     # add comm_action as last dim in actions
#     args.num_actions = [*args.num_actions, 2]
#     args.dim_actions = env.dim_actions + 1
    
# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs



parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print('-------- All Args -----------')
print(args)
print('-----------------------------')


## Policy Net
if args.gacomm:
    policy_net = GACommNetMLP(args, num_inputs)
elif args.commnet:
    if args.tarcomm:
        policy_net = TarCommNetMLP(args, num_inputs)
    else:
        policy_net = CommNetMLP(args, num_inputs) # for IC3Net hard_attn and mean_ratio=0
elif args.magic:
        policy_net = MAGIC(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)
        
if not args.display:
    display_models([policy_net])

## Specific Args for session    
args_to_print = ['mode' , 'commnet' , 'ic3net' , 'tarcomm' , 'gacomm', 'magic' ,'hard_attn' , 'mean_ratio', 'hid_size' , 'recurrent' , 'rnn_type' ,
                 'num_epochs' , 'epoch_size' , 'batch_size', 'lrate' , 'gamma' , 'value_coeff', 'seed' , 'nagents' , 'add_rate_min', 'add_rate_max',
                 'curr_start', 'curr_end', 'difficulty','comm_constraints', 'env_name',  'display' , 'max_steps' , 'comm_passes' , 
                 'noise_level', 'noise_type' , 'nprocesses' , 'plot' , 'save' , 'load', 'export' ]

for arg in args_to_print:
    print(f'{arg}: {getattr(args, arg)}')


# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()    
    
log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)

## Save
if args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
elif args.magic:
    model_dir = Path('./saved') / args.env_name / 'magic'
else:
    model_dir = Path('./saved') / args.env_name / 'other'

if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run 
    
def run(num_epochs):
    
    #TODO: add noise in name if noise is enabled
    # if args.mode == 'test':
    #     filename = f"./data/{policy_net.name}_test"
    #     if args.comm_constraints == 'simple' and args.noise_level > 0.0:
    #         filename += "_noisy" 
    #     filename += ".txt"
    # else:
    #     filename = f"./data/{policy_net.name}_train.txt"
    
    if args.mode == 'test':
        filename = f"./data/{policy_net.name}_test_{args.difficulty}"  # ------ RESTORE NAME
        if args.comm_constraints == 'simple' and args.noise_level > 0.0:
            filename += f"_{args.noise_type}{args.noise_level}"
        if args.comm_constraints == 'drops':
            filename += "_drops"
            if args.drop_prob_part != 0.0:
                filename += f"_part{args.drop_prob_part}"
            if args.drop_prob_whole != 0.0:
                filename += f"_whole{args.drop_prob_whole}"
        if args.comm_constraints == 'jumble':
            filename += "_jumble_non_norm"
            if args.jumble_prob != 0.0:
                filename += f"_{args.jumble_prob}"
        if args.comm_constraints == 'delay':
            filename += "_delay"
            if args.delay_prob != 0.0:
                filename += f"_{args.delay_prob}"
            if args.delay_step != 0:
                filename += f"_{args.delay_step}"
        filename += ".txt"
    else:
        filename = f"./data/{policy_net.name}_train_{args.difficulty}.txt" # ----- RESTORE 
        
    stats_for_export = []
    
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            
            if args.mode == 'test':
                s = trainer.test_batch(ep)
            elif args.mode == 'train':
                s = trainer.train_batch(ep)
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=4)
        
        stats_for_export.append((epoch, stat['reward'], stat['success']))
        
        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        
        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))        


        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))
    
        if args.save_every and ep and args.save and ep % args.save_every == 0:
            save(final=False, episode=ep)

        if args.save:
            save(final=True)
            
    if args.export:
        with open(filename, 'w') as file:
            np.set_printoptions(precision=4)
            # Prepare the data to write
            data_to_write = []
            # 
            for epoch_data, reward_data, success_data in stats_for_export:
                data_to_write = [
                    f"Epoch {epoch_data}\tReward {reward_data}",
                    f"Success {success_data}" ]
                file.write("\n".join(data_to_write) + "\n")
        print('Stats Exported.')
    
    if args.mode == 'train':
        print('finished training!')
        if args.save:
            print('Model saved.')
    elif args.mode == 'test':
            print('Finished testing')

def save(final, episode=0): 
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(episode)))

#
# If args.mode is 'train' and a model is loaded resume training
# if args.mode is 'test' then play the model loaded
# 
def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    # load log only when resuming training a loaded model, not when testing
    if args.mode == 'train': 
        log.update(d['log'])
        trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '' and (args.mode == "train" or args.mode == 'test'):
    load(args.load)

run(args.num_epochs)
if args.display:
    env.end_display()

if args.save:
    save(final=True)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
