#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:32:05 2025

@author: anastasios
"""

import torch
import math
import random
from typing import List, Any

def generate_noise_legacy(signal, noise_type, noise_lvl, mean=0.0):
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
    
    

def generate_noise(signal, noise_type, noise_lvl, mean=0.0, apply_prob=1.0):
    """
    signal: tensor of shape [batch, n, hid] OR [n, hid] (works either way)
    noise_type: 'gaussian' | 'uniform' | 'multiplicative' ...
    noise_lvl: interpreted as std for gaussian, as max amplitude for uniform
    apply_prob: fraction of agent messages to apply noise to (0..1).
    """
    # Ensure tensor (work with cloned copy to avoid in-place surprises)
    sig = signal.clone()
    device = sig.device
    dtype = sig.dtype

    # support both [n,hid] and [batch,n,hid]
    if sig.dim() == 2:
        sig = sig.unsqueeze(0)  # [1, n, hid]

    batch, n, hid = sig.shape

    # Decide which *messages* to apply noise to (per-agent mask)
    if apply_prob >= 1.0:
        msg_mask = torch.ones(batch, n, 1, device=device, dtype=dtype)
    elif apply_prob <= 0.0:
        msg_mask = torch.zeros(batch, n, 1, device=device, dtype=dtype)
    else:
        # keep with probability = (1-apply_prob) and apply noise where bernoulli==1
        msg_mask = torch.bernoulli(torch.full((batch, n, 1), apply_prob, device=device)).to(dtype)

    # Build noise tensor matching sig
    if noise_type == 'gaussian':
        noise = torch.normal(mean=torch.full((), mean, device=device, dtype=dtype),
                             std=torch.full((), noise_lvl, device=device, dtype=dtype),
                             size=(batch, n, hid), device=device).to(dtype)
    elif noise_type == 'uniform':
        # uniform in [mean - noise_lvl, mean + noise_lvl] (symmetric) — clearer semantics
        low = mean - float(noise_lvl)
        high = mean + float(noise_lvl)
        noise = (torch.rand(batch, n, hid, device=device, dtype=dtype) * (high - low) + low)
    elif noise_type == 'multiplicative':
        # multiplicative noise around 1: sample (1 + gaussian(0, noise_lvl))
        scale = 1.0 + torch.normal(mean=torch.full((), 0.0, device=device, dtype=dtype),
                                   std=torch.full((), noise_lvl, device=device, dtype=dtype),
                                   size=(batch, n, hid), device=device)
        noise = (scale - 1.0) * sig  # store only additive part to be applied below
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Apply noise only to selected messages (msg_mask expands to full shape)
    noise = noise * msg_mask  # broadcast (batch,n,1) -> (batch,n,hid)

    # Apply noise: for multiplicative we treated noise as (scale-1)*sig; adjust accordingly:
    if noise_type == 'multiplicative':
        out = sig + noise
    else:
        out = sig + noise

    # Restore original shape if needed
    if signal.dim() == 2:
        out = out.squeeze(0)

    return out

def snr_noise(signal, snr_db, apply_prob=1.0, eps=1e-12):
    """
    Apply SNR-targeted Gaussian noise to a fraction of messages.

    Args:
        signal: tensor [n,hid] or [batch,n,hid]
        snr_db: target SNR in decibels (float). Larger -> cleaner channel.
        apply_prob: per-message probability (0..1) that noise is applied this hop.
        eps: small value to guard against zero signal power.

    Returns:
        noisy_signal: tensor same shape as input (does NOT modify input inplace).
    """
    sig = signal.clone()
    device = sig.device
    dtype = sig.dtype

    squeezed = False
    if sig.dim() == 2:
        sig = sig.unsqueeze(0)  # [1, n, hid]
        squeezed = True
    elif sig.dim() != 3:
        raise ValueError("signal must be shape [n,hid] or [batch,n,hid]")

    batch, n, hid = sig.shape

    # Determine per-message mask (which messages receive noise)
    if apply_prob <= 0.0:
        if squeezed:
            return sig.squeeze(0)
        return sig
    if apply_prob >= 1.0:
        msg_mask = torch.ones((batch, n, 1), device=device, dtype=dtype)
    else:
        msg_mask = torch.bernoulli(torch.full((batch, n, 1), apply_prob, device=device)).to(dtype)

    # compute signal power per message (mean squared across hid)
    # shape -> [batch, n, 1]
    power = (sig ** 2).mean(dim=-1, keepdim=True)  # mean over hid

    # convert SNR dB -> linear ratio (signal_power / noise_power)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)

    # desired noise power per message: noise_power = signal_power / snr_linear
    noise_power = power / (snr_linear + 1e-30)

    # fallback for near-zero power: set noise_power to eps
    noise_power = torch.clamp(noise_power, min=eps)

    # noise std per element (same across hid dims): sqrt(noise_power)
    noise_std = torch.sqrt(noise_power).expand(batch, n, hid)

    # sample gaussian noise matching device/dtype
    noise = torch.normal(mean=torch.zeros_like(noise_std, device=device, dtype=dtype),
                         std=noise_std)

    # apply only to messages selected by msg_mask
    noisy = sig + noise * msg_mask

    if squeezed:
        noisy = noisy.squeeze(0)
    return noisy

## i.i.d. message drops
def drop_messages(comm, drop_prob_whole=0.0, drop_prob_part=0.0):
    """
    comm: tensor [batch, n, hid] OR [n, hid] (we handle both)
    drop_prob_whole: per-agent probability that the entire message vector is zeroed
    drop_prob_part: per-element probability of zeroing elements (applied after whole-message drops)
    Returns a new tensor (does not modify input).
    """
    sig = comm.clone()
    device = sig.device
    dtype = sig.dtype

    if sig.dim() == 2:
        sig = sig.unsqueeze(0)  # [1, n, hid]

    batch, n, hid = sig.shape

    # Whole-message drop mask: 1 means KEEP, 0 means DROP
    if drop_prob_whole <= 0:
        whole_mask = torch.ones((batch, n, 1), device=device, dtype=dtype)
    elif drop_prob_whole >= 1.0:
        whole_mask = torch.zeros((batch, n, 1), device=device, dtype=dtype)
    else:
        whole_mask = torch.bernoulli(torch.full((batch, n, 1), 1.0 - drop_prob_whole, device=device)).to(dtype)

    sig = sig * whole_mask  # drop entire message vectors where mask=0

    # Element-wise drop (for entries within messages that survived)
    if drop_prob_part > 0.0:
        part_mask = torch.bernoulli(torch.full((batch, n, hid), 1.0 - drop_prob_part, device=device)).to(dtype)
        sig = sig * part_mask

    if comm.dim() == 2:
        sig = sig.squeeze(0)

    return sig

class GilbertElliotChannel:
    """
    Gilbert-Elliot model for bursty channel errors/drops.
    Stores per-(batch,agent) state across hops.

    Args:
        batch_size: int (default 1)
        n_agents: int
        p_g2b: prob Good->Bad (float in [0,1])
        p_b2g: prob Bad->Good (float in [0,1])
        bad_mode: 'drop' | 'noise' | 'partial'  (behavior in Bad state)
        bad_noise_std: std of Gaussian noise to add when bad_mode == 'noise' or 'partial'
        partial_frac: for 'partial' mode, fraction of elements to corrupt when bad
        init_bad_prob: initial probability each agent starts in Bad
        device: torch device or None
    """
    def __init__(self,
                 batch_size,
                 n_agents,
                 p_g2b=0.05,
                 p_b2g=0.2,
                 bad_mode='drop',
                 bad_noise_std=1.0,
                 partial_frac=0.5,
                 init_bad_prob=0.0,
                 device=None):
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.p_g2b = float(p_g2b)
        self.p_b2g = float(p_b2g)
        self.bad_mode = bad_mode
        self.bad_noise_std = float(bad_noise_std)
        self.partial_frac = float(partial_frac)
        self.device = device if device is not None else torch.device('cpu')

        # initialize states: 0 = Good, 1 = Bad
        init = torch.bernoulli(torch.full((batch_size, n_agents), init_bad_prob, device=self.device)).bool()
        self.state = init  # bool tensor

    def reset(self, init_bad_prob=0.0):
        self.state = torch.bernoulli(torch.full((self.batch_size, self.n_agents),
                                                init_bad_prob, device=self.device)).bool()

    def step(self, comm):
        """
        Apply the GE channel to comm for one hop.
        comm: tensor [n,hid] or [batch,n,hid]
        Returns:
            new_comm: modified tensor (same shape)
            info: dict with 'bad_mask' (bool tensor [batch,n]) indicating which were in Bad state
        """
        sig = comm.clone()
        squeezed = False
        if sig.dim() == 2:
            sig = sig.unsqueeze(0)
            squeezed = True
        elif sig.dim() != 3:
            raise ValueError("comm must be shape [n,hid] or [batch,n,hid]")

        batch, n, hid = sig.shape
        if batch != self.batch_size or n != self.n_agents:
            # resize internal state if needed (maintain previous values where possible)
            self.batch_size = batch
            self.n_agents = n
            self.state = torch.zeros((batch, n), dtype=torch.bool, device=self.device)

        device = sig.device
        dtype = sig.dtype

        # update Markov state per agent
        # For Good (state==0): with prob p_g2b go to Bad (1)
        # For Bad  (state==1): with prob p_b2g go to Good (0)
        rand_mat = torch.rand((batch, n), device=device)
        new_state = self.state.clone()

        # Good -> Bad
        go_bad = (~self.state) & (rand_mat < self.p_g2b)
        # Bad -> Good
        go_good = (self.state) & (rand_mat < self.p_b2g)

        new_state[go_bad] = True
        new_state[go_good] = False
        self.state = new_state

        bad_mask = self.state  # boolean mask [batch, n]

        # Apply behavior in Bad state
        if self.bad_mode == 'drop':
            # zero-out entire message vectors for bad agents
            drop_mask = (~bad_mask).to(dtype).unsqueeze(-1)  # keep mask: 1 keep, 0 drop
            sig = sig * drop_mask

        elif self.bad_mode == 'noise':
            # add gaussian noise with std bad_noise_std to bad messages only
            std = torch.full((batch, n, hid), float(self.bad_noise_std), device=device, dtype=dtype)
            noise = torch.normal(mean=torch.zeros_like(std, device=device, dtype=dtype), std=std)
            # apply only to bad agents (bad_mask True -> apply)
            apply_mask = bad_mask.to(dtype).unsqueeze(-1)
            sig = sig + noise * apply_mask

        elif self.bad_mode == 'partial':
            # For bad agents, corrupt only a fraction of dims
            if self.partial_frac <= 0.0:
                pass
            elif self.partial_frac >= 1.0:
                # same as noise
                std = torch.full((batch, n, hid), float(self.bad_noise_std), device=device, dtype=dtype)
                noise = torch.normal(mean=torch.zeros_like(std), std=std)
                apply_mask = bad_mask.to(dtype).unsqueeze(-1)
                sig = sig + noise * apply_mask
            else:
                dim_mask = torch.bernoulli(torch.full((hid,), self.partial_frac, device=device)).bool()
                for b in range(batch):
                    bad_inds = bad_mask[b].nonzero(as_tuple=False).view(-1)
                    if bad_inds.numel() == 0:
                        continue
                    # build noise only for those bad indices and dims
                    std = torch.full((bad_inds.numel(), dim_mask.sum().item()), float(self.bad_noise_std), device=device, dtype=dtype)
                    noise = torch.normal(mean=torch.zeros_like(std), std=std)
                    # assign noise to chosen dims
                    sig[b, bad_inds][:, dim_mask] = sig[b, bad_inds][:, dim_mask] + noise

        else:
            raise ValueError(f"Unknown bad_mode {self.bad_mode}")

        if squeezed:
            sig = sig.squeeze(0)
        return sig, {'bad_mask': bad_mask}

def jumble_messages_realistic(comm,
                              jumble_prob,
                              mode='reorder_local',
                              noise_after=0.0,
                              dim_apply_prob=0.5,
                              local_k_frac=0.3,
                              collision_mix_frac=0.5,
                              overlap_alpha=0.5):
    """
    Realistic jumbling/reordering/collision behaviors for V2V abstraction layer.

    Args:
        comm: tensor [n,hid] or [batch,n,hid]
        jumble_prob: prob we apply any jumbling this hop (0..1)
        mode: 'reorder_local' | 'collision_mix' | 'overlap_delay' | 'partial'
        noise_after: gaussian std to add after jumbling (small jitter)
        dim_apply_prob: for 'partial', fraction of dims to affect
        local_k_frac: fraction of agents forming a *local* group to reorder (for 'reorder_local')
        collision_mix_frac: mixing ratio for collision (how much other signals contribute)
        overlap_alpha: blending weight for overlap (alpha*orig + (1-alpha)*older)
    Returns:
        new tensor of same shape (no in-place modification)
    """
    sig = comm.clone()
    device = sig.device
    dtype = sig.dtype

    squeezed = False
    if sig.dim() == 2:
        sig = sig.unsqueeze(0)
        squeezed = True
    elif sig.dim() != 3:
        raise ValueError("comm must be shape [n,hid] or [batch,n,hid]")

    batch, n_agents, hid = sig.shape

    # quick exit
    if jumble_prob <= 0.0 or (torch.rand(1).item() >= float(jumble_prob)):
        return sig.squeeze(0) if squeezed else sig

    for b in range(batch):
        if n_agents < 2:
            continue

        # number of agents in local group
        k = max(2, int(max(2, round(local_k_frac * n_agents))))
        # choose a random start index for a contiguous local group (wrap-around)
        start = random.randint(0, n_agents - 1)
        indices = [(start + i) % n_agents for i in range(k)]
        indices = torch.tensor(indices, device=device, dtype=torch.long)

        tmp = sig[b, indices, :].clone()  # [k, hid]

        if mode == 'reorder_local': # low probabilitty of happening with modern routing protocols
            # realistic local reordering: permute messages *within* local neighborhood
            perm = torch.randperm(k, device=device)
            new_vals = tmp[perm, :].clone()

        elif mode == 'collision_mix':
            # simulate collision: signals add and each affected message receives a mix
            # compute sum-of-signals then mix into each message
            summed = tmp.sum(dim=0, keepdim=True)  # [1, hid]
            avg = summed / float(k)
            # new = (1 - collision_mix_frac) * orig + collision_mix_frac * avg
            new_vals = ((1.0 - collision_mix_frac) * tmp) + (collision_mix_frac * avg.expand(k, hid))
            # small gaussian jitter can mimic interference
        #TODO: path to use messages from delay_buffer rather than 'surrogate' messages    
        elif mode == 'overlap_delay':
            # simulate an older message overlapping (we don't have previous_comm here)
            # so we sample other agents' messages as surrogate "older" arrivals:
            # pick other indices outside the local group or randomly within
            other_idx = torch.randperm(n_agents, device=device)[:k]
            other_tmp = sig[b, other_idx, :].clone()
            other_avg = other_tmp.mean(dim=0, keepdim=True)
            # blend original with older/other: alpha * orig + (1-alpha) * other_avg
            new_vals = (overlap_alpha * tmp) + ((1.0 - overlap_alpha) * other_avg.expand(k, hid))

        elif mode == 'partial': # simialr to i.i.d noise but sparser and affecting subsest of message dims
            # only affect some dims (simulate bit/field corruption)
            if dim_apply_prob <= 0.0:
                new_vals = tmp.clone()
            elif dim_apply_prob >= 1.0:
                # permute full vectors within the local group
                perm = torch.randperm(k, device=device)
                new_vals = tmp[perm, :].clone()
            else:
                dim_mask = torch.bernoulli(torch.full((hid,), dim_apply_prob, device=device)).bool()
                perm = torch.randperm(k, device=device)
                new_tmp = tmp.clone()
                if dim_mask.any():
                    new_tmp[:, dim_mask] = tmp[perm, :][:, dim_mask]
                new_vals = new_tmp.clone()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # optional small noise after jumbling (model channel jitter/interference)
        if noise_after and noise_after > 0.0:
            noise = torch.normal(mean=0.0, std=float(noise_after), size=new_vals.shape, device=device, dtype=dtype)
            new_vals = new_vals + noise

        # assign back to the local indices
        sig[b, indices, :] = new_vals

    return sig.squeeze(0) if squeezed else sig


def comm_delay_n_hops(comm, hop_delay_buffer, max_hop_delay, delay_prob):
    """
    comm: tensor (any shape) — we store clones in hop_delay_buffer (list).
    hop_delay_buffer: list object stored in model (persist between calls).
    max_hop_delay: integer number of hops to allow as maximum delay
    delay_prob: probability to actually apply a delay this call
    Returns: delayed_comm (tensor); also keeps hop_delay_buffer trimmed to max_hop_delay
    """

    # store clone of current comm
    hop_delay_buffer.append(comm.clone())

    # buffer size to max_hop_delay
    if max_hop_delay is not None and len(hop_delay_buffer) > max_hop_delay:
        # drop oldest
        del hop_delay_buffer[0]

    # decide if we apply a delay
    if max_hop_delay is None or max_hop_delay <= 0 or random.random() >= float(delay_prob):
        # no delay — return current comm clone
        return comm.clone()

    # choose a random delay in [1, max_hop_delay]
    delay = random.randint(1, max_hop_delay)
    if len(hop_delay_buffer) >= delay:
        # -delay gets the comm 'delay' steps ago
        delayed_comm = hop_delay_buffer[-delay].clone()
    else:
        # not enough history use oldest available
        delayed_comm = hop_delay_buffer[0].clone()

    return delayed_comm

def jumble_messages_norm(comm, jumble_prob, mode='permute', noise_after=0.0, dim_apply_prob=1.0):
    """
    Jumble messages between a random subset of agents.

    Args:
        comm: tensor either [n, hid] OR [batch, n, hid]
        jumble_prob: probability (0..1) that jumbling will be applied this call
        mode: 'permute' | 'mix' | 'sum_distribute' | 'partial'
            - 'permute': randomly permute the selected agents' messages among them
            - 'mix': replace each selected agent message by the average of selected messages
            - 'sum_distribute': sum selected messages and assign averaged sum to each (close to your original)
            - 'partial': permute only a fraction (dim_apply_prob) of dimensions per-agent
        noise_after: std of small gaussian noise added after jumbling (0.0 no noise)
        dim_apply_prob: for 'partial' mode, fraction of hidden dims to swap (0..1)
    Returns:
        new tensor with jumbling applied (does not modify input in-place)
    """
    # Clone to avoid in-place modification
    sig = comm.clone()
    device = sig.device
    dtype = sig.dtype

    # Normalize shape to [batch, n, hid]
    squeezed = False
    if sig.dim() == 2:
        sig = sig.unsqueeze(0)  # [1, n, hid]
        squeezed = True
    elif sig.dim() != 3:
        raise ValueError("comm must be shape [n,hid] or [batch,n,hid]")

    batch, n_agents, hid = sig.shape

    # Quick exit if no jumble
    if jumble_prob <= 0.0 or (torch.rand(1).item() >= float(jumble_prob)):
        return sig.squeeze(0) if squeezed else sig

    # For each batch element, pick random subset and jumble
    for b in range(batch):
        if n_agents < 2:
            continue

        # pick number of agents to jumble (at least 2)
        k = random.randint(2, n_agents)  # uniform in [2, n_agents]
        indices = torch.randperm(n_agents, device=device)[:k]

        # Work on a temporary copy for safe assignment
        tmp = sig[b, indices, :].clone()  # [k, hid]

        if mode == 'permute':
            # permute rows among the selected indices
            perm = torch.randperm(k, device=device)
            new_vals = tmp[perm, :].clone()

        elif mode == 'mix':
            # average across selected agents and assign same average to all
            avg = tmp.mean(dim=0, keepdim=True)  # [1, hid]
            new_vals = avg.expand(k, hid).clone()

        elif mode == 'sum_distribute':
            # sum and distribute equally (the original summed+divide behaviour)
            summed = tmp.sum(dim=0, keepdim=True)  # [1, hid]
            new_vals = (summed / float(k)).expand(k, hid).clone()

        elif mode == 'partial':
            # select dims to permute and leave other dims intact
            if dim_apply_prob <= 0.0:
                continue
            if dim_apply_prob >= 1.0:
                # full permute (same as permute)
                perm = torch.randperm(k, device=device)
                new_tmp = tmp[perm, :].clone()
                new_vals = new_tmp
            else:
                # pick dims to swap
                dim_mask = torch.bernoulli(torch.full((hid,), dim_apply_prob, device=device)).bool()
                perm = torch.randperm(k, device=device)
                new_tmp = tmp.clone()
                # swap only chosen dims according to permutation
                if dim_mask.any():
                    new_tmp[:, dim_mask] = tmp[perm, :][:, dim_mask]
                new_vals = new_tmp.clone()
        else:
            raise ValueError(f"Unknown mode {mode}")

        # optionally add small gaussian noise after jumbling
        if noise_after and noise_after > 0.0:
            noise = torch.normal(mean=0.0,
                                 std=float(noise_after),
                                 size=new_vals.shape,
                                 device=device,
                                 dtype=dtype)
            new_vals = new_vals + noise

        # assign back
        sig[b, indices, :] = new_vals

    return sig.squeeze(0) if squeezed else sig

# ----------- Calibration noise

DEFAULT_CALIB_NOISE_MIX = {
    'gaussian': 0.35,
    'drops': 0.25,
    'snr': 0.10,
    'jumble': 0.10,
    'delay': 0.10,
    'gilbert_elliot': 0.10
}

def _draw_from_dict(d):
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    tot = sum(vals)
    probs = [v/tot for v in vals]
    return random.choices(keys, probs, k=1)[0]


def sample_noise_config(args):
    """
    Return a small dict describing a noise configuration for calibration.
    Keep keys consistent with args.* names used in model.forward.
    """
    # 20% clean batches by default
    p_clean = getattr(args, 'calib_clean_frac', 0.2)
    if random.random() < p_clean:
        return {'comm_constraints': 'none'}

    noise_mix = DEFAULT_CALIB_NOISE_MIX
    typ = _draw_from_dict(noise_mix)

    if typ == 'gaussian':
        cfg = {
            'comm_constraints': 'additive_iid',
            'noise_type': 'gaussian',
            'noise_level': random.uniform(0.2, 0.8),
            'noise_mean': 0.0,
            'apply_prob': random.choice([0.4, 0.6, 1.0])
        }
        return cfg

    if typ == 'drops':
        cfg = {
            'comm_constraints': 'drops_iid',
            'drop_prob_whole': random.choice([0.2, 0.4, 0.6]),
            'drop_prob_part': random.choice([0.0, 0.1])
        }
        return cfg

    if typ == 'snr':
        cfg = {
            'comm_constraints': 'snr_noise',
            'snr_db': random.uniform(0.0, 10.0),
            'apply_prob': random.choice([0.4, 1.0])
        }
        return cfg

    if typ == 'jumble':
        cfg = {
            'comm_constraints': 'jumble',
            'jumble_prob': random.uniform(0.05, 0.3),
            'jumble_mode': random.choice(['reorder_local','collision_mix', 'overlap_delay']),
            'noise_after': random.uniform(0.0, 0.1),
        }
        return cfg

    if typ == 'delay':
        cfg = {
            'comm_constraints': 'delay',
            'delay_prob': random.uniform(0.1, 0.5),
            'max_hop_delay': getattr(args, 'max_hop_delay', 2)
        }
        return cfg

    if typ == 'gilbert_elliot':
        cfg = {
            'comm_constraints': 'gilbert-elliot',
            'p_g2b': random.uniform(0.01, 0.15),
            'p_b2g': random.uniform(0.05, 0.3),
            'bad_mode': random.choice(['drop', 'noise', 'partial']),
            'bad_noise_std': random.uniform(0.2, 1.0),
            'partial_frac': random.uniform(0.2, 0.8)
        }
        return cfg

    return {'comm_constraints': 'none'}


def apply_noise(comm, cfg, device=None):
    """
    Apply a single noise configuration to 'comm' using your existing functions.
    Returns corrupted comm.
    """
    if cfg is None:
        return comm

    cc = cfg.get('comm_constraints', 'none')
    device = device or comm.device

    if cc == 'none':
        return comm

    if cc == 'additive_iid':
        return generate_noise(comm,
                              cfg.get('noise_type', 'gaussian'),
                              cfg.get('noise_level', 0.0),
                              cfg.get('noise_mean', 0.0),
                              cfg.get('apply_prob', 1.0))

    if cc == 'snr_noise':
        return snr_noise(comm,
                         cfg.get('snr_db', 5.0),
                         cfg.get('apply_prob', 1.0))

    if cc == 'drops_iid':
        return drop_messages(comm,
                             drop_prob_whole=cfg.get('drop_prob_whole', 0.0),
                             drop_prob_part=cfg.get('drop_prob_part', 0.0))

    if cc == 'jumble':
        return jumble_messages_realistic(comm,
                                         jumble_prob=cfg.get('jumble_prob', 0.1),
                                         mode=cfg.get('jumble_mode', 'reorder_local'),
                                         noise_after=cfg.get('noise_after', 0.0))

    if cc == 'delay':
        # we don't have access to hop_delay_buffer here; call a simple delay helper
        return comm_delay_n_hops(comm,
                                 hop_delay_buffer=getattr(cfg, 'hop_delay_buffer', None),
                                 max_hop_delay=cfg.get('max_hop_delay', 1),
                                 delay_prob=cfg.get('delay_prob', 0.0))

    if cc == 'gilbert-elliot' or cc == 'gilbert-eliiot':
        # instantiate a simple GE channel for this batch (stateless across calls)
        # Users might want to store state across hops/iterations — this is a simple one-shot
        batch_size = cfg.get('batch_size', 1)
        n_agents = cfg.get('n_agents', comm.shape[0] if comm.ndim == 2 else comm.shape[1])
        ge = GilbertElliotChannel(batch_size=batch_size,
                                  n_agents=n_agents,
                                  p_g2b=cfg.get('p_g2b', 0.05),
                                  p_b2g=cfg.get('p_b2g', 0.2),
                                  bad_mode=cfg.get('bad_mode', 'drop'),
                                  bad_noise_std=cfg.get('bad_noise_std', 1.0),
                                  partial_frac=cfg.get('partial_frac', 0.5),
                                  init_bad_prob=0.0,
                                  device=device)
        comm_noised, info = ge.apply(comm)
        return comm_noised

    # fallback
    return comm


def apply_calibration_noise(comm, device=None):
    cfg = sample_noise_config()
    comm_noised = apply_noise(comm, cfg, device=device)
    return comm_noised
