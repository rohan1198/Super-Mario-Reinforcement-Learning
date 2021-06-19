import torch
import torch.nn as nn


def calc_loss(batch, net, tgt_net, gamma, device):
    states, actions, rewards, next_states, dones = batch

    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)
