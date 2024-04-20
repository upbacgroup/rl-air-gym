import torch


def reward_function(actual_positions, desired_position):
    # distance to target
    target_dist = torch.sqrt(actual_positions[..., 0] * actual_positions[..., 0] +
                             actual_positions[..., 1] * actual_positions[..., 1] +
                             (6.0 - actual_positions[..., 2]) * (6.0-actual_positions[..., 2]))
    pos_reward = 2.0 / (1.0 + target_dist * target_dist)

    dist_reward = (20.0 - target_dist) / 40.0

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward) + dist_reward
