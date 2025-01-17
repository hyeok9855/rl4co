from tensordict.tensordict import TensorDict
import torch

from rl4co.envs.routing.pctsp.env import PCTSPEnv
from rl4co.envs.routing.op.env import OPEnv
from rl4co.models.common.constructive.nonautoregressive.decoder import NonAutoregressiveDecoder
from rl4co.utils.decoding import Sampling, Greedy
from rl4co.utils.ops import batchify, unbatchify


@torch.no_grad()
def local_search(env: PCTSPEnv | OPEnv, td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Destroy & Repair local search, considering the solution symmetry from GFACS (Kim et al. 2024)
    Args:
        td: TensorDict, td from env with shape [batch_size,]
        actions: torch.Tensor, Tour indices with shape [batch_size, max_seq_len]
    Returns:
        torch.Tensor, Improved tour indices with shape [batch_size, max_seq_len
    """

    heatmap = kwargs["heatmap"]  # This is required for decoding

    max_iterations = kwargs.get("max_iterations", 5)
    n_symmetry = kwargs.get("n_symmetry", 2)
    assert 1 <= n_symmetry <= 2, "There are only 2 symmetric solutions for PCTSP and OP"
    destruction_rate = torch.linspace(0.2, 0.8, max_iterations)

    reward = env.get_reward(td, actions)

    actions = actions.unsqueeze(1)
    reward = reward.unsqueeze(1)
    for it in range(max_iterations):
        ###  obtain the symmetric solutions
        # Destroy the solutions
        destroyed_actions = symmetry_aware_destrction(
            actions[:, 0], n_symmetry, destruction_rate[it].item()
        )
        # [batch_size, n_symmetry, protected_len]
        destroyed_actions = destroyed_actions.transpose(0, 1).flatten(0, 1)
        # [n_symmetry * batch_size, protected_len]

        # Repair the destroyed solutions
        # Define decoding_strategy
        decoding_strategy = Greedy(**kwargs.get("decoding_kwargs", {}))
        td_batched = batchify(env.reset(td), n_symmetry)

        for _a in range(destroyed_actions.size(1)):
            td_batched.set("action", destroyed_actions[:, _a])
            td_batched = env.step(td_batched)["next"]

        while not td_batched["done"].all():
            logits, mask = NonAutoregressiveDecoder.heatmap_to_logits(td_batched, heatmap, n_symmetry)
            td_batched = decoding_strategy.step(logits, mask, td_batched)
            td_batched = env.step(td_batched)["next"]
        _, repaired_actions, td_batched, _ = decoding_strategy.post_decoder_hook(td_batched, env)

        new_actions = torch.cat([destroyed_actions, repaired_actions], dim=1)

        new_reward = env.get_reward(td_batched, new_actions)

        new_actions = unbatchify(new_actions, n_symmetry)  # [batch_size, n_symmetry, max_seq_len]
        new_reward = unbatchify(new_reward, n_symmetry)  # [batch_size, n_symmetry]

        # Padding the actions and rewards to concatenate with the topk actions
        diff_length = new_actions.size(-1) - actions.size(-1)
        if diff_length > 0:
            actions = torch.nn.functional.pad(actions, (0, diff_length), value=0)
        elif diff_length < 0:
            new_actions = torch.nn.functional.pad(new_actions, (0, -diff_length), value=0)
        actions = torch.cat([actions, new_actions], dim=1)
        reward = torch.cat([reward, new_reward], dim=1)

        # Select better actions
        best_indices = reward.argmax(dim=1)
        actions = torch.gather(actions, 1, best_indices.view(-1, 1, 1).expand(-1, -1, actions.size(-1)))
        reward = torch.gather(reward, 1, best_indices.view(-1, 1))

        # Remove tailing zeros
        actions = actions[:, :, : (actions != 0).sum(-1).max() + 1]

    return actions[:, 0]


def symmetry_aware_destrction(
    actions: torch.Tensor,
    n_symmetry: int = 2,
    destruction_rate: float = 0.5,
) -> torch.Tensor:
    """
    Destroy & Repair local search, considering the solution symmetry from GFACS (Kim et al. 2024)
    Args:
        actions: torch.Tensor, Tour indices with shape [batch_size, max_seq_len]
        n_symmetry: int, number of symmetric solutions to consider
        destruction_rate: float, rate of the solution to be destroyed

    Returns:
        torch.Tensor, Improved tour indices with shape [batch_size, n_symmetry, protected_len]
    """

    protected_len = int(actions.size(-1) * (1 - destruction_rate))

    symm_new_actions = get_symmetric_sols(actions, n_symmetry)
    # [batch_size, n_symmetry, max_seq_len]

    # Slicing is the most simple way to destroy the solutions, but there could be more sophisticated ways.
    destroyed_sols = symm_new_actions[:, :, :protected_len]
    # [batch_size, n_symmetry, protected_len]

    return destroyed_sols


def get_symmetric_sols(actions, n_symmetry):
    """
    Args:
        actions: torch.Tensor, Tour indices with shape [batch_size, max_seq_len]
    Returns:
        torch.Tensor, Symmetric solutions with shape [batch_size, n_symmetry, max_seq_len]
    """
    assert 1 <= n_symmetry <= 2, "Only 1 or 2 symmetry is supported"
    if n_symmetry == 1:
        return actions.unsqueeze(1)

    flipped_actions = flip_sequences(actions)
    return torch.stack([actions, flipped_actions], dim=1)


def flip_sequences(actions):
    """
    Flip the sequences with tailing zeros
    Args:
        actions: torch.Tensor, Tour indices with shape [batch_size, max_seq_len]
    Returns:
        torch.Tensor, Flipped tour indices with shape [batch_size, max_seq_len]
    """
    bs, max_seq_len = actions.size()

    if pad_end := not (actions[:, -1] == 0).all():
        # pad zeros to the end
        actions = torch.cat([actions, torch.zeros(bs, 1, device=actions.device)], dim=1)
        max_seq_len += 1

    nonzero_mask = actions != 0  # shape [batch_size, max_seq_len]
    lengths = nonzero_mask.sum(-1)  # shape [batch_size]
    positions = torch.arange(max_seq_len, device=actions.device)
    positions = positions.unsqueeze(0).expand(bs, max_seq_len)  # shape [batch_size, max_seq_len]
    reversed_positions = lengths.unsqueeze(-1) - 1 - positions
    mask_pos = positions < lengths.unsqueeze(-1)
    reversed_positions = torch.where(mask_pos, reversed_positions, max_seq_len - 1)
    flipped_actions = torch.gather(actions, 1, reversed_positions)

    if pad_end:
        flipped_actions = flipped_actions[:, :-1]

    return flipped_actions