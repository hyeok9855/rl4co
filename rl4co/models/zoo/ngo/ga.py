from functools import cached_property
from typing import Optional, Tuple, cast

import torch

from tensordict import TensorDict
from torch import Tensor
from tqdm import trange

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.nonautoregressive.decoder import (
    NonAutoregressiveDecoder,
)
from rl4co.utils.decoding import Sampling
from rl4co.utils.ops import batchify, get_distance_matrix, unbatchify


SAVE_MEMORY = True


class GA:
    """Implements the Genetic Algorithm (GA) under the NGO framework (work in progress).

    Args:
        log_heuristic: Logarithm of the heuristic matrix.
        n_population: Number of individuals in the population. Defaults to 100.
        n_offspring: Number of offspring to be generated. Defaults to 100.
            Note that each pair of parents generates only one offspring.
        n_parents: Number of parents to be selected for mating. Defaults to 2.
        mutation_rate: Rate of mutation. Defaults to 0.01.
        rank_coefficient: Coefficient to be used for the rank-based selection. Defaults to 0.01.
        novelty_rank_w: Weight for novelty to be used for weighted rank-based selection. Defaults to 0.0.
        use_local_search: Whether to use local_search provided by the env. Default to False.
        use_nls: Whether to use neural-guided local search provided by the env. Default to False.
        n_perturbations: Number of perturbations to be used for nls. Defaults to 5.
        local_search_params: Arguments to be passed to the local_search.
        perturbation_params: Arguments to be passed to the perturbation used for nls.
    """

    def __init__(
        self,
        log_heuristic: Tensor,
        n_population: int = 100,
        n_offspring: int = 100,
        n_parents: int = 2,
        mutation_rate: float = 0.01,
        rank_coefficient: float = 0.01,
        novelty_rank_w: float = 0.0,
        use_local_search: bool = False,
        use_nls: bool = False,
        n_perturbations: int = 1,
        local_search_params: dict = {},
        perturbation_params: dict = {},
    ):
        self.log_heuristic = log_heuristic
        self.n_nodes = log_heuristic.shape[1]
        self.batch_size = log_heuristic.shape[0]

        self.n_population = n_population
        self.n_offspring = n_offspring
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.rank_coefficient = rank_coefficient
        self.novelty_rank_w = novelty_rank_w

        self.final_actions = self.final_reward = None
        self.final_reward_cache: dict = {}

        self.use_local_search = use_local_search
        assert not (use_nls and not use_local_search), "use_nls requires use_local_search"
        self.use_nls = use_nls
        self.n_perturbations = n_perturbations
        self.local_search_params = local_search_params.copy()
        self.perturbation_params = perturbation_params.copy()

        self._batchindex = torch.arange(self.batch_size, device=log_heuristic.device)

    @cached_property
    def heuristic_dist(self) -> torch.Tensor:
        heuristic = self.log_heuristic.exp().detach().cpu() + 1e-10
        heuristic_dist = 1 / (heuristic / heuristic.max(-1, keepdim=True)[0] + 1e-5)
        heuristic_dist[:, torch.arange(heuristic_dist.shape[1]), torch.arange(heuristic_dist.shape[2])] = 0
        return heuristic_dist

    def run(
        self,
        td_initial: TensorDict,
        env: RL4COEnvBase,
        n_iterations: int,
        decoding_kwargs: dict,
    ) -> Tuple[Tensor, dict[int, Tensor]]:
        """Run the NGO algorithm for a specified number of iterations.

        Args:
            td_initial: Initial state of the problem.
            env: Environment representing the problem.
            n_iterations: Number of iterations to run the algorithm.

        Returns:
            td: The final state of the problem.
            actions: The final actions chosen by the algorithm.
            reward: The final reward achieved by the algorithm.
        """

        population = {
            "actions": torch.zeros(0, dtype=torch.long, device=td_initial.device),
            "edge_mask": torch.zeros(0, dtype=torch.bool, device=td_initial.device),
            "reward": torch.zeros(0, dtype=torch.float, device=td_initial.device),
            "novelty": torch.zeros(0, dtype=torch.float, device=td_initial.device),
        }

        for i in (pbar := trange(n_iterations, dynamic_ncols=True, desc="Running GA")):
            # reset environment
            td = td_initial.clone()
            _, _, population = self._one_step(td, env, population, decoding_kwargs)
            self.final_reward_cache[i] = self.final_reward.clone()  # type: ignore
            pbar.set_postfix({"reward": self.final_reward.mean().item()})  # type: ignore

        action_matrix = self._convert_final_action_to_matrix()
        return action_matrix, self.final_reward_cache

    def run_for_training(
        self,
        td_initial: TensorDict,
        env: RL4COEnvBase,
        n_iterations: int,
        decoding_kwargs: dict,
        actions: Optional[Tensor] = None,
        reward: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the NGO algorithm for a specified number of iterations.

        Args:
            td_initial: Initial state of the problem.
            env: Environment representing the problem.
            n_iterations: Number of iterations to run the algorithm.

        Returns:
            actions: The final actions chosen by the algorithm.
            reward: The final reward achieved by the algorithm.
        """

        population = {
            "actions": torch.zeros(0, dtype=torch.long, device=td_initial.device),
            "edge_mask": torch.zeros(0, dtype=torch.bool, device=td_initial.device),
            "reward": torch.zeros(0, dtype=torch.float, device=td_initial.device),
            "novelty": torch.zeros(0, dtype=torch.float, device=td_initial.device),
        }
        if actions is not None:
            assert actions.size(0) == reward.size(0) == self.batch_size * self.n_offspring  # type: ignore

            # reshape from (batch_size * n_offspring, ...) to (batch_size, n_offspring, ...)
            reward = unbatchify(reward, self.n_offspring)  # type: ignore
            actions = unbatchify(actions, self.n_offspring)  # type: ignore
            # update final actions and rewards
            self._update_results(actions, reward)
            # update population
            population = self._update_population(env.name, actions, reward, population)

        for _ in range(n_iterations):
            # reset environment
            td = td_initial.clone()
            _, _, population = self._one_step(td, env, population, decoding_kwargs)

        return population["actions"], population["reward"]

    def _one_step(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        population: dict[str, torch.Tensor],
        decoding_kwargs: dict,
    ) -> Tuple[Tensor, Tensor, dict[str, torch.Tensor]]:
        """Run one step of the Neural Genetic Operator

        Args:
            td: Current state of the problem.
            env: Environment representing the problem.
            population: Population of individuals.
                population["actions"].shape == (batch_size, n_population, max_seq_len)
                population["edge_mask"].shape == (batch_size, n_population, max_seq_len, max_seq_len)
                population["reward"].shape == (batch_size, n_population)
                population["novelty"].shape == (batch_size, n_population)

        Returns:
            actions: The actions chosen by the algorithm.
            reward: The reward achieved by the algorithm.
        """
        # sampling
        parents_mask = self._get_parents_mask(population)
        td, env, actions, reward = self._sampling(td, env, parents_mask, decoding_kwargs)
        # local search, reserved for extensions
        if self.use_local_search:
            actions, reward = self.local_search(td, env, actions, decoding_kwargs)

        # reshape from (batch_size * n_offspring, ...) to (batch_size, n_offspring, ...)
        reward = unbatchify(reward, self.n_offspring)
        actions = unbatchify(actions, self.n_offspring)
        # update final actions and rewards
        self._update_results(actions, reward)
        # update population
        population = self._update_population(env.name, actions, reward, population)

        return actions, reward, population  # type: ignore

    def _sampling(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        parents_mask: Optional[Tensor],
        decoding_kwargs: dict,
    ):
        decode_strategy = Sampling(**decoding_kwargs)

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        while not td["done"].all():
            logits, mask = NonAutoregressiveDecoder.heatmap_to_logits(
                td, self.log_heuristic, num_starts
            )

            # NGO masking
            if parents_mask is not None:
                batch_indices = torch.arange(parents_mask.size(0), device=parents_mask.device)
                new_mask = mask * parents_mask[batch_indices, td["current_node"].squeeze(-1)]
                # Mutation here
                invalid_indices = new_mask.sum(-1) == 0  # invalid case
                mutation_indices = torch.rand(  # stochastic mutation
                    parents_mask.size(0), device=invalid_indices.device
                ) < self.mutation_rate
                unmask_indices = invalid_indices | mutation_indices
                new_mask[unmask_indices] = mask[unmask_indices]
                mask = new_mask

            td = decode_strategy.step(logits, mask, td)
            td = env.step(td)["next"]

        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)
        reward = env.get_reward(td, actions)

        return td, env, actions, reward

    def local_search(
        self, td: TensorDict, env: RL4COEnvBase, actions: Tensor, decoding_kwargs: dict
    ) -> Tuple[Tensor, Tensor]:
        """Perform local search on the actions and reward obtained.

        Args:
            td: Current state of the problem.
            env: Environment representing the problem.
            actions: Actions chosen by the algorithm.

        Returns:
            actions: The modified actions
            reward: The modified reward
        """
        device = td.device
        if env.name in ["tsp", "cvrp"]:
            # Convert to CPU in advance to minimize the overhead from device transfer
            td = td.detach().cpu()
            # TODO: avoid or generalize this, e.g., pre-compute for local search in each env
            td["distances"] = get_distance_matrix(td["locs"])
            actions = actions.detach().cpu()
        elif env.name in ["pctsp", "op"]:  # destroy & repair local search
            self.local_search_params.update(
                {
                    "decoding_kwargs": decoding_kwargs,
                    "heatmap": batchify(self.log_heuristic, self.n_offspring),
                }
            )
        else:
            raise NotImplementedError(f"Local search not implemented for {env.name}")

        best_actions = env.local_search(td=td, actions=actions, **self.local_search_params)
        best_rewards = env.get_reward(td, best_actions)

        if self.use_nls:
            td_perturb = td.clone()
            td_perturb["distances"] = torch.tile(self.heuristic_dist, (self.n_offspring, 1, 1))
            new_actions = best_actions.clone()

            for _ in range(self.n_perturbations):
                perturbed_actions = env.local_search(
                    td=td_perturb, actions=new_actions, **self.perturbation_params
                )
                new_actions = env.local_search(
                    td=td, actions=perturbed_actions, **self.local_search_params
                )
                new_rewards = env.get_reward(td, new_actions)

                improved_indices = new_rewards > best_rewards
                best_actions = env.replace_selected_actions(best_actions, new_actions, improved_indices)
                best_rewards[improved_indices] = new_rewards[improved_indices]

        best_actions = best_actions.to(device)
        best_rewards = best_rewards.to(device)

        return best_actions, best_rewards

    def _update_results(self, actions, reward):
        # update the best-trails recorded in self.final_actions
        best_index = reward.argmax(-1)
        best_reward = reward[self._batchindex, best_index]
        best_actions = actions[self._batchindex, best_index]

        if self.final_actions is None or self.final_reward is None:
            self.final_actions = list(iter(best_actions.clone()))
            self.final_reward = best_reward.clone()
        else:
            require_update = self._batchindex[self.final_reward <= best_reward]
            for index in require_update:
                self.final_actions[index] = best_actions[index].clone()
            self.final_reward[require_update] = best_reward[require_update].clone()

        return best_index

    def _convert_final_action_to_matrix(self) -> Tensor:
        assert self.final_actions is not None
        action_count = max(len(actions) for actions in self.final_actions)
        mat_actions = torch.zeros(
            (self.batch_size, action_count),
            device=self.final_actions[0].device,
            dtype=self.final_actions[0].dtype,
        )
        for index, action in enumerate(self.final_actions):
            mat_actions[index, : len(action)] = action

        return mat_actions

    ###############################
    ##### GA specific methods #####
    ###############################

    def _update_population(
        self, env_name, actions, reward, population: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        assert actions.size(1) == self.n_offspring

        if env_name == "tsp" or len(population["actions"]) == 0:
            concated_actions = torch.cat([population["actions"], actions], dim=1)
        else:  # for other envs, we may need to pad the actions
            diff_length = actions.size(-1) - population["actions"].size(-1)
            if diff_length > 0:
                population["actions"] = torch.nn.functional.pad(population["actions"], (0, diff_length), value=0)
            elif diff_length < 0:
                actions = torch.nn.functional.pad(actions, (0, -diff_length), value=0)
            concated_actions = torch.cat([population["actions"], actions], dim=1)
        # concated_actions.shape == (batch_size, n_population + n_offspring, max_seq_len)

        new_n = concated_actions.size(1)  # new_n = n_population + n_offspring
        concated_edge_mask = self._get_edge_mask(concated_actions)
        # edge_mask.shape == (batch_size, new_n, n_nodes, n_nodes)
        pairwise_distance = self._pairwise_distance(concated_edge_mask)
        # pairwise_distance.shape == (batch_size, new_n, new_n)

        concated_reward = torch.cat([population["reward"], reward], dim=1)
        # concated_reward.shape == (batch_size, new_n)
        concated_novelty = pairwise_distance.mean(-1)
        # concated_novelty.shape == (batch_size, new_n)

        if new_n <= self.n_population:
            population["actions"] = concated_actions
            population["edge_mask"] = concated_edge_mask
            population["reward"] = concated_reward
            population["novelty"] = concated_novelty
        else:
            # remove redundant individuals
            triu_indices = torch.triu_indices(new_n, new_n)
            pairwise_distance[:, triu_indices[0], triu_indices[1]] = 1.0
            # pairwise_distance.shape == (batch_size, new_n, new_n)
            redundant_indices = torch.where(pairwise_distance < 1e-5)

            uniqueness_weights = torch.ones_like(concated_reward)
            if len(redundant_indices[0]) > 0:
                concated_reward[redundant_indices[:2]] = -1e5
                concated_novelty[redundant_indices[:2]] = -1e5
                uniqueness_weights[redundant_indices[:2]] = 0.0

            # rank-based selection
            score_ranks = torch.argsort(torch.argsort(-concated_reward, dim=1), dim=1)
            novelty_ranks = torch.argsort(torch.argsort(-concated_novelty, dim=1), dim=1)
            weighted_ranks = (1 - self.novelty_rank_w) * score_ranks + self.novelty_rank_w * novelty_ranks
            weights = 1.0 / (self.rank_coefficient * new_n + weighted_ranks)
            weights *= uniqueness_weights
            indices_to_keep = torch.multinomial(weights, self.n_population, replacement=False)
            batch_indices = self._batchindex.unsqueeze(1).expand(-1, self.n_population)
            population["actions"] = concated_actions[batch_indices, indices_to_keep]
            population["edge_mask"] = concated_edge_mask[batch_indices, indices_to_keep]
            population["reward"] = concated_reward[batch_indices, indices_to_keep]
            population["novelty"] = concated_novelty[batch_indices, indices_to_keep]

        return population

    def _get_edge_mask(self, actions: Tensor) -> Tensor:
        batch_size, n, max_seq_len = actions.size()
        n_nodes = self.n_nodes

        actions_flat = actions.reshape(batch_size * n, max_seq_len)
        # actions_flat.shape == (batch_size * n, max_seq_len)
        edge_index = torch.stack([actions_flat, torch.roll(actions_flat, shifts=-1, dims=1)], dim=2)
        row = edge_index[:, :, 0]
        col = edge_index[:, :, 1]
        batch_indices = torch.arange(batch_size * n, device=actions.device).view(-1, 1).expand(-1, max_seq_len)
        linear_indices_forward = batch_indices * (n_nodes * n_nodes) + row * n_nodes + col
        linear_indices_backward = batch_indices * (n_nodes * n_nodes) + col * n_nodes + row
        linear_indices = torch.cat([linear_indices_forward, linear_indices_backward], dim=1).reshape(-1)
        total_elements = batch_size * n * n_nodes * n_nodes
        edge_mask_flat = torch.zeros(total_elements, dtype=torch.bool, device=actions.device)
        edge_mask_flat[linear_indices] = True
        edge_mask = edge_mask_flat.view(batch_size, n, n_nodes, n_nodes)
        # edge_mask.shape == (batch_size, n, n_nodes, n_nodes)

        return edge_mask

    def _pairwise_distance(self, edge_mask: Tensor) -> Tensor:
        batch_size, n, n_nodes, _ = edge_mask.size()
        if SAVE_MEMORY:  # For-loop Version (to avoid OOM)
            pairwise_sum = torch.zeros((batch_size, n, n), dtype=torch.float, device=edge_mask.device)
            for b in range(batch_size):
                _edges = edge_mask[b].view(n, -1).to(torch.float)
                pairwise_sum[b] = torch.mm(_edges, _edges.transpose(0, 1))
        else:  # Batch Version
            mask_flat = edge_mask.view(batch_size, n, -1).float()
            # mask_flat.shape == (batch_size, n, n_nodes * n_nodes)
            pairwise_sum = torch.matmul(mask_flat, mask_flat.transpose(1, 2))

        pairwise_distance = 1 - (pairwise_sum / (2 * n_nodes))
        # pairwise_distance.shape == (batch_size, n, n)
        return pairwise_distance

    def _get_parents_mask(self, population: dict[str, Tensor]) -> Optional[Tensor]:
        if len(population["reward"]) == 0 or population["reward"].size(1) < self.n_population:
            return None

        # rank-based selection
        score_ranks = torch.argsort(torch.argsort(-population["reward"], dim=1), dim=1)
        novelty_ranks = torch.argsort(torch.argsort(-population["novelty"], dim=1), dim=1)
        weighted_ranks = (1 - self.novelty_rank_w) * score_ranks + self.novelty_rank_w * novelty_ranks
        weights = 1.0 / (self.rank_coefficient * self.n_population + weighted_ranks)
        # weights.shape == (batch_size, n_population)

        if SAVE_MEMORY:  # For-loop & Sparse Version (to avoid OOM)
            weights_expanded = cast(Tensor, batchify(weights, self.n_offspring))
            parents_indices = torch.multinomial(weights_expanded, self.n_parents, replacement=False)
            parents_indices_batched = unbatchify(parents_indices, self.n_offspring)
            # parents_indices.shape == (batch_size, n_offspring, n_parents)

            parents_edge_masks = torch.cat(
                [
                    population["edge_mask"][b, parents_indices_batched[b]].unsqueeze(1).to_sparse_coo()
                    for b in range(self.batch_size)
                ],
                dim=1,
            )  # parents_edge_masks.shape == (n_offspring, batch_size, n_parents, n_nodes, n_nodes)
            out = parents_edge_masks.sum(2).bool().to_dense().flatten(0, 1)

        else:  # Batch & Dense Version
            weights_expanded = cast(Tensor, batchify(weights, self.n_offspring))
            parents_indices = torch.multinomial(weights_expanded, self.n_parents, replacement=False)

            parents_edge_masks = cast(Tensor, batchify(population["edge_mask"], self.n_offspring))
            batch_indices = torch.arange(
                parents_edge_masks.size(0), device=parents_edge_masks.device
            ).unsqueeze(1).expand(-1, self.n_parents)
            parents_edge_masks = parents_edge_masks[batch_indices, parents_indices]
            out = parents_edge_masks.sum(1).bool()
        return out  # out.shape == (batch_size * n_offspring, n_nodes, n_nodes)
