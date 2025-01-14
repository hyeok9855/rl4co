from functools import cached_property, partial
from typing import Optional, Tuple

import torch

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.nonautoregressive.decoder import (
    NonAutoregressiveDecoder,
)
from rl4co.utils.decoding import Sampling
from rl4co.utils.ops import get_distance_matrix, unbatchify


class GA:
    """Implements the Genetic Algorithm (GA) under the NGO framework (work in progress).

    Args:
        log_heuristic: Logarithm of the heuristic matrix.
        n_population: Number of individuals in the population. Defaults to 100.
        n_offspring: Number of offspring to be generated. Defaults to 100.
            Note that each pair of parents generates only one offspring.
        mutation_rate: Rate of mutation. Defaults to 0.01.
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
        use_local_search: bool = False,
        use_nls: bool = False,
        n_perturbations: int = 5,
        local_search_params: dict = {},
        perturbation_params: dict = {},
        start_node: Optional[int] = None,
    ):
        self.log_heuristic = log_heuristic
        self.batch_size = log_heuristic.shape[0]

        self.n_population = n_population
        self.n_offspring = n_offspring
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.rank_coefficient = rank_coefficient

        self.final_actions = self.final_reward = None
        self.final_reward_cache = torch.zeros(self.batch_size, 0, device=log_heuristic.device)

        self.use_local_search = use_local_search
        assert not (use_nls and not use_local_search), "use_nls requires use_local_search"
        self.use_nls = use_nls
        self.n_perturbations = n_perturbations
        self.local_search_params = local_search_params
        self.perturbation_params = perturbation_params
        self.start_node = start_node

        self._batchindex = torch.arange(self.batch_size, device=log_heuristic.device)

    @cached_property
    def heuristic_dist(self) -> torch.Tensor:
        heuristic = self.log_heuristic.exp().detach().cpu() + 1e-10
        heuristic_dist = 1 / (heuristic / heuristic.max(-1, keepdim=True)[0] + 1e-5)
        heuristic_dist[:, torch.arange(heuristic_dist.shape[1]), torch.arange(heuristic_dist.shape[2])] = 0
        return heuristic_dist

    @staticmethod
    def select_start_node_fn(
        td: TensorDict, env: RL4COEnvBase, num_starts: int, start_node: Optional[int] = None
    ):
        if env.name == "tsp":
            if start_node is not None:
                # For now, only TSP supports explicitly setting the start node
                return start_node * torch.ones(
                    td.shape[0] * num_starts, dtype=torch.long, device=td.device
                )
            else:
                return torch.randint(td["locs"].size(1), (td.shape[0] * num_starts,), device=td.device)

        elif env.name == "cvrp":
            # TODO: start node needs to be constrained with the parents
            raise NotImplementedError("Start node selection for CVRP is not implemented")

        else:
            raise NotImplementedError(f"Start node selection for {env.name} is not implemented")

    def run(
        self, td_initial: TensorDict, env: RL4COEnvBase, n_iterations: int
    ) -> Tuple[TensorDict, Tensor, Tensor]:
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

        for _ in range(n_iterations):
            # reset environment
            td = td_initial.clone()
            _, _, population = self._one_step(td, env, population)

        action_matrix = self._convert_final_action_to_matrix()
        assert action_matrix is not None and self.final_reward is not None
        td, env = self._recreate_final_routes(td_initial, env, action_matrix)

        return td, action_matrix, self.final_reward

    def _one_step(
            self, td: TensorDict, env: RL4COEnvBase, population: dict[str, torch.Tensor]
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
        td, env, actions, reward = self._sampling(td, env, parents_mask)
        # local search, reserved for extensions
        if self.use_local_search:
            actions, reward = self.local_search(td, env, actions)

        # reshape from (batch_size * n_offspring, ...) to (batch_size, n_offspring, ...)
        reward = unbatchify(reward, self.n_offspring)
        actions = unbatchify(actions, self.n_offspring)
        # update final actions and rewards
        self._update_results(actions, reward)
        # update population
        population = self._update_population(env.name, actions, reward, population)

        return actions, reward, population  # type: ignore

    def _sampling(self, td: TensorDict, env: RL4COEnvBase, parents_mask: Optional[Tensor] = None):
        batch_indices = torch.arange(
            parents_mask.size(0), device=parents_mask.device
        ) if parents_mask is not None else None
            
        decode_strategy = Sampling(
            multistart=True,
            num_starts=self.n_offspring,
            select_start_nodes_fn=partial(self.select_start_node_fn, start_node=self.start_node),
        )

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        while not td["done"].all():
            logits, mask = NonAutoregressiveDecoder.heatmap_to_logits(
                td, self.log_heuristic, num_starts
            )

            # NGO masking
            if parents_mask is not None:
                # # Dense Version
                # new_mask = mask * parents_mask[batch_indices, td["action"]]

                # Sparse Version (to avoid OOM)
                new_mask = mask * parents_mask.to_dense()[batch_indices, td["action"]]

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
        self, td: TensorDict, env: RL4COEnvBase, actions: Tensor
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
        td_cpu = td.detach().cpu()  # Convert to CPU in advance to minimize the overhead from device transfer
        td_cpu["distances"] = get_distance_matrix(td_cpu["locs"])
        # TODO: avoid or generalize this, e.g., pre-compute for local search in each env
        actions = actions.detach().cpu()
        best_actions = env.local_search(td=td_cpu, actions=actions, **self.local_search_params)
        best_rewards = env.get_reward(td_cpu, best_actions)

        if self.use_nls:
            td_cpu_perturb = td_cpu.clone()
            td_cpu_perturb["distances"] = torch.tile(self.heuristic_dist, (self.n_offspring, 1, 1))
            new_actions = best_actions.clone()

            for _ in range(self.n_perturbations):
                perturbed_actions = env.local_search(
                    td=td_cpu_perturb, actions=new_actions, **self.perturbation_params
                )
                new_actions = env.local_search(td=td_cpu, actions=perturbed_actions, **self.local_search_params)
                new_rewards = env.get_reward(td_cpu, new_actions)

                improved_indices = new_rewards > best_rewards
                best_actions = env.replace_selected_actions(best_actions, new_actions, improved_indices)
                best_rewards[improved_indices] = new_rewards[improved_indices]

        best_actions = best_actions.to(td.device)
        best_rewards = best_rewards.to(td.device)

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
                self.final_actions[index] = best_actions[index]
            self.final_reward[require_update] = best_reward[require_update]

        self.final_reward_cache = torch.cat(
            [self.final_reward_cache, self.final_reward.unsqueeze(-1)], -1
        )
        return best_index

    def _recreate_final_routes(self, td, env, action_matrix):
        for action_index in range(action_matrix.shape[-1]):
            actions = action_matrix[:, action_index]
            td.set("action", actions)
            td = env.step(td)["next"]

        assert td["done"].all()
        return td, env

    def _convert_final_action_to_matrix(self) -> Optional[Tensor]:
        if self.final_actions is None:
            return None
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
        if env_name == "tsp":
            concated_actions = torch.cat([population["actions"], actions], dim=1)
            # concated_actions.shape == (batch_size, n_population + n_offspring, seq_len)
        else:
            raise NotImplementedError(f"Population update for {env_name} is not implemented")

        new_n = concated_actions.size(1)  # new_n = n_population + n_offspring
        pairwise_distance, concated_edge_mask = self._pairwise_distance(env_name, concated_actions)
        # pairwise_distance.shape == (batch_size, new_n, new_n)
        # edge_mask.shape == (batch_size, new_n, seq_len, seq_len)

        concated_reward = torch.cat([population["reward"], reward], dim=1)
        concated_novelty = pairwise_distance.mean(-1)
        # concated_reward.shape == concated_novelty.shape == (batch_size, new_n)

        if new_n <= self.n_population:
            population["actions"] = concated_actions
            population["edge_mask"] = concated_edge_mask
            population["reward"] = concated_reward
            population["novelty"] = concated_novelty
        else:
            # rank-based selection
            score_ranks = torch.argsort(torch.argsort(-concated_reward, dim=1), dim=1)
            novelty_ranks = torch.argsort(torch.argsort(-concated_novelty, dim=1), dim=1)
            weighted_ranks = 0.5 * score_ranks + 0.5 * novelty_ranks
            weights = 1.0 / (self.rank_coefficient * new_n + weighted_ranks)
            indices_to_keep = torch.multinomial(weights, self.n_population, replacement=False)
            batch_indices = self._batchindex.unsqueeze(1).expand(-1, self.n_population)
            population["actions"] = concated_actions[batch_indices, indices_to_keep]
            population["edge_mask"] = concated_edge_mask[batch_indices, indices_to_keep]
            population["reward"] = concated_reward[batch_indices, indices_to_keep]
            population["novelty"] = concated_novelty[batch_indices, indices_to_keep]

        return population

    def _pairwise_distance(self, env_name: str, actions: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, n, seq_len = actions.size()

        actions_flat = actions.reshape(batch_size * n, seq_len)
        # actions_flat.shape == (batch_size * n, seq_len)

        if env_name == "tsp":
            edge_index = torch.stack([actions_flat, torch.roll(actions_flat, shifts=-1, dims=1)], dim=2)
            row = edge_index[:, :, 0]
            col = edge_index[:, :, 1]
            batch_indices = torch.arange(batch_size * n, device=actions.device).view(-1, 1).expand(-1, seq_len)
            linear_indices_forward = batch_indices * (seq_len * seq_len) + row * seq_len + col
            linear_indices_backward = batch_indices * (seq_len * seq_len) + col * seq_len + row
            linear_indices = torch.cat([linear_indices_forward, linear_indices_backward], dim=1).reshape(-1)
            total_elements = batch_size * n * seq_len * seq_len
            edge_mask_flat = torch.zeros(total_elements, dtype=torch.bool, device=actions.device)
            edge_mask_flat[linear_indices] = True
            edge_mask = edge_mask_flat.view(batch_size, n, seq_len, seq_len)
            # edge_mask.shape == (batch_size, n, seq_len, seq_len)

            # # Batch Version
            # mask_flat = edge_mask.view(batch_size, n, -1).float()
            # # mask_flat.shape == (batch_size, n, seq_len * seq_len)
            # pairwise_sum = torch.matmul(mask_flat, mask_flat.transpose(1, 2))

            # For-loop Version (to avoid OOM)
            pairwise_sum = torch.zeros((batch_size, n, n), dtype=torch.float, device=actions.device)
            for b in range(batch_size):
                _edges = edge_mask[b].view(n, -1).to(torch.float)
                pairwise_sum[b] = torch.mm(_edges, _edges.transpose(0, 1))

            pairwise_distance = 1 - (pairwise_sum / (2 * seq_len))
            # pairwise_distance.shape == (batch_size, n, n)
            return pairwise_distance, edge_mask
        else:
            raise NotImplementedError(f"Pairwise distance for {env_name} is not implemented")

    def _get_parents_mask(self, population: dict[str, Tensor]) -> Optional[Tensor]:
        if len(population["reward"]) == 0 or population["reward"].size(1) < self.n_population:
            return None

        # rank-based selection
        score_ranks = torch.argsort(torch.argsort(-population["reward"], dim=1), dim=1)
        weights = 1.0 / (self.rank_coefficient * self.n_population + score_ranks)
        # weights.shape == (batch_size, n_population)

        # # Batch & Dense Version
        # weights_expanded = cast(Tensor, batchify(weights, self.n_offspring))
        # parents_indices = torch.multinomial(weights_expanded, self.n_parents, replacement=False)

        # parents_edge_masks = cast(Tensor, batchify(population["edge_mask"], self.n_offspring))
        # batch_indices = torch.arange(
        #     parents_edge_masks.size(0), device=parents_edge_masks.device
        # ).unsqueeze(1).expand(-1, self.n_parents)
        # selected_edge_masks = parents_edge_masks[batch_indices, parents_indices]
        # return selected_edge_masks.sum(1) > 0

        # For-loop & Sparse Version (to avoid OOM)
        parents_edge_masks = []
        for _ in range(self.n_offspring):
            parents_indices = torch.multinomial(weights, self.n_parents, replacement=False)
            # parents_indices.shape == (batch_size, n_parents)
            parents_edge_masks.append(
                population["edge_mask"][
                    self._batchindex.unsqueeze(1).expand(-1, self.n_parents), parents_indices
                ].to_sparse_coo()
            )
        parents_edge_masks = torch.cat(parents_edge_masks, 0)
        return parents_edge_masks.sum(1).bool()  # shape == (batch_size * n_offspring, seq_len)
