from functools import partial
from typing import Optional, Type, Union

from tensordict import TensorDict
import torch

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder, NonAutoregressivePolicy
)
from rl4co.models.zoo.ngo.ga import GA
from rl4co.models.zoo.gfacs.encoder import GFACSEncoder
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering
)
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.utils import merge_with_defaults


log = get_pylogger(__name__)


class NGONonAutoregressivePolicy(NonAutoregressivePolicy):
    """Implememts NGO policy based on :class:`NonAutoregressivePolicy`. We use GFACSEncoder by default.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        env_name: Name of the environment used to initialize embeddings
        temperature: Temperature for the softmax during decoding. Defaults to 1.0.
        ga_class: Class representing the GA algorithm to be used. Defaults to :class:`GA`.
        ga_kwargs: Additional arguments to be passed to the GA algorithm.
        train_with_local_search: Whether to train with local search. Defaults to True.
        train_with_ga: Whether to train with genetic algorithm. Defaults to False.
        n_population: Number of populations to be used in the NGO algorithm. Can be an integer or dictionary. Defaults to 100.
        n_offspring: Number of offsprings to be used in the NGO algorithm. Can be an integer or dictionary. Defaults to 10.
        encoder_kwargs: Additional arguments to be passed to the encoder.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        env_name: str = "tsp",
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        ga_class: Optional[Type[GA]] = None,
        ga_kwargs: dict = {},
        train_with_local_search: bool = True,
        train_with_ga: bool = False,
        n_population: Optional[Union[int, dict]] = None,
        n_offspring: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        start_node: Optional[int] = None,
        k_sparse: Optional[int] = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder_kwargs["z_out_dim"] = 2 if train_with_local_search else 1
            encoder_kwargs["k_sparse"] = k_sparse
            encoder = GFACSEncoder(env_name=env_name, **encoder_kwargs)

        self.decode_type = "multistart_sampling" if env_name == "tsp" else "sampling"

        super().__init__(
            encoder=encoder,
            env_name=env_name,
            temperature=temperature,
            train_decode_type=self.decode_type,
            val_decode_type=self.decode_type,
            test_decode_type=self.decode_type,
        )

        self.default_decoding_kwargs = {}
        self.default_decoding_kwargs["select_best"] = False
        if k_sparse is not None:
            self.default_decoding_kwargs["top_k"] = k_sparse + (0 if env_name == "tsp" else 1)  # 1 for depot
        if "multistart" in self.decode_type:
            select_start_nodes_fn = partial(self.select_start_node_fn, start_node=start_node)
            self.default_decoding_kwargs.update(
                {"multistart": True, "select_start_nodes_fn": select_start_nodes_fn}
            )
        else:
            self.default_decoding_kwargs.update(
                {"multisample": True}
            )

        # For now, top_p and top_k are only used to filter logits (not passed to decoder)
        self.top_p = top_p
        self.top_k = top_k

        self.ga_class = GA if ga_class is None else ga_class
        self.ga_kwargs = ga_kwargs
        self.train_with_local_search = train_with_local_search
        if train_with_local_search:
            assert self.ga_kwargs.get("use_local_search", False)
        self.train_with_ga = train_with_ga
        if train_with_ga:
            assert not self.train_with_local_search
        self.n_population = merge_with_defaults(n_population, train=30, val=30, test=100)
        self.n_offspring = merge_with_defaults(n_offspring, train=30, val=30, test=100)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=5, test=10)

    @staticmethod
    def select_start_node_fn(
        td: TensorDict, env: RL4COEnvBase, num_starts: int, start_node: Optional[int] = None
    ):
        if env.name == "tsp" and start_node is not None:
            # For now, only TSP supports explicitly setting the start node
            return start_node * torch.ones(
                td.shape[0] * num_starts, dtype=torch.long, device=td.device
            )
        return torch.multinomial(td["action_mask"].float(), num_starts, replacement=True).view(-1)

    def forward(
        self,
        td_initial: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        return_actions: bool = True,
        return_hidden: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        """
        Forward method. During validation and testing, the policy runs the GA algorithm to construct solutions.
        See :class:`NonAutoregressivePolicy` for more details during the training phase.
        """
        n_population = self.n_population[phase]
        n_offspring = self.n_offspring[phase]

        heatmap, _, logZ = self.encoder(td_initial)

        decoding_kwargs.update(self.default_decoding_kwargs)
        decoding_kwargs.update(
            {"num_starts": n_offspring} if "multistart" in self.decode_type else {"num_samples": n_offspring}
        )

        # Instantiate environment if needed
        if (phase != "train" or self.train_with_local_search) and (env is None or isinstance(env, str)):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)
        else:
            assert isinstance(env, RL4COEnvBase), "env must be an instance of RL4COEnvBase"

        if phase == "train":
            # Encoder: get encoder output and initial embeddings from initial state
            if self.train_with_local_search:
                logZ, ls_logZ = logZ[:, [0]], logZ[:, [1]]
            else:
                logZ = logZ[:, [0]]

            logprobs, actions, td, env = self.common_decoding(
                self.decode_type, td_initial, env, heatmap, actions, **decoding_kwargs
            )

            # Output dictionary construction
            outdict = {
                "logZ": logZ,
                "reward": unbatchify(env.get_reward(td, actions), n_offspring),
                "log_likelihood": unbatchify(
                    get_log_likelihood(logprobs, actions, td.get("mask", None), True), n_offspring
                )
            }

            if return_actions:
                outdict["actions"] = actions

            ########################################################################
            # Local search
            if self.train_with_local_search:
                # TODO: Refactor this so that we don't need to use the ga object
                ga = self.ga_class(
                    heatmap, n_population=n_population, n_offspring=n_offspring, **self.ga_kwargs
                )
                ls_actions, ls_reward = ga.local_search(
                    batchify(td_initial, n_offspring), env, actions, decoding_kwargs  # type:ignore
                )
                ls_decoding_kwargs = decoding_kwargs.copy()
                ls_decoding_kwargs["top_k"] = 0  # This should be 0, otherwise logprobs can be -inf
                ls_logprobs, ls_actions, td, env = self.common_decoding(
                    "evaluate", td_initial, env, heatmap, ls_actions, **ls_decoding_kwargs
                )
                outdict.update(
                    {
                        "ls_logZ": ls_logZ,
                        "ls_reward": unbatchify(ls_reward, n_offspring),
                        "ls_log_likelihood": unbatchify(
                            get_log_likelihood(ls_logprobs, ls_actions, td.get("mask", None), True),
                            n_offspring,
                        )
                    }
                )
                if return_actions:
                    outdict["ls_actions"] = ls_actions

            elif self.train_with_ga:
                # Run GA
                ga = self.ga_class(
                    heatmap, n_population=n_population, n_offspring=n_offspring, **self.ga_kwargs
                )
                ga_actions, ga_reward = ga.run_for_training(
                    td_initial, env, self.n_iterations[phase], decoding_kwargs, actions, td["reward"]
                )
                ga_decoding_kwargs = decoding_kwargs.copy()
                ga_decoding_kwargs.update(
                    {"num_starts": n_population}
                    if "multistart" in self.decode_type
                    else {"num_samples": n_population}
                )
                ga_logprobs, ga_actions, td, env = self.common_decoding(
                    "evaluate", td_initial, env, heatmap, ga_actions.transpose(0, 1).flatten(0, 1),
                    **ga_decoding_kwargs,
                )

                outdict["reward"] = torch.cat([outdict["reward"], ga_reward], dim=1)
                outdict["log_likelihood"] = torch.cat(
                    [
                        outdict["log_likelihood"],
                        unbatchify(  # type: ignore
                            get_log_likelihood(ga_logprobs, ga_actions, td.get("mask", None), True),
                            n_population,
                        ),
                    ],
                    dim=1,
                )
                if return_actions:
                    outdict["actions"] = torch.cat([outdict["actions"], ga_actions], dim=1)

            ########################################################################

            if return_hidden:
                outdict["hidden"] = heatmap

            return outdict

        heatmap /= self.temperature

        if self.top_k > 0:
            self.top_k = min(self.top_k, heatmap.size(-1))  # safety check
            heatmap = modify_logits_for_top_k_filtering(heatmap, self.top_k)

        if self.top_p > 0:
            assert self.top_p <= 1.0, "top-p should be in (0, 1]."
            heatmap = modify_logits_for_top_p_filtering(heatmap, self.top_p)

        ga = self.ga_class(
            heatmap, n_population=n_population, n_offspring=n_offspring, **self.ga_kwargs
        )
        actions, iter_rewards = ga.run(td_initial, env, self.n_iterations[phase], decoding_kwargs)

        out = {"reward": iter_rewards[self.n_iterations[phase] - 1]}
        out.update({f"reward_{i:03d}": iter_rewards[i] for i in range(self.n_iterations[phase])})
        if return_actions:
            out["actions"] = actions

        return out

    def common_decoding(
        self,
        decode_type: str | DecodingStrategy,
        td: TensorDict,
        env: RL4COEnvBase,
        hidden: TensorDict,
        actions: Optional[torch.Tensor] = None,
        max_steps: int = 1_000_000,
        **decoding_kwargs,
    ):
        decoding_strategy: DecodingStrategy = get_decoding_strategy(
            decoding_strategy=decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            select_start_nodes_fn=decoding_kwargs.pop("select_start_nodes_fn", None),
            store_all_logp=decoding_kwargs.pop("store_all_logp", False),
            **decoding_kwargs,
        )
        if actions is not None:
            assert decoding_strategy.name == "evaluate", "decoding strategy must be 'evaluate' when actions are provided"

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decoding_strategy.pre_decoder_hook(
            td, env, actions[:, 0] if actions is not None else None
        )

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 1 if "multistart" in self.decode_type else 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decoding_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decoding_strategy.post_decoder_hook(td, env)
        return logprobs, actions, td, env


# class NGOAutoregressivePolicy(AutoregressivePolicy):
#     """Implememts NGO policy based on :class:`AutoregressivePolicy`.
#     """
#     ######################################################################
#     ######################################################################
#     ######################################################################
#     ################################ TODO ################################ 
#     ######################################################################
#     ######################################################################
#     ######################################################################
#     raise NotImplementedError("NGOAutoregressivePolicy is not implemented yet.")
