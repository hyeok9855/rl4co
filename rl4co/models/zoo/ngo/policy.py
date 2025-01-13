from functools import partial
from typing import Optional, Type, Union
import math

from tensordict import TensorDict
import torch

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.autoregressive import (
    AutoregressiveEncoder, AutoregressiveDecoder, AutoregressivePolicy
)
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
        train_with_local_search: Whether to train with local search. Defaults to False.
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
        n_population: Optional[Union[int, dict]] = None,
        n_offspring: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder_kwargs["z_out_dim"] = 2 if train_with_local_search else 1
            encoder = GFACSEncoder(env_name=env_name, **encoder_kwargs)

        super().__init__(
            encoder=encoder,
            env_name=env_name,
            temperature=temperature,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",            
        )

        self.top_p = top_p
        self.top_k = top_k

        self.ga_class = GA if ga_class is None else ga_class
        self.ga_kwargs = ga_kwargs
        self.train_with_local_search = train_with_local_search
        if train_with_local_search:
            assert self.ga_kwargs.get("use_local_search", False)
        self.n_population = merge_with_defaults(n_population, train=30, val=30, test=100)
        self.n_offspring = merge_with_defaults(n_offspring, train=30, val=30, test=100)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=5, test=10)

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
        n_offspring = self.n_offspring[phase]
        # Instantiate environment if needed
        if (phase != "train" or self.train_with_local_search) and (env is None or isinstance(env, str)):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)
        else:
            assert isinstance(env, RL4COEnvBase), "env must be an instance of RL4COEnvBase"

        if phase == "train":
            # Encoder: get encoder output and initial embeddings from initial state
            hidden, init_embeds, logZ = self.encoder(td_initial)
            if self.train_with_local_search:
                logZ, ls_logZ = logZ[:, [0]], logZ[:, [1]]
            else:
                logZ = logZ[:, [0]]

            select_start_nodes_fn = partial(
                self.ga_class.select_start_node_fn, start_node=self.ga_kwargs.get("start_node", None)
            )
            decoding_kwargs.update(
                {
                    "select_start_nodes_fn": select_start_nodes_fn,
                    # These are only for inference; TODO: Are they useful for training too?
                    # "top_p": self.top_p,
                    # "top_k": self.top_k,
                }
            )
            logprobs, actions, td, env = self.common_decoding(
                "multistart_sampling", td_initial, env, hidden, n_offspring, actions, **decoding_kwargs
            )
            td.set("reward", env.get_reward(td, actions))

            # Output dictionary construction
            outdict = {
                "logZ": logZ,
                "reward": unbatchify(td["reward"], n_offspring),
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
                    hidden,
                    n_population=self.n_population[phase],
                    n_offspring=n_offspring,
                    **self.ga_kwargs,
                )
                ls_actions, ls_reward = ga.local_search(
                    batchify(td_initial, n_offspring), env, actions  # type:ignore
                )
                ls_logprobs, ls_actions, td, env = self.common_decoding(
                    "evaluate", td_initial, env, hidden, n_offspring, ls_actions, **decoding_kwargs
                )
                td.set("ls_reward", ls_reward)
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
            ########################################################################

            if return_hidden:
                outdict["hidden"] = hidden

            return outdict

        heatmap_logits, _, _ = self.encoder(td_initial)
        heatmap_logits /= self.temperature

        if self.top_k > 0:
            self.top_k = min(self.top_k, heatmap_logits.size(-1))  # safety check
            heatmap_logits = modify_logits_for_top_k_filtering(heatmap_logits, self.top_k)

        if self.top_p > 0:
            assert self.top_p <= 1.0, "top-p should be in (0, 1]."
            heatmap_logits = modify_logits_for_top_p_filtering(heatmap_logits, self.top_p)

        ga = self.ga_class(
            heatmap_logits,
            n_population=self.n_population[phase],
            n_offspring=n_offspring,
            **self.ga_kwargs,
        )
        td, actions, reward = ga.run(td_initial, env, self.n_iterations[phase])

        out = {"reward": reward}
        if return_actions:
            out["actions"] = actions

        return out

    def common_decoding(
        self,
        decode_type: str | DecodingStrategy,
        td: TensorDict,
        env: RL4COEnvBase,
        hidden: TensorDict,
        num_starts: int,
        actions: Optional[torch.Tensor] = None,
        max_steps: int = 1_000_000,
        **decoding_kwargs,
    ):
        multistart = True if num_starts > 1 else False
        decoding_strategy: DecodingStrategy = get_decoding_strategy(
            decoding_strategy=decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            num_starts=num_starts,
            multistart=multistart,
            select_start_nodes_fn=decoding_kwargs.pop("select_start_nodes_fn", None),
            store_all_logp=decoding_kwargs.pop("store_all_logp", False),
            **decoding_kwargs,
        )
        if actions is not None:
            assert decoding_strategy.name == "evaluate", "decoding strategy must be 'evaluate' when actions are provided"

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decoding_strategy.pre_decoder_hook(td, env, actions[:, 0] if actions is not None else None)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 1 if multistart else 0
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


class NGOAutoregressivePolicy(AutoregressivePolicy):
    """Implememts NGO policy based on :class:`AutoregressivePolicy`.
    """
    ######################################################################
    ######################################################################
    ######################################################################
    ################################ TODO ################################ 
    ######################################################################
    ######################################################################
    ######################################################################
    raise NotImplementedError("NGOAutoregressivePolicy is not implemented yet.")
