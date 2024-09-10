import argparse
import numpy as np
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo.ppo import PPOTrainer
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.examples.envs.classes.two_step_game import TwoStepGame
#from ray.rllib.examples.models.centralized_critic_models import \
#    CentralizedCriticModel, TorchCentralizedCriticModel
from ray.rllib.examples._old_api_stack.models.centralized_critic_models import (
    YetAnotherCentralizedCriticModel,
    YetAnotherTorchCentralizedCriticModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule # no están en tf_policy y torch_policy
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.registry import register_env
# from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v3

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

n_pursuers = 5
n_sensors = 30
obs_coord = n_sensors * (5 + 3)   # 3 for speed features enabled (default)
obs_dim = obs_coord + 2     # obs_dim size = 242 for 1 pursuer (agent)
act_dim = 2                 # act_dim size = 2 for 1 pursuer (agent)
################## TF model #################################################


class YetAnotherCentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        super(YetAnotherCentralizedCriticModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # Base of the model
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)

        self.register_variables(self.model.variables())

        n_agents = n_pursuers  # ---> opp_obs and opp_acts now consist of 4 (n_puesuers - 1) different agents
        # obs = obs_dim
        # act = 2
        opp_obs_dim = obs_dim * (n_agents - 1)
        opp_acts_dim = act_dim * (n_agents - 1)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        obs = tf.keras.layers.Input(shape=(obs_dim, ), name="obs")
        opp_obs = tf.keras.layers.Input(shape=(opp_obs_dim, ), name="opp_obs")
        opp_act = tf.keras.layers.Input(shape=(opp_acts_dim, ), name="opp_act")
        concat_obs = tf.keras.layers.Concatenate(axis=1)([obs, opp_obs, opp_act])
        central_vf_dense = tf.keras.layers.Dense(16, activation=tf.nn.tanh, name="c_vf_dense")(concat_obs)
        central_vf_out = tf.keras.layers.Dense(1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(inputs=[obs, opp_obs, opp_act], outputs=central_vf_out)

        self.register_variables(self.central_vf.variables)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    # def central_value_function(self, obs, opponent_obs, opponent_actions):
    #     return tf.reshape(
    #         self.central_vf([
    #             obs, opponent_obs,
    #             tf.one_hot(tf.cast(opponent_actions, tf.int32), 2)    # waterworld has 2 actions
    #         ]), [-1])
    def central_value_function(self, obs, opponent_obs, opponent_actions):
        return tf.reshape(
            self.central_vf([
                obs, opponent_obs, opponent_actions]), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

################## Torch model #################################################


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        n_agents = n_pursuers  # ---> opp_obs and opp_acts now consist of 4 (n_puesuers - 1) different agent information
        # obs = obs_dim
        # act = 2
        opp_obs_dim = obs_dim * (n_agents - 1)
        opp_acts_dim = act_dim * (n_agents - 1)

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs,
                             model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = obs_dim + opp_obs_dim + opp_acts_dim  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    # def central_value_function(self, obs, opponent_obs, opponent_actions):
    #     input_ = torch.cat([
    #         obs, opponent_obs,
    #         torch.nn.functional.one_hot(opponent_actions.long(), 2).float()
    #     ], 1)
    #     return torch.reshape(self.central_vf(input_), [-1])

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        input_ = torch.cat([obs, opponent_obs, opponent_actions], 1)
        return torch.reshape(self.central_vf(input_), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used
#######################################################################################################################


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=7.99)


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        # [(_, opponent_batch)] = list(other_agent_batches.values())


        # ---> opponent batch now consists of 4 SampleBatches, so I concatenate them

        concat_opponent_batch = SampleBatch.concat_samples(
            [opponent_n_batch for _, opponent_n_batch in other_agent_batches.values()])
        opponent_batch = concat_opponent_batch

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = np.concatenate([opponent_batch[SampleBatch.CUR_OBS] for
                                                     _, opponent_batch in other_agent_batches.values()],
                                                    -1) # sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = np.concatenate([opponent_batch[SampleBatch.ACTIONS] for
                                                        _, opponent_batch in other_agent_batches.values()],
                                                       -1) # sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        if args.torch:
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_ACTION], policy.device)) \
                .cpu().detach().numpy()
        else:
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                sample_batch[SampleBatch.CUR_OBS], sample_batch[OPPONENT_OBS],
                sample_batch[OPPONENT_ACTION])
    else:

        # Policy hasn't been initialized yet, use zeros.
        batch_size = sample_batch[SampleBatch.CUR_OBS].shape[0]
        sample_batch[OPPONENT_OBS] = np.zeros((batch_size,obs_dim * (n_pursuers - 1))) #sample_batch[OPPONENT_OBS] = np.zeros_like([np.zeros(obs_dim * (n_pursuers - 1))])
        sample_batch[OPPONENT_ACTION] = np.zeros((batch_size,act_dim * (n_pursuers - 1)))#sample_batch[OPPONENT_ACTION] = np.zeros_like([np.zeros(act_dim * (n_pursuers - 1))])
        ### I think I don't have to change this
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }


CCPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CCPPOTFPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_tf_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])

CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    mixins=[
        TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class(config):
    if config["framework"] == "torch":
        return CCPPOTorchPolicy


CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOTFPolicy,
    get_policy_class=get_policy_class,
)


if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()


    def env_creator(args):
        return PettingZooEnv(waterworld_v3.env(n_pursuers=5, n_evaders=5, n_sensors=30))


    env = env_creator({})
    register_env("waterworld", env_creator)

    obs_space = env.observation_space
    action_space = env.action_space
    policies = {agent: (None, obs_space, action_space, {}) for agent in env.agents}

    ModelCatalog.register_custom_model(
        "cc_model", YetAnotherTorchCentralizedCriticModel
        if args.torch else YetAnotherCentralizedCriticModel)

    config = {
        "env": "waterworld",
        "batch_mode": "complete_episodes",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 1,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: agent_id),
        },
        "model": {
            "custom_model": "cc_model",
        },
        "framework": "torch" if args.torch else "tf",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(CCTrainer, config=config, stop=stop, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
