from ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from policies import DeterministicMLPPolicy
from qfuncs import ContinuousMLPQ
from strategies import OUStrategy
from utils import SEED
import mxnet as mx

# set environment, policy, qfunc, strategy

env = normalize(CartpoleEnv())

policy = DeterministicMLPPolicy(env.spec)
qfunc = ContinuousMLPQ(env.spec)
strategy = OUStrategy(env.spec)

# set the training algorithm and train

algo = DDPG(
    env=env,
    policy=policy,
    qfunc=qfunc,
    strategy=strategy,
    ctx=mx.gpu(0),
    batch_size=32,
    max_episode_length=100,
    memory_start_size=10000,
    n_episode=10000,
    discount=0.99,
    qfunc_lr=1e-3,
    policy_lr=1e-4,
    seed=SEED)

algo.train()
