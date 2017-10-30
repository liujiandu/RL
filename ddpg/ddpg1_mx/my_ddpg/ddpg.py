from replay_mem import ReplayMem
from utils import discount_return, sample_rewards
import rllab.misc.logger as logger
import pyprind
import mxnet as mx
import numpy as np
from policy import Policy
from qfunc import Qfunc
from strategy import OUStrategy

class DDPG(object):

    def __init__(
        self,
        env,
        ctx=mx.gpu(0),
        batch_size=32,
        n_episode=1000,
        memory_size=1000000,
        memory_start_size=1000,
        discount=0.99,
        max_episode_length=1000,
        eval_samples=10000,
        qfunc_optimizer="adam",
        qfunc_lr=1e-4,
        policy_optimizer="adam",
        policy_lr=1e-4,
        soft_target_tau=1e-3,
        n_updates_per_sample=1,
        include_horizon_terminal=False,
        seed=12345):

        mx.random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.ctx = ctx
        self.batch_size = batch_size
        self.n_episode = n_episode
        self.memory_size = memory_size
        self.memory_start_size = memory_start_size
        self.discount = discount
        self.max_episode_length = max_episode_length
        self.eval_samples = eval_samples
        self.qfunc_optimizer = qfunc_optimizer
        self.qfunc_lr = qfunc_lr
        self.policy_optimizer = policy_optimizer
        self.policy_lr = policy_lr
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal = include_horizon_terminal

        self.init_net()

        
        self.qfunc_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.returns = []

    def init_net(self):
        initializer = mx.initializer.Normal()
        self.policy = Policy(env=self.env, batch_size=self.batch_size, ctx=self.ctx,
                initializer=initializer, optimizer=self.policy_optimizer,
                learning_rate=self.policy_lr)
        self.qfunc = Qfunc(env=self.env, batch_size=self.batch_size, ctx=self.ctx,
                initializer=initializer, optimizer=self.qfunc_optimizer,
                learning_rate=self.policy_lr)
        self.policy_qfunc_exe()
        self.strategy = OUStrategy(self.env)

    def policy_qfunc_exe(self):
        policy_qfunc_loss = -1.0 / self.batch_size * mx.symbol.sum(self.qfunc.qval)
        policy_qfunc_loss = mx.symbol.MakeLoss(policy_qfunc_loss, name="policy_qfunc_loss")
        
        args = {}
        for name, arr in self.qfunc.arg_dict.items():
        	if name != "yval":
        		args[name] = arr
        args_grad = {}
        qfunc_grad_dict = dict(zip(self.qfunc.loss.list_arguments(), self.qfunc.executor.grad_arrays))
        for name, arr in qfunc_grad_dict.items():
        	if name != "yval":
        		args_grad[name] = arr

        self.policy_qfunc_executor = policy_qfunc_loss.bind(ctx=self.ctx,args=args,args_grad=args_grad,grad_req="write")
        self.policy_qfunc_executor_arg_dict = self.policy_qfunc_executor.arg_dict
        self.policy_qfunc_executor_grad_dict = dict(zip(policy_qfunc_loss.list_arguments(),self.policy_qfunc_executor.grad_arrays))

        


    def train(self):
        memory = ReplayMem(obs_dim=self.env.observation_space.flat_dim, 
                           act_dim=self.env.action_space.flat_dim,
                           memory_size=self.memory_size)

        for episode in range(self.n_episode):
            logger.push_prefix("epoch #%d | " % episode)
            #logger.log("Training started")

            obs = self.env.reset()
            self.strategy.reset()
            episode_length = 0
            episode_return = 0
            end = False

            while(not end and episode_length<self.max_episode_length):

                ## note action is sampled from the policy not the target policy
                act = self.strategy.get_action(obs, self.policy)
                nxt, rwd, end, _ = self.env.step(act)

                episode_length += 1
                episode_return += rwd
                
                ##add sample in memory
                if not end and episode_length >= self.max_episode_length:
                    end = True
                    if self.include_horizon_terminal:
                        memory.add_sample(obs, act, rwd, end)
                memory.add_sample(obs, act, rwd, end)
                
                
                ##update
                if memory.size >= self.memory_start_size:
                    for update_time in range(self.n_updates_per_sample):
                        batch = memory.get_batch(self.batch_size)
                        self.do_update(episode, batch)

                ##
                obs = nxt

            #logger.log("Training finished")
            if memory.size >= self.memory_start_size:
                #self.evaluate(epoch, memory)
                print ("return ", episode_return)
                self.returns.append(episode_return)

            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
        
        ##plot
        import matplotlib.pyplot as plt
        returns = np.array(self.returns)
        num = returns.shape[0]/10*10
        ave_returns = np.mean(returns[0:num].reshape((num/10,10)),axis=1)
        plt.plot(range(len(ave_returns)), ave_returns)
        plt.show()


    def do_update(self, episode, batch):
        obss, acts, rwds, ends, nxts = batch
        ##qfunc update##
        self.policy.target_net_executor.arg_dict["obs"][:] = nxts
        self.policy.target_net_executor.forward(is_train=False)
        next_acts = self.policy.target_net_executor.outputs[0].asnumpy()
        self.qfunc.target_net_executor.arg_dict["obs"][:] = nxts
        self.qfunc.target_net_executor.arg_dict["act"][:] = next_acts
        self.qfunc.target_net_executor.forward(is_train=False)
        next_qvals = self.qfunc.target_net_executor.outputs[0].asnumpy()

        # executor accepts 2D tensors
        rwds = rwds.reshape((-1, 1))
        ends = ends.reshape((-1, 1))
        ys = rwds + (1.0 - ends) * self.discount * next_qvals

        # since policy_executor shares the grad arrays with qfunc
        # the update order could not be changed
        self.qfunc.update_params(obss, acts, ys)

        qfunc_loss = self.qfunc.executor.outputs[0].asnumpy()
        qvals = self.qfunc.executor.outputs[1].asnumpy()
        

        ##policy updata##
        policy_acts = self.policy.get_actions(obss)
        self.policy_qfunc_executor.arg_dict["obs"][:] = obss
        self.policy_qfunc_executor.arg_dict["act"][:] = policy_acts
        self.policy_qfunc_executor.forward(is_train=True)
        self.policy_qfunc_executor.backward()
        self.policy.update_params(self.policy_qfunc_executor_grad_dict["act"])

        # update target networks
        if(episode%1==0):
            for name, arr in self.policy.target_net_executor.arg_dict.items():
                if name not in self.policy.input_shapes:
                    arr[:] = (1.0 - self.soft_target_tau) * arr[:] + \
                        self.soft_target_tau * self.policy.arg_dict[name][:]
            for name, arr in self.qfunc.target_net_executor.arg_dict.items():
                if name not in self.qfunc.input_shapes:
                    arr[:] = (1.0 - self.soft_target_tau) * arr[:] + \
                        self.soft_target_tau * self.qfunc.arg_dict[name][:]

        self.qfunc_loss_averages.append(qfunc_loss)
        self.q_averages.append(qvals)
        self.y_averages.append(ys)

    def evaluate(self, epoch, memory):

        if epoch == self.n_epochs - 1:
            logger.log("Collecting samples for evaluation")
            rewards = sample_rewards(env=self.env,
                                     policy=self.policy,
                                     eval_samples=self.eval_samples,
                                     max_path_length=self.max_path_length)
            average_discounted_return = np.mean(
                [discount_return(reward, self.discount) for reward in rewards])
            returns = [sum(reward) for reward in rewards]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_qfunc_loss = np.mean(self.qfunc_loss_averages)

        logger.record_tabular('Epoch', epoch)
        if epoch == self.n_epochs - 1:
            logger.record_tabular('AverageReturn',
                              np.mean(returns))
            logger.record_tabular('StdReturn',
                              np.std(returns))
            logger.record_tabular('MaxReturn',
                              np.max(returns))
            logger.record_tabular('MinReturn',
                              np.min(returns))
            logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_qfunc_loss)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))

