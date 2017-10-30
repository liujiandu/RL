from replay_mem import ReplayMem
from utils import discount_return, sample_rewards
import rllab.misc.logger as logger
import pyprind
import mxnet as mx
import numpy as np


class DDPG(object):

    def __init__(
        self,
        env,
        policy,
        qfunc,
        strategy,
        ctx=mx.gpu(0),
        batch_size=32,
        n_episode=1000,
        memory_size=1000000,
        memory_start_size=1000,
        discount=0.99,
        max_episode_length=1000,
        eval_samples=10000,
        qfunc_updater="adam",
        qfunc_lr=1e-4,
        policy_updater="adam",
        policy_lr=1e-4,
        soft_target_tau=1e-3,
        n_updates_per_sample=1,
        include_horizon_terminal=False,
        seed=12345):

        mx.random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.ctx = ctx
        self.policy = policy
        self.qfunc = qfunc
        self.strategy = strategy
        self.batch_size = batch_size
        self.n_episode = n_episode
        self.memory_size = memory_size
        self.memory_start_size = memory_start_size
        self.discount = discount
        self.max_episode_length = max_episode_length
        self.eval_samples = eval_samples
        self.qfunc_updater = qfunc_updater
        self.qfunc_lr = qfunc_lr
        self.policy_updater = policy_updater
        self.policy_lr = policy_lr
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal = include_horizon_terminal

        self.init_net()

        # logging
        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.returns = []

    def init_net(self):

        # qfunc init
        qfunc_init = mx.initializer.Normal()
        loss_symbols = self.qfunc.get_loss_symbols()
        qval_sym = loss_symbols["qval"]
        yval_sym = loss_symbols["yval"]
        
        # define loss here
        loss = 1.0 / self.batch_size * mx.symbol.sum(
            mx.symbol.square(qval_sym - yval_sym))
        qfunc_loss = loss
        qfunc_updater = mx.optimizer.get_updater(
            mx.optimizer.create(self.qfunc_updater,
                                learning_rate=self.qfunc_lr))
        self.qfunc_input_shapes = {
            "obs": (self.batch_size, self.env.observation_space.flat_dim),
            "act": (self.batch_size, self.env.action_space.flat_dim),
            "yval": (self.batch_size, 1)}
        self.qfunc.define_loss(qfunc_loss)
        self.qfunc.define_exe(
            ctx=self.ctx,
            init=qfunc_init,
            updater=qfunc_updater,
            input_shapes=self.qfunc_input_shapes)


        # qfunc_target init
        qfunc_target_shapes = {
            "obs": (self.batch_size, self.env.observation_space.flat_dim),
            "act": (self.batch_size, self.env.action_space.flat_dim)
        	}
        self.qfunc_target = qval_sym.simple_bind(ctx=self.ctx,
                                                 **qfunc_target_shapes)
        # parameters are not shared but initialized the same
        for name, arr in self.qfunc_target.arg_dict.items():
            if name not in self.qfunc_input_shapes:
                self.qfunc.arg_dict[name].copyto(arr)

        # policy init

        policy_init = mx.initializer.Normal()
        loss_symbols = self.policy.get_loss_symbols()
        act_sym = loss_symbols["act"]
        policy_qval = qval_sym
        # note the negative one here: the loss maximizes the average return
        policy_loss = -1.0 / self.batch_size * mx.symbol.sum(policy_qval)
        policy_loss = mx.symbol.MakeLoss(policy_loss, name="policy_loss")

        policy_updater = mx.optimizer.get_updater(mx.optimizer.create(self.policy_updater, learning_rate=self.policy_lr))
        self.policy_input_shapes = {"obs": (self.batch_size, self.env.observation_space.flat_dim)}
        self.policy.define_exe(ctx=self.ctx,init=policy_init, updater=policy_updater, input_shapes=self.policy_input_shapes)
        # policy network and q-value network are combined to backpropage
        # gradients from the policy loss
        # since the loss is different, yval is not needed
        args = {}
        for name, arr in self.qfunc.arg_dict.items():
            if name != "yval":
        		args[name] = arr
        args_grad = {}
        policy_grad_dict = dict(zip(self.qfunc.loss.list_arguments(), self.qfunc.exe.grad_arrays))
        for name, arr in policy_grad_dict.items():
        	if name != "yval":
        		args_grad[name] = arr

        self.policy_executor = policy_loss.bind(
            ctx=self.ctx,
        	args=args,
            args_grad=args_grad,
            grad_req="write")
        self.policy_executor_arg_dict = self.policy_executor.arg_dict

        self.policy_executor_grad_dict = dict(zip(
            policy_loss.list_arguments(),
            self.policy_executor.grad_arrays))

        # policy_target init
        # target policy only needs to produce actions, not loss
        # parameters are not shared but initialized the same
        self.policy_target = act_sym.simple_bind(ctx=self.ctx,
                                                 **self.policy_input_shapes)
        for name, arr in self.policy_target.arg_dict.items():
            if name not in self.policy_input_shapes:
                self.policy.arg_dict[name].copyto(arr)

        #draw net,write by zhupingmin
        #mx.viz.plot_network(qval_sym).render('qval_sym')
        #mx.viz.plot_network(act_sym).render('act_sym')
        #mx.viz.plot_network(self.policy.act).render('policy')
        #mx.viz.plot_network(self.qfunc.qval).render('qfunc')

    def train(self):

        memory = ReplayMem(
            obs_dim=self.env.observation_space.flat_dim,
            act_dim=self.env.action_space.flat_dim,
            memory_size=self.memory_size)
        
        for episode in range(self.n_episode):
            logger.push_prefix("epoch #%d | " % episode)
            #logger.log("Training started")
        
            #reset the environment and stretegy when an episode ends
            obs = self.env.reset()
            self.strategy.reset()
            episode_length = 0
            episode_return = 0
            end = False

            # note action is sampled from the policy not the target policy
            while(not end and episode_length<self.max_episode_length):
                act = self.strategy.get_action(obs, self.policy)
                nxt, rwd, end, _ = self.env.step(act)

                episode_length += 1
                episode_return += rwd

                if not end and episode_length >= self.max_episode_length:
                    end = True
                    if self.include_horizon_terminal:
                        memory.add_sample(obs, act, rwd, end)
                #else
                memory.add_sample(obs, act, rwd, end)
                
                obs = nxt

                if memory.size >= self.memory_start_size:
                    for update_time in range(self.n_updates_per_sample):
                        batch = memory.get_batch(self.batch_size)
                        self.do_update(episode, batch)


            #logger.log("Training finished")
            if memory.size >= self.memory_start_size:
                #self.evaluate(eposide, memory)
                self.returns.append(episode_return)
                print("return: ", episode_return)

            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

        #self.env.terminate()
        #self.policy.terminate()
        
        
        import matplotlib.pyplot as plt
        returns = np.array(self.returns)
        print returns
        num=returns.shape[0]/10*10
        ave_returns = np.mean(returns[0:num].reshape((num/10, 10)),axis=1)
        plt.plot(range(len(ave_returns)), ave_returns)

        plt.show()
        
    
    def do_update(self, episode, batch):

        obss, acts, rwds, ends, nxts = batch

        self.policy_target.arg_dict["obs"][:] = nxts
        self.policy_target.forward(is_train=False)
        next_acts = self.policy_target.outputs[0].asnumpy()
        policy_acts = self.policy.get_actions(obss)
        self.qfunc_target.arg_dict["obs"][:] = nxts
        self.qfunc_target.arg_dict["act"][:] = next_acts
        self.qfunc_target.forward(is_train=False)
        next_qvals = self.qfunc_target.outputs[0].asnumpy()

        #print "next q:",next_qvals," reward: ",rwds#wright zhupingmin

        # executor accepts 2D tensors
        rwds = rwds.reshape((-1, 1))
        ends = ends.reshape((-1, 1))
        ys = rwds + (1.0 - ends) * self.discount * next_qvals

        # since policy_executor shares the grad arrays with qfunc
        # the update order could not be changed

        self.qfunc.update_params(obss, acts, ys)
        # in update values all computed
        # no need to recompute qfunc_loss and qvals
        qfunc_loss = self.qfunc.exe.outputs[0].asnumpy()
        qvals = self.qfunc.exe.outputs[1].asnumpy()
        self.policy_executor.arg_dict["obs"][:] = obss
        self.policy_executor.arg_dict["act"][:] = policy_acts
        self.policy_executor.forward(is_train=True)
        policy_loss = self.policy_executor.outputs[0].asnumpy()
        self.policy_executor.backward()
        self.policy.update_params(self.policy_executor_grad_dict["act"])

        # update target networks
        if(episode%10==0):
            for name, arr in self.policy_target.arg_dict.items():
                if name not in self.policy_input_shapes:
                    arr[:] = (1.0 - self.soft_target_tau) * arr[:] + \
                        self.soft_target_tau * self.policy.arg_dict[name][:]
            for name, arr in self.qfunc_target.arg_dict.items():
                if name not in self.qfunc_input_shapes:
                    arr[:] = (1.0 - self.soft_target_tau) * arr[:] + \
                        self.soft_target_tau * self.qfunc.arg_dict[name][:]

        self.qfunc_loss_averages.append(qfunc_loss)
        self.policy_loss_averages.append(policy_loss)
        self.q_averages.append(qvals)
        self.y_averages.append(ys)

    def evaluate(self, epoch, memory):

        if epoch == self.n_epochs - 1:
            logger.log("Collecting samples for evaluation")
            rewards = sample_rewards(env=self.env,
                                     policy=self.policy,
                                     eval_samples=self.eval_samples,
                                     max_episode_length=self.max_episode_length)
            average_discounted_return = np.mean(
                [discount_return(reward, self.discount) for reward in rewards])
            returns = [sum(reward) for reward in rewards]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_qfunc_loss = np.mean(self.qfunc_loss_averages)
        average_policy_loss = np.mean(self.policy_loss_averages)

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
        logger.record_tabular('AveragePolicyLoss', average_policy_loss)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))

        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
