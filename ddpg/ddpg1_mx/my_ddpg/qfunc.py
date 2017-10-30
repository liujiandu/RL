#from utils import define_qfunc
import mxnet as mx


class Qfunc(object):
    def __init__(self, env, batch_size=1, ctx=mx.cpu(),
            initializer=mx.initializer.Normal(), optimizer='adam', learning_rate=0.01):
        self.env = env
        self.ctx= ctx
        self.batch_size = batch_size

        #net
        self.obs = mx.symbol.Variable("obs")
        self.act = mx.symbol.Variable("act")
        self.qval = self.define_qfunc()
        self.yval = mx.symbol.Variable("yval")
        self.loss = self.define_loss()
        
        #updater
        self.updater = mx.optimizer.get_updater(mx.optimizer.create(optimizer,
            learning_rate=learning_rate))
        
        #executor
        self.input_shapes ={
                "obs":(self.batch_size, self.env.observation_space.flat_dim),
                "act":(self.batch_size, self.env.action_space.flat_dim),
                "yval":(self.batch_size, 1) }
        self.executor = self.loss.simple_bind(ctx=ctx, **self.input_shapes)
    	self.arg_arrays = self.executor.arg_arrays
    	self.grad_arrays = self.executor.grad_arrays
    	self.arg_dict = self.executor.arg_dict
        
        #initializer
        self.initializer = initializer
        self.initialize()
        
        #target_net_executor
        self.target_net_executor = self.target_net()

    def define_qfunc(self):
	net = mx.symbol.FullyConnected(data=self.obs, name="qfunc_fc1", num_hidden=32)
	net = mx.symbol.Activation(data=net, name="qfunc_relu1", act_type="relu")
	net = mx.symbol.FullyConnected(data=net, name="qfunc_fc2", num_hidden=32)
	net = mx.symbol.Activation(data=net, name="qfunc_relu2", act_type="relu")
	net = mx.symbol.Concat(net, self.act, name="qfunc_concat")
	net = mx.symbol.FullyConnected(data=net, name="qfunc_fc3", num_hidden=32)
	net = mx.symbol.Activation(data=net, name="qfunc_relu3", act_type="relu")
	qval = mx.symbol.FullyConnected(data=net, name="qfunc_qval", num_hidden=1)
	return qval

    def define_loss(self):
        loss =1.0/self.batch_size*mx.symbol.sum(mx.symbol.square(self.qval-self.yval))
        loss = mx.symbol.MakeLoss(loss, name="qfunc_loss")
        loss = mx.symbol.Group([loss, mx.symbol.BlockGrad(self.qval)])
        return loss

    def initialize(self):
    	for name, arr in self.arg_dict.items():
            if name not in self.input_shapes:
                self.initializer(mx.init.InitDesc(name), arr)

    def update_params(self, obs, act, yval):
        self.arg_dict["obs"][:] = obs
        self.arg_dict["act"][:] = act
        self.arg_dict["yval"][:] = yval
        self.executor.forward(is_train=True)
        self.executor.backward()
        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_qvals(self, obs, act):
        self.executor.arg_dict["obs"][:] = obs
        self.executor.arg_dict["act"][:] = act
        self.executor.forward(is_train=False)
        return self.executor.outputs[1].asnumpy()
    
    def target_net(self):
        target_net_input_shapes ={"obs":self.input_shapes["obs"],"act":self.input_shapes["act"]}
        target_net_executor = self.qval.simple_bind(ctx=self.ctx, **target_net_input_shapes)
        for name, arr in target_net_executor.arg_dict.items():
                if name not in self.input_shapes:
                    self.arg_dict[name].copyto(arr)
        return target_net_executor

if __name__=="__main__":
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.envs.normalized_env import normalize
    env = normalize(CartpoleEnv())
    initializer = mx.initializer.Normal()
    qfunc = Qfunc(env=env, ctx=mx.cpu(),batch_size=10, initializer=initializer,
             optimizer="adam", learning_rate=0.01)         
