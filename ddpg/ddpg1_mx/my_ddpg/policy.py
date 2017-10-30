import mxnet as mx
	
class Policy(object):
    def __init__(self, env, batch_size, ctx, initializer, optimizer, learning_rate):
        self.env =env
        self.batch_size = batch_size
        self.ctx =ctx
        
        #define net
	self.obs = mx.symbol.Variable("obs")
        self.act = self.define_policy(self.obs)
        
        #updater
        self.updater = mx.optimizer.get_updater(mx.optimizer.create(optimizer,learning_rate=learning_rate))
        
        #executor
        self.input_shapes = {"obs":(self.batch_size, self.env.observation_space.flat_dim)}
	self.executor = self.act.simple_bind(ctx=self.ctx, **self.input_shapes)
	self.arg_arrays = self.executor.arg_arrays
	self.grad_arrays = self.executor.grad_arrays
	self.arg_dict = self.executor.arg_dict
        
        #initializer
        self.initializer = initializer
        self.initialize()

        #executor_one
        one_input_shapes = {"obs":(1,self.input_shapes["obs"][1])}
        self.executor_one = self.executor.reshape(**one_input_shapes)
	self.arg_dict_one = self.executor_one.arg_dict
        
        #target_net_executor
        self.target_net_executor = self.target_net()

    def define_policy(self, obs):
	net = mx.symbol.FullyConnected(data=obs, name="policy_fc1", num_hidden=32)
	net = mx.symbol.Activation(data=net, name="policy_relu1", act_type="relu")
	net = mx.symbol.FullyConnected(data=net, name="policy_fc2", num_hidden=32)
	net = mx.symbol.Activation(data=net, name="policy_relu2", act_type="relu")
	net = mx.symbol.FullyConnected(data=net,name="policy_fc3",num_hidden=self.env.action_space.flat_dim)
	action = mx.symbol.Activation(data=net, name="act", act_type="tanh")
	return action

    def initialize(self): 
	for name, arr in self.arg_dict.items():
		if name not in self.input_shapes:
			self.initializer(mx.init.InitDesc(name),arr)
	
    def update_params(self, grad_form_top):
	self.executor.forward(is_train=True)
	self.executor.backward([grad_form_top])
	for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
		weight, grad = pair
		self.updater(i, grad, weight)
	
    def get_actions(self, obs):
	self.arg_dict["obs"][:]=obs
	self.executor.forward(is_train=False)
	return self.executor.outputs[0].asnumpy()


    def get_action(self, obs):
	self.arg_dict_one["obs"][:]=obs
	self.executor_one.forward(is_train=False)
	return self.executor_one.outputs[0].asnumpy()
    
    def target_net(self):
	target_net_executor = self.act.simple_bind(ctx=self.ctx, **self.input_shapes)
        for name, arr in target_net_executor.arg_dict.items():
            if name not in self.input_shapes:
                    self.arg_dict[name].copyto(arr)
        return target_net_executor

if __name__=="__main__":
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.envs.normalized_env import normalize
    env = normalize(CartpoleEnv())
    initializer = mx.initializer.Normal()
    policy = Policy(env=env, ctx = mx.cpu(), batch_size=10,
            initializer=initializer, optimizer='adam', learning_rate=0.01)


