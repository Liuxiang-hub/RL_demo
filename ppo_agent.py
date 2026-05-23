import torch
from torch import device, nn
from torch.distributions import Normal
import numpy as np
import torch.optim as optim

# 定义设备 如果有可用的GPU则使用GPU，否则使用CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

                                             # 1. 定义Actor网络
# 定义Actor网络class. Actor网络构建了四层全连接层 用于生成动作的概率分布 
# 类内方法forward 输入为状态 输出为动作的均值和标准差
# 类内方法select_action 输入为状态 调用forward方法根据动作的均值和标准差 从正态分布中采样动作  

class Actor(nn.Module):  # 类关系：Actor网络继承自nn.Module类
    def __init__(self, state_dim, action_dim, hidden_dim=256):     # 参数: 状态维度, 动作维度, 隐藏层维度默认256
        super(Actor, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 定义第一层全连接层 输入维度为状态维度 输出维度为隐藏层维度
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        # 定义第二层全连接层 输入维度为隐藏层维度 输出维度为隐藏层维度
        self.fc_mean = nn.Linear(hidden_dim, action_dim) # 定义第三层全连接层 输入维度为隐藏层维度 输出维度为动作维度
        self.fc_std = nn.Linear(hidden_dim, action_dim)  # 定义第四层全连接层 输入维度为隐藏层维度 输出维度为动作维度
        
        self.relu = nn.ReLU()  # 定义ReLU激活函数 Relu特点：输出非负，梯度恒为1
        self.tanh = nn.Tanh()  # 定义Tanh激活函数 Tanh特点：输出在-1到1之间，梯度在-1到1之间，非线性
        self.softplus= nn.Softplus()  # 定义Softplus激活函数 Softplus特点：输出非负，梯度在0到1之间，非线性


    def forward(self, x):  # 前向传播函数
        x = self.relu(self.fc1(x))  # 第一层全连接层后接ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层后接ReLU激活函数
        mean = self.tanh(self.fc_mean(x)) * 2  # 第三层全连接层后接Tanh激活函数  乘以2将输出范围调整到-2到2之间
        std = self.softplus(self.fc_std(x)) + 1e-3  # 第四层全连接层后接Softplus激活函数 输出动作的标准差 加上1e-3是为了避免标准差为0导致数值不稳定
        return mean, std      # 返回动作的均值和标准差 用于生成动作的概率分布
    
    
        # select_action()方法根据当前状态在动作空间 正态分布 中选择动作
        #直接选最大值 (greedy) ❌ → 容易陷入局部最优，缺乏探索
        #概率分布采样 ✅ → 有机会探索新动作，发现更好的策略
    def select_action(self, state):  # 根据当前状态选择动作

        # 情况	                        是否需要梯度
        # 训练 (更新权重)	           ✅ 需要，计算损失→反向传播
        # 推理/选择动作 (不更新权重)	❌ 不需要，节省显存、加速计算

        with torch.no_grad():  # 禁用梯度计算  避免计算梯度时的内存占用
            mu, sigma = self.forward(state)  # 前向传播获取动作的均值和标准差
            dist = torch.distributions.Normal(mu, sigma)  # 创建正态分布对象  
            action = dist.sample()  # 从正态分布中采样一个动作 意义：根据概率分布随机选择一个动作，提升探索能力
            action = action.clamp(-2, 2)  # 将采样的动作限制在-2到2之间  意义：确保动作在环境允许的范围内   

        return action  # 返回采样的动作和动作的对数概率 用于计算损失函数时使用



                                            # 2. 定义Critic网络  
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):     # 参数: 状态维度, 隐藏层维度默认256
        super(Critic, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 定义第一层全连接层 输入维度为状态维度 输出维度为隐藏层维度
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        # 定义第二层全连接层 输入维度为隐藏层维度 输出维度为隐藏层维度
        self.fc3 = nn.Linear(hidden_dim, 1)          # 定义第三层全连接层 输入维度为隐藏层维度 输出维度为1

        self.relu = nn.ReLU()  # 定义ReLU激活函数 Relu特点：输出非负，梯度恒为1

    def forward(self, x):  # 前向传播函数
        x = self.relu(self.fc1(x))  # 第一层全连接层后接ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层后接ReLU激活函数
        value = self.fc3(x)  # 第三层全连接层输出状态值value
        return value  # 返回状态值value 用于评估当前状态的价值
    # value是Critic网络的输出,用于评估动作的价值估计,即当前状态 不管做任何动作, 未来奖励的预估总和
    # value 会被用于计算优势函数,即当前动作的价值估计与真实奖励的差值
    # 优势函数用于衡量当前动作的好坏,如果优势函数为正,则说明当前动作比平均动作更好,反之则更差（平均·动作是指在当前状态下所有可能动作的平均表现）
    # 优势函数的计算公式为: A(t) = δ_t +  （γ入）δ_t+1 + (γ入)**2 δ_t+2 + ... + (γ入)**(T-t-1) δ_T-1  
    # 其中: δ_t = r + γV(t+1) - V(t)   δ_t是当前动作的TD误差, γ是折扣因子, δ(t+1)是下一个状态的TD误差, T是当前轮的总步骤数, t是当前步骤数
    # 其中: r是当前奖励, γ是折扣因子, V(t+1)是下一个状态的价值估计, V(t)是当前状态的价值估计
    



  
                                             # 神经网络结构图
#┌─────────────────────────────────────────────────┬─────────────────────────────────────────────────┐
#│           🎯 策略网络 (Actor)                    │           💎 价值网络 (Critic)                  │
#├─────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#│                                                 │                                                 │
#│  Input:  state_dim                              │  Input:  state_dim                              │
#│          ↓                                          ↓                                             │          
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     fc1       │ state_dim → hidden_dim      │     fc1       │ state_dim → hidden_dim          │
#│  │ Linear + ReLU │                             │ Linear + ReLU │                                 │
#│  └───────────────┘                             └───────────────┘                                 │   
#│          ↓                                            ↓                                             │
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     fc2       │ hidden_dim → hidden_dim     │     fc2       │ hidden_dim → hidden_dim        │
#│  │ Linear + ReLU │                             │ Linear + ReLU │                                 │
#│  └───────────────┘                             └───────────────┘                                 │
#│          ↓                                            ↓                                             │
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     fc3       │ hidden_dim → action_dim     │     fc3       │ hidden_dim → 1                  │
#│  │   Linear*2    │                             │   Linear      │                                 │
#│  └───────────────┘                             └───────────────┘                                 │
#│       ↓     ↓                                         ↓                                             │
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     μ, σ      │  动作均值和标准差             │    value      │  状态价值 V(s)                  │
#│  │ Normal()采样   │  用于构建动作分布            │               │  评估当前状态好坏                │
#│  └───────────────┘                             └───────────────┘                                 │
#│          │                                          │                                             │
#└──────────┼──────────────────────────────────────────┼─────────────────────────────────────────────┘
#           │                                          │
#           ▼                                          ▼
#    ┌─────────────┐                            ┌─────────────┐
#    │   action    │                            │     V(s)    │
#    │  发送给dist  │                            │  计算优势函数│
#    │  供dist采样  │                            │  A(s) = r + γV(s') - V(s)
#    └─────────────┘                            └─────────────┘ 








                                             # 3. 定义回放缓冲区 
class ReplayMemory:  # 回放缓冲区   
    def __init__(self, batch_size):  # 参数: 批次大小
        self.state_capacity = []
        self.action_capacity = []
        self.reward_capacity = []
        self.value_capacity = []
        self.done_capacity = []
        self.batch_size = batch_size  # 批次大小 用于更新智能体参数时的样本采样


    def add_memory(self, state, action, reward, value, done):  # 添加记忆
        self.state_capacity.append(state)
        self.action_capacity.append(action)
        self.reward_capacity.append(reward)
        self.value_capacity.append(value)
        self.done_capacity.append(done)


    def sample(self):  # 样本采样
        # 打乱缓冲区中的样本顺序  避免训练时样本之间的相关性  提高训练效果
        num_states = len(self.state_capacity)  # 获取当前缓冲区中状态的数量 比如50
        batch_start_points = np.arange(0, num_states, self.batch_size)  # 计算每个批次的起始索引 比如[0, 25, 50]
        memory_indices  = np.arange(num_states)  # 获取当前缓冲区中所有样本的索引 比如[0, 1, 2, ..., 49]
        np.random.shuffle(memory_indices)  # 随机打乱样本的索引 顺序 打乱后可能是[49, 48, 47, ..., 0]
        batches = [memory_indices[i:i+self.batch_size] for i in batch_start_points]  
        # batches是一个列表，里面有num_states//batch_size 个元素，每个元素是一个长度为batch_size的样本索引列表
        # 把打乱后的样本索引根据批次batch_size大小进行分组
        # 方便后续根据这些索引从缓冲区中提取对应的状态、动作、奖励等数据进行训练

        return (
            np.array(self.state_capacity),  # 返回memory中所有状态的数组（没有打乱顺序） 并转换为numpy数组 方便后续训练时使用
            np.array(self.action_capacity),  # 返回所有动作的数组 并转换为numpy数组 方便后续训练时使用
            np.array(self.reward_capacity), 
            np.array(self.value_capacity), 
            np.array(self.done_capacity),  
            batches  # 返回打乱后的批次索引列表 用于后续训练时按批次提取数据 例如[[49, 48, 47, ..., 25], [24, 23, 22, ..., 0]]
        )
       

    def clear_memory(self):  # 清空记忆
        print("clear_memory")
        self.state_capacity = []
        self.action_capacity = []
        self.reward_capacity = []
        self.value_capacity = []
        self.done_capacity = []





                                            #  计算优势函数A例子
# 优势函数的计算公式为: A(t) = δ_t +  （γ入）δ_t+1 + (γ入)**2 δ_t+2 + ... + (γ入)**(T-t-1) δ_T-1  
                    # A(t) = A(t+1) * γ入 + δ_t
    # 其中: δ_t = r + γ*V(t+1) - V(t) 
# 假设已经收集完一条T步的轨迹，拿到了所有数据
# rewards: [r0, r1, ..., rT-1]
# values:   [V(s0), V(s1), ..., V(sT)]   value来源：critic神经网络根据state输出的状态价值
# gamma (γ), lam (λ):  超参数，比如0.99和0.95

# 1. 先算所有TD误差δ_t，一步就能算完
#deltas = []
#for t in range(T):
#    delta = rewards[t] + gamma * values[t+1] - values[t]
#    deltas.append(delta)
#
## 2. 从后往前算优势A_t，也是一步循环
#advantages = [0] * T
## 最后一步的A，就是它自己的δ
#advantages[-1] = deltas[-1]
#for t in reversed(range(T-1)):  # 从T-2一直倒推到0
#    advantages[t] = deltas[t] + gamma * lam * advantages[t+1]

# 最后得到T个优势函数值advantages: [A0, A1, ..., A(T-1)]，每个A_t衡量了在状态s_t下动作a_t的好坏程度，正数表示比平均动作更好，负数表示更差
# A的作用是指导策略网络（Actor）更新参数，让它更倾向于选择那些带来更高奖励的动作，
# 同时也指导价值网络（Critic）更新参数，让它更准确地评估动作的价值。
# 优势函数A的值越大，表示该动作比平均动作更好，反之则更差。

# PPO 论文并行N个agent 即N个Actor网络，同时与环境交互,每个agent收集一条轨迹,每条轨迹包含T步数据,
# 总共收集N*T步数据,然后一起更新智能体参数,这样可以更高效地利用数据,加速训练过程。
# 具体是从NT个A(t)中随机抽取batch_size（M）个样本,然后根据这些样本更新智能体参数.




                                        # 4. 定义智能体类
class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        # 定义超参数
        self.LR_ACTOR = 1e-4  # Actor网络的学习率
        self.LR_CRITIC = 1e-4  # Critic网络的学习率
        self.GAMMA = 0.99      # 折扣因子 用于计算优势函数A和 TD误差δ_t 时的折扣因子
        self.LAMBDA = 0.95     # 折扣因子 用于计算优势函数A时的折扣因子
        self.K_EPOCHS = 10     # 重复训练的轮数
        self.NUM_EPOCHS = 10   
        # K epochs 就是：用同一批收集到的数据，重复训练 K 次。
        # 在你这篇 PPO 论文里，它是这么写的：
        # Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT
        # 翻译成大白话：
        # 针对策略参数 θ，用 K 个 epoch、小批量M 大小不超过总数据量 NT 的方式，优化PPO的 代理目标函数 L。

        self.EPSILON = 0.2     # PPO的截断参数 用于限制策略更新时的步长
        # PPO的核心思想是限制策略更新的幅度，防止过大的更新导致性能崩溃。
        # 通过设置一个截断参数 ε，当新旧策略之间的概率比超过 1+ε 或小于 1-ε 时，就会对目标函数进行截断，
        # 限制更新的步长在合理范围内。

        self.actor = Actor(state_dim, action_dim).to(device)  # 创建Actor网络实例  # 创建Actor网络实例并将其移动到设备上
        self.old_actor = Actor(state_dim, action_dim).to(device)  # 创建旧的Actor网络实例  # 创建旧的Actor网络实例并将其移动到设备上
        self.critic = Critic(state_dim).to(device)             # 创建Critic网络实例
        self.state_dim = state_dim      # 状态维度
        self.action_dim = action_dim     # 动作维度 
        self.batch_size = batch_size     # 批次大小 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)  # Actor网络的优化器
        # Adam是一种自适应学习率优化器，适用于训练神经网络。它通过计算梯度的一阶矩估计和二阶矩估计来更新参数。
        # parameters()来自Actor网络实例，继承nn.Module类，返回Actor网络中所有可训练的参数
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)  # Critic网络的优化器
        #parameters()方法返回网络中所有可训练参数的迭代器，lr是学习率，Adam是一种自适应学习率优化算法，适用于训练神经网络
        self.replay_buffer = ReplayMemory(batch_size)  # 用于存储经验回放的缓冲区     


    def get_action(self, state):  # 根据当前状态获取动作
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 将状态转换为浮点数张量并添加一个维度，
        #将其移动到设备上
        action = self.actor.select_action(state)  # 调用Actor网络的select_action方法根据当前状态获取动作
        value = self.critic.forward(state)  # 调用Critic网络的forward方法根据当前状态获取状态价值
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]  
    # 返回动作和状态价值 用于与环境交互和存储经验回放时使用
    # detach()方法用于从计算图中分离张量，返回一个新的张量，该张量与原始张量共享数据，但不参与梯度计算。
    # 这样可以将张量转换为NumPy数组，方便后续的数值计算和存储。

    def update(self):  # 更新智能体参数
        self.old_actor.load_state_dict(self.actor.state_dict())  # 将新的Actor网络参数加载到旧的Actor网络中
        #load_state_dict()方法(来源Actor.Module类)用于将一个字典加载到Actor网络实例中，字典中包含网络的所有参数
        for epoch_i in range(self.NUM_EPOCHS): # 重复训练的轮数
            states, actions, rewards, values, dones, batches = self.replay_buffer.sample()  # 从回放缓冲区中采样数据
            # sample()方法返回回放缓冲区中的状态、动作、奖励、状态值、是否结束和批次索引列表
            # 这些数据将用于更新智能体的参数
            T = len(rewards)  # 获取当前批次中状态的数量 即当前批次的样本数量
            Advantages = np.zeros(T, dtype=np.float32)  # 初始化优势函数数组 用于存储每个样本的优势函数值

            for t in range(T):  # 计算每个样本的优势函数值
                discount = 1  # 初始化折扣因子
                A_t = 0  # 初始化当前样本的优势函数值
                for k in range(t, T-1):  # 计算优势函数A
                    A_t += discount * ( rewards[k] + self.GAMMA * values[k+1] * (1 - int(dones[k])) - values[k] )   
                    # 计算TD误差δ_t并累积到A_t中
                    discount *= self.LAMBDA * self.GAMMA  # 更新折扣因子 
                    Advantages[t] = A_t # 将当前样本的优势函数值赋值给Advantages数组的第t个元素

            # 计算完所有样本的优势函数值后,就可以根据这些优势函数值来更新智能体的参数了
            with torch.no_grad():  # 禁用梯度计算  避免计算梯度时的内存占用
                Advantages_tensor = torch.FloatTensor(Advantages).unsqueeze(1).to(device)  
                # 将优势函数值转换为张量 unsqueeze作用是添加一个维度 例如原来是(50,) 变成(50, 1) 方便后续计算
                values_tensor = torch.FloatTensor(values).to(device)  # 将状态值转换为张量 并添加一个维度  

            states_tensor = torch.FloatTensor(states).to(device)  # 将状态转换为张量 
            actions_tensor = torch.FloatTensor(actions).to(device)  # 将动作转换为张量 
            for batch_indices in batches:  # 遍历每个批次的索引列表
                with torch.no_grad():  # 禁用梯度计算  避免计算梯度时的内存占用
                    old_mu, old_sigma = self.old_actor.forward(states_tensor[batch_indices])  # 获取旧的Actor网络根据状态输出的动作均值和标准差 
                    # states_tensor[batch_indices] 举例：batch_indices为[0, 1, 2, 3, 4] 从states_tensor中提取索引为0,1,2,3,4的样本状态
                    old_pi_dists = Normal(old_mu, old_sigma)  # 创建旧的Actor网络的动作分布对象
                old_log_probs = old_pi_dists.log_prob(actions_tensor[batch_indices])# 计算旧的Actor网络根据状态和动作输出的动作对数概率
                 # log_prob()方法返回动作在动作分布中的对数概率 用于计算旧的Actor网络根据状态和动作输出的动作对数概率
                 # 举例：actions_tensor[batch_indices] 举例：batch_indices为[0, 1, 2, 3, 4] 
                 # 从actions_tensor中提取索引为0,1,2,3,4的样本动作
                 # 然后 old_pi_dists假如是[Normal(0, 1), Normal(0.5, 1), Normal(-0.5, 1), Normal(1, 1), Normal(-1, 1)]
                 # actions_tensor[batch_indices]假如是[0.1, 0.6, -0.4, 1.2, -0.8]
                 # 那么old_log_probs就是[-0.5, -0.3, -0.2, -0.1, -0.04]
                 

                 

                mu , sigma = self.actor.forward_action(states_tensor[batch_indices])  # 获取新的Actor网络根据状态输出的动作均值和标准差
                pi_dists = Normal(mu, sigma)  # 创建新的Actor网络的动作分布对象
                log_probs = pi_dists.log_prob(actions_tensor[batch_indices])  # 计算新的Actor网络根据状态和动作输出的动作对数概率

                ratio = torch.exp(log_probs - old_log_probs)  # 计算新旧策略的概率比
                #exp()方法用于计算指数函数，log_probs - old_log_probs表示新旧策略的对数概率之差，exp()将其转换为概率比
                 # 举例：log_probs为[-0.5, -0.3, -0.2, -0.1, -0.04] old_log_probs为[-0.55, -0.27, -0.21, -0.095, -0.48] 
                 # 那么log_probs - old_log_probs就是[0.05, -0.03, 0.01, -0.005, 0.44] 
                 # ratio就是[1.0513, 0.9704, 1.0101, 0.9950, 1.5534]
                surr1 = ratio * Advantages_tensor[batch_indices]  # 计算PPO的代理目标函数L的第一部分
                # 举例：batch_indices为[0, 1, 2, 3, 4] 从Advantages_tensor中提取索引为0,1,2,3,4的样本优势函数值
                # 然后 ratio假如是[1.1, 0.9, 1.05, 0.95, 1.2] Advantages_tensor[batch_indices]假如是[0.5, -0.3, 0.2, -0.1, 0.4] 
                # 那么surr1就是[0.55, -0.27, 0.21, -0.095, 0.48]
                surr2 = torch.clamp(ratio, 1.0 - self.EPSILON, 1.0 + self.EPSILON) * Advantages_tensor[batch_indices]# 计算PPO的代理目标函数L的第二部分
                # clamp()方法用于限制ratio的值在1-ε和1+ε之间，防止过大的更新导致性能崩溃


                actor_loss = -torch.min(surr1, surr2).mean()  # 计算Actor网络的损失函数 负号表示要最小化损失函数

                batch_old_values = self.critic(states_tensor[batch_indices])    # 从状态值张量中提取当前批次的旧状态值
                batch_returns = Advantages_tensor[batch_indices] + values_tensor[batch_indices]  
                # 计算当前批次的回报值 即优势函数值加上状态值 用于计算Critic网络的损失函数
                 # 举例：batch_indices为[0, 1, 2, 3, 4] 从Advantages_tensor和values_tensor中提取索引为0,1,2,3,4的样本值
                critic_loss = nn.MSELoss()(batch_old_values,batch_returns)  # 计算Critic网络的损失函数
                # MSELoss()方法用于计算均方误差损失函数，
                # batch_old_values是Critic网络根据当前批次状态输出的状态值，batch_returns是当前批次的回报值


                self.actor_optimizer.zero_grad()  # 清空Actor网络的梯度
                actor_loss.backward()  # 反向传播计算Actor网络的梯度
                self.actor_optimizer.step()  # 更新Actor网络的参数

                self.critic_optimizer.zero_grad()  # 清空Critic网络的梯度
                critic_loss.backward()  # 反向传播计算Critic网络的梯度
                self.critic_optimizer.step()  # 更新Critic网络的参数    

        self.replay_buffer.clear_memory()  # 清空回放缓冲区中的数据  为下一轮的经验收集做好准备
          


    def save_policy(self):  # 保存模型参数
        torch.save(self.actor.state_dict(), "ppo_policy_pendulum_v1.params")  # 保存Actor网络的参数到文件
        # 保存的是Actor网络的参数，因为在PPO算法中，Actor网络是用来生成动作的核心组件，保存它的参数可以在之后加载模型时恢复智能体的策略。
        # state_dict()方法返回一个字典，包含了网络的所有参数和缓存状态
        # state_dict()是PyTorch中用于保存和加载模型参数的标准方法，保存的是网络的权重和偏置等参数，而不是整个模型结构，这样可以在加载时灵活地构建模型结构并加载参数。