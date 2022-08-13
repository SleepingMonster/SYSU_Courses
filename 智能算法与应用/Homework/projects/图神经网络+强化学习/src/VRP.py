#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
from torch import nn
import math
import tqdm


# In[3]:


BATCH_SIZE = 32
device = torch.device('cuda:2')


# In[4]:


torch.cuda.is_available()
print(torch.cuda.device_count())
print(torch.cuda.current_device())
torch.cuda.get_device_capability(device)


# In[5]:


class  AttentionEncoder(nn.Module):
    def __init__(self,hidden_dim):
        super(AttentionEncoder,self).__init__()
        self.hidden_dim=hidden_dim
        self.softmax=torch.nn.Softmax(dim=2)
        
    def forward(self,query,key,value):
        '''
        x对应query: [batch_size, node_num, node_hidden_dim]
        neighbour对应key: [batch_size, node_num, k, node_hidden_dim]
        neighbour对应value: [batch_size, node_num, k, node_hidden_dim]
        '''
        t = query.unsqueeze(2)
        score = torch.matmul(t, key.transpose(-1,-2))/math.sqrt(key.shape[3])  # 使用缩放点积模型计算score
        score = score.squeeze(2) # 从[batch_size,node_num,1, k]到[batch_size,node_num,k]
        score = self.softmax(score)  # score: [batch_size, node_num, k]
        score = score.unsqueeze(3).repeat(1,1,1,query.shape[2])  # [batch_size, node_num, k, node_hidden_dim]
        result=torch.sum(score*value,dim=2)  # 求和:[batch_size, node_num, node_hidden_dim]
        return result
        
        


# In[19]:


class environment(object): #类里还需对distance进行加一维的操作
    def __init__(self, batch_size, file_path, k,is_train):
        super(environment,self).__init__()
        # 0 常量参数
        self.batch_size = batch_size    # 1个batch中的数据量
        self.k_nearest = k      # k近邻参数
        self.total_graph, self.total_demand, self.total_distance = self.readData(file_path,is_train)  # 读文件
        self.total_A,self.total_index = self.knn()
        self.batch_num = self.total_graph.shape[0]//self.batch_size   # batch数目
        self.node_num = self.total_graph.shape[1]     # 客户数量：21(warehouse included)
        
        #self.initial_capacity = torch.max(torch.mean(self.total_demand) * 7,torch.max(self.total_demand)).to(device) # 车的初始容量：1.1108(假设大概可以3趟跑完),同时要大于最大的容量
        self.initial_capacity = torch.tensor(1.3, dtype=torch.float).to(device)
        # 1 整个epoch遍历完才更新：
        self.count = 0   # 记录当前访问到的Batch数  
        
        # 2 每个batch遍历完更新
        self.pi = torch.zeros([batch_size,1]).long()   # 设置成batch_size行，使用Cat来拼接当前时刻的访问节点集合
        self.visited = torch.zeros((batch_size, self.node_num),dtype=torch.bool) # 用来存哪个点已经被访问，仓库取值恒为1
        self.visited[ : ,0] = 1   # 将仓库赋值为1，表示仓库已被访问（具体在mask中更改）
        self.reward = torch.zeros((batch_size)) #累计reward：[batch_size]
        self.capacity = torch.tensor([[self.initial_capacity]]*batch_size)    # capacity剩余容量：[batch_size, 1]
        # 得到第0个batch的数据：graph [batch_size,node_num,2] demand[batch_size, node_num] distance[batch_size,node_num,node_num]
        self.graph, self.demand, self.distance, self.A, self.index = self.getData()  
        # 初始化mask，将仓库设置为不可访问（t=1不可访问仓库）
        self.mask = torch.zeros([batch_size, self.graph.shape[1]],dtype=torch.bool) #[batch_size, node_num]
        self.mask[:, 0] = 1
        # 调用KNN：A为邻接矩阵：[batch_size,node_num,node_num,1]; index为邻居的index: [batch_size,node_num,k,1]
        # sequential_decoder中只有初始一次
        #self.A, self.index = self.knn()
        self.graph, self.demand, self.distance = self.graph.to(device), self.demand.to(device), self.distance.to(device)
        self.A, self.index = self.A.to(device), self.index.to(device)
        self.count=0 ########新增
        
    
    def readData(self, file_path,is_train):
        '''
        读文件:返回graph, demand, distance
        '''
        if is_train==True:
            data = np.load(file_path)
            graph=torch.from_numpy(data['graph'][:,:11,:])
            # 处理demand[50000,21]变为[50000,20,1]
            demand = data['demand'][:,:11]
            demand = torch.from_numpy(demand)
            # 处理distance[50000,21,21]变为[50000,21,21,1]
            distance = torch.from_numpy(data['dis'][:,:11,:11])
            return graph, demand, distance
        else:
            data = np.load(file_path)
            graph=torch.from_numpy(data['graph'])
            # 处理demand[50000,21]变为[50000,20,1]
            demand = data['demand']
            demand = torch.from_numpy(demand)
            # 处理distance[50000,21,21]变为[50000,21,21,1]
            distance = torch.from_numpy(data['dis'])
            return graph, demand, distance
    
    def getData(self):
        '''
        返回分batch_size之后的数据集
        '''
        count = self.count
        self.count += 1   
        if (count+1)*self.batch_size <= self.total_graph.shape[0]:
            return self.total_graph[count*self.batch_size:(count+1)*self.batch_size], self.total_demand[count*self.batch_size:(count+1)*self.batch_size].clone(),self.total_distance[count*self.batch_size:(count+1)*self.batch_size],self.total_A[count*self.batch_size:(count+1)*self.batch_size], self.total_index[count*self.batch_size:(count+1)*self.batch_size]
        else:
            return self.total_graph[count*self.batch_size: ], self.total_demand[count*self.batch_size: ].clone(), self.total_distance[count*self.batch_size: ],self.total_A[count*self.batch_size: ],self.total_index[count*self.batch_size: ]
    
    def step(self, action): #step这里可以返回是否已经全部访问过了
        '''
        输入agent选择的动作，更新车辆状态，客户demand，返回奖励，并根据action修改self.visited元素
        @param action: [batch_size, 1]
        '''
        self.pi = torch.cat((self.pi, action), dim=1)  # 竖着拼接
        now_demand = torch.gather(self.demand, 1, action)  # now_demand为本次选出来的下一个节点，[batch_size, 1]
        # 对capacity进行更新：访问新节点则减去对应的demand，访问仓库则设置为初始capacity
        self.capacity[now_demand>0] = self.capacity[now_demand>0] - now_demand[now_demand>0]  # compacity: [batch_size, 1]
        self.capacity[now_demand==0] = self.initial_capacity
        # 更新完之后demand变为0（即已服务）
        #这里的demand修改方法改过
        action1= torch.squeeze(action,1)  
        index = torch.arange(0,self.batch_size)
        self.demand[index ,action1] = 0    
        #得到累计reward
        self.reward += self.get_reward(action,self.pi[:,-2]) #[batch_size]
        
        #根据action改visited（use stupid way to update)
#         index = torch.arange(0,self.batch_size)
#         action1 = torch.squeeze(action,1)    # action1: [batch_size]
        self.visited[index, action1] = 1
        
    def get_reward(self,action, last_visited):
        '''根据当前选择的动作action和上次访问的节点，返回这次路径的负长度(reward定义是负的)
        action: [batch_size,1]
        last_visited: 上次访问的节点，[batch_size]
        return: 这次访问距离的负值 [batch_size]
        self.distance [batch_size,node_num,node_num]'''
        index = torch.arange(0, self.batch_size)
        action1 = torch.squeeze(action,1)    # action1: [batch_size]
        reward = self.distance[index,last_visited, action1]
        #reward=torch.index_select(torch.index_select(self.distance,0,last_visited1),1,action1)
        return reward
    
    def knn(self):
        '''
        self.distance：[batch_size,node_num,node_num]
        k: k近邻参数
        return: A[batch_size,node_num,node_num,1]
                index[batch_size,node_num,k,1]
        '''
        index = torch.argsort(self.total_distance, dim=-1, descending=True)
        index = index[:, :, 1:self.k_nearest+1]  # 取前k个
        A = torch.zeros((self.total_distance.shape[0], self.total_distance.shape[1], self.total_distance.shape[2]))  # 初始化A
        A[:,] = torch.eye(self.total_distance.shape[1],dtype=torch.int) # 生成对角矩阵1
        A = 0 - A  # 将对角线元素变成-1
        # 设置knn
        # ?trick
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                A[i,j,index[i,j]] = 1
        A = A.reshape((A.shape[0], A.shape[1], A.shape[2], 1))
        index=index.reshape((index.shape[0],index.shape[1],index.shape[2],1))
        return A, index
    
    def reset(self):
        '''
        在上个batch训练完之后，重新对数据赋值
        '''
        #更新graph,demand,distance A index
        self.graph, self.demand, self.distance,self.A,self.index = self.getData()
        self.graph = self.graph.to(device)
        self.demand = self.demand.to(device)
        self.distance = self.distance.to(device)
        self.A = self.A.to(device)
        self.index=self.index.to(device)
        self.pi = torch.zeros([self.batch_size,1]).long() .to(device)
        
        self.mask = torch.zeros([self.batch_size, self.node_num],dtype=torch.bool).to(device) #[batch_size, node_num]
        self.mask[:,0]=1
        self.reward = torch.zeros((self.batch_size)).to(device)
        self.visited = torch.zeros((self.batch_size,self.node_num), dtype=torch.bool).to(device) #用来存哪个点已经被访问，仓库取值恒为1
        self.visited[:0] = 1
        self.capacity = torch.tensor([[self.initial_capacity]] * self.batch_size).to(device)  # 更新env中的capacity
    
    def reset_in_sequential(self):
        '''
        After sample policy，reset for greedy policy.
        '''
        self.pi = torch.zeros([self.batch_size,1]).long().to(device)   # 设置成batch_size行，使用Cat来拼接当前时刻的访问节点集合
        self.visited = torch.zeros((self.batch_size,self.node_num),dtype=torch.bool).to(device) # 用来存哪个点已经被访问，仓库取值恒为1
        self.visited[ : ,0] = 1   # 将仓库赋值为1，表示仓库已被访问（具体在mask中更改）
        self.reward = torch.zeros((self.batch_size)).to(device) #累计reward：[batch_size]
        self.mask = torch.zeros([self.batch_size, self.graph.shape[1]],dtype=torch.bool).to(device) #[batch_size, node_num]
        self.mask[:, 0] = 1
        self.capacity = torch.tensor([[self.initial_capacity]] * self.batch_size).to(device)  # 更新env中的capacity
        self.count-=1;
        _,self.demand,_,_,_ = self.getData(); #########新增
        self.demand = self.demand.to(device)
        
    def update_mask(self,action):
        '''
        action [batch_size,1]
        self.mask [batch_size, node_num]
        '''
        self.mask = self.visited.clone()   # 先对visited进行深拷贝，将已经访问过的客户mask掉
        action1 = action.squeeze(-1)  # action1: [batch_size]
        self.mask[action1!=0, 0] = 0  # 对仓库进行处理：只要本点不是仓库，下次就可以访问
        self.mask[(self.capacity - self.demand)<0] = 1    # 对超出需求的客户：mask掉
        is_all_visit = torch.min(self.visited, dim=1)[0]     # 计算出该行中的客户是不是都被访问: [batch_size, 1]
        self.mask[is_all_visit==1, 0] = 0    # 如果该行的所有客户都被访问，则设置仓库可以访问
        
    def is_all_end(self):
        '''
        判断batch中的每一条数据是不是都已经访问过所有客户了，返回bool
        '''
        min_value = torch.min(self.visited)   # 整个batch的最小值，如果是0表示还有客户没被满足
        #修改：
        #如果访问完了，让pi加上仓库，让reward算上最后一个客户到仓库的距离
        if min_value!=0:
            temp = torch.zeros((self.batch_size,1)).long().to(device)
            self.pi = torch.cat((self.pi,temp), dim=1)
            self.reward += self.get_reward(temp,self.pi[:,-2])
            return True
        return False
        #return False if min_value==0 else True
    
    def get_result_matrix(self):
        '''
        根据self.pi生成一个[batch_size, node_num, node_num]大小的0,1矩阵。1 - 边被选（且不用管后面的0）
        self.pi: [batch_size, max_length]
        '''
        max_length = self.pi.shape[1]
        result = torch.zeros((self.batch_size, self.node_num, self.node_num))
        start_node = self.pi[: ,0:max_length-1]   # 起始点，[batch_size, max_length-1]
        end_node = self.pi[: , 1:max_length]      # 终止点，[batch_size, max_length-1]
        index = torch.arange(0, self.batch_size)
        last_visit_node = self.pi[:,-1] # 取出pi最后访问的点，并设为1（即形成回路）
        result[index,last_visit_node,0] = 1  
        index = index.unsqueeze(-1).repeat(1, max_length-1)
        result[index, start_node, end_node] = 1
        result[:,0,0] = 0    # 去掉不合法的(0,0)
        return result


# In[20]:


# env = environment(32, "VRP/data/G-20-training.npz", k=3)
# print(env.graph.shape)
# print(torch.mean(env.demand))

# action=torch.ones((BATCH_SIZE,1), dtype=torch.int)
# env.step(action.long())


# In[21]:


class GCNLayer(nn.Module):
    def __init__(self,hidden_dim):
        super(GCNLayer,self).__init__()
        self.W_node = nn.Linear(hidden_dim,hidden_dim)
        self.V_node_in = nn.Linear(hidden_dim,hidden_dim)
        self.V_node = nn.Linear(2*hidden_dim,hidden_dim)
        self.attn = AttentionEncoder(hidden_dim)
        self.Relu = nn.ReLU()
        self.ln1_node = nn.LayerNorm(hidden_dim)
        self.ln2_node = nn.LayerNorm(hidden_dim)
        
        #edge
        self.W_edge = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge = nn.Linear(2*hidden_dim, hidden_dim)
        self.W1_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W2_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W3_edge = nn.Linear(hidden_dim, hidden_dim)
        self.Relu = nn.ReLU()
        self.ln1_edge = nn.LayerNorm(hidden_dim)
        self.ln2_edge = nn.LayerNorm(hidden_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x, e, neighbor_index):
        '''
        @param x [batch_size, node_num, node_hidden_dim]
        @param e [batch_size, node_num, node_num, edge_hidden_dim]
        @param neighbor_index [batch_size,node_num,邻居数]?
        node embedding
        '''
        batch_size = x.size(0)
        node_num = x.size(1)
        node_hidden_dim = x.size(-1)
        t = x.unsqueeze(1).repeat(1,node_num,1,1) #t [batch_size,node_num, node_num,node_hidden_dim]
       
        #neighbor_index = neighbor_index.unsqueeze(3).repeat(1,1,1,node_hidden_dim)
        neighbor_index = torch.tensor((neighbor_index.long()).repeat(1,1,1,node_hidden_dim)).long()
        neighbor = t.gather(2, neighbor_index)  #选出邻居的隐状态
        neighbor = neighbor.view(batch_size,node_num,-1,node_hidden_dim) #[batch_size,node_num,邻居数，node_hidden_dim]
        
        h_nb_node = self.ln1_node(x+self.Relu(self.W_node(self.attn(x,neighbor,neighbor))))
        h_node = self.ln2_node(h_nb_node+self.Relu(self.V_node(torch.cat([self.V_node_in(x),h_nb_node],dim=-1))))
        
        #edge embedding
        x_from = x.unsqueeze(2).repeat(1,1,node_num,1)
        x_to=x.unsqueeze(1).repeat(1,node_num,1,1)
        h_nb_edge=self.ln1_edge(e+self.Relu(self.W_edge(self.W1_edge(e)+self.W2_edge(x_from)+self.W3_edge(x_to))))
        h_edge=self.ln2_edge(h_nb_edge+self.Relu(self.V_edge(torch.cat([self.V_edge_in(e),h_nb_edge],dim=-1))))
        
        return h_node, h_edge


# In[22]:


class GCN(nn.Module):
    def __init__(self, node_input_dim,edge_input_dim, node_hidden_dim,edge_hidden_dim,gcn_num_layers,k):
        super(GCN,self).__init__()
        self.k = k
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.gcn_num_layers = gcn_num_layers
        
        self.node_W1 = nn.Linear(2,self.node_hidden_dim)
        self.node_W2 = nn.Linear(2,self.node_hidden_dim//2)
        self.node_W3 = nn.Linear(1,self.node_hidden_dim//2)
        self.edge_W4 = nn.Linear(1,self.edge_hidden_dim//2)
        self.edge_W5 = nn.Linear(1,self.edge_hidden_dim//2)
        
        self.nodes_embedding = nn.Linear(self.node_hidden_dim, self.node_hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(self.edge_hidden_dim,self.edge_hidden_dim,bias=False)
        self.gcn_layers = nn.ModuleList([GCNLayer(self.node_hidden_dim) for i in range(self.gcn_num_layers) ])
        self.Relu = nn.ReLU()
        
    def forward(self, x_c, x_d, A, distance,index):
        '''
        x_c [batch_size,node_num(+1),2]
        x_d [batch_size,node_num]
        '''
        # 1 Input Given
        #n = x_d.shape[1]
        # node: 求出X_0和X_1和X
        x_d = x_d.unsqueeze(-1)
        X_0 = self.Relu(self.node_W1(x_c[:,0,:]))
        X_0 = X_0.view(X_0.shape[0],1,X_0.shape[1])
        X_1 = self.Relu(torch.cat((self.node_W2(x_c[:,1:]),self.node_W3(x_d[:,1:])),dim=2)) # 是按照第3维拼接
        X = torch.cat((X_0,X_1), dim=1)  # 拼接出x
        # edge：计算Y
        # print(distance.shape)
        distance = distance.unsqueeze(-1)
        Y = self.Relu(torch.cat((self.edge_W4(distance), self.edge_W5(A)), dim=3))
        
        # 2 GCN Encoder
        # node and edge embeddings
        H_0 = self.nodes_embedding(X) # 点的embedding
        H_e0 = self.edges_embedding(Y) # 边的embedding
        # GCN layers: 2 layers
        for layer in self.gcn_layers:
            H_0, H_e0 = layer(H_0, H_e0, index)
        
        
        return H_0, H_e0


# In[23]:


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()
        
        v = torch.FloatTensor(dim)
        if use_cuda:
            v = v.to(device)
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)) , 1. / math.sqrt(dim))
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        return: 
            logits: 到下一个点的概率？
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits


# In[24]:


class SequentialDecoder(nn.Module):
    def __init__(self, hidden_dim, decode_type, use_cuda=False):
        super(SequentialDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.decode_type = decode_type
        self.use_cuda = use_cuda
        self.sm = nn.Softmax(1)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2) # 2层GRU
        self.tanh = nn.Tanh()
        self.h = nn.Linear(hidden_dim, 1)
        self.W = nn.Linear(2, 1)
        self.pointer = Attention(hidden_dim, use_tanh=True, use_cuda=self.use_cuda)
        
    def forward(self, x, last_node, hidden, mask):
        '''
        @param x : [batch_size, node_num, node_hidden_dim]
        @param last_node: [batch_size, 1]
        @param hidden: [2,batch_size,node_hidden_dim]
        @param mask: [batch_size, node_num] 类型是torch.bool,要mask的是true，不用的是false（表示此时刻unvalid节点）
        return:
        ind: [batch_size, 1]，表示采样后下一个点的下标
        probability,squeeze: [batch_size]，ind对应的概率值
        hidden: [2,batch_size,node_hidden_dim] 表示此时刻GRU的隐状态
        '''
        batch_size = x.size(0)
        batch_idx = torch.arange(0, batch_size).unsqueeze(1).to(device)
        last_x = x[batch_idx, last_node] # [batch_size,1,node_hidden_dim] 取出上次预测的点对应GCN隐状态
        last_x = last_x.permute(1, 0, 2)
        
        _, hidden = self.gru(last_x,hidden)  # gru参数：[seq_len, batch, input_size], hidden(num_layers * num_directions, batch, hidden_size)
        z = hidden[-1] #第二层GRU的隐状态 [1,batch_size,node_hidden_dim]
        
        _, u = self.pointer(z, x.permute(1, 0, 2))  # u: [batch_size, node_num]
        u = u.masked_fill_(mask, -np.inf)  # 掩码：u: [batch_size, node_num]
        probs = self.sm(u)  # softmax后，取下一个点对应的概率：[batch_size, node_num]
        if self.decode_type == "sample":
            ind = torch.multinomial(probs, num_samples=1)  # sample采样：根据probs，对每一行进行1次采样（类似轮盘赌）
        else:   
            ind = torch.max(probs, dim=1)[1].unsqueeze(1)  # greedy采样：增维后[batch_size, 1]
        probability = probs[batch_idx, ind].to(device)  # 取出ind对应的概率值
        return ind, probability.squeeze(1), hidden


# In[25]:


class ClassificationDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.sm = nn.Softmax(-1)
    
    def forward(self, e):
        a = self.MLP(e)
        a = a.squeeze(-1)
        out = self.sm(a)
        return out


# In[26]:


class Model(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k):
        super(Model, self).__init__()
        self.GCN = GCN(node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k).to(device)
        self.sequential_decoder_sample = SequentialDecoder(node_hidden_dim, "sample").to(device)
        self.sequential_decoder_greedy = SequentialDecoder(node_hidden_dim, "greedy").to(device)
        self.classification_decoder = ClassificationDecoder(edge_hidden_dim).to(device)
        
    def forward(self, env):
        '''
        return:
        prob_sum_sample: sample policy得到的概率总和, [batch_size]
        sample_distance: sample policy得到的距离总和，[batch_size]
        greedy_distance: greedy policy得到的距离总和，[batch_size]
        predict_matrix: classification decoder output, [batch_size, node_num, node_num]
        solution_matrix: sequential decoder(sample policy) output[batch_size, node_num, node_num]
        '''
        # get data from env
        x_c, x_d, A, index, distance = env.graph, env.demand, env.A, env.index, env.distance
        x_c, x_d, A, index, distance = x_c.to(device), x_d.to(device), A.to(device), index.to(device), distance.to(device)
        # encoder
        H_node, H_edge = self.GCN(x_c, x_d, A, distance,index)
        # classification decoder
        predict_matrix = self.classification_decoder(H_edge).to(device)
        
        
        hidden_sample=torch.zeros((2,env.batch_size,50)).to(device)
        hidden_greedy=torch.zeros((2,env.batch_size,50)).to(device)
        ind_sample= torch.zeros((env.batch_size,1)).long().to(device)
        ind_greedy= torch.zeros((env.batch_size,1)).long().to(device)
        prob_sum_sample = torch.zeros(env.batch_size).to(device)
        # prob_sum_greedy = torch.zeros(env.batch_size)
        
        
        while env.is_all_end() is False:    # sample policy
            #print("sample")
            ind_sample, probability_sample, hidden_sample = self.sequential_decoder_sample(H_node,ind_sample, hidden_sample, env.mask)
            env.step(ind_sample)
            env.update_mask(ind_sample)
            prob_sum_sample += torch.log(probability_sample)
            #print(ind_sample[0],env.demand[0],env.capacity[0])
        solution_matrix = env.get_result_matrix().to(device)   # 得到sample的结果矩阵: [batch_size, node_num, node_num]
        sample_distance = env.reward
        sample_pi = env.pi
    
        
        env.reset_in_sequential()   # reset
        
        while env.is_all_end() is False:    # greedy policy
            #print("greedy")
            ind_greedy, _, hidden_greedy = self.sequential_decoder_greedy(H_node,ind_greedy, hidden_greedy, env.mask)
            env.step(ind_greedy)
            env.update_mask(ind_greedy)
            # prob_sum_greedy += torch.log(probability_greedy)
        greedy_distance = env.reward
        
        
        return prob_sum_sample, sample_distance, greedy_distance, predict_matrix, solution_matrix, sample_pi


# In[27]:


env = environment(batch_size=32, file_path='../tc/G-20-training.npz', k=3,is_train=True)
env_test=environment(batch_size=32,file_path='../tc/G-20-testing.npz',k=3,is_train=False)
crossEntropy = torch.nn.CrossEntropyLoss()
model = Model(node_input_dim=2, edge_input_dim=1, node_hidden_dim=50, edge_hidden_dim=50, gcn_num_layers=2, k=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[28]:


def test():
    total_loss=0
    mean_distance=0
    env_test.count = 0
    sample_pi=torch.zeros(1)
    for i in range(env_test.batch_num):   # 内循环：遍历batch
        
        with torch.no_grad():
            env_test.reset()
            sample_logprob, sample_distance, greedy_distance, predict_matrix, solution_matrix,sample_pi = model(env_test)
            predict_matrix = predict_matrix.view(-1,2)
            solution_matrix = solution_matrix.view(-1).long()
            classification_loss = crossEntropy(predict_matrix, solution_matrix)     # 可以直接算较熵吗？（同长？
            advantage = (sample_distance - greedy_distance).detach()
            reinforce = advantage * sample_logprob
            sequential_loss = reinforce.sum()
            loss = sequential_loss + classification_loss
            total_loss += loss
            mean_distance+=torch.mean(sample_distance).item()
    return total_loss/env_test.batch_num,mean_distance/env_test.batch_num,sample_pi


# In[29]:



def main(epochs):
    # 新建环境
    loss_list = []
    loss_test_list=[]
    mean_dis_list=[]
    mean_dis_test_list=[]
    sample_pi=torch.zeros(1)
    test_pi=torch.zeros(1)
   
    for epoch in range(epochs):   # 外循环：遍历次数
        env.count = 0    # 更新env中的count
        total_loss = 0
        mean_distance=0
        mean_distance_greedy=0
        for i in range(env.batch_num):   # 内循环：遍历batch
            optimizer.zero_grad()
            env.reset()
            sample_logprob, sample_distance, greedy_distance, predict_matrix, solution_matrix,sample_pi = model(env)
            predict_matrix = predict_matrix.view(-1,2)
            solution_matrix = solution_matrix.view(-1).long()
            classification_loss = crossEntropy(predict_matrix, solution_matrix)     # 可以直接算较熵吗？（同长？
            advantage = (sample_distance - greedy_distance).detach()
            reinforce = advantage * sample_logprob
            sequential_loss = reinforce.sum()
            loss = sequential_loss + classification_loss
            total_loss += loss
            mean_distance+=torch.mean(sample_distance).item()
            mean_distance_greedy+=torch.mean(greedy_distance).item()
            loss.backward()
            optimizer.step()

        test_loss,test_dis_mean,test_pi=test()
        print("loss:", total_loss/env.batch_num,test_loss,"distance:",mean_distance/env.batch_num,test_dis_mean)
        loss_list.append(total_loss/env.batch_num)
        loss_test_list.append(test_loss)
        mean_dis_list.append(mean_distance/env.batch_num)
        mean_dis_test_list.append(test_dis_mean)
        if mean_distance < mean_distance_greedy: #如果sample比greedy好，就更新greedy参数
            model.sequential_decoder_greedy.load_state_dict(model.sequential_decoder_sample.state_dict())
            print("update")
            
    return sample_pi,test_pi,loss_list,loss_test_list,mean_dis_list,mean_dis_test_list


# In[30]:


def write_file(file_name,loss_list,loss_test_list,mean_dis_list,mean_dis_test):
    f = open(file_name, "a")
    for i in range(len(loss_list)):
        f.write("%.6f " %loss_list[i])
        f.write("%.6f "%loss_test_list[i])
        f.write("%.6f "%mean_dis_list[i])
        f.write("%.6f\n"%mean_dis_test[i])
    f.close()
def write_pi(file_name,sample_pi):
    f = open(file_name, "a")
    for i in range(len(sample_pi)):
        for j in range(len(sample_pi[i])):
            f.write("%d "%sample_pi[i][j])
        f.write("\n")
    f.write("\n")
    f.close()


# In[31]:


sample_pi_list=[]
for i in range(20):
    sample_pi,test_pi,loss_list,loss_test_list,mean_dis_list,mean_dis_test=main(epochs=5)
    sample_pi_list.append(sample_pi)
    write_file("result.txt",loss_list,loss_test_list,mean_dis_list,mean_dis_test)
    write_pi("pi_train.txt",sample_pi)
    write_pi("pi_test.txt",test_pi)


# In[34]:





# In[19]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# model=GCN(2,1, 50,50,2,3)
# temp=model(graph,demand, A, distance,index)


# In[ ]:



        

