""" Meta Learner """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from feature_extractor import FeatureExtractor

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, config, z_dim):
        super().__init__()
        self.config = config
        self.z_dim = z_dim #208 # 240 for new model
        self.vars = nn.ParameterList()
        self.fc2_w = nn.Parameter(torch.ones([self.config['way'], self.z_dim]))
        # config['way'] --> Way number, how many classes in a task
        torch.nn.init.kaiming_normal_(self.fc2_w)
        self.vars.append(self.fc2_w)
        self.fc2_b = nn.Parameter(torch.zeros(self.config['way']))
        self.vars.append(self.fc2_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc2_w = the_vars[0]
        fc2_b = the_vars[1]
        net = F.softmax(F.linear(input_x, fc2_w, fc2_b), dim=1)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, config, mode='meta'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.update_lr = config['base_lr']
        self.update_step = config['update_step']
        #z_dim = filter_sizing*D*13 ~ 8*2*13 = 208 # 240 for new model
        z_dim = 240
        if self.config['clstype'] == 'legs':
            num_cls = 2
        elif self.config['clstype'] == 'multiclass':
            num_cls = 4  
        self.base_learner = BaseLearner(config, z_dim)

        if self.mode == 'meta':
            self.encoder = FeatureExtractor(self.config)  
        else:
            self.encoder = FeatureExtractor(self.config, mtl=False)  
            self.pre_fc = nn.Sequential(nn.Linear(z_dim, num_cls))

    def forward(self, inp):
        if self.mode=='pre' or self.mode=='origval':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query, type = inp
            return self.meta_forward(data_shot, label_shot, data_query, type)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        return F.softmax(self.pre_fc(self.encoder(inp)), dim=1)    

    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.005 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(2):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.005 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q

    def meta_forward(self, data_shot, label_shot, data_query, type = None):
        embedding_shot = self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        params=self.base_learner.parameters()
        optimizer=optim.Adam(params)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)
        optimizer.step()
        logits_q = self.base_learner(embedding_query)

        for _ in range(10):
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True)
            optimizer.step()
            logits_q = self.base_learner(embedding_query)
        if type == 'test':
          return logits_q, self.base_learner.state_dict()
        else :
          return logits_q