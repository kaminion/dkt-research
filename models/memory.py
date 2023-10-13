import torch
from torch import nn

class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write 
        if self.is_write:
            # write에서는 erase / add를 사용하므로
            self.erase = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constnat_(self.add.bias, 0)
            
    def addressing(self, control_input, memory): # Corrleation Weight에 해당
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim) 
            memory:                 Shape (memory_size, memory_state_dim)
        Returns 
            correlation_weight:     Shape (batch_size, memory_size)
        """ # key의 embedding 벡터 
        # torch.t는 마지막 차원을 2차로 만드는 함수임
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = torch.nn.functional.softmax(similarity_score, dim=1) # Shape: (batch_size, memory_size) 
        
        return correlation_weight
    
    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters 
            control_input:      Shape (batch_size, control_state_dim)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
            read_weight:        Shape (batch_size, memory_size)
        Returns
            read_content:       Shape (batch_size, memory_size)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)    
        return read_content
        
        
    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write # 값이 없으면 나감
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
            
        # 논문에 나와있듯이 operation 임
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        # view
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        # add와 erase 식에 weight 까지 넣어서 미리 계산해줌
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        # 결국 마지막 보면 식이 아래처럼 하나로 정리됨
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory