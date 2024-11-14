import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdapterElement(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=64,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar": # Learnable scaler factor
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar) # Fixed scaler factor

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
                
    def forward(self, x):
        #print("Use lion adaptor")
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        return up

class AdapterRouter(nn.Module):
    def __init__(self,
        d_model,
        bottleneck=256,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="learnable_scalar",
        adapter_layernorm_option="none",
        num_adapters = 2,
    ):
        super().__init__()
        self.adapters = nn.ModuleList([])
        for _ in range(num_adapters):
            self.adapters.append(AdapterElement(
                d_model=d_model,
                bottleneck=bottleneck,
                dropout=dropout,
                init_option=init_option,
                adapter_scalar=adapter_scalar,
                adapter_layernorm_option=adapter_layernorm_option
            ))
        if num_adapters > 1:
            self.router_ratio1 = nn.Parameter(
                torch.tensor([[1],[0]],dtype=torch.float32).repeat(1,d_model)
            )
            self.router_ratio2 = nn.Parameter(
                torch.tensor([[0],[1]],dtype=torch.float32).repeat(1,d_model)
            )
        self.num_adapters = num_adapters
        self.router_idx = 0 # Determine which initial ratio pairs to utilize
        """
        router_idx = 0:
        router_ratio1[0] = [1,1,1,...]  
        router_ratio2[0] = [0,0,0,...]  

        #router_idx = 1:
        router_ratio1[1] = [0,0,0,...]  
        router_ratio2[1] = [1,1,1,...]  
        """
    def forward(self, x):
        assert self.router_idx in [0,1]
        output1 = self.adapters[0](x)
        if self.num_adapters == 1:
            return output1
        
        output2 = self.adapters[1](x)
        ratio1 = self.router_ratio1[self.router_idx]
        ratio2 = self.router_ratio2[self.router_idx]
        return output1 * ratio1 + output2 * ratio2
