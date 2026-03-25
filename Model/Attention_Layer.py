import torch.nn as nn
from Model.Attention_Family import series_decomp

class Attention_Layer(nn.Module):
    def __init__(self,
           attention,
           input_dim,        
           output_dim,
           type_attention,
           d_model_type,
           n_heads_type,
           d_keys = None,
           d_values = None,
           causal_kernel_size = 3,
           value_kernel_size = 1,
           resid_drop = 0.1,
           auto_moving_avg = 25):

        super(Attention_Layer, self).__init__()

        self.d_keys = d_keys or int(d_model_type/n_heads_type)
        self.d_values = d_values or int(d_model_type/n_heads_type)

        self.n_heads_type = n_heads_type
        self.type_attention = type_attention

        self.causal_kernel_size = causal_kernel_size
        self.value_kernel_size= value_kernel_size

        if self.type_attention != "FFT":
            self.query_projection = nn.Conv1d(in_channels   = input_dim, 
                                              out_channels  = self.d_keys * self.n_heads_type, 
                                              kernel_size   = self.causal_kernel_size)
            
            self.key_projection = nn.Conv1d(in_channels     = input_dim, 
                                            out_channels    = self.d_keys * self.n_heads_type, 
                                            kernel_size     = self.causal_kernel_size)
            
            self.value_projection = nn.Conv1d(in_channels   = input_dim, 
                                              out_channels  = self.d_values * self.n_heads_type, 
                                              kernel_size   = self.value_kernel_size) 

        if self.type_attention == "FFT":
            self.fft_projection = nn.Conv1d(in_channels     = input_dim,
                                            out_channels    = output_dim,
                                            kernel_size     = self.value_kernel_size)

        self.inner_attention = attention
        self.out_projection = nn.Conv1d(in_channels     = self.d_values * self.n_heads_type,
                                        out_channels    = output_dim,
                                        kernel_size     = self.value_kernel_size)
        self.activation = nn.ReLU(inplace=True)
        self.resdi_dropout = nn.Dropout(resid_drop) 

        if self.type_attention == "Auto":
            self.decomp_block = series_decomp(auto_moving_avg)


    def forward(self, queries, keys, values):
        """
        input x: [batch, length, channel], channel=input_dim
        return y: [batch, length, output_dim]
        """
        B, L_Q, I_Q = queries.shape
        _, L_K, I_K = keys.shape
        _, L_V, I_V = values.shape
        H = self.n_heads_type 
        if self.type_attention == "FFT":
            fft_values = values
            fft_padding_size = int(self.value_kernel_size/2)
            padding_fft = nn.functional.pad(fft_values.permute(0, 2, 1),
                                            pad     = (fft_padding_size, fft_padding_size),
                                            mode    = 'replicate')
            fft_values = self.fft_projection(padding_fft).permute(0, 2, 1) # [B, L, output_dim] 
            
            out, attn = self.inner_attention(fft_values)
            return out, attn

        else:
            # query, key, value projection
            queries_padding_size =  int(self.causal_kernel_size/2)
            padding_queries = nn.functional.pad(queries.permute(0, 2, 1),
                                                pad     = (queries_padding_size, queries_padding_size),
                                                mode    = 'replicate')
            queries = self.query_projection(padding_queries).permute(0, 2, 1) # [B, L, d_k*n]   

            keys_padding_size =  int(self.causal_kernel_size/2)
            padding_keys = nn.functional.pad(keys.permute(0, 2, 1),
                                             pad    = (keys_padding_size, keys_padding_size),
                                             mode   = 'replicate')
            keys = self.key_projection(padding_keys).permute(0, 2, 1) # [B, L, d_k*n]        

            values_padding_size = int(self.value_kernel_size/2)
            padding_values = nn.functional.pad(values.permute(0, 2, 1),
                                               pad  = (values_padding_size, values_padding_size),
                                               mode = 'replicate')
            values = self.value_projection(padding_values).permute(0, 2, 1) # [B, L, d_v*n]        

            query = queries.view(B, L_Q, H, -1)
            key  = keys.view(B, L_K, H, -1)
            value = values.view(B, L_V, H, -1)
        
            out, attn = self.inner_attention(query, key, value)

            out = out.view(B, L_Q, -1)
            padding_out = nn.functional.pad(out.permute(0, 2, 1),
                                            pad     = (values_padding_size, values_padding_size),
                                            mode    = 'replicate')
             
            out = self.activation(self.out_projection(padding_out)).permute(0, 2, 1)
            out = self.resdi_dropout(out)

            if self.type_attention == "Auto":
                out, _ = self.decomp_block(out)
                return out, attn

            else:    
                return out, attn