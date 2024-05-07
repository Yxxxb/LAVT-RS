import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from lib.backbone import LangProject


# the "classifier" in _utils.py and segmentation.py
class VLTFuseAndClassify(nn.Module):
    def __init__(self, d_model=256, nhead=8, d_hid=256, nlayers=2, args=None):
        super().__init__()
        # bookkeeping
        self.top_vis_in_channs = 1024  # hard code to swin base configurations
        self.middle_vis_in_channs = 512  # applies to vlt_swin_transformer and vlt_unified_swin_transformer_f
        self.bottom_vis_in_channs = 256  # applies to vlt_swin_transformer and vlt_unified_swin_transformer_f
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid  # feedforward layers inside of the decoder; let's set to the same as self.d_model per VLT
        self.dropout = args.fusion_drop
        self.num_queries = 16
        self.size = args.img_size // 16
        self.joint_dim = self.top_vis_in_channs  # following VLT
        print('VLT size is {}'.format(args.img_size))

        # preprocessing features:
        # project channels and resize THREE input vis. feats. maps to those of the middle one
        # and then sum three into one vis feats. map
        # reduce channels of language features
        # aggregate over words to obtain sentence-level feature vector
        # each is a 1x1, 3x3, 1x1 bottleneck
        '''
            y = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)))(x)
        x = Add()([x, y])
        '''
        self.vis_reduce_chann_1 = nn.Sequential(nn.Conv2d(self.top_vis_in_channs, self.top_vis_in_channs//2, 1, bias=False),
                                                nn.BatchNorm2d(self.top_vis_in_channs//2),
                                                nn.ReLU(),

                                                nn.Conv2d(self.top_vis_in_channs//2, self.top_vis_in_channs, 3, padding=1, bias=False),
                                                nn.BatchNorm2d(self.top_vis_in_channs),
                                                nn.ReLU(),
                                                )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.vis_reduce_chann_2 = nn.Sequential(nn.Conv2d(self.middle_vis_in_channs, self.middle_vis_in_channs, 1, bias=False),
                                                nn.BatchNorm2d(self.middle_vis_in_channs),
                                                nn.ReLU(),
                                                )

        self.fuse_1_2 = nn.Sequential(nn.Conv2d(self.joint_dim + self.middle_vis_in_channs, self.joint_dim//2, 1, bias=False),
                                                nn.BatchNorm2d(self.joint_dim//2),
                                                nn.ReLU(),
                                                )

        self.down = nn.AvgPool2d(2)  # stride size is kernel size by default; this assumes feature maps are of even sizes

        self.vis_reduce_chann_3 = nn.Sequential(nn.Conv2d(self.bottom_vis_in_channs, self.bottom_vis_in_channs, 1, bias=False),
                                                nn.BatchNorm2d(self.bottom_vis_in_channs),
                                                nn.ReLU(),
                                                )
        self.fuse_2_3 = nn.Sequential(nn.Conv2d(self.joint_dim//2 + self.bottom_vis_in_channs, self.joint_dim//2, 1, bias=False),
                                      nn.BatchNorm2d(self.joint_dim//2),
                                      nn.ReLU(),
                                      )
        self.hallucinate_result_of_23 = nn.Sequential(nn.Conv2d(self.joint_dim//2, self.joint_dim//4, 1, bias=False),
                                                      nn.BatchNorm2d(self.joint_dim//4),
                                                      nn.ReLU(),
                                                      nn.Conv2d(self.joint_dim // 4, self.joint_dim//2, 3, padding=1, bias=False),
                                                      nn.BatchNorm2d(self.joint_dim//2),
                                                      nn.ReLU(),
                                      )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.project_again = nn.Sequential(nn.Conv2d(self.joint_dim, self.joint_dim//2, 1, bias=False),
                                                nn.BatchNorm2d(self.joint_dim//2),
                                                nn.ReLU(),
                                                )
        self.fuse_again = nn.Sequential(nn.Conv2d(self.joint_dim + self.joint_dim//2, self.d_model, 1, bias=False),
                                                nn.BatchNorm2d(self.d_model),
                                                nn.ReLU(),
                                                )
        self.last_project = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, bias=False),
                                                nn.BatchNorm2d(self.d_model),
                                                nn.ReLU(),
                                                )
        ##########################################################################
        self.lang_proj = nn.Sequential(nn.Linear(768, self.joint_dim),
                                       nn.BatchNorm1d(self.joint_dim),
                                       nn.ReLU(),
                                       )
        self.joint_threshold = nn.Sequential(nn.BatchNorm2d(self.joint_dim),
                                             nn.ReLU(),
                                             )

        self.query_generation = QueryGenerationModule(self.joint_dim//2, self.d_model,
                                                      h=self.size, w=self.size, num_queries=self.num_queries)
        # (self, visual_dim, dim, h=26, w=26, lang_dim=768, num_queries=16)
        # (self, x, l, l_mask)
        self.transformer_fusion = TransformerModel(d_model, nhead, d_hid, nlayers, self.dropout, self.size, self.size)
        # (self, d_model, nhead, d_hid, nlayers, dropout=0.0)
        # (self, src, tgt, averaged_sentence_features)
        self.query_balancing = QueryBalancingModule(self.d_model)
        # (self, dim)
        # (self, not_decoded_queries, decoded_queries)

        # four building blocks with 1 link:
        # 1. query generation module;
        # 2. transformer encoder-decoder fusion module;
        # 3. query balancing module
        # 4. final decoding and classification layers
        # link: convert number of queries to HW and then permute and then reshape, then 3x3 conv for spatial refinement
        self.q_to_spatial = nn.Sequential(nn.Conv1d(self.d_model, self.size*self.size, 1, bias=False),
                                          nn.ReLU(),
                                          )
        self.spatial_refine = nn.Sequential(nn.Conv2d(self.num_queries, self.d_model, 3, padding=1, bias=False),
                                            nn.BatchNorm2d(self.d_model),
                                            nn.ReLU(),
                                            )

        self.decoding = ProgressiveDecoding(self.d_model, self.d_model)
        # (self, c4_dim, hidden_size)

    def forward(self, x_c4, x_c3, x_c2, l, l_mask):
        """
        Args:
            x_c4, x_c3, x_c2: Tensor, shape [batch_size, swin_dim, hi, wi]
            l: (B, 768, #words)
            l_mask: (B, #words, 1)
            l_mask is used for obtaining an averaged sentence-level language vector
        Returns:
            final score maps (of 2 classes). This class minx the process of multi-modal fusion with segmentation decoding
        """
        #x = x_c4 + x_c3 + x_c2  # (B, d_model, 26, 26)
        #assert(x.size(-1) == 26)
        #assert (x.size(-2) == 26)
        #######################################
        # language embeddings average pooling #
        #######################################
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        sentence_l = (l * l_mask).sum(dim=-1).div(l_mask.sum(dim=-1))  # (B, 768)
        #######################################
        sentence_l = self.lang_proj(sentence_l).unsqueeze(-1).unsqueeze(-1)  # (B, self.joint_dim, 1, 1)
        x_c4 = x_c4 + self.vis_reduce_chann_1(x_c4)  # according to VLT (B, joint_dim, h4, w4)
        # cannot use += here, because it is an in_place operation
        x_mm_c4 = x_c4 * sentence_l
        x_mm_c4 = self.joint_threshold(x_mm_c4)  # (B, joint_dim, h4, w4)

        temp = self.up(x_mm_c4)
        temp2 = self.vis_reduce_chann_2(x_c3)
        temp_cat_1 = torch.cat([temp, temp2], dim=1)
        Fm_mid_query = self.fuse_1_2(temp_cat_1)  # # (B, joint_dim//2, h3, w3)

        x_c2 = self.down(x_c2)
        x_c2 = self.vis_reduce_chann_3(x_c2)
        Fm_query = torch.cat([Fm_mid_query, x_c2], dim=1)  # (B, joint_dim//2 + bottom_vis_in_channs, h3, w3)
        Fm_query = self.fuse_2_3(Fm_query)  # (B, joint_dim//2, h3, w3)

        temp3 = self.hallucinate_result_of_23(Fm_query)  # (B, joint_dim//2, h3, w3)
        Fm_mid_tf = torch.cat([temp3, Fm_mid_query], dim=1)  # (joint_dim, h3, w3)

        # F_tf = up_proj_cat_proj(Fm, Fm_mid_tf, K.int_shape(Fm)[-1] // 2)
        temp4 = self.up2(x_mm_c4)  # (B, joint_dim, h3, w3)
        temp5 = self.project_again(Fm_mid_tf)  # (B, joint_dim//2, h3, w3)
        temp6 = torch.cat([temp4, temp5], dim=1)
        F_tf = self.fuse_again(temp6)  # (B, d_model, h3, w3)

        # F_tf = V.DarknetConv2D_BN_Leaky(config.hidden_dim, (1, 1))(F_tf)
        F_tf = self.last_project(F_tf)  # (B, d_model, h3, w3)

        # four building components and one link
        # no more m_mask reversing in rest of pipeline because now it is reversed.
        not_decoded_queries = self.query_generation(Fm_query, l, l_mask)
        # (num_queries, B, self.dim)
        decoded_queries = self.transformer_fusion(F_tf, not_decoded_queries)
        # (num_queries, B, self.dim)
        balanced_decoded_queries = self.query_balancing(not_decoded_queries, decoded_queries)
        # (B, d_model, num_queries)

        out = self.q_to_spatial(balanced_decoded_queries)  # (B, size*size, num_queries)
        out = out.permute(0, 2, 1).view(-1, self.num_queries, self.size, self.size)
        # (B, num_queries, h, w)
        out = self.spatial_refine(out)  # (B, d_model, h, w)

        out = self.decoding(out)
        return out


'''
    output_layer = L.Dense(spatial_size, activation='relu')(weighted_output)
    output_layer = L.Reshape((num_query, feat_size, feat_size))(output_layer)
    output_layer = L.Permute((2, 3, 1))(output_layer)

    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)

'''


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (676, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        # this interleaves sin and cos across all positions in the sequence
        # positions alternate between sin and cos: one sin value, one cosine value, one sin value, ...
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # extended slices [start:stop:step]. [::-1] is used to reverse a list

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, d_hid, nlayers, dropout=0.0, h=26, w=26):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.h = h
        self.w = w

        self.pos_encoder = PositionalEncoding(d_model)
        #self.pos_decoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
        # nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)

    def forward(self, src, tgt):
        """
        Args:
            src: Tensor, shape [batch_size, dim, h, w], key and value
            tgt: Tensor, shape [num_queries, batch_size, dim], query
        Returns:
            output Tensor of shape [num_queries, batch_size, dim]
        """
        src = src.view(-1, self.d_model, self.h*self.w).permute(2, 0, 1)  # spatial size has already been checked in upper layer
        src = self.pos_encoder(src)
        #src = src * averaged_sentence_features.unsqueeze(0)  # (h*w, B, d_model)
        memory = self.transformer_encoder(src)  # memory is the output from the last layer of the encoder
        # tgt is query, indicates the length of the sequence we want to obtain finally
        # src is input to encoder and encoder's output is memory, which is key and value, like a src, the length of
        # which disappear in the decoder layers.
        # forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        # (num_queries, B, self.dim)
        return output


# the following is VLT's custom spatial coords concat. method
def vlt_concat_coords(x):
    device = 'cuda:'+str(x.get_device())
    in_feats = x  #  [B, C, H, W]

    batch_size, h, w = x.size(0), x.size(-2), x.size(-1)
    #float_h = K.cast(h, 'float32')
    #float_w = K.cast(w, 'float32')

    y_range = torch.arange(0, h, dtype=torch.float32, device=device)  # [h, ]
    y_range = 2.0 * y_range / (h - 1.0) - 1.0
    x_range = torch.arange(0, w, dtype=torch.float32, device=device)  # [w, ]
    x_range = 2.0 * x_range / (w - 1.0) - 1.0
    x_range = x_range[None, :]   # [1, w]
    y_range = y_range[:, None]   # [h, 1]
    x = x_range.expand(h, w)     # [h, w]
    y = y_range.expand(h, w)     # [h, w]

    x = x[None, None, :, :]   # [1, 1, h, w]
    y = y[None, None, :, :]   # [1, 1, h, w]
    x = x.expand(batch_size, 1, h, w)   # [N, 1, h, w]
    y = y.expand(batch_size, 1, h, w)   # [N, 1, h, w]

    out = torch.cat([in_feats, x, x, x, y, y, y], dim=1)   # [N, c+6, h, w]

    return out


# define the query generation module #
class QueryGenerationModule(nn.Module):
    def __init__(self, visual_dim, dim, h=26, w=26, lang_dim=768, num_queries=16):
        super(QueryGenerationModule, self).__init__()
        self.visual_dim = visual_dim
        self.dim = dim
        self.h = h
        self.w = w
        self.lang_dim = lang_dim
        self.num_queries = num_queries
        self.project_1 = nn.Sequential(
            nn.Conv2d(self.visual_dim + 6, self.visual_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.visual_dim),
            nn.ReLU(),

            nn.Conv2d(self.visual_dim, self.visual_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.visual_dim),
            nn.ReLU(),

            nn.Conv2d(self.visual_dim, self.visual_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.visual_dim),
            nn.ReLU(),
        )

        self.project_2 = nn.Conv2d(self.visual_dim, self.num_queries, 1, bias=False)

        self.project_query = nn.Sequential(nn.Conv1d(self.h * self.w, self.dim, 1, bias=False),
                                           nn.ReLU()
                                           )
        self.project_lang = nn.Sequential(nn.Conv1d(self.lang_dim, self.dim, 1, bias=False),
                                          nn.ReLU()
                                          )
        self.pos_encoder = PositionalEncoding(self.dim)
        self.query_gen = nn.MultiheadAttention(self.dim, 8)

    def forward(self, x, l, l_mask):
        # x shape: B, visual_dim, H, W
        # l: B, 768, # words
        # l_mask: B, 1, # words
        x = vlt_concat_coords(x)
        x = self.project_1(x)  # preprocess coords and original feats. (B, visual_dim, H, W)
        x = self.project_2(x)  # reduce channels to number of queries. (B, num_queries, H, W)
        # flatten visual features x
        x = x.view(-1, self.num_queries, self.h * self.w).permute(0, 2, 1)  # (B, h*w, num_queries)

        # q: project spatial dimensions into the channel dimension
        vis_x = self.project_query(x)  # (B, self.dim, num_queries)
        x = self.pos_encoder(vis_x.permute(2, 0, 1))
        # k and v: project language features
        l = self.project_lang(l)  # (B, self.dim, #words)
        l = self.pos_encoder(l.permute(2, 0, 1))

        # convert sentence mask to a padding_mask by inverting 0s and 1s, and convert to ByteTensor,
        # ByteTensor is deprecated, use BoolTensor instead
        # and squeeze to singleton dimension
        l_mask = (1 - l_mask.squeeze(1)).bool()
        # query, key, value, key_padding_mask=l_mask, need_weights=False
        out, _ = self.query_gen(x, l, l, key_padding_mask=l_mask, need_weights=False)
        # even if specified need_weights=False, still returns the additional attention weight matrix, strange
        # (num_queries, B, self.dim)
        # according to VLT's code, add lang feats back with vision feats
        out = out + vis_x.permute(2, 0, 1)
        return out  # (num_queries, B, self.dim)


'''
key_padding_mask: if provided, specified padding elements in the key will be ignored by the attention. 
This is an binary mask. When the value is True, 
the corresponding value on the attention layer will be filled with -inf.
key_padding_mask: (N, S), ByteTensor, where N is the batch size, S is the source sequence length
torch.uint8
Inputs:
query: (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
key: (S, N, E)(S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
value: (S, N, E)(S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.
key_padding_mask: (N, S)(N,S) , ByteTensor, where N is the batch size, S is the source sequence length.
attn_mask: 2D mask (L, S)(L,S) where L is the target sequence length, S is the source sequence length. 3D mask (N*num_heads, L, S)
where N is the batch size, L is the target sequence length, S is the source sequence length.
Outputs:
attn_output: (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
attn_output_weights: (N, L, S)(N,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length.
'''


# define the query balancing module #
class QueryBalancingModule(nn.Module):
    def __init__(self, dim):
        super(QueryBalancingModule, self).__init__()
        self.dim = dim
        self.not_decoded_query_proj = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1, bias=False),
                                                    nn.ReLU()
                                                    )
        self.decoded_query_proj = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1, bias=False),
                                                nn.ReLU()
                                           )

        self.gate_proj = nn.Sequential(nn.Conv1d(2 * self.dim, self.dim, 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv1d(self.dim, 1, 1, bias=False),
                                  nn.Sigmoid(),
                                  )

    def forward(self, not_decoded_queries, decoded_queries):
        # both inputs have shape: (num_queries, B, self.dim)
        x = self.not_decoded_query_proj(not_decoded_queries.permute(1, 2, 0))
        # (B, self.dim, num_queries)
        y = self.decoded_query_proj(decoded_queries.permute(1, 2, 0))
        # (B, self.dim, num_queries)
        yx = torch.cat([y, x], dim=1)  # (B, 2*self.dim, num_queries)
        gates = self.gate_proj(yx)
        # (B, 1, num_queries)
        return gates * y  # (B, self.dim, num_queries)


'''
    if balance:
        query_proj = L.Dense(hidden_dim, activation='relu')(decoder_input)
        output_proj = L.Dense(hidden_dim, activation='relu')(decoded_layer)
        output_proj = L.Concatenate()([output_proj, query_proj])
        output_proj = L.Dense(hidden_dim, activation='relu')(output_proj)
        query_confident = L.Dense(1, activation='sigmoid')(output_proj)

        weighted_output = L.Multiply()([query_confident, decoded_layer])
    else:
        weighted_output = decoded_layer
'''


##################################################################
##################################################################
##################################################################
##################################################################


class ProgressiveDecoding(nn.Module):
    def __init__(self, c4_dim, hidden_size):
        super(ProgressiveDecoding, self).__init__()

        self.conv1_4 = nn.Conv2d(c4_dim, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.up_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()

        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(hidden_size)
        self.relu1_1 = nn.ReLU()

        self.classifier = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x):
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)

        x = self.up_4(x)

        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)

        x = self.up_3(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.up_2(x)

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        return self.classifier(x)
