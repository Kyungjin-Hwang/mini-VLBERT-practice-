import torch
import torch.nn as nn
from previous.modeling_BERT import BertEncoder, BertPooler, ACT2FN, BertOnlyMLMHead, BertLayerNorm

NUM_SPECIAL_WORDS = 1000
vocab_size = 30522
hidden_size = 512
visual_size = 512
intermediate_size = 3072
num_hidden_layers = 12
hidden_act = "gelu"
with_pooler = False
num_attention_heads = 12
max_position_embeddings = 512
type_vocab_size = 3
hidden_dropout_prob = 0.1
attention_probs_dropout_prob: 0.1
visual_scale_text_init = 0.0
visual_scale_object_init = 0.0

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented

class VisualLinguisticBert(BaseModel):
    def __init__(self, language_pretrained_model_path=None):
        super(VisualLinguisticBert, self).__init__()
            # embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.end_embedding = nn.Embedding(1, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.embedding_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.embedding_dropout = nn.Dropout(hidden_dropout_prob)
        self.with_pooler = with_pooler
        self.visual_size = visual_size
        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None

        visual_scale_text = nn.Parameter(torch.as_tensor(self.visual_scale_text_init, dtype=torch.float),
                                         requires_grad=True)
        self.register_parameter('visual_scale_text', visual_scale_text)
        visual_scale_object = nn.Parameter(torch.as_tensor(self.visual_scale_object_init, dtype=torch.float),
                                           requires_grad=True)
        self.register_parameter('visual_scale_object', visual_scale_object)

        self.encoder = BertEncoder(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, intermediate_size, hidden_act, num_hidden_layers)

        if self.with_pooler:
            self.pooler = BertPooler(hidden_size)

        # init weights
        self.apply(self.init_weights)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

    def word_embeddings_wrapper(self, input_ids):
        self.word_embeddings(input_ids)


    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_visual_embeddings,
                text_mask,
                object_vl_embeddings,
                object_mask,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False,
                output_attention_probs=False):

        # 임베딩 연결 및 mask
        embedding_output, attention_mask, text_mask_new, object_mask_new = self.embedding(text_input_ids,
                                                                                          text_token_type_ids,
                                                                                          text_visual_embeddings,
                                                                                          text_mask,
                                                                                          object_vl_embeddings,
                                                                                          object_mask)
        # 2D 텐서 마스크로부터 3D 어텐션 마스크를 만듬.
        # Size는 [batch_size, 1, 1, to_seq_length]이었음.
        # 본 모델은 이를 [batch_size, num_heads, from_seq_length, to_seq_length]로 확장함.

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Attention_mask는 본 논문이 attend 하려는 position에서는 1.0이고 마스크된 위치에 대해 0.0이므로
        # 본 논문은 attend 하려는 position에 대해 0.0이고 마스크된 위치에 대해 -10000.0인 텐서를 생성.
        # softmax 이전에 raw score에 추가하기 때문에 이것은 완전히 제거하는 것과 사실상 동일.

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if output_attention_probs:
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers=output_all_encoded_layers,
                                                           output_attention_probs=output_attention_probs)
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers=output_all_encoded_layers,
                                          output_attention_probs=output_attention_probs)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.with_pooler else None

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if output_text_and_object_separately:
            if not output_all_encoded_layers:
                encoded_layers = [encoded_layers]
            encoded_layers_text = []
            encoded_layers_object = []
            for encoded_layer in encoded_layers:
                max_text_len = text_input_ids.shape[1]
                max_object_len = object_vl_embeddings.shape[1]
                encoded_layer_text = encoded_layer[:, :max_text_len]
                encoded_layer_object = encoded_layer.new_zeros(
                    (encoded_layer.shape[0], max_object_len, encoded_layer.shape[2]))
                encoded_layer_object[object_mask] = encoded_layer[object_mask_new]
                encoded_layers_text.append(encoded_layer_text)
                encoded_layers_object.append(encoded_layer_object)
            if not output_all_encoded_layers:
                encoded_layers_text = encoded_layers_text[0]
                encoded_layers_object = encoded_layers_object[0]
            if output_attention_probs:
                return encoded_layers_text, encoded_layers_object, pooled_output, attention_probs
            else:
                return encoded_layers_text, encoded_layers_object, pooled_output
        else:
            if output_attention_probs:
                return encoded_layers, pooled_output, attention_probs
            else:
                return encoded_layers, pooled_output

    def embedding(self,
                  text_input_ids,
                  text_token_type_ids,
                  text_visual_embeddings,
                  text_mask,
                  object_vl_embeddings,
                  object_mask):
        text_linguistic_embedding = self.word_embeddings_wrapper(text_input_ids)
        if self.visual_1x1_text is not None:
            text_visual_embeddings = self.visual_1x1_text(text_visual_embeddings)
        else:
            text_visual_embeddings *= self.visual_scale_text
        text_vl_embeddings = text_linguistic_embedding + text_visual_embeddings

        object_visual_embeddings = object_vl_embeddings[:, :, :self.visual_size]
        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(object_visual_embeddings)
        else:
            object_visual_embeddings *= self.visual_scale_object
        object_linguistic_embeddings = object_vl_embeddings[:, :, self.visual_size:]
        object_vl_embeddings = object_linguistic_embeddings + object_visual_embeddings

        bs = text_vl_embeddings.size(0)
        vl_embed_size = text_vl_embeddings.size(-1)
        max_length = (text_mask.sum(1) + object_mask.sum(1)).max() + 1
        grid_ind, grid_pos = torch.meshgrid(torch.arange(bs, dtype=torch.long, device=text_vl_embeddings.device),
                                            torch.arange(max_length, dtype=torch.long, device=text_vl_embeddings.device))
        text_end = text_mask.sum(1, keepdim=True)
        object_end = text_end + object_mask.sum(1, keepdim=True)

        # seamlessly concatenate visual linguistic embeddings of text and object
        _zero_id = torch.zeros((bs,), dtype=torch.long, device=text_vl_embeddings.device)
        vl_embeddings = text_vl_embeddings.new_zeros((bs, max_length, vl_embed_size))
        vl_embeddings[grid_pos < text_end] = text_vl_embeddings[text_mask]
        vl_embeddings[(grid_pos >= text_end) & (grid_pos < object_end)] = object_vl_embeddings[object_mask]
        vl_embeddings[grid_pos == object_end] = self.end_embedding(_zero_id)

        # token type embeddings/ segment embeddings
        token_type_ids = text_token_type_ids.new_zeros((bs, max_length))
        token_type_ids[grid_pos < text_end] = text_token_type_ids[text_mask]
        token_type_ids[(grid_pos >= text_end) & (grid_pos <= object_end)] = 2
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # position embeddings
        position_ids = grid_pos + self.position_padding_idx + 1
        position_ids[(grid_pos >= text_end) & (grid_pos < object_end)] \
            = text_end.expand((bs, max_length))[
                  (grid_pos >= text_end) & (grid_pos < object_end)] + self.position_padding_idx + 1

        position_ids[grid_pos == object_end] = (text_end + 1).squeeze(1) + self.position_padding_idx + 1

        position_embeddings = self.position_embeddings(position_ids)
        mask = text_mask.new_zeros((bs, max_length))
        mask[grid_pos <= object_end] = 1

        embeddings = vl_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_LayerNorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings, mask, grid_pos < text_end, (grid_pos >= text_end) & (grid_pos < object_end)

class VisualLinguisticBertForPretraining(VisualLinguisticBert):
    def __init__(self, language_pretrained_model_path=None,
                 with_rel_head=True, with_mlm_head=True, with_mvrc_head=True):

        super(VisualLinguisticBertForPretraining, self).__init__(language_pretrained_model_path=None)

        self.with_rel_head = with_rel_head
        self.with_mlm_head = with_mlm_head
        self.with_mvrc_head = with_mvrc_head
        if with_rel_head:
            self.relationsip_head = VisualLinguisticBertRelationshipPredictionHead()
        if with_mlm_head:
            self.mlm_head = BertOnlyMLMHead(self.word_embeddings.weight)
        if with_mvrc_head:
            self.mvrc_head = VisualLinguisticBertMVRCHead()

        # init weights
        self.apply(self.init_weights)


        # load language pretrained model
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

    def forward(self, text_input_ids, text_token_type_ids,
                text_visual_embeddings, text_mask, object_vl_embeddings,
                object_mask, output_all_encoded_layers=True,
                output_text_and_object_separately=False):
        text_out, object_out, pooled_rep = super(VisualLinguisticBertForPretraining, self).forward(
            text_input_ids,
            text_token_type_ids,
            text_visual_embeddings,
            text_mask,
            object_vl_embeddings,
            object_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_text_and_object_separately=output_text_and_object_separately
        )

        if self.with_rel_head:
            relationship_logits = self.relationship_head(pooled_rep)
        else:
            relationship_logits = None
        if self.with_mlm_head:
            mlm_logits = self.mlm_head(text_out)
        else:
            mlm_logits = None
        if self.with_mvrc_head:
            mvrc_logits = self.mvrc_head(object_out)
        else:
            mvrc_logits = None

        return relationship_logits, mlm_logits, mvrc_logits

class VisualLinguisticBertMVRCHeadTransform(BaseModel):
    def __init__(self, hidden_size, hidden_act):
        super(VisualLinguisticBertMVRCHeadTransform, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[hidden_act]

        self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)

        return hidden_states


class VisualLinguisticBertMVRCHead(BaseModel):
    def __init__(self, hidden_size, hidden_act, visual_region_classes):
        super(VisualLinguisticBertMVRCHead, self).__init__()

        self.transform = VisualLinguisticBertMVRCHeadTransform(hidden_size, hidden_act)
        self.region_cls_pred = nn.Linear(hidden_size, visual_region_classes)
        self.apply(self.init_weights)

    def forward(self, hidden_states):

        hidden_states = self.transform(hidden_states)
        logits = self.region_cls_pred(hidden_states)

        return logits


class VisualLinguisticBertRelationshipPredictionHead(BaseModel):
    def __init__(self, hidden_size):
        super(VisualLinguisticBertRelationshipPredictionHead, self).__init__()

        self.caption_image_relationship = nn.Linear(hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, pooled_rep):

        relationship_logits = self.caption_image_relationship(pooled_rep)

        return relationship_logits

