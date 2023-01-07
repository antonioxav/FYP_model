# Transformer Model Architecture
import torch
from torch import nn



class Sign_Classifier(nn.Module):
    """Transformer Baseline v0.0.0"""

    def __init__(self, backbone, transformer, num_classes, num_queries):  # , aux_loss=False):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # final predictions
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        # # outputs two values for bounding box, 3 layers
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # CNN --> Transformer projector
        self.input_proj = nn.Linear(backbone.num_channels, hidden_dim)
        self.backbone = backbone
        # self.aux_loss = aux_loss

    def forward(self, samples: torch.Tensor):
        """The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched images, of shape [batch_size x C x S]
              It returns a dict with the following elements:
        - "pred_logits": the classification logits (including no-object) for all queries (how
         many events are detected)
                         Shape= [batch_size x num_queries x (num_classes + 1)]
        - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                        (center_x, center_y, height, width). These values are normalized in
                        [0, 1],
                        relative to the size of each individual image (disregarding possible
                        padding).
                        See PostProcess for information on how to retrieve the unnormalized
                        bounding box.
        - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a
        list of
                         dictionaries containing the two above keys for each decoder layer.
        """
        # changed to only account for tensors
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # src, mask = features[-1].decompose()
        src = features
        # no attention mask needed
        # assert mask is not None
        mask = None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        # for no reasons currently
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        # #not needed for our purposes
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out