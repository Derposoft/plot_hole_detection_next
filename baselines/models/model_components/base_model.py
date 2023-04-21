"""
Base model for all matching model
"""
from torch import nn
import typing
import torch
import torch.nn.functional as F
import numpy as np

from libraries.matchzoo.engine.param_table import ParamTable
from libraries.matchzoo.engine.param import Param
from libraries.matchzoo.engine import hyper_spaces

from baselines.models.model_components.base_components import LSTM
from baselines.models.model_components.self_attention import (
    MultiHeadSelfAttentionICLR2017Extend,
)
import baselines.preprocess as parse
import baselines.utils as torch_utils
from baselines.setting_keywords import KeywordSettings
import torchtext.vocab as vocab


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def get_default_params(
        self, with_embedding=False, with_multi_layer_perceptron=False
    ):
        """
        Model default parameters.

        The common usage is to instantiate :class:`matchzoo.engine.ModelParams`
            first, then set the model specific parametrs.

        Examples:
            >>> class MyModel(BaseModel):
            ...     def build(self):
            ...         print(self._params['num_eggs'], 'eggs')
            ...         print('and', self._params['ham_type'])
            ...
            ...
            ...     def get_default_params(self):
            ...         params = ParamTable()
            ...         params.add(Param('num_eggs', 512))
            ...         params.add(Param('ham_type', 'Parma Ham'))
            ...         return params
            >>> my_model = MyModel()
            >>> my_model.build()
            512 eggs
            and Parma Ham

        Notice that all parameters must be serialisable for the entire model
        to be serialisable. Therefore, it's strongly recommended to use python
        native data types to store parameters.

        :return: model parameters

        """
        params = ParamTable()
        params.add(
            Param(
                name="model_class",
                value=self.__class__.__name__,
                desc="Model class. Used internally for save/load. "
                "Changing this may cause unexpected behaviors.",
            )
        )
        params.add(
            Param(
                name="input_shapes",
                desc="Dependent on the model and data. Should be set manually.",
            )
        )
        params.add(
            Param(name="task", desc="Decides model output shape, loss, and metrics.")
        )
        params.add(
            Param(
                name="optimizer",
                value="adam",
            )
        )
        if with_embedding:
            params.add(
                Param(
                    name="with_embedding",
                    value=True,
                    desc="A flag used help `auto` module. Shouldn't be changed.",
                )
            )
            params.add(
                Param(
                    name="embedding_input_dim",
                    desc="Usually equals vocab size + 1. Should be set manually.",
                )
            )
            params.add(
                Param(name="embedding_output_dim", desc="Should be set manually.")
            )
            params.add(
                Param(
                    name="embedding_trainable",
                    value=True,
                    desc="`True` to enable embedding layer training, "
                    "`False` to freeze embedding parameters.",
                )
            )
        if with_multi_layer_perceptron:
            params.add(
                Param(
                    name="with_multi_layer_perceptron",
                    value=True,
                    desc="A flag of whether a multiple layer perceptron is used. "
                    "Shouldn't be changed.",
                )
            )
            params.add(
                Param(
                    name="mlp_num_units",
                    value=128,
                    desc="Number of units in first `mlp_num_layers` layers.",
                    hyper_space=hyper_spaces.quniform(8, 256, 8),
                )
            )
            params.add(
                Param(
                    name="mlp_num_layers",
                    value=3,
                    desc="Number of layers of the multiple layer percetron.",
                    hyper_space=hyper_spaces.quniform(1, 6),
                )
            )
            params.add(
                Param(
                    name="mlp_num_fan_out",
                    value=64,
                    desc="Number of units of the layer that connects the multiple "
                    "layer percetron and the output.",
                    hyper_space=hyper_spaces.quniform(4, 128, 4),
                )
            )
            params.add(
                Param(
                    name="mlp_activation_func",
                    value="relu",
                    desc="Activation function used in the multiple "
                    "layer perceptron.",
                )
            )
        return params

    def _make_perceptron_layer(
        self,
        in_features: int = 0,
        out_features: int = 0,
        activation: nn.Module = nn.ReLU,
    ) -> nn.Module:
        """:return: a perceptron layer."""
        return nn.Sequential(nn.Linear(in_features, out_features), activation)

    def _make_output_layer(
        self, in_features: int = 0, activation: typing.Union[str, nn.Module] = None
    ) -> nn.Module:
        """:return: a correctly shaped torch module for model output."""
        if activation:
            return nn.Sequential(
                nn.Linear(in_features, 1), parse.parse_activation(activation)
            )
        else:
            return nn.Linear(in_features, 1)

    def _make_default_embedding_layer(self, _params, use_torchtext=True) -> nn.Module:
        """:return: an embedding module."""
        if use_torchtext:
            # MatchZoo sucks -- use torchtext.
            glove = vocab.GloVe(name="6B", dim=100)
            return nn.Embedding.from_pretrained(glove.vectors)

        if isinstance(_params["embedding"], np.ndarray):
            _params["embedding_input_dim"] = _params["embedding"].shape[0]
            _params["embedding_output_dim"] = _params["embedding"].shape[1]
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(_params["embedding"]),
                freeze=_params["embedding_freeze"],
            )
        else:
            return nn.Embedding(
                num_embeddings=_params["embedding_input_dim"],
                embedding_dim=_params["embedding_output_dim"],
            )

    def _make_default_char_embedding_layer(self, _params) -> nn.Module:
        """:return: an embedding module."""
        if isinstance(_params["char_embedding"], np.ndarray):
            _params["char_embedding_input_dim"] = _params["char_embedding"].shape[0]
            _params["char_embedding_output_dim"] = _params["char_embedding"].shape[1]
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(_params["char_embedding"]),
                freeze=_params["char_embedding_freeze"],
            )
        else:
            return nn.Embedding(
                num_embeddings=_params["char_embedding_input_dim"],
                embedding_dim=_params["char_embedding_output_dim"],
            )

    def _make_entity_embedding_layer(
        self, matrix: np.ndarray, freeze: bool
    ) -> nn.Module:
        """:return: an embedding module."""
        return nn.Embedding.from_pretrained(
            embeddings=torch.Tensor(matrix), freeze=freeze
        )

    def predict(
        self, query: np.ndarray, doc: np.ndarray, verbose: bool = False, **kargs
    ) -> np.ndarray:
        self.train(False)  # very important, to disable dropout
        if verbose:
            print("query: ", query)
            print("doc: ", doc)
            print("================ end of query doc =================")
        out = self(query, doc, verbose, **kargs)
        return torch_utils.cpu(out).detach().numpy().flatten()

    def forward(self, *input):
        pass


class BasicFCModel(BaseModel):
    """
    Basic Fact-checking model used for all other models
    """

    def __init__(self, params):
        super(BaseModel, self).__init__()
        self._params = params
        self.embedding = self._make_default_embedding_layer(params)
        self.num_classes = self._params["num_classes"]
        self.fixed_length_right = self._params["fixed_length_right"]
        self.fixed_length_left = self._params["fixed_length_left"]
        self.use_claim_source = self._params["use_claim_source"]
        self.use_article_source = self._params["use_article_source"]
        self._use_cuda = self._params["cuda"]
        self.num_heads = 1  # self._params["num_att_heads"]
        self.dropout_left = self._params["dropout_left"]
        self.dropout_right = self._params["dropout_right"]
        self.hidden_size = self._params["hidden_size"]
        if self.use_claim_source:
            self.claim_source_embs = self._make_entity_embedding_layer(
                self._params["claim_source_embeddings"], freeze=False
            )  # trainable
            self.claim_emb_size = self._params["claim_source_embeddings"].shape[1]

        if self.use_article_source:
            self.article_source_embs = self._make_entity_embedding_layer(
                self._params["article_source_embeddings"], freeze=False
            )  # trainable
            self.article_emb_size = self._params["article_source_embeddings"].shape[1]

        D = self._params["embedding_output_dim"]
        # self.linear1 = nn.Sequential(
        #     nn.Linear(self._params["embedding_output_dim"] + 3 * D, 1),
        #     # self.activation
        #     nn.Tanh()
        # )
        # self.linear1[0].apply(torch_utils.init_weights)
        self.bilstm = LSTM(
            input_size=D,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout_left,
        )
        self.query_bilstm = LSTM(
            input_size=D,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout_right,
        )
        input_size = (
            4 * self.hidden_size + self.claim_emb_size
            if self.use_claim_source
            else 4 * self.hidden_size
        )
        self.self_att_word = MultiHeadSelfAttentionICLR2017Extend(
            inp_dim=input_size, out_dim=2 * self.hidden_size, num_heads=self.num_heads
        )

        evd_input_size = 4 * self.hidden_size * self.num_heads
        if self.use_article_source:
            evd_input_size += self.article_emb_size
            input_size += self.article_emb_size
        if self.use_claim_source:
            evd_input_size += self.claim_emb_size
            # input_size += self.claim_emb_size
        self.self_att_evd = MultiHeadSelfAttentionICLR2017Extend(
            inp_dim=evd_input_size,
            out_dim=2 * self.hidden_size,
            num_heads=self.num_heads,
        )
        self.out = nn.Sequential(
            nn.Linear(input_size * self.num_heads, self.hidden_size),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            nn.Linear(self.hidden_size, 1),
            # nn.ReLU(),  # no one uses ReLU at the end of a linear layer
            # nn.Sigmoid()
        )
        self.out[0].apply(torch_utils.init_weights)
        self.out[1].apply(torch_utils.init_weights)

    def forward(
        self, query: torch.Tensor, document: torch.Tensor, verbose=False, **kargs
    ):
        pass

    def _pad_left_tensor(self, left_tsr: torch.Tensor, **kargs):
        """pad left tensor of shape (B, H) to tensor of shape (n1 + n2 + ... + nx, H)"""
        evd_count_per_query = kargs[KeywordSettings.EvidenceCountPerQuery]
        B, H = left_tsr.size()
        assert evd_count_per_query.size(0) == left_tsr.size(0)
        ans = []
        for num_evd, tsr in zip(evd_count_per_query, left_tsr):
            # num_evd = evd_count_per_query[idx] # int(torch_utils.cpu(evd_count_per_query[idx]).detach().numpy())
            tmp = tsr.clone()
            tsr = tmp.expand(num_evd, H)
            ans.append(tsr)
        ans = torch.cat(ans, dim=0)  # (n1 + n2 + ... + nx, H)
        return ans

    @classmethod
    def _pad_right_tensor(self, tsr: torch.Tensor, **kargs):
        """
        padding the output evidences. I avoid input empty sequence into lstm due to exception. I tried to make add mask
        to empty sequence but I don't have much belief in it.
        Parameters
        ----------
        bilstm_out: `torch.Tensor` (n1 + n2 + ... + n_B, H)
        doc_src: `torch.Tensor` (B, n) where n is the maximum number of evidences
        Returns
        -------

        """
        max_num_evd = kargs[KeywordSettings.FIXED_NUM_EVIDENCES]
        evd_count_per_query = kargs[KeywordSettings.EvidenceCountPerQuery]
        batch_size = evd_count_per_query.size(0)
        b_prime, H = tsr.size()
        last = 0
        ans = []
        for idx in range(batch_size):
            num_evd = int(torch_utils.cpu(evd_count_per_query[idx]).detach().numpy())
            hidden_vectors = tsr[last : last + num_evd]  # (n1, H)
            padded = F.pad(
                hidden_vectors, (0, 0, 0, max_num_evd - num_evd), "constant", 0
            )
            ans.append(padded)
            last += num_evd
        ans = torch.stack(ans, dim=0)
        assert ans.size() == (batch_size, max_num_evd, H)
        return ans


if __name__ == "__main__":
    print("here")
