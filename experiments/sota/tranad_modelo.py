"""TranAD (Tuli, Casale e Jennings, VLDB 2022) — as classes do MODELO copiadas sem alteração de TSB-AD 1.5 (`TSB_AD/models/TranAD.py`, Apache-2.0,
https://github.com/TheDatumOrg/TSB-AD), que por sua vez adapta o repositório oficial (https://github.com/imperial-qore/TranAD, BSD-3).
Copiadas para não instalar no worker as dependências pesadas do TSB-AD (transformers, numba, tslearn). O que MUDA está só em `TranADCausal`, no fim do arquivo:
escala min-max do TREINO (como no repositório oficial; o TSB-AD padroniza cada chamada pelos próprios dados, o que apagaria mudança de nível do mês testado) e escore
atribuído ao FIM de cada janela (o TSB-AD preenche as bordas de forma centrada, o que usa dado futuro). O treino segue o `fit` do TSB-AD: perda em duas fases, AdamW
(lr 1e-4, decaimento 1e-5), StepLR(5; 0,9), MSE, 20% finais para validação, parada antecipada com paciência 3, até 50 épocas, lote 128.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, *args, **kwargs):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, *args, **kwargs):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TranADModel(nn.Module):
    def __init__(self, batch_size, feats, win_size):
        super(TranADModel, self).__init__()
        self.name = "TranAD"
        self.batch = batch_size
        self.n_feats = feats
        self.n_window = win_size
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranADCausal:
    """fit(W) com janelas (n, w, d) de treino; score(W) = MSE da fase 2 no último instante de cada janela, média entre as variáveis."""
    def __init__(self, feats, win_size=10, semente=0, epocas=50, paciencia=3, lr=1e-4, lote=128, validacao=0.2):
        torch.manual_seed(semente); torch.set_num_threads(1)
        self.feats, self.w, self.epocas, self.paciencia, self.lote, self.val = feats, win_size, epocas, paciencia, lote, validacao
        self.rng = np.random.default_rng(semente)
        self.model = TranADModel(batch_size=lote, feats=feats, win_size=win_size)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5); self.sch = torch.optim.lr_scheduler.StepLR(self.opt, 5, 0.9)
        self.crit = nn.MSELoss()

    def _esc(self, W):
        return np.clip((W - self.lo) / self.amp, -10, 10).astype("float32")

    def _passo(self, xb, epoca, treino):
        x = torch.tensor(xb).permute(1, 0, 2); elem = x[-1, :, :].view(1, x.shape[1], self.feats)
        z = self.model(x, elem); loss = (1 / epoca) * self.crit(z[0], elem) + (1 - 1 / epoca) * self.crit(z[1], elem)
        if treino:
            self.opt.zero_grad(); loss.backward(retain_graph=True); self.opt.step()
        return float(loss)

    def fit(self, W):
        flat = W.reshape(-1, W.shape[-1]); self.lo = flat.min(axis=0); self.amp = np.where(flat.max(axis=0) - self.lo > 0, flat.max(axis=0) - self.lo, 1.0)
        X = self._esc(W); n = len(X); nv = int(self.val * n); Xt, Xv = X[: n - nv], X[n - nv:]; melhor, cont, estado = np.inf, 0, None
        for ep in range(1, self.epocas + 1):
            self.model.train()
            for b in np.array_split(self.rng.permutation(len(Xt)), max(1, len(Xt) // self.lote)):
                self._passo(Xt[b], ep, True)
            self.model.eval()
            with torch.no_grad():
                lv = np.mean([self._passo(Xv[i:i + 1024], ep, False) for i in range(0, len(Xv), 1024)]) if len(Xv) else 0.0
            self.sch.step()
            if lv < melhor - 1e-4:
                melhor, cont, estado = lv, 0, {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                cont += 1
                if cont >= self.paciencia:
                    break
        if estado is not None:
            self.model.load_state_dict(estado)
        self.model.eval(); return self

    def score(self, W):
        X = self._esc(W); out = np.empty(len(X))
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                x = torch.tensor(X[i:i + 2048]).permute(1, 0, 2); elem = x[-1, :, :].view(1, x.shape[1], self.feats)
                _, z = self.model(x, elem); out[i:i + 2048] = ((z - elem) ** 2)[0].mean(dim=-1).numpy()
        return out
