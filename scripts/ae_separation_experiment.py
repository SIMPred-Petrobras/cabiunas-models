#!/usr/bin/env python
"""
Experimento local de separação normal vs anomalia para o AE de detecção.

Objetivo: medir AUC-ROC com um RÓTULO LIMPO (positivos = pré-alarme curto,
negativos = longe de qualquer alarme) e iterar rápido em CPU sobre os levers:
bottleneck, janela temporal, pureza do treino e arquitetura.

Uso:
  python scripts/ae_separation_experiment.py --time_steps 60 --f1 16 --f2 4 \
      --pre_hours 8 --gap_hours 72 --epochs 15 --tag baseline
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd

RAW = "../dados/sensores_brutos_2025_30s.csv"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
TIME_COL = "data_datetime"
RUNNING_COL = "RUNNING_A"
SENSORS = [
    "T5_AVG_A","TC382_01_A","TC382_02_A","TC382_03_A","TC382_04_A","TC382_05_A","TC382_06_A",
    "TV_351X_A","TV_351Y_A","TV_352X_A","TV_352Y_A","TV_353X_A","TV_353Y_A",
    "TV_354X_A","TV_354Y_A","TV_355X_A","TV_355Y_A",
]
OUT = "ae_separation_out"


def log(m): print(f"[EXP] {m}", flush=True)


def load(priority=None):
    df = pd.read_csv(RAW)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    al = pd.read_csv(ALARM)
    al["t"] = pd.to_datetime(al["Data da Ocorrência"], errors="coerce")
    al = al[(al["t"] >= df[TIME_COL].min()) & (al["t"] <= df[TIME_COL].max())]
    # onsets reais (descarta os 'OK' = retorno ao normal)
    ons = al[al["Condição do Alarme"] != "OK"].copy()
    if priority:  # ex: ['HIGH','MEDIUM','LOLO']
        ons = ons[ons["Prioridade"].isin(priority)]
    ons = ons.sort_values("t")
    # dedup em incidentes (gap>4h vira novo incidente) -> usa o 1o timestamp
    g = (ons["t"].diff().dt.total_seconds() / 3600 > 4).cumsum()
    inc = ons.groupby(g)["t"].min().sort_values().reset_index(drop=True)
    log(f"raw={len(df)} pts | onsets={len(ons)} | incidentes={len(inc)} | priority={priority or 'todas'}")
    return df, inc


def make_seqs(values, ts, stride):
    n = len(values); idx = list(range(0, n - ts + 1, stride))
    out = np.stack([values[i:i+ts] for i in idx], axis=0)
    return out, np.array(idx)


def build_ae(ts, nf, f1, f2, s1, s2, dropout, l2):
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    reg = keras.regularizers.l2(l2) if l2 > 0 else None
    inp = keras.Input((ts, nf))
    x = layers.Conv1D(f1, 7, padding="same", strides=s1, activation="relu", kernel_regularizer=reg)(inp)
    if dropout > 0: x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(f2, 7, padding="same", strides=s2, activation="relu", kernel_regularizer=reg)(x)
    latent = int(np.ceil(ts/(s1*s2))) * f2
    x = layers.Conv1DTranspose(f2, 7, padding="same", strides=s2, activation="relu", kernel_regularizer=reg)(x)
    if dropout > 0: x = layers.Dropout(dropout)(x)
    x = layers.Conv1DTranspose(f1, 7, padding="same", strides=s1, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Conv1DTranspose(nf, 3, padding="same")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m, latent


def mse_per_seq(model, x, bs=512):
    out = np.empty(len(x), np.float32)
    for s in range(0, len(x), bs):
        xb = x[s:s+bs]; pb = np.asarray(model.predict_on_batch(xb))
        out[s:s+len(xb)] = np.mean((pb-xb)**2, axis=(1,2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_steps", type=int, default=60)
    ap.add_argument("--f1", type=int, default=16)
    ap.add_argument("--f2", type=int, default=4)
    ap.add_argument("--s1", type=int, default=2)
    ap.add_argument("--s2", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--pre_hours", type=float, default=6.0)
    ap.add_argument("--post_hours", type=float, default=1.0)
    ap.add_argument("--gap_hours", type=float, default=12.0)
    ap.add_argument("--priority", type=str, default="all", help="all | hi (HIGH/MEDIUM/LOLO)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_train", type=int, default=40000)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--plot_series", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    prio = ["HIGH", "MEDIUM", "LOLO"] if args.priority == "hi" else None
    df, at = load(priority=prio)
    n = len(df)
    dt = 30.0  # segundos por amostra
    # distancia (em horas) de cada ponto ao alarme mais proximo, com sinal:
    #   <0 = antes do proximo alarme | >0 = depois do alarme anterior
    # conversao robusta a unidade (pandas 3.0 usa datetime64[us], nao [ns])
    tsec = df[TIME_COL].values.astype("datetime64[s]").astype("int64")
    asec = pd.DatetimeIndex(at).values.astype("datetime64[s]").astype("int64")
    pos = np.searchsorted(asec, tsec)
    dist_next = np.where(pos < len(asec), (asec[np.clip(pos,0,len(asec)-1)]-tsec), np.inf)/3600.0
    dist_prev = np.where(pos > 0, (tsec-asec[np.clip(pos-1,0,len(asec)-1)]), np.inf)/3600.0
    dist_any = np.minimum(dist_next, dist_prev)
    running = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy()

    # matriz de sensores
    X = df[SENSORS].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)

    ts = args.time_steps
    seqs, starts = make_seqs(X, ts, args.stride)
    ends = starts + ts - 1
    # rotulos por sequencia (usa o ponto final da janela)
    d_next_end = dist_next[ends]; d_any_end = dist_any[ends]; run_end = running[ends]
    is_pos = (d_next_end >= -args.post_hours) & (d_next_end <= args.pre_hours)
    is_neg = (d_any_end > args.gap_hours) & (run_end > 0.5)
    log(f"seqs={len(seqs)} | positivos={is_pos.sum()} | negativos(puros)={is_neg.sum()}")

    # treino = negativos puros, split temporal (primeiros 90%)
    neg_idx = np.where(is_neg)[0]
    n_tr = int(0.9*len(neg_idx)); tr_idx = neg_idx[:n_tr]; va_idx = neg_idx[n_tr:]
    if len(tr_idx) > args.max_train:
        rng = np.random.default_rng(42); tr_idx = rng.choice(tr_idx, args.max_train, replace=False)
    # normalizacao: fit SO no treino
    flat = seqs[tr_idx].reshape(-1, seqs.shape[-1])
    mu = flat.mean(0); sd = flat.std(0); sd[sd==0]=1.0
    norm = lambda a: (a-mu)/sd
    x_tr = norm(seqs[tr_idx]); x_va = norm(seqs[va_idx])
    x_pos = norm(seqs[is_pos]); x_neg_eval = norm(seqs[va_idx])  # negativos de avaliacao = val

    model, latent = build_ae(ts, seqs.shape[-1], args.f1, args.f2, args.s1, args.s2, args.dropout, args.l2)
    in_dim = ts*seqs.shape[-1]
    log(f"latente={latent} vs entrada={in_dim} (ratio={latent/in_dim:.3f}) | x_tr={x_tr.shape}")
    from tensorflow import keras
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    model.fit(x_tr, x_tr, validation_data=(x_va, x_va), epochs=args.epochs, batch_size=256, verbose=2, callbacks=cb)

    mse_pos = mse_per_seq(model, x_pos)
    mse_neg = mse_per_seq(model, x_neg_eval)

    # AUC-ROC (Mann-Whitney U via ranking, sem sklearn)
    y = np.concatenate([np.ones(len(mse_pos)), np.zeros(len(mse_neg))])
    s = np.concatenate([mse_pos, mse_neg])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s)+1)
    # corrige empates pela media de rank
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sum_r = np.zeros(len(cnt)); np.add.at(sum_r, inv, ranks); ranks = (sum_r/cnt)[inv]
    n_pos = len(mse_pos); n_neg = len(mse_neg)
    auc = (ranks[:n_pos].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    # varredura de threshold -> ROC + melhor F1
    thr_grid = np.quantile(s, np.linspace(0,1,400))
    fpr=[]; tpr=[]; best_f1=0.0; thr_best=float(np.median(mse_neg)); prec=0.0; rec=0.0
    for t in thr_grid:
        pred = s>=t
        tp=int((pred&(y==1)).sum()); fp=int((pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum())
        tn=int((~pred&(y==0)).sum())
        fpr.append(fp/max(fp+tn,1)); tpr.append(tp/max(tp+fn,1))
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f=2*p*r/max(p+r,1e-9)
        if f>best_f1: best_f1=f; thr_best=float(t); prec=p; rec=r
    fpr=np.array(fpr); tpr=np.array(tpr)
    order2=np.argsort(fpr); fpr=fpr[order2]; tpr=tpr[order2]
    f1_best=best_f1

    res = dict(tag=args.tag, auc=float(auc), f1=float(f1_best), precision=float(prec), recall=float(rec),
               threshold=float(thr_best), n_pos=int(len(mse_pos)), n_neg=int(len(mse_neg)),
               latent=int(latent), latent_ratio=float(latent/in_dim),
               mse_pos_med=float(np.median(mse_pos)), mse_neg_med=float(np.median(mse_neg)),
               sep_ratio_med=float(np.median(mse_pos)/max(np.median(mse_neg),1e-9)),
               params=vars(args), secs=round(time.time()-t0,1))
    log("RESULT " + json.dumps({k:res[k] for k in ["tag","auc","f1","precision","recall","latent_ratio","sep_ratio_med","n_pos","n_neg","secs"]}))

    # plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,4))
    ax[0].hist(mse_neg, bins=80, density=True, alpha=.6, label="normal")
    ax[0].hist(mse_pos, bins=80, density=True, alpha=.6, label="anomalia (pre-alarme)")
    ax[0].axvline(thr_best, color="r", ls="--", label=f"thr={thr_best:.4g}")
    ax[0].set_title(f"Erro recon normal vs anomalia ({args.tag})"); ax[0].legend(); ax[0].set_xlabel("MSE/seq")
    ax[1].plot(fpr,tpr,label=f"AUC={auc:.3f}"); ax[1].plot([0,1],[0,1],"k--",alpha=.3)
    ax[1].set_title("ROC"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{args.tag}_hist_roc.png", dpi=130); plt.close(fig)

    # serie temporal: erro de reconstrucao ao longo do tempo, com threshold e incidentes
    if args.plot_series:
        mse_all = mse_per_seq(model, norm(seqs))
        t_end = df[TIME_COL].to_numpy()[ends]
        figs, axs = plt.subplots(figsize=(15,4))
        axs.plot(t_end, mse_all, lw=0.5, color="steelblue")
        axs.axhline(thr_best, color="r", ls="--", label=f"threshold={thr_best:.3g}")
        det = mse_all >= thr_best
        axs.scatter(t_end[det], mse_all[det], s=4, color="orange", label="detectado", zorder=3)
        for it in pd.DatetimeIndex(at):
            axs.axvline(it, color="green", alpha=0.25, lw=0.8)
        axs.set_title(f"Erro de reconstrucao ao longo de 2025 ({args.tag}) | verde=incidente")
        axs.set_xlabel("tempo"); axs.set_ylabel("MSE/seq"); axs.legend(loc="upper right")
        figs.tight_layout(); figs.savefig(f"{OUT}/{args.tag}_serie_tempo.png", dpi=130); plt.close(figs)
        log(f"salvo {OUT}/{args.tag}_serie_tempo.png")

    with open(f"{OUT}/{args.tag}_result.json","w") as f: json.dump(res,f,indent=2)
    log(f"salvo {OUT}/{args.tag}_*.png/json")


if __name__ == "__main__":
    main()
