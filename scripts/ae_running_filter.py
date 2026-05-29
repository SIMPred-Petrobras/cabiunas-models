#!/usr/bin/env python
"""
Filtro de operação (running_a) sobre as detecções do AE.

Investiga se quedas por DESLIGAMENTO dominam as detecções e aplica máscara de
operação. Proxy: running_a (1=ligada, 0=desligada). Substituível por NGP_A.

Uso: python scripts/ae_running_filter.py
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from ae_separation_experiment import load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL

# ----------------------------------------------------------------------------
# Flag de operação parametrizada. Quando NGP_A estiver disponível no dataset,
# basta trocar para OPERATION_FLAG = "NGP_A" (e ajustar o limiar se contínuo).
OPERATION_FLAG = RUNNING_COL  # "RUNNING_A" (proxy de NGP_A)
# ----------------------------------------------------------------------------

TS = 60; STRIDE = 10; F1 = 16; F2 = 3; S1 = 2; S2 = 2
PRE_H = 6.0; POST_H = 1.0; GAP_H = 12.0
THR = 0.228            # threshold definido no relatório de separação
BUFFER_MIN = 5.0       # buffer de transição (min) antes/depois de mudar de estado
OUT = "ae_running_out"


def log(m): print(f"[RUN] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, inc = load(priority=["HIGH", "MEDIUM", "LOLO"])
    n = len(df)

    op = pd.to_numeric(df[OPERATION_FLAG], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    X = df[SENSORS].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)

    # distancia (h) ao incidente mais proximo (unidade-robusta p/ pandas 3.0)
    tsec = df[TIME_COL].values.astype("datetime64[s]").astype("int64")
    asec = pd.DatetimeIndex(inc).values.astype("datetime64[s]").astype("int64")
    pos = np.searchsorted(asec, tsec)
    dnext = np.where(pos < len(asec), (asec[np.clip(pos,0,len(asec)-1)]-tsec), np.inf)/3600.0
    dprev = np.where(pos > 0, (tsec-asec[np.clip(pos-1,0,len(asec)-1)]), np.inf)/3600.0
    dany = np.minimum(dnext, dprev)

    # buffer de transicao: marca pts a < BUFFER_MIN de qualquer mudanca de estado
    buf_pts = int(BUFFER_MIN*60/30)
    trans = np.where(np.abs(np.diff(op)) > 0)[0]
    in_buffer = np.zeros(n, bool)
    for k in trans:
        in_buffer[max(0,k-buf_pts):min(n,k+buf_pts+1)] = True

    # sequencias
    seqs, starts = make_seqs(X, TS, STRIDE)
    ends = starts + TS - 1
    run_frac = np.array([op[s:s+TS].mean() for s in starts])      # fracao ligada na janela
    run_all = run_frac >= 0.999                                    # 100% ligada
    buf_seq = np.array([in_buffer[s:s+TS].any() for s in starts])  # toca transicao
    d_end = dnext[ends]; dany_end = dany[ends]; t_end = df[TIME_COL].to_numpy()[ends]

    is_pos = (d_end >= -POST_H) & (d_end <= PRE_H) & run_all
    is_neg = (dany_end > GAP_H) & run_all

    # treino = negativos puros (ligado, longe de incidente), split temporal
    neg_idx = np.where(is_neg)[0]; n_tr = int(0.9*len(neg_idx))
    tr_idx = neg_idx[:n_tr]
    rng = np.random.default_rng(42)
    if len(tr_idx) > 40000: tr_idx = rng.choice(tr_idx, 40000, replace=False)
    flat = seqs[tr_idx].reshape(-1, seqs.shape[-1]); mu = flat.mean(0); sd = flat.std(0); sd[sd==0]=1
    norm = lambda a: (a-mu)/sd
    model, latent = build_ae(TS, seqs.shape[-1], F1, F2, S1, S2, 0.1, 1e-4)
    log(f"latente={latent} ratio={latent/(TS*seqs.shape[-1]):.3f} | treino={len(tr_idx)} seqs (todas ligadas)")
    from tensorflow import keras
    cb=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    va_idx = neg_idx[n_tr:]
    model.fit(norm(seqs[tr_idx]), norm(seqs[tr_idx]),
              validation_data=(norm(seqs[va_idx]), norm(seqs[va_idx])),
              epochs=20, batch_size=256, verbose=2, callbacks=cb)

    mse_all = mse_per_seq(model, norm(seqs))
    det_raw = mse_all >= THR

    # ===================== INVESTIGAÇÃO (b) =====================
    res = {}
    n_det = int(det_raw.sum())
    det_off = int((det_raw & (run_frac < 0.999)).sum())     # janela toca desligamento
    det_off_end = int((det_raw & (op[ends] < 0.5)).sum())   # ponto final desligado
    log("=== (b) deteccoes brutas vs operacao ===")
    log(f"deteccoes brutas (mse>={THR}): {n_det}")
    log(f"  com janela NAO-100%-ligada: {det_off} ({100*det_off/max(n_det,1):.1f}%)")
    log(f"  com ponto final desligado : {det_off_end} ({100*det_off_end/max(n_det,1):.1f}%)")
    res["b_det_raw"]=n_det; res["b_det_janela_off"]=det_off; res["b_pct_off"]=round(100*det_off/max(n_det,1),1)

    # ===================== CORREÇÃO: máscaras =====================
    det_mask_op = det_raw & run_all                       # so 100% ligada
    det_mask_opbuf = det_raw & run_all & (~buf_seq)        # + fora de buffer de transicao
    log("=== correcao: deteccoes apos filtro ===")
    log(f"  apos mascara operacao (100% ligada): {int(det_mask_op.sum())} (elimina {n_det-int(det_mask_op.sum())})")
    log(f"  apos operacao + buffer {BUFFER_MIN:.0f}min: {int(det_mask_opbuf.sum())} (elimina {n_det-int(det_mask_opbuf.sum())})")
    res["c_det_op"]=int(det_mask_op.sum()); res["c_det_op_buf"]=int(det_mask_opbuf.sum())
    res["eliminados_abs"]=n_det-int(det_mask_opbuf.sum())
    res["eliminados_pct"]=round(100*(n_det-int(det_mask_opbuf.sum()))/max(n_det,1),1)

    # ===================== métricas em operação =====================
    def metrics(scores, y):
        order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
        np_,nn=int(y.sum()),int((~y).sum())
        auc=(ranks[y].sum()-np_*(np_+1)/2)/(np_*nn)
        pred=scores>=THR
        tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
        p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);f=2*p*r/max(p+r,1e-9)
        return dict(auc=float(auc),precision=float(p),recall=float(r),f1=float(f),tp=tp,fp=fp,fn=fn)
    # eval so em operacao: positivos vs negativos (ambos 100% ligados)
    ev = is_pos | is_neg
    m_op = metrics(mse_all[ev], is_pos[ev])
    res["metrics_operacao"]=m_op
    log(f"=== metricas SO em operacao: AUC={m_op['auc']:.3f} P={m_op['precision']:.3f} R={m_op['recall']:.3f} F1={m_op['f1']:.3f}")

    # FP/dia antes (inclui off) vs depois (so operacao+buffer). FP = deteccao fora de janela pos.
    span_days = (df[TIME_COL].max()-df[TIME_COL].min()).total_seconds()/86400
    # cada seq cobre STRIDE*30s; converte contagem de seq->"pts" equivalentes p/ taxa diaria comparavel
    fp_before = int((det_raw & ~is_pos).sum())
    fp_after = int((det_mask_opbuf & ~is_pos).sum())
    res["fp_per_day_before"]=round(fp_before/span_days,1)
    res["fp_per_day_after"]=round(fp_after/span_days,1)
    log(f"=== FP/dia (seq): antes={fp_before/span_days:.1f} depois={fp_after/span_days:.1f}")

    # ===================== plots =====================
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (c) serie + running + deteccoes filtradas
    fig,ax=plt.subplots(2,1,figsize=(15,7),sharex=True)
    ax[0].plot(t_end, mse_all, lw=.5, color="steelblue"); ax[0].axhline(THR,color="r",ls="--",label=f"thr={THR}")
    ax[0].scatter(t_end[det_raw],mse_all[det_raw],s=4,color="gray",label="deteccao BRUTA",zorder=2)
    ax[0].scatter(t_end[det_mask_opbuf],mse_all[det_mask_opbuf],s=6,color="orange",label="deteccao EM OPERACAO",zorder=3)
    ax[0].set_ylabel("MSE/seq"); ax[0].legend(loc="upper right"); ax[0].set_title("Antes (cinza) vs depois do filtro running_a (laranja)")
    # running overlay
    opf=np.array([op[s:s+TS].mean() for s in starts])
    ax[1].fill_between(t_end,0,1,where=(opf<0.999),color="#ef6c00",alpha=.4,step="mid",label="janela c/ desligamento")
    ax[1].plot(t_end,opf,lw=.5,color="green"); ax[1].set_ylabel("frac ligada"); ax[1].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/serie_running_filtro.png",dpi=120); plt.close(fig)

    # histograma do erro por running
    fig2,a2=plt.subplots(figsize=(9,4))
    e_off=mse_all[run_frac<0.5]; e_on=mse_all[run_frac>=0.999]
    a2.hist(e_on,bins=80,density=True,alpha=.6,label=f"ligada ({len(e_on)})")
    a2.hist(e_off,bins=80,density=True,alpha=.6,label=f"desligada ({len(e_off)})")
    a2.axvline(THR,color="r",ls="--",label=f"thr={THR}")
    a2.set_title("Erro de reconstrucao por estado de operacao"); a2.set_xlabel("MSE/seq"); a2.legend()
    a2.set_yscale("log")
    fig2.tight_layout(); fig2.savefig(f"{OUT}/hist_erro_por_running.png",dpi=120); plt.close(fig2)
    res["err_med_ligada"]=float(np.median(e_on)); res["err_med_desligada"]=float(np.median(e_off))
    log(f"=== erro mediano: ligada={np.median(e_on):.4f} desligada={np.median(e_off):.4f}")

    json.dump(res, open(f"{OUT}/running_filter_result.json","w"), indent=2)
    log(f"salvo {OUT}/*.png e running_filter_result.json")


if __name__ == "__main__":
    main()
