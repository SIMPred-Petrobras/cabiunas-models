#!/usr/bin/env python
"""
Camada PREDITIVA sobre o health-index do AE (Fase 2 do roadmap).

Converte o erro de reconstrução (detecção pontual) em ALERTA ANTECIPADO via
EWMA + persistência, e mede a capacidade preditiva pela curva
lead-time x falso-alarme/dia contra os incidentes genuínos (proxy HIGH/MED/LOLO).

Não depende da lista de tags da operação — usa o proxy validado; trocar depois é trivial.
Uso: python scripts/ae_predictive_layer.py
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from ae_separation_experiment import load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL

TS=60; STRIDE=10; F1=16; F2=3; S1=2; S2=2
HORIZON_H=float(os.getenv("HORIZON_H","24"))   # janela de antecipação considerada "acerto"
GAP_H=12.0              # negativos de treino: >12h de incidente
ALPHA=0.02              # EWMA (memória ~1/alpha janelas)
ALERT_DEBOUNCE_H=8.0    # episódios de alerta separados por >8h = distintos
OUT="ae_pred_out"


def log(m): print(f"[PRED] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, inc = load(priority=None)
    n=len(df)
    op=pd.to_numeric(df[RUNNING_COL],errors="coerce").fillna(0).to_numpy().astype(np.float32)
    X=df[SENSORS].apply(pd.to_numeric,errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)
    tsec=df[TIME_COL].values.astype("datetime64[s]").astype("int64")
    asec=pd.DatetimeIndex(inc).values.astype("datetime64[s]").astype("int64")
    pos=np.searchsorted(asec,tsec)
    dnext=np.where(pos<len(asec),(asec[np.clip(pos,0,len(asec)-1)]-tsec),np.inf)/3600.0
    dprev=np.where(pos>0,(tsec-asec[np.clip(pos-1,0,len(asec)-1)]),np.inf)/3600.0
    dany=np.minimum(dnext,dprev)

    seqs,starts=make_seqs(X,TS,STRIDE); ends=starts+TS-1
    run_all=np.array([op[s:s+TS].mean() for s in starts])>=0.999
    d_end=dnext[ends]; dany_end=dany[ends]; t_end=df[TIME_COL].to_numpy()[ends]; t_end_s=tsec[ends]

    # treino: negativos puros (ligado, longe de incidente)
    neg=np.where((dany_end>GAP_H)&run_all)[0]; n_tr=int(0.9*len(neg))
    rng=np.random.default_rng(42); tr=neg[:n_tr]
    if len(tr)>40000: tr=rng.choice(tr,40000,replace=False)
    flat=seqs[tr].reshape(-1,seqs.shape[-1]); mu=flat.mean(0); sd=flat.std(0); sd[sd==0]=1
    norm=lambda a:(a-mu)/sd
    model,latent=build_ae(TS,seqs.shape[-1],F1,F2,S1,S2,0.1,1e-4)
    from tensorflow import keras
    cb=[keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True)]
    va=neg[n_tr:]
    model.fit(norm(seqs[tr]),norm(seqs[tr]),validation_data=(norm(seqs[va]),norm(seqs[va])),
              epochs=20,batch_size=256,verbose=2,callbacks=cb)

    # health-index = MSE por janela; SÓ em operação (NaN fora)
    health=mse_per_seq(model,norm(seqs))
    health_op=np.where(run_all,health,np.nan)

    # EWMA (ignora NaN: mantém o último valor durante desligamento)
    ew=np.empty(len(health_op)); prev=np.nan
    for i,v in enumerate(health_op):
        if np.isnan(v): ew[i]=prev
        else: prev = v if np.isnan(prev) else (ALPHA*v+(1-ALPHA)*prev); ew[i]=prev
    ew=pd.Series(ew).ffill().bfill().to_numpy()

    # ---------- curva lead-time x falso-alarme/dia ----------
    span_days=(df[TIME_COL].max()-df[TIME_COL].min()).total_seconds()/86400
    inc_s=asec
    grid=np.quantile(ew[run_all], np.linspace(0.20,0.999,45))
    rows=[]
    deb_pts=int(ALERT_DEBOUNCE_H*3600/ (STRIDE*30))
    H=HORIZON_H*3600
    for thr in grid:
        alert = (ew>=thr) & run_all
        idx=np.where(alert)[0]
        # episodios debounced -> (inicio_s, fim_s)
        episodes=[]
        if len(idx):
            cur=[idx[0]]
            for j in idx[1:]:
                if j-cur[-1] <= deb_pts: cur.append(j)
                else: episodes.append((t_end_s[cur[0]],t_end_s[cur[-1]])); cur=[j]
            episodes.append((t_end_s[cur[0]],t_end_s[cur[-1]]))
        alert_s=t_end_s[idx]  # timestamps com alerta ativo
        # recall: incidente pego se HA alerta ATIVO em [t_inc-H, t_inc]; lead = 1o alerta na janela
        hits=0; leads=[]
        for ti in inc_s:
            w=alert_s[(alert_s>=ti-H)&(alert_s<=ti)]
            if len(w): hits+=1; leads.append((ti-w.min())/3600.0)
        recall=hits/len(inc_s)
        # falso-alarme: episodio que NAO intersecta nenhuma janela [t_inc-H, t_inc]
        fa=0
        for (s0,s1) in episodes:
            useful=(((inc_s-H)<=s1)&(inc_s>=s0)).any()  # intervalo [s0,s1] cruza [ti-H,ti]
            if not useful: fa+=1
        rows.append(dict(thr=float(thr),recall=float(recall),
                         fa_per_day=float(fa/span_days),
                         median_lead_h=float(np.median(leads)) if leads else 0.0,
                         n_episodes=len(episodes)))
    res=pd.DataFrame(rows)
    res.to_csv(f"{OUT}/lead_vs_fa_curve.csv",index=False)

    # ponto de operação: melhor recall com fa/dia <= 1
    feas=res[res["fa_per_day"]<=1.0]
    op_pt=feas.iloc[feas["recall"].argmax()] if len(feas) else res.iloc[res["recall"].argmax()]
    log(f"latente_ratio={latent/(TS*seqs.shape[-1]):.3f} | incidentes={len(inc_s)} | EWMA alpha={ALPHA}")
    log(f"PONTO OPERACIONAL (fa/dia<=1): recall={op_pt['recall']:.2f} "
        f"fa/dia={op_pt['fa_per_day']:.2f} lead_mediano={op_pt['median_lead_h']:.1f}h thr={op_pt['thr']:.3f}")
    for r in [0.5,0.7,0.8,0.9]:
        sub=res[res["recall"]>=r]
        if len(sub):
            b=sub.iloc[sub["fa_per_day"].argmin()]
            log(f"  recall>={r:.0%}: fa/dia minimo={b['fa_per_day']:.2f} (lead {b['median_lead_h']:.1f}h)")

    # ---------- plots ----------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(14,4.5))
    ax[0].plot(res["fa_per_day"],res["recall"],marker=".")
    ax[0].axvline(1.0,color="gray",ls=":",label="1 falso-alarme/dia")
    ax[0].set_xlabel("falso-alarmes / dia"); ax[0].set_ylabel(f"recall incidentes (horizonte {HORIZON_H:.0f}h)")
    ax[0].set_title("Curva preditiva: recall vs falso-alarme"); ax[0].set_xlim(0,10); ax[0].legend()
    ax[1].plot(res["fa_per_day"],res["median_lead_h"],marker=".",color="purple")
    ax[1].set_xlabel("falso-alarmes / dia"); ax[1].set_ylabel("lead time mediano (h)")
    ax[1].set_title("Antecipação vs falso-alarme"); ax[1].set_xlim(0,10)
    fig.tight_layout(); fig.savefig(f"{OUT}/curva_preditiva.png",dpi=130); plt.close(fig)

    # health-index EWMA no tempo + incidentes + ponto operacional
    fig2,a2=plt.subplots(figsize=(15,4))
    a2.plot(t_end,ew,lw=.6,color="steelblue",label="health-index EWMA")
    a2.axhline(op_pt["thr"],color="r",ls="--",label=f"alerta (thr={op_pt['thr']:.3f})")
    for ti in pd.DatetimeIndex(inc): a2.axvline(ti,color="green",alpha=.3,lw=.8)
    a2.set_title(f"Health-index EWMA | verde=incidente | recall={op_pt['recall']:.0%} fa/dia={op_pt['fa_per_day']:.2f} lead={op_pt['median_lead_h']:.1f}h")
    a2.set_ylabel("EWMA(MSE)"); a2.legend(loc="upper right")
    fig2.tight_layout(); fig2.savefig(f"{OUT}/health_index_ewma.png",dpi=120); plt.close(fig2)

    json.dump(dict(operating_point=op_pt.to_dict(),horizon_h=HORIZON_H,alpha=ALPHA,
                   n_incidents=int(len(inc_s))),open(f"{OUT}/pred_result.json","w"),indent=2)
    log(f"salvo {OUT}/*.png /*.csv /*.json")


if __name__=="__main__":
    main()
