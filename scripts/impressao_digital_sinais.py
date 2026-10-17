#!/usr/bin/env python3
"""Impressão digital de um parquet de sinais do v2 (t, p, sp, vb, mask, reset): contagens exatas e estatísticas por coluna. Serve de referência pequena (vai para o git) para o worker conferir que reproduz os sinais locais sem ter o arquivo."""
import json, sys
import numpy as np, pandas as pd


def impressao(S):
    d = {"linhas": int(len(S)), "inicio": str(S.index.min()), "fim": str(S.index.max()), "colunas": {}}
    for c in ("t", "p", "sp", "vb"):
        x = S[c].astype(float); d["colunas"][c] = dict(nan=int(x.isna().sum()), soma=float(np.nansum(x)), media=float(np.nanmean(x)), dp=float(np.nanstd(x)), q50=float(np.nanquantile(x, .5)), q99=float(np.nanquantile(x, .99)), maximo=float(np.nanmax(x)))
    for c in ("mask", "reset"):
        d["colunas"][c] = dict(verdadeiros=int(S[c].astype(bool).sum()))
    return d


if __name__ == "__main__":
    S = pd.read_parquet(sys.argv[1]); json.dump(impressao(S), open(sys.argv[2], "w"), indent=1, ensure_ascii=False); print(json.dumps(impressao(S))[:400])
