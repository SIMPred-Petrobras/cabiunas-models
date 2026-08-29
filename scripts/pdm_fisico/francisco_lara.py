#!/usr/bin/env python3
"""O que a documentacao do Francisco e da Lara (26/08/2026) pode agregar aqui.

O detector deles: PCA por familia, baseline rolante de 3.000 h de operacao,
limiar = percentil 99,9 do proprio baseline suavizado, sustentacao 30 min,
voto 2 de 4 sinais (temperatura, pressao_oleo, mancal_spread, selagem_z).
Placar: 6/8, lead 18,3 h, 0,94 FP/mes de operacao.

O nosso: mesmas familias mas escore MAXIMO por sensor (normalizado pelo p99 do
proprio sensor, piso phi=0,10), janela de 667 h, limiar absoluto k*base, canal
duplo degrau|CUSUM, voto >=2, refratario 48 h, duracao minima 120 min.
Placar: 8/8, lead medio 29,0 h, 1,12 FP/mes.

Este script testa as quatro ideias deles que AINDA NAO tinham medicao aqui.
Nao muda o detector: so mede. Cada teste imprime o que ganha e o que custa.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from publica_clearml import (reproduz, GRID, BLACKOUT, SUSTAIN, SIN, HL, BASE, K,
                             KAPPA, H_CUSUM, CARGA, REFRAT_H, DUR_MIN, T0)

PRE = "954005_624_"
TEMP = [PRE+x for x in ["TI_0301","TI_0303","TI_0305","TI_0307","TI_0315","TI_0317","TI_0325"]] \
     + ["TC382_0%d_A" % i for i in range(1,7)] + ["T5_AVG_A"]
PRESS = [PRE+x for x in ["PDIT_0305","PDI_0301","PDI_0302","PDI_0317","PDI_0338",
                         "PI_0307","PI_0308","PI_0315","PI_0319","PI_0339","PI_0340"]] + ["PI_5134001"]
VIB = ["TV_35%d%s_A" % (i, e) for i in range(1,6) for e in ("X","Y")]
FAM = {"t": TEMP, "p": PRESS, "vb": VIB}


def linha(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


def custo(al, mask, alvo):
    m = AV.avalia(al, alvo, mask)
    return m


def main():
    fin, mask, alvo, ON, idx, sel = reproduz()
    g = pd.read_parquet("grade2min.parquet")
    base = custo(fin, mask, alvo)
    print(f"referencia: {base['det']}/{base['n_ev']}  eps={base['episodios']}  "
          f"fp={base['fp']}  {base['fp_mes']:.2f}/mes  {base['h_fp_mes']:.1f} h/mes  "
          f"lead {base['lead_med']:.1f} h")

    # ---------------------------------------------------------------- teste 1
    linha("TESTE 1 - a vibracao separa? (replica do metodo deles nos NOSSOS sinais)")
    print("Eles poem as 10 tags TV_* em quarentena: nas 48 h antes das falhas ficam no")
    print("percentil 79-85 de 315 janelas de controle, ou seja nao separam. Mas eles usam")
    print("o NIVEL cru, que deriva; o nosso vb e z robusto contra referencia rolante de")
    print("400 h, que e justamente a correcao da deriva. Entao a medida precisa ser refeita.\n")
    sig = pd.DataFrame({c: v for c, v in
                        {c: None for c in []}.items()})  # placeholder
    # reconstroi os quatro escores suavizados (mesma cadeia do detector)
    z = np.load("piso_fisico_cache.npz")
    sp_ = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp_, "vb": vbz}, index=idx)
    E = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in HL.items()}

    jan = [(t - pd.Timedelta(hours=48), t) for t in alvo]
    horas_op = mask.rolling(1).sum()
    # janelas de controle: ancoras a cada 6 h em operacao, sem tocar nenhuma falha
    anc = idx[(np.arange(len(idx)) % 180 == 0)]
    ctrl = [(a - pd.Timedelta(hours=48), a) for a in anc
            if mask.loc[a - pd.Timedelta(hours=48):a].sum() * 2 / 60 >= 24
            and not any((a - pd.Timedelta(hours=48) <= t1) and (a >= t0) for t0, t1 in jan)]
    print(f"{len(ctrl)} janelas de controle de 48 h (>=24 h de operacao dentro)\n")
    print(f"{'sinal':6s} {'percentil da janela pre-falha entre os controles':50s}  mediana")
    for c in SIN:
        est_c = np.array([np.nanmax(E[c].loc[a:b].to_numpy()) for a, b in ctrl])
        est_c = est_c[np.isfinite(est_c)]
        pcs = []
        for a, b in jan:
            v = np.nanmax(E[c].loc[a:b].to_numpy())
            pcs.append(100 * (est_c < v).mean() if np.isfinite(v) else np.nan)
        s = " ".join(f"{p:5.1f}" if np.isfinite(p) else "  --  " for p in pcs)
        print(f"{c:6s} {s:50s}  {np.nanmedian(pcs):5.1f}")
    print("\n(uma coluna por falha, na ordem de falhas.csv; 100 = maior que todo controle)")

    # ---------------------------------------------------------------- teste 2
    linha("TESTE 2 - veto de instrumento travado (PASSO 5 deles)")
    print("Regra deles: se algum sensor de uma familia fica com desvio-padrao zero por")
    print("30 min, o escore daquela familia e anulado ali. Sensor constante por natureza")
    print("(>5% do tempo de operacao congelado) e dispensado do teste. Nao temos nada")
    print("disso; e um redutor de falso positivo puro, que nao toca deteccao por desenho.\n")
    n30 = SUSTAIN
    veto = {}
    for fam, tags in FAM.items():
        cong = {}
        for tg in tags:
            if tg not in g.columns:
                continue
            d = g[tg].diff().abs()
            f = (d.fillna(0) == 0).rolling(n30, min_periods=n30).min().fillna(0).astype(bool)
            cong[tg] = f
        frac = {tg: 100 * float((f & mask).sum() / max(mask.sum(), 1)) for tg, f in cong.items()}
        disp = [tg for tg, v in frac.items() if v > 5.0]
        for tg, v in sorted(frac.items(), key=lambda x: -x[1])[:4]:
            print(f"  {fam:3s} {tg:24s} congelado {v:5.1f}% do tempo de operacao"
                  + ("   -> DISPENSADO" if v > 5.0 else ""))
        usa = [tg for tg in cong if tg not in disp]
        veto[fam] = pd.Series(False, index=idx) if not usa else \
            pd.concat([cong[tg] for tg in usa], axis=1).any(axis=1)
        print(f"  {fam:3s} veto apaga {100*float((veto[fam]&mask).sum()/max(mask.sum(),1)):.2f}%"
              f" do tempo util da familia\n")
    veto["sp"] = veto["t"]                      # spread e temperatura
    ONv = {c: ON[c] & ~veto.get(c, pd.Series(False, index=idx)) for c in SIN}
    voto_v = pd.Series(sum(ONv[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    fin_v = pos(voto_v, idx) & sel
    m = custo(fin_v, mask, alvo)
    print(f"  com veto: {m['det']}/{m['n_ev']}  eps={m['episodios']}  {m['fp_mes']:.2f} FP/mes"
          f"  {m['h_fp_mes']:.1f} h/mes  lead {m['lead_med']:.1f} h")
    print(f"  sem veto: {base['det']}/{base['n_ev']}  eps={base['episodios']}  "
          f"{base['fp_mes']:.2f} FP/mes  {base['h_fp_mes']:.1f} h/mes  lead {base['lead_med']:.1f} h")

    # ---------------------------------------------------------------- teste 3
    linha("TESTE 3 - a terceira caixa da regua deles: alerta antes de parada real")
    print("Na regua deles um alerta seguido de parada real de >=2 h nas 48 h seguintes nao")
    print("conta como falso positivo -- fica fora das duas contas (2 episodios la). A nossa")
    print("regua conta tudo que nao pega falha catalogada como FP. Quanto do nosso custo e")
    print("isso?\n")
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    paradas = [(a, b) for a, b in AV.episodios(~op & (op.index >= T0))
               if (b - a).total_seconds() / 3600 >= 2.0]
    print(f"  {len(paradas)} paradas reais de >=2 h na serie a partir de {T0:%Y-%m-%d}")
    eps = AV.episodios(fin)
    fp_eps = [(a, b) for a, b in eps
              if not any((a <= t1) and (b >= t0) for t0, t1 in jan)]
    antes, resta, h_antes = [], [], 0.0
    for a, b in fp_eps:
        seg = [s for s, _ in paradas if a <= s <= a + pd.Timedelta(hours=48)]
        if seg:
            antes.append((a, b, seg[0])); h_antes += (b - a).total_seconds()/3600 + 2/60
        else:
            resta.append((a, b))
    meses = base["horas_op"] / 730.0
    print(f"  dos {len(fp_eps)} FP, {len(antes)} sao seguidos de parada real em <=48 h\n")
    for a, b, s in antes:
        print(f"    {a:%Y-%m-%d %H:%M}  dur {(b-a).total_seconds()/3600:6.1f} h"
              f"  -> parada em {s:%Y-%m-%d %H:%M} (+{(s-a).total_seconds()/3600:.1f} h)")
    print(f"\n  FP na nossa regua : {len(fp_eps)/meses:.2f}/mes  {base['h_fp_mes']:.1f} h/mes")
    print(f"  FP na regua deles : {len(resta)/meses:.2f}/mes  "
          f"{(base['h_fp_mes']*meses - h_antes)/meses:.1f} h/mes")

    # ---------------------------------------------------------------- teste 4
    linha("TESTE 4 - confirmacao POR MECANISMO (a v8 deles) contra voto cego")
    print("Hoje o voto trata todos os pares como iguais. `t` e `p` sobem juntos numa")
    print("manobra de carga por construcao; `sp` e `vb` olham o mancal por vias distintas.")
    print("Exigir que pelo menos um dos dois sinais de mancal esteja no voto muda o que")
    print("conta como confirmacao, sem mexer em limiar.\n")
    regras = {
        "voto>=2 (atual)": lambda: pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx),
        "voto>=2 + >=1 mancal (sp|vb)": lambda: (pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & (ON["sp"] | ON["vb"])),
        "voto>=2 sem o par t+p": lambda: (pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & ~(ON["t"] & ON["p"] & ~ON["sp"] & ~ON["vb"])),
        "voto>=2 so em {t,p,sp} (vb fora)": lambda: pd.Series(sum(ON[c].astype(int) for c in ["t","p","sp"]) >= 2, index=idx),
        "voto>=2 so em {t,p,vb} (sp fora)": lambda: pd.Series(sum(ON[c].astype(int) for c in ["t","p","vb"]) >= 2, index=idx),
        "voto>=3": lambda: pd.Series(sum(ON[c].astype(int) for c in SIN) >= 3, index=idx),
    }
    print(f"{'regra':34s} {'det':>5s} {'eps':>5s} {'FP/mes':>8s} {'h/mes':>8s} {'lead':>7s}  perdidos")
    for nome, f in regras.items():
        v = f() & mask
        a = pos(v, idx) & sel
        m = custo(a, mask, alvo)
        perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
        print(f"{nome:34s} {m['det']:>3d}/8 {m['episodios']:>5d} {m['fp_mes']:>8.2f} "
              f"{m['h_fp_mes']:>8.1f} {m['lead_med']:>7.1f}  {','.join(perd)}")


def pos(voto, idx):
    """Refratario de 48 h + duracao minima de 120 min -- identico ao detector."""
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin


if __name__ == "__main__":
    main()


# ==================================================================== teste 5
def teste5():
    """Limiar ADAPTATIVO (percentil do proprio baseline do mes) x absoluto k*base.

    E a ideia mais estrutural do documento deles, e a que ataca direto a nossa
    fragilidade conhecida: os nossos limiares sao constantes calibradas UMA vez
    no alvo inteiro (k=1,7/2,2), e o LOEO ja mostrou que o ponto de operacao
    depende do desempate. Um limiar que e percentil do baseline daquele mes nao
    tem constante para calibrar: ele significa a mesma coisa em qualquer escala
    e em qualquer maquina. Se der o mesmo resultado, e melhor por portabilidade.
    """
    import numpy as np, pandas as pd, avalia as AV
    from publica_clearml import reproduz, SIN, HL, BASE, K, KAPPA, H_CUSUM, CARGA, T0
    fin, mask, alvo, ON, idx, sel = reproduz()
    g = pd.read_parquet("grade2min.parquet")
    z = np.load("piso_fisico_cache.npz")
    sp_ = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan); vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp_, "vb": vbz}, index=idx)
    E = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in HL.items()}
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    estavel = (op & (g["T5_AVG_A"] > 300)).to_numpy()
    part = (op & ~op.shift(fill_value=False)).to_numpy()
    reset = (~mask.to_numpy()) | part

    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    N_FIT = 20_000

    def limiares(q):
        """Limiar em escada: percentil q do escore ja suavizado do baseline do mes."""
        TH = {c: pd.Series(np.nan, index=idx) for c in SIN}
        for i, m0 in enumerate(meses):
            m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta("2min")
            pre = (idx < m0) & estavel
            if pre.sum() < N_FIT // 4:
                continue
            jj = np.flatnonzero(pre)[-N_FIT:]
            selm = (idx >= m0) & (idx < m1)
            for c in SIN:
                v = E[c].to_numpy()[jj]
                v = v[np.isfinite(v)]
                if len(v) > 100:
                    TH[c][selm] = float(np.percentile(v, q))
        return TH

    linha("TESTE 5 - limiar adaptativo por percentil do baseline (o PASSO 8 deles)")
    print("Hoje: limiar = k * base, constante calibrada uma vez no alvo inteiro.")
    print("Deles: limiar = percentil do escore suavizado do baseline daquele mes,")
    print("recalculado a cada retreino. Nao tem constante para calibrar.\n")
    print(f"{'limiar':28s} {'det':>5s} {'eps':>5s} {'FP/mes':>8s} {'h/mes':>8s} {'lead':>7s}  perdidos")
    ref = AV.avalia(fin, alvo, mask)
    print(f"{'k*base (atual)':28s} {ref['det']:>3d}/8 {ref['episodios']:>5d} "
          f"{ref['fp_mes']:>8.2f} {ref['h_fp_mes']:>8.1f} {ref['lead_med']:>7.1f}")
    for q in [99.0, 99.5, 99.9, 99.95, 99.99]:
        TH = limiares(q)
        ONq = {}
        for c in SIN:
            thr = TH[c]
            deg = ((E[c] > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
            x = ((E[c] / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
            S = np.empty(len(x)); acc = 0.0
            for i in range(len(x)):
                acc = acc * CARGA if reset[i] else max(0.0, acc + x[i])
                S[i] = acc
            ONq[c] = (deg | pd.Series(S > H_CUSUM, index=idx)) & mask
        v = pd.Series(sum(ONq[c].astype(int) for c in SIN) >= 2, index=idx) & mask
        a = pos(v, idx) & sel
        m = AV.avalia(a, alvo, mask)
        perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
        print(f"{'percentil ' + format(q, '.2f'):28s} {m['det']:>3d}/8 {m['episodios']:>5d} "
              f"{m['fp_mes']:>8.2f} {m['h_fp_mes']:>8.1f} {m['lead_med']:>7.1f}  {','.join(perd)}")
