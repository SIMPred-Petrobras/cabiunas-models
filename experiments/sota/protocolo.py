"""PROTOCOLO ÚNICO DE AVALIAÇÃO DO BRIEF "estado da arte" (fixado em 22/10/2026, antes de qualquer modelo novo rodar). Vale para TODO modelo comparado.

ALVO (decisão do usuário, 22/10): os 8 trips do registro a partir de 27/02/2025. A régua de alarme por sensor do pipeline de AEs é só secundária.
PERÍODO: 01/02/2025 a 21/04/2026 (fim do ledger do controle). VIGILÂNCIA: a máscara do v2 (RUNNING_A > 0,5, T5 > 300 e blackout de 6 h após a partida) — alarme fora
dela não conta, para nenhum modelo (a NGP_A não existe depois de out/2025, então ela não pode ser a máscara comum).
CONVENÇÕES (as de `nv.avalia`, iguais às da pesquisa): janela [T − 48 h, T]; DETECÇÃO "de pé" = um episódio intersecta a janela; "INÍCIO" = um episódio nasce dentro dela
(a régua do Diego); BANDA = nasce com >= 4 h de antecedência; FP = episódio que não intersecta nenhuma janela; FP/mês = FP por 730 h vigiadas; FP/dia = FP por 24 h vigiadas;
horas de alarme totais e fora das janelas SEMPRE junto com a contagem.
LEAD É CENSURADO EM 48 H: o lead de um evento é T − max(nascimento, T − 48 h); um episódio que começou antes da janela conta 48 h. Média e mediana vêm com o número de
eventos censurados ao lado, e nenhuma conclusão se apoia na média do lead sozinha.
F1 POR EVENTO: recall = detectados / 8; precisão = episódios que intersectam alguma janela / episódios; F1 = média harmônica.
INCERTEZA: IC 95% por bootstrap. Eventos (detecção, início, banda, lead) reamostrados POR EVENTO; custo (FP/mês, horas por 730 h) reamostrado em BLOCOS MENSAIS
(`nv.avalia.ic_blocos_mensais`). 2.000 réplicas, semente fixa.
REGRA DE COMPARAÇÃO: entre dois modelos, bootstrap PAREADO (mesmos eventos, mesmos meses). Só se declara diferença de detecção com |Δ| >= 2 eventos E IC pareado sem o
zero; diferença de custo só com IC pareado sem o zero. Diferença de 1 detecção NUNCA declara vencedor. Escolha informada pelos eventos exige LOEO e placebo.
LIMIARES: nenhum modelo novo calibra limiar com eventos ou com o período de teste; só percentis do normal de treino.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from nv.avalia import avalia, ic_blocos_mensais
from nv.avalia.metricas import JANELA, TMIN, episodios_de_serie

INICIO = pd.Timestamp("2025-02-01", tz="UTC"); FIM = pd.Timestamp("2026-04-21 00:56", tz="UTC"); ALVO0 = pd.Timestamp("2025-02-27", tz="UTC")
N_BOOT = 2000; SEMENTE = 20261022


def trips_alvo(registro):
    return [t for t in registro if ALVO0 <= t <= FIM]


def por_evento(al, trips):
    """Detecção, início, banda e lead (censurado) de cada trip."""
    eps = episodios_de_serie(al.loc[INICIO:FIM]); out = []
    for t in trips:
        inter = [(a, b) for a, b in eps if a <= t and b >= t - JANELA]; nasc = [(a, b) for a, b in inter if a >= t - JANELA]
        lead = (t - max(inter[0][0], t - JANELA)).total_seconds() / 3600 if inter else np.nan
        out.append(dict(trip=t, det=bool(inter), inicio=bool(nasc), banda=any(a <= t - TMIN for a, _ in nasc), lead_h=lead, censurado=bool(inter) and inter[0][0] <= t - JANELA))
    return pd.DataFrame(out)


def mede(al, vig, trips):
    """Todas as métricas do protocolo de uma série de alarme (booleana, grade de 2 min)."""
    al = al.reindex(vig.index).fillna(False) & vig.astype(bool)
    r = avalia(al, vig, trips, inicio_periodo=INICIO, fim_periodo=FIM); ev = por_evento(al, trips)
    eps = episodios_de_serie(al.loc[INICIO:FIM]); tp = sum(any(a <= t and b >= t - JANELA for t in trips) for a, b in eps)
    prec = tp / len(eps) if eps else np.nan; rec = r.det / len(trips)
    lead = ev.loc[ev.det, "lead_h"]
    return dict(det=r.det, n=len(trips), inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, fp_dia=r.fp / (r.horas_vigiadas / 24.0),
                horas=r.horas_alarme, fora=r.horas_alarme_fora, horas_mes=r.horas_alarme / (r.horas_vigiadas / 730.0), episodios=len(eps),
                lead_medio=float(lead.mean()) if len(lead) else np.nan, lead_mediano=float(lead.median()) if len(lead) else np.nan,
                n_censurado=int(ev.censurado.sum()), precisao=prec, recall=rec, f1=(2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0)


def ic(al, vig, trips, n=N_BOOT, semente=SEMENTE):
    """IC 95%: eventos por bootstrap de eventos; custo por blocos mensais."""
    al = al.reindex(vig.index).fillna(False) & vig.astype(bool)
    ev = por_evento(al, trips); rng = np.random.default_rng(semente); k = len(ev); B = {c: [] for c in ("det", "inicio", "banda", "lead_mediano")}
    for _ in range(n):
        s = ev.iloc[rng.integers(0, k, k)]
        B["det"].append(s.det.sum()); B["inicio"].append(s.inicio.sum()); B["banda"].append(s.banda.sum())
        B["lead_mediano"].append(float(s.loc[s.det, "lead_h"].median()) if s.det.any() else np.nan)
    q = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
    out = {c: q(v) for c, v in B.items()}
    cm = ic_blocos_mensais(al, vig, trips, n=n, semente=semente, inicio_periodo=INICIO, fim_periodo=FIM)
    out.update({k_: v for k_, v in cm.items() if k_ in ("fp_mes", "horas_alarme_730h")})
    return out


def compara(al_a, al_b, vig, trips, n=N_BOOT, semente=SEMENTE):
    """Bootstrap PAREADO A − B: Δdetecção por evento, ΔFP/mês e Δhoras por 730 h em blocos mensais. Aplica a regra de comparação do protocolo."""
    A = al_a.reindex(vig.index).fillna(False) & vig.astype(bool); Bs = al_b.reindex(vig.index).fillna(False) & vig.astype(bool)
    ea, eb = por_evento(A, trips), por_evento(Bs, trips); rng = np.random.default_rng(semente); k = len(ea)
    dd = [int(ea.det.iloc[i].sum() - eb.det.iloc[i].sum()) for i in (rng.integers(0, k, k) for _ in range(n))]
    sel = (vig.index >= INICIO) & (vig.index <= FIM); meses = sorted(set(vig.index[sel].tz_localize(None).to_period("M")))
    def por_mes(al):
        eps = episodios_de_serie(al.loc[INICIO:FIM]); d = {m: [0.0, 0, 0.0] for m in meses}
        mv = vig.loc[INICIO:FIM]; hv = mv.groupby(mv.index.tz_localize(None).to_period("M")).sum() * 2 / 60
        for m in meses:
            d[m][0] = float(hv.get(m, 0.0))
        for a, b in eps:
            m = a.tz_localize(None).to_period("M"); d[m][2] += (b - a).total_seconds() / 3600 + 2 / 60
            if not any(a <= t and b >= t - JANELA for t in trips):
                d[m][1] += 1
        return d
    pa, pb = por_mes(A), por_mes(Bs); dfp, dh = [], []
    for _ in range(n):
        s = [meses[i] for i in rng.integers(0, len(meses), len(meses))]; h = sum(pa[m][0] for m in s) / 730.0
        if h <= 0:
            continue
        dfp.append((sum(pa[m][1] for m in s) - sum(pb[m][1] for m in s)) / h); dh.append((sum(pa[m][2] for m in s) - sum(pb[m][2] for m in s)) / h)
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    d_det = int(ea.det.sum() - eb.det.sum()); ic_det, ic_fp, ic_h = q(dd), q(dfp), q(dh)
    fora0 = lambda c: c[0] > 0 or c[1] < 0
    return dict(d_det=d_det, ic_d_det=ic_det, d_fp_mes=(float(np.mean(dfp))), ic_d_fp_mes=ic_fp, d_horas_mes=float(np.mean(dh)), ic_d_horas_mes=ic_h,
                difere_det=abs(d_det) >= 2 and fora0(ic_det), difere_fp=fora0(ic_fp), difere_horas=fora0(ic_h))
