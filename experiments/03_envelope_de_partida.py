#!/usr/bin/env python3
"""EXP 03 — ATLAS, PARTE 3: O ENVELOPE NORMAL DE PARTIDA (escala local) E OS TRIPS PERTO DA PARTIDA.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. Exploratório.

POR QUE EXISTE. O EXP 02 mostrou que 4 de 8 trips ocorrem a <= 32 h da partida (3 deles 9 a 15 h depois) e que a acomodação medida com o sigma GLOBAL não enxerga o transiente (ele é menor que a deriva de meses). Duas perguntas, agora com escala LOCAL e condicionadas ao tempo desde a partida (tau):
  P3  ACOMODAÇÃO (refeita). Em quantas horas cada tag, em corridas normais, chega ao seu estado permanente LOCAL (o da própria corrida) e fica dentro de +-3 sigma local por 12 h?
  P4  ENVELOPE DE PARTIDA. A trajetória z(tau) de cada tag nas horas depois de uma partida é reprodutível entre partidas normais? Os 4 trips a <= 32 h (07/04, 29/04, 04/11, 09/12/2025) desviam desse envelope MAIS do que as partidas normais desviam do envelope das outras?

DEFINIÇÕES (fixadas agora).
  · Corridas, ligada suave, partidas: as do EXP 02. A corrida de um trip é a que contém T. Corridas normais: as que NÃO terminam em trip (até 2 h depois do fim), com >= 12 h.
  · z de cada amostra ligada: contra a referência causal do EXP 01 (mediana e MAD das amostras vigiadas de [dia - 28 d, dia - 7 d)); z horário = mediana das amostras da hora de tau (>= 10 amostras), tau = 0..47 h.
  · P3: só corridas normais com >= 30 h. Estado permanente local da tag = mediana do x nas amostras com tau em [18, 30] h da corrida; escala local = 1,4826 x MAD dessas amostras (piso de 5% da mediana de |x|, mínimo 1e-6). Acomodação = a primeira hora de tau a partir da qual os 12 valores horários SEGUINTES de mediana ficam todos dentro de +-3 escalas locais do estado permanente (nunca = não acomoda).
  · P4: envelope por tag e por hora de tau = mediana e MAD (1,4826) do z horário entre as corridas normais com dado nessa hora (>= 15 corridas). Escore de uma corrida numa tag = max sobre as horas de tau em [2, tau_T - 2] de |z - mediana do envelope| / max(MAD do envelope, 0,5). Para os trips, tau_T = a hora do trip; para uma corrida normal, o mesmo intervalo [2, w] com w = floor(tau_T - 2) de cada trip, e o envelope SEM ela (leave-one-out).
    Percentil do trip numa tag = fração das corridas normais (que duram >= w + 2 h) com escore MENOR. Por trip: nº de tags com percentil >= 0,95 e, como comparação, a distribuição do mesmo nº entre as corridas normais (cada uma contra as outras): sob "nada de especial" o nº esperado de tags >= 0,95 é ~5% de 35.

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  · P3: a acomodação mediana de T5_AVG_A é <= 2 h; das sondas de vibração >= 3 h; dos mancais (TI_0301/0303/0305/0307) e do óleo (TI_0325) >= 6 h; pelo menos 10 das 36 tags têm mediana >= 3 h.
  · P4: pelo menos 2 dos 4 trips têm >= 3 tags com percentil >= 0,95 (contra 1 a 2 de uma corrida normal típica); o trip de 04/11 (partida a frio) desvia na vibração; dos três trips depois de parada breve, pelo menos 2 têm alguma tag a >= 0,95.
  · Probabilidade de que ALGUM dos 4 trips fique fora do que as corridas normais fazem de forma clara (>= 5 tags a >= 0,95): ~40%.

LIMITES. 46 corridas normais para o envelope (e menos nas horas tardias); 4 trips; a referência causal de z mistura estado de partida e permanente nos 28 d (os z de partida são deslocados e o envelope absorve isso); o escore é o máximo sobre horas (sensível a pico único); o percentil dos trips contra corridas normais não tem correção para as 35 tags.

RESULTADO (15/10/2026) -- EXPLORATÓRIO. 96 partidas; corridas normais com >= 12 h: 48 (39 com >= 30 h); corridas dos 4 trips a <= 32 h.
  P3 ACOMODAÇÃO COM ESCALA LOCAL: 35 de 36 tags têm mediana de acomodação de 0 h (T5_AVG_A, as 10 vibrações, os mancais TI_03 e o óleo TI_0325 inclusive); só o PI_0319 foge, e o PDIT_0305 não acomoda em 8% das corridas. Os valores da mediana e do p90 ficam em 0 h e 0 a 20 h.
        LEITURA: depois que RUNNING_A sobe, a mediana horária de cada tag já está dentro de +-3 sigma LOCAL do estado permanente da própria corrida: não há transiente térmico visível a 1 h de resolução nas corridas normais. A partida "vive" ANTES do RUNNING_A (a máquina já está quente quando o sinal sobe), e as primeiras horas ligadas são tão estáveis quanto as seguintes.
        Consequência: o blackout fixo de 6 h do v2 NÃO é justificado por acomodação térmica das tags; ele vem do desenho da máscara/EWMA (o que a pesquisa chamou de "a borda do blackout").
  P4 ENVELOPE DE PARTIDA (escore = max |z - mediana do envelope| / MAD do envelope sobre tau em [2, tau_T - 2]; percentil entre as corridas normais, cada uma contra as outras):
        07/04/2025 (tau 13,0 h; janela 2 a 11 h; 48 corridas): 14 tags com percentil >= 0,95 (corridas normais: mediana 0, p90 5, máx 16): TV_355X/Y (1,00; escores 6,6 e 8,6), TC382_03/06/01 e T5_AVG_A (0,98; escores 3,4 a 3,9): ANÔMALA, fora do envelope, multifamília (vibração + termopares).
        29/04/2025 (tau 31,1 h): 1 tag (PDI_0302 0,97; normais: mediana 0, p90 2): dentro do normal. 04/11/2025 (tau 15,3 h): 2 tags (TV_352X 1,00 com escore 4,3 e TV_352Y 0,96; normais: mediana 1, p90 6): dentro do normal; o escore do PI_0319 (1.855) é artefato de sensor de estado. 09/12/2025 (tau 9,3 h): 1 tag (TV_353Y 0,96; normais: p90 5): dentro do normal.
        SÓ 1 DOS 4 TRIPS A <= 32 h (07/04) DESVIA DO ENVELOPE NORMAL DE PARTIDA nas horas que antecedem o trip; os outros três (dois deles depois de parada breve e a partida a frio de 04/11) são indistinguíveis de uma partida normal até 2 h antes. O 07/04 já estava anômalo nas primeiras 11 h ligadas.
PLACAR: ACERTEI: o 04/11 desvia na vibração (TV_352X a 1,00, mas dentro da faixa do normal em contagem) e ALGUM trip fica claramente fora do normal (07/04; previ ~40%). ERREI: previ acomodação >= 3 h para vibração e >= 6 h para mancais e óleo (0 h); >= 10 tags com mediana >= 3 h (1); >= 2 dos 4 trips com >= 3 tags a >= 0,95 (1 de 4, o 07/04); previ que 2 dos 3 trips depois de parada breve teriam alguma tag a >= 0,95 (todos têm 1 ou mais, mas isso NÃO os distingue: as corridas normais têm p90 de 2 a 5).
HIPÓTESES geradas: H5 o precursor do 07/04 já aparece nas primeiras horas ligadas (falha latente presente na partida); H6 a concentração de risco entre 6 e 24 h NÃO vem de transiente das tags nem de assinatura visível no envelope (3 dos 4 trips perto da partida são indistinguíveis): é risco de partida sem precursor observável nas tags (mecânico/operacional), o que limita QUALQUER detector baseado só nestas tags nesse regime.

Uso:  PYTHONPATH=src .venv/bin/python experiments/03_envelope_de_partida.py
"""
from __future__ import annotations
import importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E1 = _carrega("exp01", "01_anatomia_trips.py"); E2 = _carrega("exp02", "02_ciclo_de_partida.py")
from nv import io as nvio
from nv.causal import carrega_registro

H = 1 / 30


def main():
    g = nvio.carrega_grade()
    fim = g.index[g["RUNNING_A"].notna() & g[[c for c in g.columns if c.startswith("TV_")]].notna().any(axis=1)][-1]
    g = g.loc[pd.Timestamp("2024-02-01", tz="UTC"):fim]; idx = g.index; lig = E2.ligada_suave(g); n = len(g)
    N = E1.Normal(g); tags = [c for c in g.columns if c not in ("RUNNING_A", "HSX_6240001A")]
    N.tags = tags; N.X = g[tags].to_numpy("float64")
    d = np.diff(np.concatenate(([0], lig.astype(int), [0]))); ini = np.flatnonzero(d == 1); fimr = np.flatnonzero(d == -1)
    corridas = [(a, b) for a, b in zip(ini, fimr) if a > 0]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    print(f"{len(corridas)} corridas (partida até a parada)", flush=True)

    def horario(a, b, maxh=48):
        """z horário (tau 0..maxh-1) de todas as tags, contra a referência causal; None se não há referência."""
        b = min(b, a + int(maxh / H)); ix = idx[a:b]; dias = ix.normalize(); z = np.full((b - a, len(tags)), np.nan)
        for dia in pd.unique(dias):
            r = N.ref(dia)
            if r is None:
                return None
            sel = dias == dia
            z[sel] = (N.X[a:b][sel] - r[0]) / r[1]                           # N.X já inclui T5_AVG_A: a referência causal cobre todas as tags de análise
        h = (np.arange(b - a) * H).astype(int)
        df = pd.DataFrame(z, columns=tags); df["h"] = h
        g_ = df.groupby("h"); zh = g_[tags].median(); zh[g_[tags].count() < 10] = np.nan
        return zh.reindex(range(int(min(maxh, (b - a) * H) + 1)))

    # corrida de cada trip
    trip_run = {}
    for T in reg:
        i = idx.searchsorted(T) - 1
        for q, (a, b) in enumerate(corridas):
            if a <= i < b:
                trip_run[T] = q
    normais = [q for q, (a, b) in enumerate(corridas) if q not in trip_run.values() and (b - a) * H >= 12]
    print(f"corridas normais com >= 12 h: {len(normais)} | corridas dos trips: {len(trip_run)}", flush=True)

    Z = {}
    for q in list(normais) + list(trip_run.values()):
        a, b = corridas[q]; zh = horario(a, b)
        if zh is not None:
            Z[q] = zh
    normais = [q for q in normais if q in Z]

    print("\nP3. ACOMODAÇÃO COM ESCALA LOCAL (corridas normais >= 30 h)")
    res = {t: [] for t in tags}
    longas = [q for q in normais if (corridas[q][1] - corridas[q][0]) * H >= 30]
    for q in longas:
        a, b = corridas[q]; tau = (np.arange(b - a) * H)
        X = N.X[a:b]; perm = (tau >= 18) & (tau < 30)
        med = np.nanmedian(X[perm], axis=0); sig = np.maximum(np.maximum(1.4826 * np.nanmedian(np.abs(X[perm] - med), axis=0), 0.05 * np.nanmedian(np.abs(X[perm]), axis=0)), 1e-6)
        hh = pd.DataFrame(np.abs(X - med) / sig, columns=tags).groupby((tau).astype(int)).median().to_numpy()
        for j, t in enumerate(tags):
            z = hh[:, j]; ts = np.nan
            for k0 in range(0, len(z) - 12):
                if np.isfinite(z[k0:k0 + 12]).all() and (z[k0:k0 + 12] <= 3.0).all():
                    ts = float(k0); break
            res[t].append(ts)
    S = pd.DataFrame({t.replace('954005_624_', ''): dict(corridas=len(v), nunca=f"{100 * np.mean(np.isnan(v)):.0f}%", mediana_h=round(float(np.nanmedian(v)), 1) if np.isfinite(v).any() else np.nan,
                                                         p90_h=round(float(np.nanquantile(v, 0.9)), 1) if np.isfinite(v).any() else np.nan) for t, v in res.items()}).T
    print(S.sort_values("mediana_h", ascending=False).to_string())
    med_h = {t: (np.nanmedian(v) if np.isfinite(v).any() else np.nan) for t, v in res.items()}
    print(f"\n   tags com acomodação mediana >= 3 h: {sum(1 for v in med_h.values() if v >= 3)} de {len(tags)} | >= 6 h: {sum(1 for v in med_h.values() if v >= 6)} | T5_AVG_A: {med_h['T5_AVG_A']:.1f} h | vibração (mediana das 10): {np.nanmedian([med_h[t] for t in tags if t.startswith('TV_')]):.1f} h"
          f" | mancais TI_03: {np.nanmedian([med_h[t] for t in tags if any(t.endswith(x) for x in ('TI_0301', 'TI_0303', 'TI_0305', 'TI_0307'))]):.1f} h | óleo TI_0325: {med_h['954005_624_TI_0325']:.1f} h", flush=True)

    print("\nP4. ENVELOPE DE PARTIDA E OS TRIPS A <= 32 h")
    def envelope(excl=None):
        qs = [q for q in normais if q != excl]; med = np.full((48, len(tags)), np.nan); mad = np.full((48, len(tags)), np.nan)
        for h in range(48):
            M = np.array([Z[q].iloc[h].to_numpy() if h < len(Z[q]) else np.full(len(tags), np.nan) for q in qs])
            ok = np.isfinite(M).sum(axis=0) >= 15
            m = np.nanmedian(M, axis=0); md = 1.4826 * np.nanmedian(np.abs(M - m), axis=0)
            med[h] = np.where(ok, m, np.nan); mad[h] = np.where(ok, md, np.nan)
        return med, np.maximum(mad, 0.5)

    def escore(zh, med, mad, w):
        h = np.arange(2, w + 1); h = h[h < len(zh)]
        dev = np.abs(zh.iloc[h].to_numpy() - med[h]) / mad[h]
        return np.nanmax(np.where(np.isfinite(dev), dev, np.nan), axis=0) if np.isfinite(dev).any() else np.full(len(tags), np.nan)

    envT, madT = envelope()
    for T in reg:
        q = trip_run.get(T)
        i = idx.searchsorted(T) - 1; tauT = (i - corridas[q][0]) * H if q is not None else np.nan
        if q is None or q not in Z or tauT > 32:
            continue
        w = int(np.floor(tauT - 2)); st = escore(Z[q], envT, madT, w)
        base = [qq for qq in normais if len(Z[qq]) >= w + 3]
        sn = []
        for qq in base:
            e, m_ = envelope(excl=qq); sn.append(escore(Z[qq], e, m_, w))
        sn = np.array(sn)
        pct = np.array([np.mean(sn[:, j][np.isfinite(sn[:, j])] < st[j]) if np.isfinite(st[j]) else np.nan for j in range(len(tags))])
        n_trip = int(np.nansum(pct >= 0.95))
        n_norm = np.array([int(sum(np.mean(sn[:, j][np.isfinite(sn[:, j])] < sn[r, j]) >= 0.95 for j in range(len(tags)) if np.isfinite(sn[r, j]))) for r in range(len(sn))])
        topo = np.argsort(-np.nan_to_num(pct, nan=-1))[:6]
        print(f"== {T:%Y-%m-%d} (tau_T {tauT:.1f} h; janela de tau [2, {w}] h; {len(base)} corridas normais): tags com percentil >= 0,95: {n_trip} | corridas normais (cada uma contra as outras): mediana {np.median(n_norm):.0f}, p90 {np.quantile(n_norm, 0.9):.0f}, máx {n_norm.max()}")
        for j in topo:
            print(f"     {tags[j].replace('954005_624_', ''):12s} percentil {pct[j]:.2f} | escore {st[j]:6.1f}")


if __name__ == "__main__":
    main()
