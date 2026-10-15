#!/usr/bin/env python3
"""EXP 01 — ATLAS, PARTE 1: ANATOMIA DOS 8 TRIPS (Fase 1 da nova frente).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. É ANÁLISE EXPLORATÓRIA: gera hipóteses, não conclui nada sobre detectores. Cada hipótese que sair daqui ganha pré-registro próprio antes de ser testada.

PERGUNTAS.
  Q1  O INSTANTE T. O registro diz T; o que a máquina estava fazendo? Quanto tempo havia desde a última partida, quando a temperatura T5 começou a cair e quando RUNNING_A zerou, e qual fração das 48 h e dos 7 dias anteriores era vigiada (ligada e T5 > 300)?
  Q2  O QUE É INCOMUM. Em cada trip, que tags (e que famílias) desviam do normal mais do que desviariam numa janela normal qualquer? "Incomum" é medido contra 450 janelas normais sorteadas com o MESMO esquema (percentil do evento entre as janelas normais), não contra um limiar fixo.
  Q3  O HORIZONTE. Com que antecedência o desvio sustentado de cada tag aparece antes de T (primeira vez em que |z| >= 3 por >= 3 horas seguidas), e esse horizonte é distinguível do que janelas normais produzem?
  Q4  CONSISTÊNCIA ENTRE EVENTOS. Há tags (ou famílias) incomuns em vários eventos? (O M31 da pesquisa mostrou que NÃO há direção linear compartilhada nos níveis; isto olha MAGNITUDE por tag, em módulo.)

O QUE SE FAZ.
  · Dado: a grade de 2 min (aceita no G0), 38 tags; vigiado = ligada e T5 > 300. Eventos: os 8 trips do registro a partir de 27/02/2025 (o de 16/01/2024 fica de fora).
  · Referência CAUSAL de cada tag: mediana e MAD robusto (1,4826) das amostras vigiadas em [dia - 28 d, dia - 7 d) do dia civil do instante (o último dia da referência termina 7 dias antes: a degradação em desenvolvimento não entra nela), com piso no MAD (5% da mediana de |x|, mínimo 1e-6) para sensores de estado.
  · z por amostra; agregação por hora (mediana das amostras vigiadas, >= 10 amostras na hora). Janela de análise do evento: [T - 7 d, T - 2 h]; das 2 h finais, de transiente de trip, ficam de fora.
  · Estatísticas por tag e janela: zmax = max |z horário|; persistência = maior corrida de horas seguidas com |z| >= 3; lead = T menos o início da primeira corrida de >= 3 horas seguidas com |z| >= 3.
  · JANELAS NORMAIS (nulo): 450 instantes E sorteados entre 2024-03-01 e 2026-04-19, vigiados em >= 40% de [E - 7 d, E - 2 h] e com no mínimo 24 h vigiadas, sem trip em [E - 7 d, E + 2 d], semente fixa; mesmas estatísticas com E no lugar de T.
  · Percentil do evento = fração das janelas normais com zmax MENOR que a do evento (idem para a persistência). Evento sem vigilância >= 20% na janela de 7 d tem as estatísticas marcadas como não aplicáveis (é o caso da partida a frio de 04/11/2025).
  · Consistência: nº de eventos em que cada tag tem percentil >= 0,95. Sob o nulo, P(>= 4 de 8) ~ 4,6e-4 por tag, abaixo do limiar de Bonferroni (0,05/38 = 1,3e-3): >= 4 eventos é "consistente"; 3 eventos é só sugestivo.

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  · Q1: em pelo menos 2 dos 8 eventos o instante T do registro difere do instante em que a máquina parou (RUNNING_A zerou) por mais de 1 h; ~60% de que em pelo menos 5 dos 8 a máquina estava estável 24 h antes de T.
  · Q2: pelo menos 6 dos 8 eventos têm alguma tag com percentil >= 0,95; a maioria dos extremos fica em sensores de estado de MAD pequeno (selagem, filtros, PI_0319), não em vibração nem em temperatura de mancal.
  · Q3: o lead mediano da primeira corrida sustentada da tag mais incomum fica entre 6 e 40 h; em pelo menos 2 eventos nenhuma tag tem corrida sustentada antes de T - 2 h além do que uma janela normal tem em ~25% dos casos.
  · Q4: nenhuma tag individual é incomum (>= 0,95) em >= 4 eventos (~80%); a vibração (alguma sonda TV) é a candidata mais provável a passar (~35%).

LIMITES. 8 eventos; janela de 7 d e referência de 28 d escolhidas aqui (não otimizadas); o percentil mede incomum em magnitude e não em sinal; 38 tags x 8 eventos = 304 percentis (sem correção formal para o relatório descritivo; só a consistência usa o limiar de Bonferroni).

Uso:  PYTHONPATH=src .venv/bin/python experiments/01_anatomia_trips.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro

SEMENTE, N_NULOS = 20261015, 450
JAN, FIM_H = pd.Timedelta(days=7), pd.Timedelta(hours=2)
SAIDA = Path(__file__).resolve().parents[1] / "data_local" / "atlas"
LIMITE_Z, MIN_H = 3.0, 3


class Normal:
    """Referência causal por dia civil e estatísticas por janela, com cache."""
    def __init__(self, g: pd.DataFrame):
        self.g = g; self.idx = g.index; self.tags = [c for c in g.columns if c not in nvio.TAGS_MASCARA]
        self.X = g[self.tags].to_numpy("float64"); m = nvio.mascaras(g); self.est = m["estavel"].to_numpy(bool); self.lig = m["ligada"].to_numpy(bool)
        self._ref = {}

    def ref(self, instante: pd.Timestamp):
        dia = instante.normalize()
        if dia not in self._ref:
            i0 = self.idx.searchsorted(dia - pd.Timedelta(days=28)); i1 = self.idx.searchsorted(dia - pd.Timedelta(days=7))
            w = self.X[i0:i1][self.est[i0:i1]]
            if len(w) < 720:
                self._ref[dia] = None
            else:
                med = np.nanmedian(w, axis=0); mad = 1.4826 * np.nanmedian(np.abs(w - med), axis=0)
                self._ref[dia] = (med, np.maximum(np.maximum(mad, 0.05 * np.nanmedian(np.abs(w), axis=0)), 1e-6))
        return self._ref[dia]

    def zhoras(self, E: pd.Timestamp):
        """z horário (mediana das amostras vigiadas, >= 10 por hora) em [E - 7 d, E - 2 h]; None se não houver referência."""
        i0 = self.idx.searchsorted(E - JAN); i1 = self.idx.searchsorted((E - FIM_H).floor("h"), side="left")       # só horas COMPLETAS (ver a emenda no fim do arquivo)
        ix = self.idx[i0:i1]; est = self.est[i0:i1]
        if len(ix) == 0:
            return None
        dias = ix.normalize(); z = np.full((len(ix), len(self.tags)), np.nan)
        for d in pd.unique(dias):
            r = self.ref(d)
            if r is None:
                continue
            sel = (dias == d) & est
            z[sel] = (self.X[i0:i1][sel] - r[0]) / r[1]
        df = pd.DataFrame(z, index=ix, columns=self.tags)
        h = df.groupby(df.index.floor("h")); zh = h.median(); zh[h.count() < 10] = np.nan
        return zh, float(est.mean())

    def estat(self, E):
        r = self.zhoras(E)
        if r is None:
            return None
        zh, frac = r
        if frac < 0.20 or zh.notna().any(axis=1).sum() < 24:
            return dict(frac=frac, aplicavel=False)
        a = zh.abs(); alto = (a >= LIMITE_Z).to_numpy(); out = {}
        for j, t in enumerate(self.tags):
            col = alto[:, j]; run = best = 0; first = None; cur_ini = None
            for k, v in enumerate(col):
                if v:
                    run += 1
                    if run == 1:
                        cur_ini = k
                    if run == MIN_H and first is None:
                        first = cur_ini
                else:
                    run = 0
                best = max(best, run)
            zt = zh[t]
            out[t] = dict(zmax=float(a[t].max()) if a[t].notna().any() else np.nan, persist=best,
                          lead_h=float((E - zh.index[first]).total_seconds() / 3600) if first is not None else np.nan,
                          zsinal=float(zt.loc[a[t].idxmax()]) if a[t].notna().any() else np.nan)
        return dict(frac=frac, aplicavel=True, tags=out, horas=zh.index)


def sorteia_nulos(N: Normal, trips, n=N_NULOS, semente=SEMENTE):
    rng = np.random.default_rng(semente)
    ini, fim = pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2026-04-19", tz="UTC")
    cand = N.idx[(N.idx >= ini) & (N.idx <= fim)]
    out, tent = [], 0
    while len(out) < n and tent < 60000:
        tent += 1
        E = cand[int(rng.integers(0, len(cand)))].floor("h")
        if any(f - pd.Timedelta(days=2) <= E <= f + JAN for f in trips):                 # sem trip em [E - 7 d, E + 2 d]
            continue
        s = N.estat(E)
        if s is None or not s.get("aplicavel") or s["frac"] < 0.40:
            continue
        out.append((E, s))
    return out


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    g = nvio.carrega_grade()
    N = Normal(g); reg = carrega_registro(); eventos = [t for t in reg if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    m = nvio.mascaras(g)

    print("Q1. O INSTANTE T E O ESTADO DA MÁQUINA", flush=True)
    linhas = []
    for T in eventos:
        i = g.index.searchsorted(T)
        est = m["estavel"].to_numpy(); lig = m["ligada"].to_numpy(); part = m["partida"].to_numpy()
        ant = np.flatnonzero(est[:i + 1]); ult_est = g.index[ant[-1]] if len(ant) else pd.NaT
        parts = np.flatnonzero(part[:i + 1]); ult_part = g.index[parts[-1]] if len(parts) else pd.NaT
        depois = np.flatnonzero(~lig[i - 60:i + 360]) if i > 60 else []                 # RUNNING_A desligada entre T-2 h e T+12 h
        off = g.index[i - 60 + depois[0]] if len(depois) else pd.NaT
        t5 = g["T5_AVG_A"].to_numpy(float)
        ref_t5 = np.nanmedian(t5[max(0, i - 1440):max(1, i - 180)])
        queda = np.flatnonzero(t5[i - 360:i + 60] < 0.95 * ref_t5) if ref_t5 == ref_t5 else []
        t_queda = g.index[i - 360 + queda[0]] if len(queda) else pd.NaT
        f48 = float(est[max(0, i - 1440):i].mean()); f7 = float(est[max(0, i - 5040):i].mean())
        linhas.append(dict(T=f"{T:%Y-%m-%d %H:%M}", desde_partida_h=round((T - ult_part).total_seconds() / 3600, 1) if ult_part is not pd.NaT else np.nan,
                           ult_estavel_antes_min=round((T - ult_est).total_seconds() / 60) if ult_est is not pd.NaT else np.nan,
                           queda_T5_antes_min=round((T - t_queda).total_seconds() / 60) if t_queda is not pd.NaT else np.nan,
                           RUNNING_off_apos_min=round((off - T).total_seconds() / 60) if off is not pd.NaT else np.nan, vig48=round(f48, 2), vig7d=round(f7, 2)))
    print(pd.DataFrame(linhas).to_string(index=False), flush=True)

    print("\nsorteando e medindo janelas normais ...", flush=True)
    nulos = sorteia_nulos(N, reg)
    print(f"   {len(nulos)} janelas normais", flush=True)
    tags = N.tags
    nz = {t: np.array([s["tags"][t]["zmax"] for _, s in nulos]) for t in tags}
    npers = {t: np.array([s["tags"][t]["persist"] for _, s in nulos]) for t in tags}
    nlead_has = {t: float(np.mean([not np.isnan(s["tags"][t]["lead_h"]) for _, s in nulos])) for t in tags}

    print("\nQ2/Q3. O QUE É INCOMUM EM CADA TRIP (percentil do zmax entre as janelas normais; lead = horas antes de T da 1ª corrida sustentada |z|>=3)")
    pct = {}; info = {}
    for T in eventos:
        s = N.estat(T)
        if not s.get("aplicavel"):
            print(f"\n== {T:%Y-%m-%d}: vigilância {100 * s['frac']:.0f}% da janela de 7 d: NÃO APLICÁVEL"); continue
        p = {t: float(np.mean(nz[t] < s["tags"][t]["zmax"])) for t in tags}
        pp = {t: float(np.mean(npers[t] < s["tags"][t]["persist"])) for t in tags}
        pct[T] = p; info[T] = s
        topo = sorted(tags, key=lambda t: -p[t])[:6]
        print(f"\n== {T:%Y-%m-%d}: vigilância {100 * s['frac']:.0f}% | tags com percentil >= 0,95: {sum(v >= 0.95 for v in p.values())} de {len(tags)}")
        for t in topo:
            d = s["tags"][t]
            print(f"   {t.replace('954005_624_', ''):12s} percentil {p[t]:.2f} | zmax {d['zmax']:7.1f} ({'+' if d['zsinal'] >= 0 else '-'}) | persistência {d['persist']:3d} h (percentil {pp[t]:.2f}) | lead {d['lead_h'] if d['lead_h'] == d['lead_h'] else float('nan'):6.1f} h | janelas normais com corrida sustentada {100 * nlead_has[t]:.0f}%")

    print("\nQ4. CONSISTÊNCIA ENTRE EVENTOS (nº de eventos com percentil >= 0,95; sob o nulo, >= 4 de 8 tem p ~ 4,6e-4)")
    if pct:
        M = pd.DataFrame(pct); M.columns = [f"{c:%m-%d}" for c in M.columns]
        cont = (M >= 0.95).sum(axis=1).sort_values(ascending=False)
        print(f"   eventos aplicáveis: {M.shape[1]} de {len(eventos)}")
        print("   tags com >= 2 eventos:", {k.replace('954005_624_', ''): int(v) for k, v in cont.items() if v >= 2})
        fam = nvio.grupos(tags)
        for nome in ("exaustao_tc", "vibracao", "processo"):
            n_ev = int((M.loc[[t for t in fam[nome] if t in M.index]] >= 0.95).any(axis=0).sum())
            print(f"   família {nome:12s}: eventos com >= 1 tag incomum {n_ev} de {M.shape[1]}")
        M.round(2).to_csv(SAIDA / "percentis_tag_evento.csv")
    # lead da tag mais incomum
    if info:
        leads = []
        for T, s in info.items():
            top = max(tags, key=lambda t: pct[T][t])
            leads.append(s["tags"][top]["lead_h"])
        print(f"\nQ3. lead da tag mais incomum por evento (h): {[round(x, 1) if x == x else None for x in leads]} | mediana {np.nanmedian(leads) if len(leads) else float('nan'):.1f}")

    # figuras: top-5 tags por evento
    for T, s in info.items():
        zh, _ = N.zhoras(T)
        top = sorted(tags, key=lambda t: -pct[T][t])[:5]
        fig, ax = plt.subplots(figsize=(11, 3.6))
        for t in top:
            ax.plot(zh.index, zh[t].clip(-30, 30), lw=1, label=f"{t.replace('954005_624_', '')} (pct {pct[T][t]:.2f})")
        ax.axhline(3, color="grey", ls=":"); ax.axhline(-3, color="grey", ls=":"); ax.set_ylabel("z horário (recortado em ±30)"); ax.set_title(f"trip {T:%Y-%m-%d %H:%M}: 5 tags mais incomuns, 7 d antes")
        ax.legend(fontsize=7, ncol=3, loc="upper left"); fig.tight_layout(); fig.savefig(SAIDA / f"ev_{T:%Y%m%d}.png", dpi=90); plt.close(fig)
    print(f"\nfiguras e percentis em {SAIDA}")


if __name__ == "__main__":
    main()

# EMENDA (15/10/2026, depois do 1.º resultado do EXP 01b): `zhoras` incluía a hora final INCOMPLETA da janela. Os instantes normais são sorteados em hora cheia (floor("h")), então a última hora tinha 1 amostra (< 10) e virava NaN SÓ nos nulos, enquanto os eventos (minutos quebrados) mantinham a hora parcial.
# No EXP 01b isso zerou a corrida terminal de todas as janelas normais (0,0% em todas as famílias) e deu "percentil 1,00" a qualquer evento: resultado inválido. No EXP 01 o efeito era de uma hora de assimetria (zmax e persistência). Corrigido para eventos e nulos igualmente: a janela termina na última hora completa antes de E - 2 h.
# O EXP 01 e o 01b foram REPETIDOS com a correção; valem os números repetidos.
