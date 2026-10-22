#!/usr/bin/env python3
"""Relatório final do brief "estado da arte" (`reports/estado_da_arte_comparativo.md` + figuras). Lê os artefatos das tarefas do ClearML (só leitura) e os caches locais.

O relatório e as figuras ficam FORA do git (regra do usuário: só código no repositório). Uso:  PYTHONPATH=src:experiments .venv/bin/python -m sota.relatorio
"""
from __future__ import annotations
import ast, contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv.avalia import carrega_controle
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from sota import protocolo as P

SAIDA = RAIZ / "reports"; FIG = SAIDA / "figs"
TAREFAS = dict(baseline="0d0309cd896f4a55ac30d48763ea12ef", h6="6e11bc034e0949ba8a39e1241c566357", triagem="6cf66f6e124c412a9509cd3331c927ac",
               h5="2ae653c1125145409740df89a3551c90", h1b="6d81032933e74281b336c272feb4b505", ae="fd1a4ac1de444463b7b2a7ba8042f4ea", tranad="df6909b4cf96429c8c4b3513549c424a")
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; TINTA, TINTA2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def artefatos():
    from clearml import Task
    A = {}
    for k, tid in TAREFAS.items():
        t = Task.get_task(task_id=tid)
        for nome, art in t.artifacts.items():
            A[(k, nome)] = pd.read_csv(art.get_local_copy())
    return A


def tup(x):
    return ast.literal_eval(x) if isinstance(x, str) and x.startswith("(") else x


def estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRADE)
    ax.tick_params(colors=TINTA2, labelsize=8); ax.grid(axis="y", color=GRADE, lw=0.6); ax.set_axisbelow(True); ax.set_facecolor(FUNDO)


def fig_canais(ci, det):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    S = pd.read_parquet(RAIZ / "data_local" / "sinais_v2_8ev.parquet"); S = S.loc[P.INICIO:P.FIM]
    al, vig = carrega_controle("atual"); trips = P.trips_alvo(carrega_registro()); eps = episodios_de_serie(al.loc[P.INICIO:P.FIM])
    nomes = {"t": "t — PCA de 14 temperaturas", "p": "p — PCA de 12 pressões", "sp": "sp — spread do mancal LNA (TI_0305)", "vb": "vb — z robusto das 10 sondas de vibração"}
    fig, axs = plt.subplots(4, 1, figsize=(11, 8.2), sharex=True, facecolor=FUNDO)
    for ax, c in zip(axs, ("t", "p", "sp", "vb")):
        thr = det["base"][c] * det["k_nivel_a"][c]; y = (S[c] / thr).where(S["mask"].astype(bool)).clip(upper=6)
        for a, b in eps:
            tp = any(a <= t and b >= t - JANELA for t in trips); ax.axvspan(a, b, color=AZUL if tp else LARANJA, alpha=0.18, lw=0)
        ax.plot(y.index, y.to_numpy(), color=TINTA2, lw=0.6); ax.axhline(1.0, color=TINTA, lw=0.8, ls="--")
        for t in trips:
            ax.axvline(t, color=TINTA, lw=1.0)
        ax.set_title(nomes[c], loc="left", fontsize=9, color=TINTA); ax.set_ylim(0, 6); estilo(ax)
        ax.set_ylabel("sinal / limiar A", fontsize=8, color=TINTA2)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Patch(color=AZUL, alpha=0.35, label="episódio que pega trip"), Patch(color=LARANJA, alpha=0.35, label="falso positivo"),
                           Line2D([], [], color=TINTA, lw=1, label="trip"), Line2D([], [], color=TINTA, lw=0.8, ls="--", label="limiar A")],
                  fontsize=7, ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.suptitle("PCA v2 — os 4 canais em vigilância (cortados em 6x o limiar), episódios de alarme e trips, fev/2025 a abr/2026", fontsize=10, color=TINTA, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig1_canais_pca_v2.png", dpi=150); plt.close(fig)


def fig_modelos(L):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=FUNDO)
    cores = {"PCA v2": AZUL, "AE (17 univariados)": LARANJA, "TranAD": AQUA}
    for ax, (col, ic, rot) in zip(axs, (("fp_mes", "ic_fp", "FP por mês de operação, com IC 95% (escala log)"), ("horas_mes", None, "horas de alarme por mês de operação (escala log)"))):
        for i, r in enumerate(L):
            x = r[col]; ax.scatter([x], [i], s=40, color=cores[r["familia"]], zorder=3, edgecolor=FUNDO, lw=1.5)
            if ic and r.get(ic) is not None:
                ax.plot(r[ic], [i, i], color=cores[r["familia"]], lw=2, zorder=2)
            xl = max(x, r[ic][1]) if (ic and r.get(ic) is not None) else x
            ax.text(xl * 1.12, i, f"{r['det']}/8", fontsize=7, color=TINTA2, va="center")
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
        if col == "horas_mes":
            ax.set_xticks([100, 200, 300, 500, 700])
        ax.set_yticks(range(len(L))); ax.set_yticklabels([r["rotulo"] for r in L] if ax is axs[0] else [], fontsize=7, color=TINTA)
        ax.invert_yaxis(); ax.set_title(rot, fontsize=9, loc="left", color=TINTA); estilo(ax); ax.grid(axis="x", color=GRADE, lw=0.6)
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in cores.items()], fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Modelos no protocolo único — o rótulo ao lado de cada ponto é a detecção de 8 trips; IC por blocos mensais", fontsize=10, color=TINTA, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_modelos.png", dpi=150); plt.close(fig)


def fig_retreino(pol, dec):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=FUNDO, gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]
    pts = {}
    for r in pol.itertuples():
        pts.setdefault((r.retreinos, round(r.fp_mes, 3), r.det), []).append(r.politica.split(" ", 1)[1].replace(".", ","))
    desl = {("mensal",): (-4, 6), ("ADWIN",): (-6, -12)}
    for (x, y, d), nomes in pts.items():
        cor = LARANJA if nomes == ["mensal"] else AZUL
        ax.scatter([x], [y], s=40, color=cor, zorder=3, edgecolor=FUNDO, lw=1.5)
        dx, dy = desl.get(tuple(nomes), (5, 3)); ha = "right" if dx < 0 else "left"
        ax.annotate(f"{' e '.join(nomes)} ({d}/8)", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=TINTA2, ha=ha)
    ax.set_xlim(-0.8, 15.2); ax.set_ylim(0.70, 2.25)
    ax.set_xlabel("retreinos em 14 viradas de mês", fontsize=8, color=TINTA2); ax.set_ylabel("FP por mês de operação", fontsize=8, color=TINTA2)
    ax.set_title("Política de retreino: quantos retreinos x quanto FP (rótulo: detecção)", fontsize=9, loc="left", color=TINTA); estilo(ax)
    ax = axs[1]; M = dec.set_index(dec.columns[0]).T
    ax.imshow(M.to_numpy(), aspect="auto", cmap=matplotlib.colors.ListedColormap(["#e4e3df", "#2a78d6"]), vmin=0, vmax=1)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=7, color=TINTA); ax.set_xticks(range(len(M.columns))); ax.set_xticklabels([c[2:] for c in M.columns], fontsize=7, rotation=90, color=TINTA2)
    ax.set_xticks(np.arange(-.5, len(M.columns), 1), minor=True); ax.set_yticks(np.arange(-.5, len(M.index), 1), minor=True); ax.grid(which="minor", color="#ffffff", lw=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Decisão em cada virada de mês (azul = retreina; cinza = mantém o bundle)", fontsize=9, loc="left", color=TINTA)
    fig.tight_layout(); fig.savefig(FIG / "fig3_retreino.png", dpi=150); plt.close(fig)


def tempo_inferencia(ci, modelos):
    """Segundos por dia de dado para pontuar 75 d de grade com as arquiteturas reais (o peso não importa para o tempo)."""
    import torch
    from sota.ae_walkforward import modelo as cnn, SENSORES, L
    from sota.tranad_modelo import TranADCausal
    torch.set_num_threads(1); n = 75 * 720; out = {}
    t0 = time.perf_counter()
    for _ in SENSORES:
        m = cnn(0).eval(); x = torch.randn(n, 1, L)
        with torch.no_grad():
            for i in range(0, n, 4096):
                m(x[i:i + 4096])
    out["AE (17 univariados)"] = (time.perf_counter() - t0) / 75
    t0 = time.perf_counter()
    for d in (14, 12):
        m = TranADCausal(d, 10, epocas=1); m.lo = np.zeros(d); m.amp = np.ones(d); m.score(np.random.default_rng(0).random((n, 10, d)))
    out["TranAD"] = (time.perf_counter() - t0) / 75
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; A = artefatos()
    B = A[("baseline", "baseline_pca_v2")]; AE = A[("ae", "ae_walkforward")]; TR = A[("tranad", "h3_tranad")]; POL = A[("h5", "politicas")]; DEC = A[("h5", "decisoes_por_virada")]
    TRI = A[("triagem", "triagem_fp")]; H6 = A[("h6", "h6_top3")]; R2 = A[("h1b", "r2_por_canal")]
    L = []
    for r in B.itertuples():
        L.append(dict(familia="PCA v2", rotulo=f"PCA v2 — {r.base}", fp_mes=r.fp_mes, ic_fp=tup(r.ic_fp_mes), horas_mes=r.horas_mes, det=r.det))
    for r in AE[AE.nivel == "alarm"].itertuples():
        L.append(dict(familia="AE (17 univariados)", rotulo=f"AE ALARM — semente {r.semente}", fp_mes=r.fp_mes, ic_fp=tup(r.ic_fp_mes), horas_mes=r.horas_mes, det=r.det))
    for r in TR[TR.variante == "t+p"].itertuples():
        L.append(dict(familia="TranAD", rotulo=f"TranAD t+p — semente {r.semente}", fp_mes=r.fp_mes, ic_fp=tup(r.ic_fp_mes), horas_mes=r.horas_mes, det=r.det))
    fig_canais(ci, det); fig_modelos(L); fig_retreino(POL, DEC); TI = tempo_inferencia(ci, modelos)
    GL.update(B=B, AE=AE, TR=TR, POL=POL, TRI=TRI, H6=H6, R2=R2, TI=TI)
    escreve(); print(f"relatório em {SAIDA / 'estado_da_arte_comparativo.md'}; figuras em {FIG}", flush=True)


GL = {}


def br(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ic_txt(c, nd=2):
    c = tup(c)
    return f"[{br(c[0], nd)}; {br(c[1], nd)}]" if isinstance(c, (tuple, list)) else "—"


def lista(s):
    return ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in s.value_counts().items())


def escreve():
    B, AE, TR, POL, TRI, H6, R2, TI = (GL[k] for k in ("B", "AE", "TR", "POL", "TRI", "H6", "R2", "TI"))
    pca = B[B.base == "atual"].iloc[0]; L = []
    L += ["# Estado da arte no pipeline de Cabiúnas (TC-33003A) — relatório comparativo", "",
          "Brief `BRIEF_estado_da_arte_cabiunas.md`. Branch `exp/sota` (wt-nova-visao), experimentos em `experiments/sota/`, todos rodados no ClearML (projeto `Cabiunas`, tags `sota-*`),",
          "com pré-registro commitado antes de cada execução. Gerado em 22/10/2026 por `experiments/sota/relatorio.py`.", "",
          "## 1. Resumo e recomendação", "",
          f"- **O PCA v2 continua o melhor detector testado.** No protocolo único (8 trips, janela de 48 h, fev/2025–abr/2026) ele dá {pca.det}/8 com {br(pca.fp_mes, 3)} FP por mês de operação. "
          "O TranAD (H3) é EQUIVALENTE pela regra e pior em ponto em todas as sementes; o pipeline de 17 AEs CNN-1D, refeito sem calibrar no teste, NÃO CHEGA PERTO (alarme ligado ~80% do tempo).",
          "- **Retreino sob demanda não se sustenta.** Os detectores de drift objetivos (ADWIN, Page-Hinkley) pedem retreino em 13 e 12 das 14 viradas de mês; os gatilhos que economizam retreinos "
          "pagam com detecção ou horas; congelar e a trava pioram com diferença declarada. **Recomendação: retreino mensal automático** (1,5 s, sem intervenção) — ver a seção 5.",
          "- **Os 9 FP do PCA v2:** 6 são de partida/parada (5 nascem 6,5 h após a partida, na borda da vigilância), 2 coincidem com alarme HI/HIHI do TC382_03_A da planta e 1 é puxado pela PI_5134001, "
          "uma tag de OUTRA unidade (ar de instrumentação) que está no canal de pressão.",
          "- **Vale pedir a NGP_A e a NPT_A no export de produção** (barato): a rotação reduz o erro sobretudo no canal do mancal, que o T5 não explica; não há evidência ainda de ganho em detecção.",
          "- **Diferença de 1 detecção não declara vencedor** (regra do protocolo): com 8 eventos o IC de detecção de qualquer modelo cobre 2 a 4 eventos.", "",
          "## 2. Protocolo (igual para todos os modelos)", "",
          "- Alvo: os **8 trips** de 27/02/2025 a 26/02/2026; a régua de alarme por sensor do pipeline de AEs é só secundária.",
          "- Período 01/02/2025 a 21/04/2026; só operação válida (máscara do v2: RUNNING_A, T5 > 300 e 6 h de blackout após a partida — a NGP_A não existe depois de out/2025).",
          "- Detecção: um episódio intersecta [T − 48 h, T]; *início*: nasce dentro da janela; *banda*: nasce com ≥ 4 h de antecedência. FP: episódio fora de qualquer janela. "
          "FP/mês = FP por 730 h vigiadas. Horas de alarme sempre ao lado da contagem.",
          "- **O lead é censurado em 48 h**: um episódio que já estava aceso antes da janela conta 48 h. A média e a mediana do lead vêm sempre com o número de eventos censurados; "
          "nenhuma conclusão se apoia no lead sozinho.",
          "- F1 por evento: recall = detectados/8; precisão = episódios que pegam trip / episódios.",
          "- IC 95% por bootstrap: eventos reamostrados por evento; custo em blocos mensais. Comparação pareada; detecção só difere com |Δ| ≥ 2 e IC sem o zero; custo só com IC sem o zero.",
          "- Nenhum limiar calibrado com eventos ou com o período de teste.", "",
          "## 3. Modelos no mesmo protocolo", "",
          "| modelo | detecção [IC] | início | banda | FP | FP/mês [IC] | horas (h/mês) | lead mediano (censurados) | F1 | contra o PCA v2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in B.itertuples():
        L.append(f"| PCA v2 — {r.base} | {r.det}/8 {ic_txt(r.ic_det, 0)} | {r.inicio} | {r.banda} | {r.fp} | {br(r.fp_mes, 3)} {ic_txt(r.ic_fp_mes)} | {br(r.horas, 0)} ({br(r.horas_mes, 1)}) | "
                 f"{br(r.lead_mediano, 1)} h ({r.n_censurado} de {r.det}) | {br(r.f1)} | {'referência' if r.base == 'atual' else 'ver seção 3.1'} |")
    for r in AE[AE.nivel == "alarm"].itertuples():
        L.append(f"| AE 17 univariados, ALARM — semente {r.semente} | {r.det}/8 {ic_txt(r.ic_det, 0)} | {r.inicio} | {r.banda} | {r.fp} | {br(r.fp_mes, 3)} {ic_txt(r.ic_fp_mes)} | "
                 f"{br(r.horas, 0)} ({br(r.horas_mes, 1)}) | {br(r.lead_mediano, 1)} h ({r.n_censurado} de {r.det}) | {br(r.f1)} | FP e horas PIORES (declarado) |")
    for r in TR[TR.variante == "t+p"].itertuples():
        piora = (r.difere_det and r.d_det < 0) or (r.difere_fp and r.d_fp_mes > 0) or (r.difere_horas and r.d_horas_mes > 0)
        L.append(f"| TranAD (t e p) — semente {r.semente} | {r.det}/8 {ic_txt(r.ic_det, 0)} | {r.inicio} | {r.banda} | {r.fp} | {br(r.fp_mes, 3)} {ic_txt(r.ic_fp_mes)} | "
                 f"{br(r.horas, 0)} ({br(r.horas_mes, 1)}) | {br(r.lead_mediano, 1)} h ({r.n_censurado} de {r.det}) | {br(r.f1)} | {'PIOR (declarado)' if piora else 'indistinguível'} |")
    L += ["", f"Tempo de inferência (CPU, 1 núcleo, segundos por dia de dado): PCA v2 0,005 · AE de 17 sensores {br(TI['AE (17 univariados)'], 3)} · TranAD {br(TI['TranAD'], 3)} "
          "(AE e TranAD medidos com as arquiteturas reais e pesos aleatórios, pontuando 75 d de grade; o peso não muda o tempo). Os três rodam em tempo real com folga.", "",
          "![Modelos](figs/fig2_modelos.png)", "",
          "**Leitura.** O AE de 17 sensores pega 8/8 porque fica em alarme ~80% do tempo vigiado: cada sensor passa do próprio q 0,90 em 22–27% do tempo (o limiar tirado da validação "
          "não se sustenta no mês seguinte) e o voto de 2 em 17 sensores acende quase sempre. O pipeline original tirava esse quantil da PRÓPRIA série avaliada (`pipeline_multi.py`), "
          "o que é calibrar com o teste. O TranAD, no mesmo lugar do PCA, perde de 1 a 3 detecções e soma 4 FP em todas as sementes. Isso fecha a frente de modelos (K-Means, FA, LSTM-AE, "
          "CNN-1D, Optuna com 300 configurações, modelo por grupo e agora TranAD e o AE univariado): o PCA não é superado com estes dados.", "",
          "### 3.1 Variantes do PCA v2 contra a base atual (pareado)", "",
          "Base atual 7/8 e 9 FP; G168 8/8 e 14 FP (horas +49,8 h/mês, declarado); B800 8/8 e 12 FP (indistinguível); retreino com n = 19.000 5/8 e 11 FP (Δdet −2, IC [−4; 0], "
          "indistinguível). O ruído de retreino cobre 2 detecções — por isso nenhuma diferença de 1 evento é tratada como resultado.", "",
          "![Canais do PCA v2](figs/fig1_canais_pca_v2.png)", "",
          "## 4. Os 9 falsos positivos do PCA v2 (triagem) e quem puxou cada alarme (H6)", "",
          "Causa provável pela precedência fixada antes: alarme real da planta sem trip > instrumento (sensor travado) > partida/parada (nasce na borda ou termina em parada) > sem explicação.", "",
          "| início (local) | duração | causa provável | alarme analógico da planta | horas desde a partida | borda | termina em parada | carga L | 3 tags que mais pesam (canal, razão ao limiar) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in TRI.itertuples():
        L.append(f"| {r.inicio_local} | {br(r.dur_h, 1)} h | {r.causa_provavel} | {r.alarmes_analogicos} | {br(r.h_desde_partida, 1)} | {'sim' if r.borda else 'não'} | {'sim' if r.termina_em_parada else 'não'} | "
                 f"{br(r.carga_L)} | {'; '.join(str(x).replace('.', ',') for x in (r.top1, r.top2, r.top3))} |")
    t1 = H6["top1"].str.split(" ").str[0]
    L += ["", "- **Partida/parada (6):** cinco nascem exatamente 6,5 h após a partida — a abertura da vigilância depois do blackout. O de 26/10/2025 termina numa parada seguida de 96 h de "
          "manutenção (05/11) e pode não ser falso para a operação; a consulta à operação (EXP 58) já inclui esta janela.",
          "- **Alarme real sem trip (2):** 14/10/2025 e 08/04/2026 têm TC382_03_A HI/HIHI durante o episódio, mas a tag que puxou o alarme do detector é outra (TI_0305/TI_0317 e PI_0308). "
          "Evidência fraca: o TC382_03_A é a tag que mais alarma na planta.",
          "- **Sem explicação (1):** 09/01/2026, puxado pela PI_5134001 (131x o limiar), PDI_0338 e PI_0319. A PI_5134001 é do ar de instrumentação (outra unidade): candidata a sair do canal p.",
          f"- **H6 não separa FP de acerto:** tag do topo nos FP: {lista(t1[H6.rotulo == 'FP'])}; nos acertos: {lista(t1[H6.rotulo != 'FP'])}. "
          "Serve para o operador saber o que olhar, não para filtrar.", "",
          "## 5. Retreino sob demanda (H5)", "",
          "Todas as políticas só trocam de bundle numa virada de mês, com os bundles de dia 1 (mudar o dia do corte muda muito o detector), partindo do bundle de fev/2025; 14 viradas.", "",
          "| política | retreinos | forçados | meses com bundle vencido | detecção | FP | FP/mês | horas | contra a mensal |", "|---|---|---|---|---|---|---|---|---|"]
    for r in POL.itertuples():
        piora = (tup(r.cmp_difere_det) and r.cmp_d_det < 0) or (tup(r.cmp_difere_fp) and r.cmp_d_fp_mes > 0) or (tup(r.cmp_difere_horas) and r.cmp_d_horas_mes > 0)
        L.append(f"| {r.politica} | {r.retreinos} | {r.forcados} | {r.meses_vencidos} | {r.det}/8 | {r.fp} | {br(r.fp_mes, 3)} | {br(r.horas, 0)} | "
                 f"{'referência' if r.politica == 'P0 mensal' else ('PIOR (declarado)' if piora else 'sem piora declarada')} |")
    L += ["", "![Retreino](figs/fig3_retreino.png)", "",
          "### Argumento para o cliente", "",
          "1. **O próprio detector de drift diz quando retreinar — e diz quase todo mês.** ADWIN e Page-Hinkley, com parâmetros padrão e sem calibração nos eventos, sobre o escore do modelo, "
          "pedem retreino em 13 e 12 das 14 viradas, com o mesmo desempenho do mensal. A demanda medida é mensal: a planta muda de regime (carga, campanha, manutenção) nessa escala.",
          "2. **Pular retreinos tem custo medido.** Congelar o modelo leva o FP de 0,81 para 2,15 por mês e as horas de alarme de 1.457 para 2.615 (diferença declarada). Os gatilhos que "
          "economizam retreinos (2 a 7 em vez de 14) perdem de 1 a 3 trips.",
          "3. **Retreinar é barato e automático.** O retreino roda em ~1,5 s com o histórico que já chega; a inferência já recusa bundle vencido (62 d). O \"sob demanda\" que faz sentido é "
          "**automático e mensal**, com o ADWIN registrado como evidência de que a troca era necessária — não um retreino manual esporádico.",
          "4. **A trava (só trocar se o baseline novo estiver limpo e a semana de sombra não piorar) não deve ir para produção**: rejeita a maioria dos bundles, segura o velho até a troca forçada "
          "e piora o FP (diferença declarada). É coerente com um achado antigo: o baseline precisa conter a variação normal, inclusive a desconfortável.", "",
          "## 6. Vale pedir NGP_A/NPT_A no export? (H1(b), descritivo)", "",
          "A NGP_A e a NPT_A só existem em cadência real de jan a out/2025 (dataset ClearML 424e5b58, hora local rotulada como UTC). R² fora da amostra (walk-forward mensal, LightGBM) "
          "de log(1 + sinal) de cada canal, só com o T5 contra T5 + NGP + NPT:", "", "| canal | R² (T5) | R² (T5 + rotações) | ΔR² |", "|---|---|---|---|"]
    for r in R2.itertuples():
        L.append(f"| {r.canal} | {br(r.r2_T5, 3)} | {br(r.r2_T5_NGP_NPT, 3)} | {br(r.delta, 3)} |")
    L += ["", "Pela regra fixada antes, **vale pedir** (ΔR² ≥ 0,10 no sp e no t). Ressalva: todos os R² são negativos — nenhum modelo prevê melhor que a média do mês; a rotação reduz o "
          "erro sobretudo no spread do mancal, que depende da rotação do eixo. Nenhum transiente de rotação escondido do T5 apareceu nos nascimentos (0 FP, 0 acertos). O pedido é barato e é "
          "pré-requisito para testar a H1 e a H2 de verdade.", "",
          "## 7. O que não foi rodado, e por quê", "",
          "- **H1 completa (modelo de comportamento esperado):** sem NGP/NPT depois de out/2025; a versão com T5 como variável de operação já foi refutada (`avalia_residuo_carga`), e os trips "
          "acontecem em carga alta — tirar a carga tira sinal.",
          "- **H2 (AE condicionado ao regime):** depende da mesma variável; um AE condicional ao regime (tempo desde a partida e nível recente) já foi testado (EXP 07) sem ganho.",
          "- **H4 (híbrido PCA + deep):** unir o PCA a outro modelo não ganhou (EXP 49); com o TranAD e o AE sem ganho isolado, não há segundo modelo para o híbrido.", "",
          "## 8. Limitações", "",
          "- 8 trips: qualquer IC de detecção cobre 2 a 4 eventos; ganhos pequenos são indistinguíveis do ruído de retreino.",
          "- Lead censurado em 48 h (3 de 7 eventos do PCA v2 estão no teto).",
          "- O AE foi refeito sem busca de hiperparâmetros (timebox de 1 dia) e em PyTorch; o TranAD usou os hiperparâmetros do TSB-AD/oficial.",
          "- A H1(b) cobre 5 trips agrupados e 9 meses; não mede detecção.",
          "- A triagem depende do diário de alarmes (alarmes suprimidos não aparecem); as ordens do SAP não foram cruzadas no worker.", "",
          "## 9. Rastreabilidade (ClearML, projeto `Cabiunas`)", ""]
    for k, tid in TAREFAS.items():
        L.append(f"- `{k}`: tarefa `{tid}`")
    (SAIDA / "estado_da_arte_comparativo.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
