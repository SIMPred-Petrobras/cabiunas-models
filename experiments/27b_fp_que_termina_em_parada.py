#!/usr/bin/env python3
"""EXP 27b — OS FP REMOVÍVEIS TERMINAM NUMA PARADA? (alarme que dura até a máquina parar por outro motivo)

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. (Pergunta nascida da tabela do EXP 27, depois de H-COMB e H-MANUT refutadas; os critérios abaixo são escritos antes desta rodada.)

OBSERVAÇÃO QUE MOTIVA. No EXP 27 o FP de 26/10/2025 (112,4 h) termina por volta de 31/10/2025 09:40, e a lista de paradas longas tem uma parada de 77 h que começa em 31/10/2025; o de 15/10/2025 (135 h) tem t, sp e vb acesos 100% do tempo. A memória 'trip-denominador-parada-real' mostra que 18 dos 20 'episódios TRIP' da pesquisa antiga eram alarmes com a máquina parada, e 'o-alvo-esta-certo': 13 das 14 paradas fora do alvo são utilidade ou gás. Hipótese H-PARADA: os FP longos (o alarme não acaba por melhora do sinal, acaba porque a máquina PARA, por um motivo que não é o trip-alvo: parada operacional ou de manutenção). Se for, o episódio 'FP' pode ser, na prática, um alarme que precedeu uma parada não-alvo, e a remoção desses FP pelo E192/consenso é a remoção de alarmes com valor operacional possível, o que mudaria a leitura 'ganho'.
O QUE SE FAZ (sem mexer no detector). Os 9 FP e os 5 episódios de trip do ponto publicado (caminho exato). Para cada episódio: (i) COMO TERMINA: o episódio termina no fechamento da máscara de vigilância (a máquina parou ou T5 caiu), com a RUNNING_A virando 0 ou o T5 < 300 em até 2 h depois do fim do episódio; ou termina com a máquina em marcha; (ii) a duração da parada que se segue (horas até o religamento) quando termina em parada; (iii) se há trip registrado a <= 7 dias depois do fim (o episódio seria um precursor tardio de um trip-alvo fora da janela de 48 h); (iv) a taxa de base: fração de TODOS os episódios do detector publicado que terminam em parada.
CRITÉRIOS (fixados antes). H-PARADA se confirma para um FP se termina em parada de >= 24 h; para o conjunto dos FP removíveis (15/10/2025, 26/10/2025, 09/01/2026) se >= 2 dos 3 terminam em parada >= 24 h; e é ESPECÍFICA se a fração de FP removíveis que terminam em parada >= 24 h é >= 2x a dos demais FP.
EXPECTATIVAS: o de 26/10/2025 termina na parada de 31/10 ~90%; o de 15/10/2025 termina em parada ~35%; o de 09/01/2026 termina em parada >= 24 h ~25%; o conjunto confirma H-PARADA ~30%; específica ~20%; o fato de os FP removíveis serem longos (> 100 h) explica por si o terminar em parada (episódios longos têm mais chance de pegar uma parada) ~60%.
INTERPRETAÇÃO PREVISTA. Se confirma e é específica, os FP longos são candidatos a 'alarme que durou até a parada' (valor operacional possível) e remover esses episódios custa o que um operador poderia ter usado: pede checar com a operação o motivo das paradas de 31/10/2025 e de janeiro/2026. Se não, são FP puros e a conclusão do EXP 26 se mantém.
(Nota: a 1.ª execução calculou a duração da parada com um laço que saía no mesmo instante e deu 0,0 h em todos os casos; corrigido antes de qualquer leitura da duração.)
LIMITES. 14 episódios; uma parada por outro motivo não prova causalidade (pode ser coincidência com o ciclo de manutenção); a classificação 'termina em parada' depende da máscara (o episódio termina quando a vigilância fecha); não há registro das causas das paradas; nada se adota.

RESULTADO (15/10/2026) -- H-PARADA NÃO SE CONFIRMA (1 de 3), mas um dos três é um alarme de 112 h que termina numa parada de 77 h seguida de trip-alvo 4 dias depois.
  (Nota: a 1.ª execução imprimiu a duração da parada como 0,0 h em todos os casos por um erro de laço meu; corrigido e reexecutado antes de ler qualquer duração. Os resultados abaixo são os da versão corrigida.)
  Como cada episódio termina (parada de >= 24 h entre parênteses): 15/10/2025 (135 h, FP): termina com a máquina parando por 0,6 h (oscilação), NÃO em parada longa; 26/10/2025 (112 h, FP): termina em 31/10 09:40 e a máquina fica PARADA 77,4 h; o trip-alvo de 04/11/2025 vem 4 dias depois; 09/01/2026 (34 h, FP): termina com a máquina em marcha. Os demais FP: 08/08 e 23/08 (20 h) param por 8,2 h; 16/01/2026 (7 h) 0,1 h; 02/04/2026 (52 h) termina numa parada de 92,9 h; 31/03, 16/06 e 08/04/2026 terminam em marcha. Taxa de base: 10 de 16 episódios (62%) terminam em parada.
  CRITÉRIOS: FP removíveis que terminam em parada >= 24 h: 1 de 3 (limite >= 2 de 3): NÃO; demais FP: 1 de 6 (17%), razão 2,0x, sobre 1 e 1 casos: sem poder.
  Cobertura do 'Histórico Operações': as datas de execução vão só até 04/2025 e as datas planejadas até 24/11/2025 (preventivas em 06/11 e 20/11/2025, depois do trip); NÃO há registro de execução que explique a parada de 31/10/2025.
LEITURA. (1) O FP de 26/10/2025 é um alarme de quase cinco dias que termina numa parada de 77 h, e o trip de 04/11 vem logo depois da religação: pode ser um precursor distante da cadeia 31/10 -> 04/11 (fora da janela de 48 h do alvo e por isso 'FP'), pode ser coincidência. Os dados atuais não resolvem; é a única observação desta sessão em que um FP removível tem um desfecho físico plausível. (2) Os outros dois FP removíveis (15/10/2025 e 09/01/2026) não terminam em parada longa. (3) Consequência: a remoção pelo E192 ou pelo consenso apaga o alarme de 26/10 e, se ele for precursor, essa 'melhoria' custa um aviso; pede-se à operação o motivo da parada de 31/10/2025. Nada aqui muda o alvo (trip = 2º estágio).
PLACAR: ACERTEI: o de 26/10 termina na parada de 31/10 (~90%); o de 15/10 NÃO termina em parada >= 24 h (~65%); o de 09/01 NÃO termina em parada >= 24 h (~75%); o conjunto não confirma H-PARADA (~70%). NÃO CONCLUSIVO: específica (razão 2,0x sobre 1 e 1 casos).

Uso:  PYTHONPATH=src .venv/bin/python experiments/27b_fp_que_termina_em_parada.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E25 = _carrega("exp25", "25_consenso_de_vizinhanca.py"); E6 = E25.E6; E5 = E25.E5
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG

INICIO = E25.INICIO; GL = E25.GL; JAN = pd.Timedelta(hours=48)
REMOVIVEIS = ("2025-10-15", "2025-10-26", "2026-01-09")


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual")
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    pub = E25.alarme({}); eps = episodios_de_serie(pd.Series(pub, index=idx).loc[INICIO:])
    r = (g["RUNNING_A"] > 0.5).fillna(False); t5 = g["T5_AVG_A"]; linhas = []
    for a, b in eps:
        tr = [t for t in reg if a <= t and b >= t - JAN]; tipo = ("trip " + str(tr[0])[:10]) if tr else "FP"
        pos = g.loc[b:b + pd.Timedelta(hours=2)]; para = bool((~r.loc[pos.index]).any() or (t5.loc[pos.index] < 300).any())
        dur_parada = np.nan
        if para:
            j = g.index.searchsorted(b); i = j
            while i < len(g) and (r.iloc[i] and t5.iloc[i] > 300) and i - j < 60:      # primeiro instante parado (até 2 h depois do fim do episódio)
                i += 1
            ini_p = i
            while i < len(g) and not (r.iloc[i] and t5.iloc[i] > 300) and i - ini_p < 40000:   # até o religamento
                i += 1
            dur_parada = (g.index[min(i, len(g) - 1)] - g.index[min(ini_p, len(g) - 1)]).total_seconds() / 3600
        depois = [t for t in reg if b < t <= b + pd.Timedelta(days=7)]
        linhas.append(dict(inicio=f"{a:%Y-%m-%d %H:%M}", fim=f"{b:%Y-%m-%d %H:%M}", dur_h=round((b - a).total_seconds() / 3600 + 2 / 60, 1), tipo=tipo, termina_em_parada=para, parada_h=round(dur_parada, 1) if para else None, trip_ate_7d_depois=str(depois[0])[:10] if depois else ""))
    T = pd.DataFrame(linhas); print(T.to_string(index=False), flush=True)
    base = float(T.termina_em_parada.mean()); fp = T[T.tipo == "FP"]; rem = fp[fp.inicio.str[:10].isin(REMOVIVEIS)]; outros = fp[~fp.inicio.str[:10].isin(REMOVIVEIS)]
    longa = lambda d: float(np.mean([bool(x) and (y is not None and y >= 24) for x, y in zip(d.termina_em_parada, d.parada_h)])) if len(d) else float("nan")
    print(f"\ntaxa de base: {100 * base:.0f}% de todos os episódios ({int(T.termina_em_parada.sum())} de {len(T)}) terminam em parada", flush=True)
    print(f"FP removíveis que terminam em parada >= 24 h: {int(round(longa(rem) * len(rem)))} de {len(rem)} | demais FP: {int(round(longa(outros) * len(outros)))} de {len(outros)} ({100 * longa(outros):.0f}%) -> razão {longa(rem) / max(longa(outros), 1e-9):.1f}x", flush=True)
    # as paradas fora do alvo nas datas dos FP
    print("\nOperações (Histórico Operações): cobertura de datas", flush=True)
    try:
        d = pd.read_excel(Path(__file__).resolve().parents[2] / "Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models/Histórico Operações TSC33003A.xlsx")
        print("   colunas de data:", [c for c in d.columns if "ata" in str(c)][:6], flush=True)
        for c in [c for c in d.columns if "ata" in str(c)][:3]:
            x = pd.to_datetime(d[c], errors="coerce"); print(f"   {c}: {x.min()} a {x.max()}", flush=True)
    except Exception as e:
        print("   não lido:", repr(e)[:120], flush=True)


if __name__ == "__main__":
    main()
