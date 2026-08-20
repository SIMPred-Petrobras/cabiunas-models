#!/usr/bin/env python3
"""
avalia_pos_processamento.py
Aplica ao detector de 4 sinais duas correções vindas de frentes diferentes e mede
o efeito de cada uma isoladamente e das duas juntas.

CORREÇÃO A — portão de "desligado" pelo piso físico do sensor-alvo (Diego, EXP10)
    O `RUNNING_A` fica preso em ~1 durante desligamentos reais. No episódio de
    19–23/08/2025 a turbina passou 3,5 dias a ~30 °C com a flag em 1. Medido no
    detector de 4 sinais: 99,4% dos pontos abaixo de 150 °C são tratados como
    operação, e 24% de TODO o alerta do período OOS cai nesse único episódio.
    O critério de piso é independente do RUNNING_A — um pode falhar sem o outro.

CORREÇÃO B — janela pós-evento na definição de falso positivo (Diego, implícito)
    A régua original conta como falso positivo qualquer episódio que não anteceda
    um evento. Um alerta que CONTINUA depois do trip é contado como erro — mas a
    máquina acabou de falhar, e o detector estar aceso ali não é engano. O
    `normal_alert_rate` do Diego já exclui ±24h em torno do alarme; a régua de
    episódios não excluía nada depois.

Por que as duas juntas e não só a A: aplicada sozinha, a correção A PIORA a
contagem de episódios (11→16 no ponto panorâmico). O motivo não é o portão, é o
agrupamento — o portão mascara o resfriamento pós-trip e parte em dois os
episódios em torno das falhas reais de 27/02, 18/03, 07/04, 11/04 e 29/04. Os
fragmentos posteriores viram "falso positivo" pela régua antiga. Só com a
correção B o ganho aparece.

Uso:
    PYTHONPATH=. python scripts/avalia_pos_processamento.py
    PYTHONPATH=. python scripts/avalia_pos_processamento.py --piso 200 --pos_evento 12
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(os.environ.get(
    "PDM_REPLAY_DIR",
    "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados"
    "-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad"))
HORIZONTE_H = 48.0
GAP_EPISODIO = pd.Timedelta(hours=2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--piso", type=float, default=150.0,
                   help="piso físico do sensor-alvo, em °C, abaixo do qual a máquina "
                        "é considerada desligada independentemente do RUNNING_A")
    p.add_argument("--pos_evento", type=float, default=24.0,
                   help="horas após o evento em que o alerta não conta como falso positivo")
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--out", default="eval_predictive_out/pos_processamento.csv")
    return p.parse_args()


def episodios(alerta: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    idx = alerta.index[alerta.fillna(False).values]
    if not len(idx):
        return []
    saida, ini, ant = [], idx[0], idx[0]
    for t in idx[1:]:
        if t - ant > GAP_EPISODIO:
            saida.append((ini, ant))
            ini = t
        ant = t
    saida.append((ini, ant))
    return saida


def metricas(alerta: pd.Series, eventos: list[pd.Timestamp], dias: float,
             pos_evento_h: float) -> dict:
    sp = episodios(alerta)
    ev = np.array([t.timestamp() for t in eventos])
    ini = np.array([a.timestamp() for a, _ in sp]) if sp else np.empty(0)
    hz = HORIZONTE_H * 3600.0

    detectados = [t for t in ev if ini.size and np.any((ini <= t) & (ini >= t - hz))]
    leads = [(t - ini[(ini <= t) & (ini >= t - hz)].min()) / 3600.0 for t in detectados]

    fp = []
    for a, b in sp:
        ta, tb = pd.Timestamp(a).timestamp(), pd.Timestamp(b).timestamp()
        antecipa = bool(np.any((ev >= ta) & (ev <= tb + hz)))
        rescaldo = bool(np.any((ev <= ta) & (ev >= ta - pos_evento_h * 3600.0))) \
            if pos_evento_h > 0 else False
        if not (antecipa or rescaldo):
            fp.append((a, b))
    horas = sum((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 3600.0 for a, b in fp)
    meses = dias / 30.44
    return dict(detectadas=len(detectados),
                lead_h=float(np.median(leads)) if leads else float("nan"),
                episodios=len(sp), fp=len(fp), fp_mes=len(fp) / meses,
                horas_fp_mes=horas / meses, pontos=int(alerta.fillna(False).sum()))


def main() -> None:
    a = parse_args()
    import joblib
    sys.path.insert(0, str(SCRATCH / "pdm" / "src"))
    os.environ.setdefault("CABIUNAS_PDM_ROOT", str(SCRATCH / "pdm"))

    linhas = []
    for nome, arq in [("panoramico", "replay_panoramico.joblib"),
                      ("conservador", "replay_conservador.joblib")]:
        caminho = SCRATCH / arq
        if not caminho.exists():
            print(f"[skip] {nome}: {caminho.name} ausente")
            continue
        r = joblib.load(caminho)
        eventos = [pd.Timestamp(e["inicio"]) for e in r.events]
        dias = float(r.metrics["dias_avaliados"])
        alvo = r.sensors[a.sensor].reindex(r.alert.index, method="nearest")
        base = r.alert.fillna(False)
        com_portao = base & (alvo >= a.piso)

        print(f"\n=== {nome}  ({dias:.0f} dias, {len(eventos)} falhas) ===")
        print(f"{'cenário':30s} {'det':>4s} {'lead':>6s} {'FP':>4s} "
              f"{'FP/mês':>7s} {'h FP/mês':>9s} {'pontos':>8s}")
        for rot, al, pos in [("original", base, 0.0),
                             ("A: portão de piso", com_portao, 0.0),
                             ("B: janela pós-evento", base, a.pos_evento),
                             ("A+B", com_portao, a.pos_evento)]:
            m = metricas(al, eventos, dias, pos)
            linhas.append(dict(ponto=nome, cenario=rot, **m))
            print(f"  {rot:28s} {m['detectadas']:>4d} {m['lead_h']:>6.1f} {m['fp']:>4d} "
                  f"{m['fp_mes']:>7.2f} {m['horas_fp_mes']:>9.1f} {m['pontos']:>8d}")

    if not linhas:
        raise SystemExit("nenhum replay encontrado — rode o replay antes.")
    df = pd.DataFrame(linhas)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\ngravado: {a.out}")


if __name__ == "__main__":
    main()
