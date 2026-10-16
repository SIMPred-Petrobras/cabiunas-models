#!/usr/bin/env python3
"""EXP 38 — OS FP DO DETECTOR COINCIDEM COM ALARMES REGISTRADOS DA PLANTA? (diário real de alarmes de 2025-2026 contra os episódios do publicado, 8 eventos)

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

DADO. `Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv`: o diário de alarmes do sistema de controle (SCADA) da turbina A, de 04/01/2022 a 18/04/2026, 6.851 eventos (3.757 ACT/UNACK e 3.094 INACT/UNACK), com a coluna `Tag Alarme` JÁ nos nomes das nossas colunas, a `Condição do Alarme` (HI, HIHI, UNDER, LOLO, OVER, CFN = alarme discreto 'ATUADO', OK = retorno), `Prioridade` e `Suprimido`; 1.814 eventos em 2025 e 258 em 2026.
FUSO (regra fixada antes): a data do diário é naive; escolhe-se o deslocamento (-3 h, 0 ou +3 h somado à data do diário para obter UTC) que MAXIMIZA o número de eventos ACT no intervalo [T - 10 min, T + 60 min] em torno dos instantes T dos 9 trips do registro (a cascata de alarmes quando a máquina cai); o xlsx de 30 s tinha hora local, então a expectativa é +3 h.
O QUE SE FAZ. Os 18 episódios do detector publicado (sinais de 8 eventos; 9 FP e 9 de trip), e para cada um: (i) os eventos ACT no instante do disparo, em [nascimento - 30 min, nascimento + 30 min]; (ii) os ACT durante o episódio e nas 6 h antes do nascimento; (iii) as tags e condições. 'Alarme analógico' = HI, HIHI, UNDER, LOLO, OVER; 'discreto' = CFN. LINHA DE BASE: 2.000 instantes aleatórios do tempo vigiado do período avaliado (a partir de 01/02/2025, até o último dia coberto pelo diário): a probabilidade de haver ao menos um ACT (analógico) em +-30 min, e o número esperado de FP com alarme se os FP fossem instantes aleatórios.
CRITÉRIOS (fixados antes). Os FP 'COINCIDEM COM ALARME DA PLANTA' se >= 5 dos 9 têm ao menos um ACT analógico em +-30 min do nascimento E a proporção é >= 2x a da linha de base (teste binomial unilateral p <= 0,10). 'NÃO COINCIDEM' se <= 2 dos 9. Idem para os 9 episódios de trip, só descritivo.
EXPECTATIVAS: o deslocamento +3 h vence ~80%; os FP têm alarme analógico da planta em +-30 min em <= 3 dos 9 ~65% e sem enriquecimento sobre o acaso ~70% (o detector aponta desvios do padrão antes que os limites do sistema de controle sejam cruzados); os episódios de trip têm alarme analógico em +-30 min em >= 3 dos 9 ~55%; a probabilidade de base de alarme analógico em +-30 min fica entre 5% e 30% ~65%.
INTERPRETAÇÃO PREVISTA. Se os FP não coincidem com alarmes da planta, a maioria dos nossos FP é 'sub-alarme': o detector avisa de um desvio que o operador não via; o FP não é, então, um 'alarme incômodo' redundante, mas um aviso novo sem desfecho conhecido. Se coincidem, o detector está repetindo o que o sistema de controle já alarmou, e o valor desses episódios é baixo. Em ambos os casos o resultado só é descritivo: o alvo continua sendo o trip de 2º estágio.
LIMITES. 18 episódios; o diário pode ter eventos suprimidos ou perdidos; o diário termina em 18/04/2026; o deslocamento de fuso é escolhido com 9 trips; a coincidência temporal não prova causalidade; nada se adota.

RESULTADO (16/10/2026) -- OS FP NÃO COINCIDEM COM ALARMES REGISTRADOS DA PLANTA: nenhum dos 9 teve alarme analógico no instante do disparo; o diário de alarmes da planta é raro e o detector avisa ANTES (ou sem) alarme da planta.
  FUSO: o diário está em UTC: deslocamento 0 h = 23 eventos ACT em [T-10 min, T+60 min] ao redor dos trips, +-3 h = 0 eventos (a expectativa de +3 h estava errada; a regra escrita foi aplicada). O diário tem 3.757 ativações (ACT) de 04/01/2022 a 18/04/2026; em 2025-2026 as analógicas são poucas: HI 35, HIHI 27, LOLO 10, UNDER 157 (229 em 16 meses) contra 867 discretas (CFN: 'ATUADO').
  18 episódios do publicado cobertos pelo diário (até 18/04/2026). NO INSTANTE DO DISPARO (+-30 min): 0 de 9 FP e 0 de 9 episódios de trip têm ACT analógico; linha de base (2.000 instantes vigiados aleatórios): P(>= 1 ACT analógico em +-30 min) = 0,005; VEREDITO (FP): NÃO COINCIDEM (0 de 9). Nas 6 h ANTES do nascimento: 0 de 9 FP (e 1 de 9 trips). DURANTE o episódio, ACT analógico: 2 de 9 FP (15/10/2025: TC382_03_A HI e HIHI; 08/04/2026: TC382_03_A HI e HIHI) e 2 de 9 trips (a corrida de 647 h e o onset de 26/02/2026).
  COMPLEMENTO DESCRITIVO (pós-hoc, não pré-registrado), contando também os alarmes DISCRETOS (CFN): em +-2 h do disparo só 2 dos 9 FP têm algum evento (09/01/2026 e 08/04/2026, 1 evento cada); durante os episódios, 6 dos 9 FP têm algum evento (7 em 15/10/2025: PDI_0302, PI_0315 e PI_0319 discretos e TC382_03_A HI/HIHI; 5 em 08/04/2026: TC382_03_A HI e HIHI; 2 em 23/08/2025 e 1 em 16/01/2026 e 02/04/2026, todos discretos de PI_0319 ou PDI_0302); em 31/03/2025, 16/06/2025, 26/10/2025 e 09/01/2026 nenhum evento durante o episódio.
LEITURA. (1) O FP do detector NÃO é um alarme redundante do sistema de controle: em nenhum dos 9 houve alarme analógico da planta no momento do disparo, e a probabilidade ao acaso é de 0,5% (a planta alarma pouco, 229 analógicos em 16 meses): é um aviso de 'sub-alarme' (desvio do padrão antes de qualquer limite configurado). (2) O mesmo vale para os trips: 0 de 9 episódios de trip têm alarme da planta no onset: o detector precede o sistema de controle também nos acertos; logo a presença de alarme da planta não separa FP de acerto no disparo e não serve como veto nem como confirmação. (3) Em dois FP (15/10/2025 e 08/04/2026) a planta alarmou HI e HIHI do termopar TC382_03_A DURANTE o episódio: houve um evento térmico real que cruzou o limite do sistema de controle; esses dois merecem pergunta à operação (e o de 15/10/2025 é um dos dois FP longos removidos pelo consenso/E192). (4) Os alarmes discretos (CFN) durante os FP são de pressão de gás combustível (PI_0315, 'Pr. Baixa Gas Combustivel') e de PI_0319/PDI_0302: coerente com 'selagem e gás combustível são tags independentes' e com o canal p ser o que vive aceso nos FP só-B (p + vb).
PLACAR: ACERTEI: os FP têm alarme analógico da planta em +-30 min em <= 3 de 9 (~65%: 0) e sem enriquecimento (~70%). ERREI: o fuso +3 h vence (~80%: o diário está em UTC); os episódios de trip têm alarme analógico em +-30 min em >= 3 de 9 (~55%: 0 de 9); a probabilidade de base entre 5% e 30% (~65%: 0,5%).

Uso:  PYTHONPATH=src .venv/bin/python experiments/38_alarmes_da_planta_nos_fp.py
"""
from __future__ import annotations
import contextlib, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48)
LOG = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv"); ANALOG = {"HI", "HIHI", "UNDER", "LOLO", "OVER"}


def main():
    d = pd.read_csv(LOG); d["t0"] = pd.to_datetime(d["Data da Ocorrência"]); d = d[d["Status"] == "ACT/UNACK"].copy(); trips = [t for t in carrega_registro()]
    print(f"diário: {len(d)} ativações (ACT) de {d.t0.min()} a {d.t0.max()}; condições: {d['Condição do Alarme'].value_counts().to_dict()}", flush=True)
    print("\nFUSO: eventos ACT em [T-10 min, T+60 min] ao redor dos 9 trips, por deslocamento", flush=True); best = None
    for off in (-3, 0, 3):
        t = (d.t0 + pd.Timedelta(hours=off)).dt.tz_localize("UTC"); n = sum(int(((t >= T - pd.Timedelta(minutes=10)) & (t <= T + pd.Timedelta(minutes=60))).sum()) for T in trips if d.t0.min() <= T.tz_localize(None) <= d.t0.max()); print(f"   deslocamento {off:+d} h: {n} eventos", flush=True)
        if best is None or n > best[1]:
            best = (off, n)
    off = best[0]; print(f"   -> deslocamento escolhido: {off:+d} h", flush=True); d["t"] = (d.t0 + pd.Timedelta(hours=off)).dt.tz_localize("UTC"); d = d.sort_values("t"); an = d[d["Condição do Alarme"].isin(ANALOG)]
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        R = ci.prever({"detector": det}, proc)
    al = R["is_anomaly"].astype(bool); fim_log = d.t.max(); eps = [(a, b) for a, b in episodios_de_serie(al.loc[INICIO8:]) if a <= fim_log]
    ta = an.t.to_numpy(); janela = lambda x, lo, hi: an[(an.t >= x + lo) & (an.t <= x + hi)]
    print(f"\nEPISÓDIOS DO PUBLICADO cobertos pelo diário (até {fim_log:%Y-%m-%d}): {len(eps)}", flush=True); linhas = []
    for a, b in eps:
        tr = [t for t in reg if a <= t and b >= t - JAN]; w = janela(a, -pd.Timedelta(minutes=30), pd.Timedelta(minutes=30)); du = an[(an.t >= a) & (an.t <= b)]; an6 = janela(a, -pd.Timedelta(hours=6), pd.Timedelta(0))
        linhas.append(dict(inicio=f"{a:%Y-%m-%d %H:%M}", dur_h=round((b - a).total_seconds() / 3600 + 2 / 60, 1), tipo=("trip " + str(tr[0])[:10]) if tr else "FP", alarmes_30min=len(w), durante=len(du), antes_6h=len(an6), tags_30min=", ".join(sorted({f"{x.split('_', 2)[-1]}:{c}" for x, c in zip(w["Tag Alarme"], w["Condição do Alarme"])}))[:90]))
    T = pd.DataFrame(linhas); pd.set_option("display.width", 250); print(T.to_string(index=False), flush=True)
    vm = proc["mask"].to_numpy(bool) & (proc.index >= INICIO8) & (proc.index <= fim_log); pos = np.flatnonzero(vm); rng = np.random.default_rng(20261016); idx = proc.index; hits = []
    for _ in range(2000):
        x = idx[int(rng.choice(pos))]; hits.append(len(janela(x, -pd.Timedelta(minutes=30), pd.Timedelta(minutes=30))) > 0)
    p0 = float(np.mean(hits)); print(f"\nLINHA DE BASE: probabilidade de >= 1 ACT analógico em +-30 min de um instante vigiado aleatório = {p0:.3f}", flush=True)
    from scipy.stats import binom
    for tipo in ("FP", "trip"):
        s = T[T.tipo.str.startswith(tipo)]; k = int((s.alarmes_30min > 0).sum()); n = len(s); pv = float(binom.sf(k - 1, n, p0)) if k > 0 else 1.0
        print(f"   {tipo}: {k} de {n} têm >= 1 ACT analógico em +-30 min do nascimento (esperado ao acaso {n * p0:.1f}; razão {k / max(n * p0, 1e-9):.1f}x; p binomial unilateral {pv:.3f}) | com ACT durante o episódio: {int((s.durante > 0).sum())} de {n} | nas 6 h antes: {int((s.antes_6h > 0).sum())} de {n}", flush=True)
    fp = T[T.tipo == "FP"]; k = int((fp.alarmes_30min > 0).sum()); print(f"   VEREDITO (FP): {'COINCIDEM' if k >= 5 and k / max(len(fp) * p0, 1e-9) >= 2 else 'NÃO COINCIDEM' if k <= 2 else 'INCONCLUSIVO'} ({k} de {len(fp)})", flush=True)


if __name__ == "__main__":
    main()
