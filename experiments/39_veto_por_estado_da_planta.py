#!/usr/bin/env python3
"""EXP 39 — TERCEIRA VIA: VETO POR ESTADO DE ALARME DA PLANTA (falha de instrumento declarada; distúrbio de suprimento de gás).

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. EXP 36: os FP passam pelo nível B e têm o padrão de canais de trips; EXP 37/37b: vetos derivados dos dados (transiente de carga, sensores travados) falharam; EXP 38: o diário real de alarmes da planta não tem alarme analógico no disparo (0 de 9 FP, 0 de 9 trips), mas tem alarmes DISCRETOS (CFN) com significado físico explícito na `Descrição Alarme`: 'Falha PI Gás Motor p/Partida' (PI_0319), 'Falha no PDT Gás Combustível' (PDI_0317), 'Falha PDI Vazam. Selo Prim.' (PDIT_0305) = FALHA DE INSTRUMENTO declarada pela planta; 'Pressão Baixa Gás Comb.' (PI_0315), 'PrDif.Bx.Gás Linha Balanc.' (PDI_0302) = DISTÚRBIO DE SUPRIMENTO DE GÁS. São condições com RAZÃO FÍSICA definida pelo sistema de controle, fixadas AGORA pela descrição e não pelos FP: se o instrumento declarou falha, o desvio do canal não é saúde da máquina; se o gás combustível está instável, o desvio é de processo (as paradas de utilidade e gás são as 13 de 14 paradas fora do alvo).
FAMÍLIAS (fixadas pelo texto da descrição): F_falha = os alarmes cuja `Descrição Alarme` contém 'Falha'; F_gas = os alarmes cuja descrição contém 'Gás Comb' ou 'Gás Linha' e não contém 'Falha'. ESTADO: de uma ativação (ACT/UNACK) até a desativação seguinte (INACT/UNACK) da mesma tag e condição; uma ativação sem desativação nos 14 dias seguintes fecha em 14 dias (o diário tem ativações sem par); só se consideram eventos analisados a partir de 01/02/2025 até o fim do diário (18/04/2026).
O QUE SE FAZ. (1) DUTY de cada família: fração do tempo vigiado com algum estado ativo. (2) ESTADO NO NASCIMENTO dos 18 episódios do publicado (sinais de 8 eventos): quantos FP e quantos trips nascem com a família ativa (e em algum momento durante o episódio). (3) PERMUTAÇÃO por DESLOCAMENTO CIRCULAR do estado da família por 1.000 deslocamentos aleatórios de pelo menos 14 dias (preserva o duty e a estrutura de blocos): a probabilidade de, ao acaso, >= o número observado de FP com a família ativa no nascimento. (4) SE a família for enriquecida nos FP: o veto no episódio (episódio desconsiderado se a família está ativa no nascimento, sem fechar a vigilância): detecção de 8, FP.
CRITÉRIOS (fixados antes). A família é ENRIQUECIDA nos FP se >= 3 dos 9 FP nascem com ela ativa, com <= 1 dos 9 trips, e p unilateral da permutação <= 0,10. O veto é ÚTIL se, sendo enriquecida, a detecção fica >= 7/8 e os FP <= 7. NOCIVO se perde detecção sem cortar FP.
EXPECTATIVAS: F_falha tem duty <= 15% ~55%; F_falha enriquecida nos FP ~35% (o PI_0319 'Falha' aparece DURANTE 3 dos FP no EXP 38); F_gas enriquecida ~25%; algum veto ÚTIL ~10%.
INTERPRETAÇÃO PREVISTA. Se a falha de instrumento declarada concentra os FP e poupa os trips, é um veto de qualidade de dado fisicamente justificado e implementável (a planta já emite o alarme), a confirmar na sombra. Se nenhuma família se enriquece, a terceira via com informação da planta não separa FP de acerto neste conjunto.
LIMITES. 9 FP e 9 episódios de trip; o diário tem ativações sem par e termina em 18/04/2026; 2 famílias fixadas por texto (a descrição por tag vem do catálogo); o p da permutação com 9 FP tem pouco poder; nada se adota sem sombra.

RESULTADO (16/10/2026) -- INVÁLIDO: o ESTADO de alarme reconstruído do diário não é utilizável (duty de 84% e 94%), então nenhum veto por estado pôde ser avaliado.
  F_falha ficou ATIVA em 84,0% do tempo vigiado e F_gas em 93,8% (o critério escrito esperava <= 15%); 7 de 9 FP e 8 de 9 trips nascem com F_falha ativa (permutação circular: P(>= 7 FP) = 0,90, média ao acaso 7,7), e 9 de 9 FP e 9 de 9 trips com F_gas ativa (P = 0,57): não separam nada porque estão ligadas quase sempre. O veto 'descartar os episódios que nascem com a família ativa' descarta quase todos (F_falha: 1/8 e 2 FP; F_gas: 0/8): artefato de duty, não resultado.
  CAUSA. O diário tem 1.056 ativações sem desativação seguinte (28% das 3.757) e os discretos CFN (PI_0319 313 vezes em 16 meses, PI_0315 233, PI_0339 86...) alternam com frequência: pareando ACT com o INACT seguinte (ou fechando em 14 dias), os estados cobrem quase todo o tempo. (A listagem de F_gas impressa saiu vazia porque usou a primeira descrição de cada par tag/condição; o cálculo usou todos os eventos.)
  Duty de F_falha <= 15% (~55%): ERREI (84%). F_falha enriquecida nos FP (~35%): não avaliável.
LEITURA. A reconstrução do ESTADO pelo diário é inviável com os dados atuais (pares ACT/INACT incompletos); o EVENTO de ativação é confiável (o fuso e a cascata dos trips bateram no EXP 38). A pergunta passa a ser feita com eventos de ativação em uma janela anterior ao disparo (EXP 39b).

Uso:  PYTHONPATH=src .venv/bin/python experiments/39_veto_por_estado_da_planta.py
"""
from __future__ import annotations
import contextlib, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48); CAP = pd.Timedelta(days=14)
LOG = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv")


def estados(d, idx, escolhe):
    """Série booleana na grade: algum estado ativo (ACT até o INACT seguinte, no máximo CAP) entre as tags/condições escolhidas."""
    out = np.zeros(len(idx), bool); x = d[escolhe(d["Descrição Alarme"].astype(str))].sort_values(["Tag Alarme", "Condição do Alarme", "t"])
    for (tag, cond), g in x.groupby(["Tag Alarme", "Condição do Alarme"]):
        for k, (t, st) in enumerate(zip(g.t, g.Status)):
            if st != "ACT/UNACK":
                continue
            fim = t + CAP; prox = g[(g.t > t) & (g.Status == "INACT/UNACK")]
            if len(prox):
                fim = min(fim, prox.t.iloc[0])
            out[idx.searchsorted(t):idx.searchsorted(fim)] = True
    return out


def main():
    d = pd.read_csv(LOG); d["t"] = pd.to_datetime(d["Data da Ocorrência"]).dt.tz_localize("UTC"); fim_log = d.t.max()
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index
    with contextlib.redirect_stdout(io.StringIO()):
        R = ci.prever({"detector": det}, proc)
    al = R["is_anomaly"].astype(bool); eps = [(a, b) for a, b in episodios_de_serie(al.loc[INICIO8:]) if a <= fim_log]; tipo = [("trip" if any(a <= t and b >= t - JAN for t in reg) else "FP") for a, b in eps]
    pub = avalia(al.reindex(al_c.index).fillna(False), vig.reindex(al_c.index).fillna(False), reg, inicio_periodo=INICIO8); print(f"PUBLICADO: {pub.det}/8, FP {pub.fp}; {len(eps)} episódios cobertos pelo diário ({tipo.count('FP')} FP, {tipo.count('trip')} trip)", flush=True)
    desc = d.drop_duplicates(["Tag Alarme", "Condição do Alarme"])[["Tag Alarme", "Condição do Alarme", "Descrição Alarme"]]
    fam = {"F_falha": lambda s: s.str.contains("Falha", case=False), "F_gas": lambda s: (s.str.contains("Gás Comb", case=False) | s.str.contains("Gás Linha", case=False)) & ~s.str.contains("Falha", case=False)}
    for nome, f in fam.items():
        sel = desc[f(desc["Descrição Alarme"].astype(str))]; print(f"\n{nome}: alarmes incluídos (tag, condição, descrição): {[(a.replace('954005_624_', ''), c, ds[:42]) for a, c, ds in zip(sel['Tag Alarme'], sel['Condição do Alarme'], sel['Descrição Alarme'])][:8]}", flush=True)
    vm = proc["mask"].to_numpy(bool) & (idx >= INICIO8) & (idx <= fim_log); nasc = np.array([idx.searchsorted(a) for a, b in eps]); rng = np.random.default_rng(20261016); n = len(idx); i_ini = idx.searchsorted(INICIO8); i_fim = idx.searchsorted(fim_log)
    for nome, f in fam.items():
        st = estados(d, idx, f); duty = float(st[vm].mean()); no_nasc = st[nasc]; durante = np.array([bool(st[idx.searchsorted(a):idx.searchsorted(b) + 1].any()) for a, b in eps])
        kfp = int(sum(1 for x, t in zip(no_nasc, tipo) if x and t == "FP")); ktr = int(sum(1 for x, t in zip(no_nasc, tipo) if x and t == "trip")); dfp = int(sum(1 for x, t in zip(durante, tipo) if x and t == "FP")); dtr = int(sum(1 for x, t in zip(durante, tipo) if x and t == "trip"))
        sh = []; L = i_fim - i_ini
        for _ in range(1000):
            off = int(rng.integers(14 * 720, L - 14 * 720)); s2 = np.concatenate([st[i_ini:i_fim][off:], st[i_ini:i_fim][:off]]); full = np.zeros(n, bool); full[i_ini:i_fim] = s2
            sh.append(int(sum(1 for i, t in zip(nasc, tipo) if full[i] and t == "FP")))
        p = float(np.mean(np.array(sh) >= kfp)); enr = kfp >= 3 and ktr <= 1 and p <= 0.10
        print(f"\n{nome}: duty {100 * duty:.1f}% do tempo vigiado | NASCEM com a família ativa: {kfp} de {tipo.count('FP')} FP e {ktr} de {tipo.count('trip')} trips | ATIVA em algum momento do episódio: {dfp} FP e {dtr} trips | permutação circular: P(>= {kfp} FP ao acaso) = {p:.3f} (média ao acaso {np.mean(sh):.2f}) -> {'ENRIQUECIDA' if enr else 'não enriquecida'}", flush=True)
        print(f"   episódios que nascem com ela ativa: {[f'{a:%Y-%m-%d} {t}' for (a, b), t, x in zip(eps, tipo, no_nasc) if x]}", flush=True)
        al2 = al.copy()
        for (a, b), x in zip(eps, no_nasc):
            if x:
                al2.loc[a:b] = False
        r = avalia(al2.reindex(al_c.index).fillna(False), vig.reindex(al_c.index).fillna(False), reg, inicio_periodo=INICIO8)
        print(f"   veto no episódio (descartar os que nascem com a família ativa): {r.det}/8, FP {r.fp} (publicado {pub.det}/8, FP {pub.fp}) -> {'ÚTIL' if enr and r.det >= 7 and r.fp <= 7 else 'NOCIVO' if r.det < pub.det and r.fp >= pub.fp else 'NEUTRO/INCONCLUSIVO'}", flush=True)


if __name__ == "__main__":
    main()
