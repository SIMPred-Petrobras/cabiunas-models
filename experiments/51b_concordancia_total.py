#!/usr/bin/env python3
"""EXP 51b — CONCORDÂNCIA TOTAL NO NASCIMENTO: o teste que o EXP 51 não chegou a fazer (correção de desenho).

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUÊ. No EXP 51 a hipótese P3 ficou NÃO TESTADA por erro meu: a condição escolhida (exigir duração >= 240 min do episódio que nasce com os 4 canais acesos) não filtra nada, porque os 4 episódios com k=4 duram de 17,0 h a 135,1 h. A hipótese em si continua de pé e é a única do diagnóstico de hoje com um mecanismo físico independente:
  MEDIDO: nos 18 episódios desde 01/02/2025, o nº de canais do nível A acesos no NASCIMENTO separa — k=2: 3 trips e 4 FP; k=3: 6 trips e 1 FP; k=4: 0 trips e 4 FP (225 h = 15% das horas de alarme). p nominal 0,041 (hipergeométrica, unilateral), PÓS-HOC.
  MECANISMO: uma falha incipiente é LOCAL (começa num subsistema e se espalha); os 9 episódios de trip nascem com 2 ou 3 canais, nunca com 4. Quatro canais simultâneos são assinatura de MUDANÇA DE ESTADO. O próprio pacote já afirma isso para um caso particular — o comentário da máscara em `cabiunas_inference.py` diz que o transiente de religamento "imita degradação em todos os quatro canais de uma vez" — e a memória `drift-abrupto-pos-manutencao` registra "normal novo em 14 tags de uma vez" depois da manutenção. A proposta é generalizar o que a máscara já faz numa janela fixa de 6 h, usando a simultaneidade em vez do relógio.

O QUE SE FAZ. Mesmo caminho do EXP 51 (o `proc` do cache passa por `ci.prever` para obter os acendimentos, a força e o vigiado; o voto e o pós-processamento são recalculados fora do pacote, com V3 provando igualdade exata com `prever`). Duas CONDIÇÕES, desta vez com efeito garantido:
  VETO   — o episódio que nasce com k canais acesos é suprimido por inteiro.
  ATRASO — o episódio não nasce enquanto k estiver no valor alvo; o nascimento é empurrado para o primeiro instante com k menor (não remove o episódio, atrasa o início).
  Cada uma nas 2 POSIÇÕES do portão de força P1/P1b: ANTES do refratário e DEPOIS. Alvo da hipótese: k=4. CONTROLES NEGATIVOS pré-registrados: a MESMA condição em k=3 (que contém 6 dos 9 episódios de trip) e em k=2 (3 trips, 4 FP). Se "concordância total" fosse só ruído, os três k custariam o mesmo.
MEDIDAS: detecção de 8, início, FP, FP/mês, horas totais e fora, lista nominal dos episódios removidos ou atrasados (a Regra C não limita duração: um episódio que cresce até encostar numa parada vira TP e a "carga" cai), placebo (+45 a +225 d) e vizinhança (k do nível A e do B x 0,90 0,95 1,00 1,05 1,10).
CRITÉRIOS (fixados antes). GANHO = FP <= 6 E detecção >= 6/8 E horas <= 1.457 E acima do placebo E >= 3 dos 5 pontos de vizinhança com detecção >= 6 e FP <= 9 E o controle negativo em k=3 custando MAIS (menos detecção ou mais FP) que o k=4. PIOR = detecção <= 5/8 OU FP >= 12. EQUIVALENTE no resto. META DO USUÁRIO à parte: FP/mês <= 0,55 com detecção >= 5/8.
EXPECTATIVAS: o VETO em k=4 remove os 4 episódios e leva o FP a 5 (~70%); mas a posição "antes" faz nascerem episódios novos, como no portão de força P1 (~60%); o GANHO completo, com vizinhança e controle negativo, ~30%; o controle negativo k=3 custa >= 1 detecção (~75%); o ATRASO não muda o FP (o episódio continua existindo) e só move o início (~80%); alguma variante atinge a meta do usuário (~35%).
LIMITES. 8 eventos; a regra é PÓS-HOC (p nominal 0,041 em 18 episódios) e a casa tem histórico de padrões assim morrendo no teste (E192, consenso de vizinhança, piso de força, portão de força P1b); 4 episódios é amostra pequena para qualquer conclusão; nada se adota sem a sombra.

RESULTADO (18/10/2026; rodado localmente; commit do pré-registro f7093c7). O MELHOR PONTO DA SÉRIE — e ainda assim uma HIPÓTESE, não um achado, pelas razões do fim deste bloco.
  V3 PASSA (o controle reimplementado dá 7/8, início 6, FP 9, 1.457 h, 407 h fora).
  VETO, POSIÇÃO "DEPOIS", k=4:  7/8 · início 6 · FP 5 · 0,448/mês · 1.232 h · 183 h fora · placebo 1,0/8
  CONTROLE:                      7/8 · início 6 · FP 9 · 0,807/mês · 1.457 h · 407 h fora · placebo 1,3/8
  Ou seja: -4 FP (-44%), -0,359 FP/mês, -225 h de alarme e -224 h fora das janelas, com a MESMA detecção e o MESMO início. Atinge a meta do usuário (FP/mês <= 0,55 com detecção >= 5/8), a primeira configuração desta frente a consegui-lo.
  1) OS 4 EPISÓDIOS REMOVIDOS SÃO TODOS FALSOS POSITIVOS, nenhum trip tocado: 31/03/2025 (17,0 h), 23/08/2025 (20,1 h), 15/10/2025 (135,1 h) e 02/04/2026 (52,4 h). O de 15/10/2025 é um dos dois FP longos da série.
  2) OS CONTROLES NEGATIVOS SEPARAM, e na direção prevista pelo mecanismo: o mesmo veto em k=3 leva a detecção de 7/8 para 2/8 (toca 15 episódios); em k=2, para 5/8 (toca 12). Só o k=4 remove FP sem tocar em trip — consistente com "falha incipiente é local, modo comum é mudança de estado".
  3) A POSIÇÃO IMPORTA, como no portão de força P1/P1b: "antes" do refratário toca 8 episódios em vez de 4 e dá FP 7 (contra 5), porque remover o episódio antes do refratário deixa outros nascerem. "Depois" é a posição certa.
  4) O ATRASO (deixar nascer só quando k cai) NÃO SERVE: "depois" dá FP 9 e 1.447 h — praticamente o controle, porque o episódio continua existindo e só o início se move (0,1 a 7,8 h); "antes" PIORA (FP 11).
  5) O CRITÉRIO DE VIZINHANÇA QUE PRÉ-REGISTREI ERA INVÁLIDO e isso é erro meu: exigi ">= 3 dos 5 fatores com detecção >= 6 e FP <= 9", mas O PRÓPRIO CONTROLE só cumpre em 1 dos 5 (0,90: 7/20 · 0,95: 7/15 · 1,00: 7/9 · 1,05: 6/11 · 1,10: 5/12). Um critério que a configuração em produção não passa não discrimina nada. A comparação PAREADA, que é a válida, favorece o veto em toda a vizinhança: 0,90 7/20 -> 6/16 (-4 FP, -1 det) · 0,95 7/15 -> 7/11 (-4) · 1,00 7/9 -> 7/5 (-4) · 1,05 6/11 -> 6/10 (-1) · 1,10 5/12 -> 5/10 (-2). Reduz FP nos 5 pontos e mantém a detecção em 4 dos 5.
  POR QUE ISTO AINDA É HIPÓTESE, E NÃO ACHADO: (a) a regra é PÓS-HOC — k=4 foi escolhido depois de ver a tabela dos 18 episódios; (b) o p nominal é 0,041 (hipergeométrica: 4 de 4 removidos serem FP, com 9 FP em 18 episódios), e foram testadas 12 combinações (3 valores de k x 2 modos x 2 posições), o que sob Bonferroni dá ~0,49 — não significativo; (c) 4 episódios é amostra pequena; (d) a casa tem histórico de padrões assim morrendo no teste (E192, consenso de vizinhança, piso de força, portão de força P1b, piso de ruído da régua). O que distingue este dos anteriores é ter um mecanismo físico declarado ANTES (o próprio pacote trata o modo comum na janela de blackout) e controles negativos que separam. O teste decisivo é PROSPECTIVO: entra no sombra.py como variante, não no contrato.
PLACAR: ACERTEI: o veto em k=4 remove os 4 episódios e leva o FP a 5 (~70%, acertei o número exato); a posição "antes" faz nascerem episódios (~60%); o controle negativo k=3 custa >= 1 detecção (~75%; custou 5); o atraso não muda o FP e só move o início (~80%); alguma variante atinge a meta do usuário (~35%). ERREI: o critério de vizinhança que escrevi era inválido, então a classificação "GANHO" ficou inalcançável por construção (~30% de GANHO completo) — a falha é de desenho meu, não do resultado; previ que o atraso em "antes" seria neutro e ele PIOROU (FP 11).

Uso:  PYTHONPATH=src .venv/bin/python experiments/51b_concordancia_total.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E51 = _carrega("exp51", "51_estrutura_do_voto.py")
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import episodios_de_serie, JANELA
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG

ROOT = E51.ROOT; INICIO8 = E51.INICIO8; SIN = E51.SIN; FATORES = E51.FATORES; GL = E51.GL


def pos_processa_k(voto, forca, det, k_serie, modo=None, kalvo=None, posicao=None):
    """Como `E51.pos_processa`, mas com VETO ou ATRASO por nº de canais no nascimento."""
    ci = GL["ci"]; idx = voto.index
    f = forca.fillna(0.0).to_numpy(); v = voto.to_numpy().copy()
    n_idade = int(det["escalada_idade_h"] * ci.POR_HORA); n_gap = int(ci.GAP_EPISODIO_H * ci.POR_HORA) + 1
    dentro, inicio, ja_forte = False, 0, False
    for i in range(len(v)):
        if not v[i]:
            dentro, ja_forte = False, False; continue
        agora_forte = f[i] > det["escalada_abs"]
        if not dentro:
            dentro, inicio, ja_forte = True, i, agora_forte; continue
        if agora_forte and not ja_forte and (i - inicio) >= n_idade:
            v[max(inicio + 1, i - n_gap):i] = False; inicio = i
        ja_forte = agora_forte
    voto = pd.Series(v, index=idx)
    tocados = []

    def aplica(eps):
        if modo is None:
            return eps
        out = []
        for a, b in eps:
            if int(k_serie.loc[a]) != kalvo:
                out.append((a, b)); continue
            if modo == "veto":
                tocados.append((a, b, "removido")); continue
            w = k_serie.loc[a:b]; menor = w[w < kalvo]
            if len(menor) == 0:
                tocados.append((a, b, "removido (nunca cai)")); continue
            novo = menor.index[0]
            if novo >= b:
                tocados.append((a, b, "removido (cai no fim)")); continue
            tocados.append((a, b, f"atrasado {((novo - a).total_seconds() / 3600):.1f} h")); out.append((novo, b))
        return out

    eps_voto = [tuple(x) for x in ci._episodios(voto)]
    if posicao == "antes":
        eps_voto = aplica(eps_voto)
    al = pd.Series(False, index=idx); bloq = ini_bloq = None; fortes = []
    for a, b in eps_voto:
        forte = float(forca.loc[a:b].max()) > det["escalada_abs"]
        velho = (ini_bloq is not None and (a - ini_bloq).total_seconds() / 3600 >= det["escalada_idade_h"])
        if bloq is not None and a <= bloq and not (forte and velho):
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=det["refratario_h"]); ini_bloq = a
        if forte:
            fortes.append((a, b))
    eps_al = [tuple(x) for x in ci._episodios(al)]
    if posicao == "depois":
        eps_al = aplica(eps_al)
    final = pd.Series(False, index=idx)
    for a, b in eps_al:
        dur = (b - a).total_seconds() / 60 + 2
        e_forte = any(x >= a and y <= b for x, y in fortes)
        if dur >= det["duracao_min"] or (e_forte and dur >= det["duracao_min_forte"]):
            final.loc[a:b] = True
    return final, tocados


def alarme(det, fator=1.0, modo=None, kalvo=None, posicao=None):
    ci = GL["ci"]; proc = GL["proc"]; mask = proc["mask"].astype(bool); reset = proc["reset"].to_numpy()
    A = {c: ci._acende(proc[c], det["base"][c] * det["k_nivel_a"][c] * fator, mask, reset, det["kappa"], det["h_cusum"], det["carga_reset"]) for c in SIN}
    B = {c: ci._acende(proc[c], det["base"][c] * det["k_nivel_b"][c] * fator, mask, reset, det["kappa"], det["h_cusum"], det["carga_reset"]) for c in SIN}
    voto = (E51.regra_atual_A(A) & mask) | (E51.regra_atual_B(B) & mask)
    forca = pd.concat([proc[c].where(mask) / (det["base"][c] * det["k_nivel_b"][c] * fator) for c in SIN], axis=1).max(axis=1)
    k_serie = sum(A[c].astype(int) for c in SIN)
    return pos_processa_k(voto, forca, det, k_serie, modo=modo, kalvo=kalvo, posicao=posicao)


def main():
    ci = _ci(); GL["ci"] = ci
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in E51.SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV)
    base_al, _ = alarme(det); L0 = E51.linha(base_al)
    print(f"V3 — controle reimplementado: {L0['det']}/8, início {L0['inicio']}, FP {L0['fp']} ({L0['fp_mes']:.3f}/mês), {L0['horas']:.0f} h, {L0['fora']:.0f} h fora", flush=True)
    if not (L0["det"] == 7 and L0["fp"] == 9):
        print("   V3 FALHA — parando.", flush=True); return
    print("   V3 PASSA\n", flush=True)
    res = {}
    print("CONDIÇÕES (detecção/8 | início | FP | FP/mês | horas totais | h fora | placebo | episódios tocados)", flush=True)
    for modo in ("veto", "atraso"):
        for posicao in ("antes", "depois"):
            for kalvo in (4, 3, 2):
                al, toc = alarme(det, modo=modo, kalvo=kalvo, posicao=posicao); L = E51.linha(al); res[(modo, posicao, kalvo)] = (L, toc)
                tag = "HIPÓTESE" if kalvo == 4 else "controle-"
                print(f"   {modo:6s} {posicao:6s} k={kalvo} {tag:9s} | {L['det']}/8 | {L['inicio']} | FP {L['fp']:3d} | {L['fp_mes']:.3f} | {L['horas']:6.0f} h | {L['fora']:5.0f} h | {L['placebo']:.1f} | {len(toc)}", flush=True)
    print("\nEPISÓDIOS TOCADOS pela hipótese (k=4), por modo e posição:", flush=True)
    for modo in ("veto", "atraso"):
        for posicao in ("antes", "depois"):
            toc = res[(modo, posicao, 4)][1]
            for a, b, o in toc:
                assoc = [t for t in reg if a <= t and b >= t - JANELA]
                print(f"   {modo:6s} {posicao:6s} {a.strftime('%d/%m/%Y %H:%M')} ({(b - a).total_seconds() / 3600:6.1f} h) {'TRIP ' + assoc[0].strftime('%d/%m') if assoc else 'FP':12s} -> {o}", flush=True)
    print("\nVIZINHANÇA (fator nos k; detecção/FP; * = detecção >= 6 e FP <= 9)", flush=True)
    for modo in ("veto", "atraso"):
        for posicao in ("antes", "depois"):
            viz = []
            for f_ in FATORES:
                al, _ = alarme(det, fator=f_, modo=modo, kalvo=4, posicao=posicao); Lv = E51.linha(al)
                viz.append(f"{f_}:{Lv['det']}/{Lv['fp']}" + ("*" if Lv["det"] >= 6 and Lv["fp"] <= 9 else ""))
            print(f"   {modo:6s} {posicao:6s} k=4 | " + " ".join(viz), flush=True)
    print("\nCLASSIFICAÇÃO (GANHO = FP <= 6, det >= 6, horas <= 1457, acima do placebo, >= 3/5 na vizinhança e k=3 custando mais)", flush=True)
    for (modo, posicao, kalvo), (L, _) in res.items():
        if kalvo != 4:
            continue
        L3 = res[(modo, posicao, 3)][0]
        pior_neg = (L3["det"] < L["det"]) or (L3["fp"] > L["fp"])
        viz_ok = sum(1 for f_ in FATORES if (lambda x: x["det"] >= 6 and x["fp"] <= 9)(E51.linha(alarme(det, fator=f_, modo=modo, kalvo=4, posicao=posicao)[0]))) >= 3
        cl = "GANHO" if (L["fp"] <= 6 and L["det"] >= 6 and L["horas"] <= 1457 and L["det"] > L["placebo"] and viz_ok and pior_neg) else "PIOR" if (L["det"] <= 5 or L["fp"] >= 12) else "EQUIVALENTE"
        meta = L["fp_mes"] <= 0.55 and L["det"] >= 5
        print(f"   {modo:6s} {posicao:6s} -> {cl} | vizinhança {'ok' if viz_ok else 'reprova'} | controle k=3 {'custa mais' if pior_neg else 'NÃO custa mais'}{'  | META DO USUÁRIO' if meta else ''}", flush=True)
    pd.DataFrame([dict(modo=m, posicao=p, k=k, tocados=len(t), **L) for (m, p, k), (L, t) in res.items()]).to_csv(ROOT / "exp51b_resultados.csv", index=False)


if __name__ == "__main__":
    main()
