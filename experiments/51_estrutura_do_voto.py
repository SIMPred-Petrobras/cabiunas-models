#!/usr/bin/env python3
"""EXP 51 — ESTRUTURA DO VOTO: canal-portão contra canal-evidência (P1) e concordância total como condição de nascimento (P3).

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUÉ. A busca por modelo fechou (EXP 44-50: 0 ganho em 44 sinais). Sobra a regra de voto, e o diagnóstico de hoje mostra que ela viola a premissa que faz um voto k-de-n reduzir FP — a de que os votantes trazem evidência:
  (i) PODER DISCRIMINANTE por canal (fração do tempo aceso nas 48 h pré-trip contra o resto, nível A): t 58,5%/33,9% = 1,73x; vb 82,0%/58,6% = 1,40x; p 33,6%/35,2% = 0,95x; sp 57,5%/59,7% = 0,96x. Sob LOEO do próprio diagnóstico o lift é estável em t (1,51-2,03) e vb (1,35-1,58) e cruza 1 em p (0,62-1,12) e sp (0,84-1,13). Coerente com a razão-vs-acaso já registrada no DETECTOR_V2.md §7 (sp 0,55x e p 0,83x, ABAIXO do acaso). O M4 (`v3_independencia_dos_votos.py`) mediu INDEPENDÊNCIA (n_efetivo 3,54 de 4) e concluiu "nenhuma alteração do voto se propõe"; não mediu poder discriminante. Dois votantes podem ser independentes e ambos inúteis.
  (ii) O voto "3 de 4 simétrico" já é assimétrico na prática: 93,8% dos instantes de voto A incluem sp E vb; só 2,6% dispensam o sp e 3,6% o vb. sp e vb ficam acesos ~60% do tempo e preenchem o quórum sem trazer evidência.
  (iii) CONCORDÂNCIA TOTAL nasce FP: nos 18 episódios desde 01/02/2025, k=2 no nascimento dá 3 trips e 4 FP; k=3 dá 6 trips e 1 FP; k=4 dá 0 trips e 4 FP (225 h = 15% das horas de alarme; p nominal 0,041, hipergeométrica, PÓS-HOC). Mecanismo independente: o comentário da máscara em `cabiunas_inference.py` diz que o transiente de religamento "imita degradação em todos os quatro canais de uma vez", e a memória `drift-abrupto-pos-manutencao` registra "normal novo em 14 tags de uma vez".
PRECEDENTE. `scripts/pdm_fisico/_tmp_voto_obrigatorio.py` varreu as 15 combinações de canal obrigatório x quórum {2,3}, MAS: arquivo temporário, sem log nem CSV salvos, no ponto v1 (limiar único, KB=1,7, 31/08), ANTES do gatilho de dois níveis. Não é veredito. Fica registrado como precedente.

O QUE SE FAZ. O `proc` do cache (`sinais_v2_8ev.parquet`) passa por `ci.prever` UMA vez para obter os acendimentos por canal (`a_{c}`, `b_{c}`), a força e o vigiado — tudo com o CUSUM e a máscara de produção. O voto é então recalculado fora do `prever` pela regra da variante, e a escalada por idade, o refratário de 72 h e a duração mínima são reaplicados COPIANDO a lógica do pacote (verificação V3 abaixo). O pacote de produção NÃO é editado.
  REGRAS (fixadas aqui). Nível A (hoje: >= 3 de 4) e nível B (hoje: >= 2 de 4 com portão sp|vb). Cada regra é aplicada em 3 POSIÇÕES: só no A, só no B, nos dois (os FP vêm do B — EXP 36: 9 de 9 FP têm B ligado, 4 só por ele —, então mexer só no A pode não reduzir FP).
    a  "vb obrigatório + >= 2 de {t,p,sp}"      (atribuição por LIFT: vb é evidência)
    b  ">= 1 de {t,vb} + >= 2 no total"          (atribuição por LIFT: t e vb são a evidência)
    c  "vb obrigatório + >= 1 de {t,p}"          (mais frouxa, para mapear a fronteira)
    d  "sp E vb + >= 1 de {t,p}"                 (atribuição por TEMPO ACESO, SEM rótulos: portão = os acesos > 50%)
  PLACEBOS DE DESENHO (mesma forma, papéis invertidos; se a estrutura não importar, custam o mesmo):
    a' "sp obrigatório + >= 2 de {t,p,vb}"   ·   b' ">= 1 de {p,sp} + >= 2 no total"   ·   c' "p obrigatório + >= 1 de {sp,vb}"
  P3 (concordância total): episódio que NASCE com os 4 canais do nível A acesos exige duração >= 240 min (o dobro do atual). Nas duas posições do portão de força P1/P1b: ANTES do refratário (P3a) e DEPOIS (P3b) — o P1 reprovou justamente porque remover o episódio antes do refratário faz nascerem 17 episódios novos. CONTROLE NEGATIVO: a mesma regra aplicada a k=3 (que contém 6 dos 9 episódios de trip); se "k alto" fosse ruído, as duas custariam o mesmo.
VERIFICAÇÕES (todas precisam passar).
  V1 — `ci.prever` sobre o cache reproduz o controle: 7/8, FP 9 (+-1), 1.457 h, e coincide >= 97% com a série do ledger nos instantes vigiados.
  V3 — a regra ATUAL reimplementada fora do `prever` devolve EXATAMENTE a mesma série de alarme (diferença 0 instantes) e os mesmos 18 episódios.
MEDIDAS: detecção de 8, início na janela, FP, FP/mês, horas de alarme TOTAIS e fora das janelas, lista dos episódios que mudam (a Regra C não limita duração: um episódio que cresce até encostar numa parada vira TP e a "carga" cai — por isso as horas totais vão em toda tabela); placebo por deslocamento dos eventos (+45 a +225 d); vizinhança (k do nível A e do B multiplicados por 0,9, 0,95, 1,0, 1,05, 1,1); LOEO (para cada evento deixado de fora, a regra é escolhida pelos 7 restantes — maior detecção, desempate por menor FP — e avaliada no evento de fora).
CRITÉRIOS (fixados antes). GANHO = FP <= 6 (<= 0,54/mês) E detecção >= 6/8 E horas totais <= 1.457 E acima do placebo da própria regra E >= 3 dos 5 pontos de vizinhança mantendo detecção >= 6 e FP <= 9 E, para as regras informadas por lift (a, b, c), LOEO >= 6/8. PIOR = detecção <= 5/8 OU FP >= 12. EQUIVALENTE no resto. META DO USUÁRIO, relatada à parte: FP/mês <= 0,55 com detecção >= 5/8.
  São ~25 variantes: qualquer GANHO é HIPÓTESE e só vale com o placebo de desenho CUSTANDO MAIS que a regra correspondente (senão o que ganhou foi o acaso da forma, não a atribuição de papéis).
EXPECTATIVAS: algum GANHO ~20%; a regra (d), sem rótulos, GANHO ~10%; o placebo de desenho custa igual ou menos que a regra correspondente em pelo menos uma delas ~45% (seria o sinal de que a atribuição não importa); a posição "só B" reduz mais FP que "só A" ~70%; P3a reprova por episódios novos (como o P1) ~65%; P3b corta >= 2 FP ~55%; o controle negativo (k=3) custa MAIS detecção que o k=4 ~80%; nenhuma variante atinge a meta do usuário ~65%; LOEO >= 6/8 em alguma regra ~40%.
INTERPRETAÇÃO PREVISTA. Se nenhuma variante ganha, a estrutura do voto fecha junto com o modelo, e o limite passa a ser inteiramente informação sobre o desfecho (tags de nível, operação). Se alguma ganha e o placebo de desenho custa mais, vira HIPÓTESE para a sombra, com pré-registro próprio.
LIMITES. 8 eventos (+-2; o FP do controle oscila +-3 na vizinhança); as regras (a), (b) e (c) são informadas pelo lift medido nos mesmos 8 eventos (daí o LOEO); a regra (a) sacrifica o 29/04/2025 por construção (vb aceso em 14% da janela), teto realista 7/8; P3 é pós-hoc com p nominal 0,041 em 18 episódios; o nível B mantém o portão sp|vb quando a regra não o substitui; nada se adota sem a sombra.

RESULTADO (18/10/2026; rodado localmente, 25 variantes; commit do pré-registro f2280fc). NENHUM GANHO; nenhuma variante atinge a meta do usuário (melhor FP/mês 0,717 contra o teto de 0,55). V1 PASSA (7/8, FP 9, 1.457 h, coincidência 100,00% com o ledger) e V3 PASSA com 0 instantes diferentes do `prever` e 18 episódios contra 18 — a reimplementação do voto fora do pacote é exata.
  CLASSIFICAÇÃO: 0 GANHO, 8 EQUIVALENTE, 17 PIOR.
  1) O MELHOR PONTO É UM PICO DE UM SÓ PASSO. A regra (b) ">= 1 de {t,vb} + >= 2 no total" aplicada SÓ NO NÍVEL B dá 7/8, início 6, FP 8, 0,717/mês, 1.423 h, 373 h fora — melhor que o controle (7/8, início 6, FP 9, 0,807/mês, 1.457 h, 407 h) em TODOS os eixos, mas por UM falso positivo. E a vizinhança reprova: dos 5 fatores (0,90 0,95 1,00 1,05 1,10) só o 1,00 cumpre detecção >= 6 e FP <= 9 (7/17, 7/13, **7/8**, 6/11, 5/12) — o critério exigia 3 de 5. É a assinatura de sobreajuste que o EXP 24 já havia medido para os k.
  2) A POSIÇÃO IMPORTA MAIS QUE A REGRA, e confirma o EXP 36 (os FP vêm do nível B): aplicadas só no B as 7 regras ficam entre 8 e 12 FP; só no A, entre 9 e 22. Nenhuma regra no nível A reduz FP.
  3) O PLACEBO DE DESENHO SALVA A ESTRUTURA EM 2 DAS 3 FAMÍLIAS, MAS NÃO NA PRIMEIRA. Para (b) e (c) a regra custa menos que o seu placebo nas 3 posições (b só B: FP 8/det 7 contra o placebo FP 10/det 5). Para (a) o placebo é IGUAL OU MELHOR nas 3 posições (a só A: FP 11/det 7 contra placebo FP 9/det 7) — ou seja, na família (a) o que decide é a forma da regra, não a atribuição de papéis. A atribuição SEM rótulos, (d) "sp E vb + >= 1 de {t,p}", é indistinguível de (a) nas três posições (7/8 FP 11, 4/8 FP 9, 5/8 FP 11).
  4) UM PONTO INTERESSANTE FORA DO ALVO: (c) "vb obrigatório + >= 1 de {t,p}" no nível A dá 8/8 com início 7 — a única configuração desta série que detecta os OITO eventos, inclusive o 04/11/2025, e com o melhor início já medido. Custa 17 FP (1,525/mês) e 1.752 h. Não serve como alarme; é a fronteira do lado oposto, e só teria uso como nível de atenção (a mesma leitura do vb com Ridge no EXP 50).
  5) LOEO = 7/8, IGUAL AO CONTROLE: escolher a regra pelos 7 eventos de treino não ganha nada. Em 7 das 8 dobras a escolhida é (c) só A, e a dobra que perde é sempre o 04/11/2025 — o mesmo evento que cai em `loeo-e-frageil-a-desempate`.
  6) ERRO DE DESENHO MEU NA P3, REGISTRADO: a condição escolhida (exigir duração >= 240 min do episódio que nasce com k=4) NÃO FILTRA NADA — "antes" e "depois" devolvem exatamente o controle (7/8, FP 9, 1.457 h, 407 h). Os 4 episódios que nascem com k=4 duram de 17,0 h a 135,1 h, todos muito acima de 4 h. A hipótese P3 continua NÃO TESTADA; o que foi testado é uma condição inócua. O controle negativo, porém, informa: k=3 na posição "antes" PIORA (FP 9 -> 12, 1.541 h), reproduzindo o mecanismo do portão de força P1 (remover episódio antes do refratário faz nascerem outros), enquanto "depois" é inócuo nos dois k.
  CONCLUSÃO: reorganizar quem vota não é alavanca. O único ponto que domina o controle o faz por 1 FP e não sobrevive a um passo de 5% nos k. Junto com os EXP 44-50, fecha também a estrutura do voto; o que resta é informação sobre o desfecho.
PLACAR: ACERTEI: nenhum GANHO (~20% de haver); a regra (d) sem rótulos não ganha (~10%); o placebo de desenho custa igual ou menos em ao menos uma família (~45%; foi a família (a), nas 3 posições); "só B" reduz mais FP que "só A" (~70%); nenhuma variante atinge a meta do usuário (~65%); LOEO >= 6/8 (~40%; deu 7/8, mas igual ao controle). ERREI: previ que P3b cortaria >= 2 FP (~55%) — cortou ZERO, por erro meu de desenho; previ que o controle negativo k=3 custaria MAIS DETECÇÃO que o k=4 (~80%) — a detecção ficou 7/8 nos dois e o que mudou foi o FP (12 contra 9, só na posição "antes"); previ que P3a reprovaria por episódios novos (~65%) — reprovou só com k=3, e com k=4 não teve efeito nenhum.

Uso:  PYTHONPATH=src .venv/bin/python experiments/51_estrutura_do_voto.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, itertools, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py")
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG

ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; SIN = ["t", "p", "sp", "vb"]
FATORES = (0.9, 0.95, 1.0, 1.05, 1.1); GL = {}


# ───────────────────────── regras de voto ─────────────────────────
def _conta(X, canais):
    return sum(X[c].astype(int) for c in canais)


def regra_atual_A(X):
    return _conta(X, SIN) >= 3


def regra_atual_B(X):
    return (_conta(X, SIN) >= 2) & (X["sp"] | X["vb"])


def _obrig(obrigatorios, resto, k):
    """Todos os `obrigatorios` acesos E pelo menos `k` do `resto`."""
    def f(X):
        o = X[obrigatorios[0]].copy()
        for c in obrigatorios[1:]:
            o = o & X[c]
        return o & (_conta(X, resto) >= k)
    return f


def _um_de_mais_total(conjunto, total):
    """Pelo menos um do `conjunto` E pelo menos `total` no geral."""
    def f(X):
        o = X[conjunto[0]].copy()
        for c in conjunto[1:]:
            o = o | X[c]
        return o & (_conta(X, SIN) >= total)
    return f


REGRAS = {
    "a  vb + >=2 de {t,p,sp}": _obrig(["vb"], ["t", "p", "sp"], 2),
    "b  >=1 de {t,vb} + >=2": _um_de_mais_total(["t", "vb"], 2),
    "c  vb + >=1 de {t,p}": _obrig(["vb"], ["t", "p"], 1),
    "d  sp+vb + >=1 de {t,p}": _obrig(["sp", "vb"], ["t", "p"], 1),
    "a' sp + >=2 de {t,p,vb}": _obrig(["sp"], ["t", "p", "vb"], 2),
    "b' >=1 de {p,sp} + >=2": _um_de_mais_total(["p", "sp"], 2),
    "c' p + >=1 de {sp,vb}": _obrig(["p"], ["sp", "vb"], 1),
}
PLACEBO_DE = {"a  vb + >=2 de {t,p,sp}": "a' sp + >=2 de {t,p,vb}", "b  >=1 de {t,vb} + >=2": "b' >=1 de {p,sp} + >=2", "c  vb + >=1 de {t,p}": "c' p + >=1 de {sp,vb}"}
POR_LIFT = ("a  vb + >=2 de {t,p,sp}", "b  >=1 de {t,vb} + >=2", "c  vb + >=1 de {t,p}")


# ──────────────────── pós-processamento (cópia fiel do pacote) ────────────────────
def pos_processa(voto, forca, det, k_nasc=None, p3=None):
    """Escalada por idade, refratário e duração mínima — a lógica de `prever`, linhas 638-685.

    `p3`: (posicao, k_alvo, dur_min_exigida). posicao 'antes' aplica a exigência aos episódios do VOTO
    (antes do refratário, como o portão de força P1); 'depois', aos episódios já filtrados (como o P1b).
    """
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

    def exige(eps):
        """Aplica a condição do P3 a uma lista de episódios, devolvendo os que sobrevivem."""
        if p3 is None:
            return eps
        pos, kalvo, dur_min = p3; out = []
        for a, b in eps:
            if int(k_nasc.loc[a]) == kalvo and ((b - a).total_seconds() / 60 + 2) < dur_min:
                continue
            out.append((a, b))
        return out

    eps_voto = [tuple(x) for x in ci._episodios(voto)]
    if p3 is not None and p3[0] == "antes":
        eps_voto = exige(eps_voto)
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
    if p3 is not None and p3[0] == "depois":
        eps_al = exige(eps_al)
    final = pd.Series(False, index=idx)
    for a, b in eps_al:
        dur = (b - a).total_seconds() / 60 + 2
        e_forte = any(x >= a and y <= b for x, y in fortes)
        if dur >= det["duracao_min"] or (e_forte and dur >= det["duracao_min_forte"]):
            final.loc[a:b] = True
    return final


def alarme_de_regra(rA, rB, det, fator=1.0, p3=None):
    """Recalcula A/B com os k multiplicados por `fator`, aplica as regras e o pós-processamento."""
    ci = GL["ci"]; proc = GL["proc"]; mask = proc["mask"].astype(bool); reset = proc["reset"].to_numpy()
    A = {c: ci._acende(proc[c], det["base"][c] * det["k_nivel_a"][c] * fator, mask, reset, det["kappa"], det["h_cusum"], det["carga_reset"]) for c in SIN}
    B = {c: ci._acende(proc[c], det["base"][c] * det["k_nivel_b"][c] * fator, mask, reset, det["kappa"], det["h_cusum"], det["carga_reset"]) for c in SIN}
    voto = ((rA(A) & mask) | (rB(B) & mask))
    forca = pd.concat([proc[c].where(mask) / (det["base"][c] * det["k_nivel_b"][c] * fator) for c in SIN], axis=1).max(axis=1)
    k_nasc = sum(A[c].astype(int) for c in SIN)
    return pos_processa(voto, forca, det, k_nasc=k_nasc, p3=p3)


def mede(al, evs=None):
    al_c = GL["al_c"]; a = al.reindex(al_c.index).fillna(False)
    return avalia(a, GL["vig_c"], evs if evs is not None else GL["EV"]["real"], inicio_periodo=INICIO8)


def linha(al):
    r = mede(al); pl = []
    for k, ev in GL["EV"].items():
        if k == "real" or not ev:
            continue
        pl.append(avalia(al.reindex(GL["al_c"].index).fillna(False), GL["vig_c"], ev, inicio_periodo=INICIO8).det / len(ev))
    return dict(det=r.det, inicio=r.inicio, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora, placebo=float(np.median(pl)) * 8 if pl else np.nan)


def main():
    ci = _ci(); GL["ci"] = ci
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV)
    print(f"EXP 51: {len(REGRAS)} regras x 3 posições + P3 | eventos {({k: len(v) for k, v in EV.items()})}", flush=True)

    # ── V1 ──
    res0 = ci.prever({"detector": det}, proc); al0 = res0["is_anomaly"].astype(bool); r0 = mede(al0)
    v = vig_c.loc[INICIO8:]; conc = float((al0.reindex(al_c.index).fillna(False).loc[INICIO8:][v] == al_c.loc[INICIO8:][v]).mean())
    print(f"\nV1 — `prever` sobre o cache: {r0.det}/8 | início {r0.inicio} | FP {r0.fp} | {r0.fp_mes:.3f}/mês | {r0.horas_alarme:.0f} h | {r0.horas_alarme_fora:.0f} h fora | coincidência com o ledger {100 * conc:.2f}%", flush=True)
    v1 = r0.det == 7 and abs(r0.fp - 9) <= 1 and conc >= 0.97; print(f"   V1 {'PASSA' if v1 else 'FALHA'}", flush=True)

    # ── V3: a regra atual reimplementada ──
    alA = alarme_de_regra(regra_atual_A, regra_atual_B, det)
    j = alA.index.intersection(al0.index); dif = int((alA.loc[j] != al0.loc[j]).sum())
    e0 = len(episodios_de_serie(al0.loc[INICIO8:])); e1 = len(episodios_de_serie(alA.loc[INICIO8:]))
    print(f"\nV3 — regra atual reimplementada: {dif} instantes diferentes de `prever` | episódios {e1} contra {e0}", flush=True)
    v3 = dif == 0 and e0 == e1; print(f"   V3 {'PASSA' if v3 else 'FALHA'}", flush=True)
    if not (v1 and v3):
        print("\n   VERIFICAÇÃO FALHOU — o resto não vale. Parando.", flush=True); return
    base = linha(alA); print(f"\nCONTROLE: {base['det']}/8, início {base['inicio']}, FP {base['fp']} ({base['fp_mes']:.3f}/mês), {base['horas']:.0f} h, {base['fora']:.0f} h fora, placebo {base['placebo']:.1f}/8", flush=True)

    # ── P1: regras x posições ──
    print("\nP1 — REGRAS DE VOTO (detecção/8 | início | FP | FP/mês | horas totais | h fora | placebo)", flush=True)
    out = {}
    for nome, f in REGRAS.items():
        for pos in ("só A", "só B", "A+B"):
            rA = f if pos in ("só A", "A+B") else regra_atual_A
            rB = f if pos in ("só B", "A+B") else regra_atual_B
            al = alarme_de_regra(rA, rB, det); L = linha(al); out[(nome, pos)] = (L, al)
            marca = "  <= placebo de desenho" if nome in PLACEBO_DE.values() else ""
            print(f"   {nome:26s} {pos:5s} | {L['det']}/8 | {L['inicio']} | FP {L['fp']:3d} | {L['fp_mes']:.3f} | {L['horas']:6.0f} h | {L['fora']:5.0f} h | {L['placebo']:.1f}{marca}", flush=True)

    # ── P3: concordância total ──
    print("\nP3 — CONCORDÂNCIA TOTAL NO NASCIMENTO (exige duração >= 240 min; controle negativo em k=3)", flush=True)
    for pos in ("antes", "depois"):
        for kalvo in (4, 3):
            al = alarme_de_regra(regra_atual_A, regra_atual_B, det, p3=(pos, kalvo, 240.0)); L = linha(al)
            tag = "k=4 (hipótese)" if kalvo == 4 else "k=3 (CONTROLE NEGATIVO)"
            print(f"   {pos:6s} {tag:24s} | {L['det']}/8 | {L['inicio']} | FP {L['fp']:3d} | {L['fp_mes']:.3f} | {L['horas']:6.0f} h | {L['fora']:5.0f} h | placebo {L['placebo']:.1f}", flush=True)
            out[(f"P3 {tag}", pos)] = (L, al)

    # ── vizinhança ──
    print("\nVIZINHANÇA (k do nível A e do B x fator; pontos com detecção >= 6 e FP <= 9)", flush=True)
    for (nome, pos), (L, _) in sorted(out.items(), key=lambda x: (x[1][0]["fp"], -x[1][0]["det"]))[:8]:
        bons = []
        for fator in FATORES:
            rA = REGRAS[nome] if (nome in REGRAS and pos in ("só A", "A+B")) else regra_atual_A
            rB = REGRAS[nome] if (nome in REGRAS and pos in ("só B", "A+B")) else regra_atual_B
            p3 = None if nome in REGRAS else (pos, 4 if "k=4" in nome else 3, 240.0)
            Lv = linha(alarme_de_regra(rA, rB, det, fator=fator, p3=p3)); bons.append(f"{fator}:{Lv['det']}/{Lv['fp']}" + ("*" if Lv["det"] >= 6 and Lv["fp"] <= 9 else ""))
        print(f"   {nome:26s} {pos:6s} | " + " ".join(bons), flush=True)

    # ── LOEO para as regras informadas por lift ──
    print("\nLOEO (regra escolhida pelos 7 eventos de treino — maior detecção, desempate por menor FP — e avaliada no 8º)", flush=True)
    cand = [(n, p) for n in REGRAS for p in ("só A", "só B", "A+B")]
    evs = EV["real"]; acertos = 0; escolhas = []
    for i, t_ in enumerate(evs):
        treino = [t for j, t in enumerate(evs) if j != i]
        melhor = None
        for key in cand:
            L, al = out[key]; r = mede(al, treino)
            cmp = (r.det, -r.fp)
            if melhor is None or cmp > melhor[0]:
                melhor = (cmp, key)
        key = melhor[1]; al = out[key][1]; fora = mede(al, [t_]).det; acertos += int(fora); escolhas.append((t_.strftime("%d/%m/%Y"), key[0].split()[0], key[1], int(fora)))
    print(f"   LOEO = {acertos}/8   (controle, sem seleção: {base['det']}/8)", flush=True)
    for e, n, p, f_ in escolhas:
        print(f"      {e}: escolheu {n:3s} {p:5s} -> {'detectou' if f_ else 'PERDEU'}", flush=True)

    # ── classificação ──
    print("\nCLASSIFICAÇÃO (GANHO = FP <= 6 e det >= 6 e horas <= 1457 e acima do placebo; PIOR = det <= 5 ou FP >= 12)", flush=True)
    for (nome, pos), (L, _) in out.items():
        cl = "GANHO" if (L["fp"] <= 6 and L["det"] >= 6 and L["horas"] <= 1457 and L["det"] > L["placebo"]) else "PIOR" if (L["det"] <= 5 or L["fp"] >= 12) else "EQUIVALENTE"
        meta = L["fp_mes"] <= 0.55 and L["det"] >= 5
        if cl != "EQUIVALENTE" or meta:
            print(f"   {nome:26s} {pos:6s} -> {cl}{'  | META DO USUÁRIO' if meta else ''}", flush=True)
    print("\n   comparação com o placebo de desenho (a regra tem de custar MENOS que o seu placebo):", flush=True)
    for nome, plc in PLACEBO_DE.items():
        for pos in ("só A", "só B", "A+B"):
            L = out[(nome, pos)][0]; P = out[(plc, pos)][0]
            print(f"      {nome.split()[0]} {pos:5s}: FP {L['fp']:3d} det {L['det']} | placebo FP {P['fp']:3d} det {P['det']} -> {'regra melhor' if (L['fp'], -L['det']) < (P['fp'], -P['det']) else 'PLACEBO IGUAL OU MELHOR'}", flush=True)
    pd.DataFrame([dict(regra=n, posicao=p, **L) for (n, p), (L, _) in out.items()]).to_csv(ROOT / "exp51_resultados.csv", index=False)


if __name__ == "__main__":
    main()
