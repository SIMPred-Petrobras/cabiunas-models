#!/usr/bin/env python3
"""Features de textura de vibracao (EXP7 item 2, Diego) como 5o/6o sinal do detector.

Motivo. O item 2 do EXP7 foi o unico ganho limpo da serie inteira do Diego: kurtosis,
skewness e crest factor melhoraram hit_rate (87,5->92,5%), falso alerta (2,21->1,94%)
e sem-deteccao (5->3) AO MESMO TEMPO. Ganho simultaneo em eixos que normalmente se
opoem e assinatura de sinal genuino, nao de reparametrizacao -- e a justificativa
fisica e padrao em condition monitoring: degradacao incipiente de mancal aparece como
pico esporadico (cauda pesada) antes de mexer no nivel RMS.

Nosso detector usa a vibracao so como NIVEL (max do z robusto das 10 sondas contra
referencia rolante). Se a textura carrega informacao que o nivel nao carrega, ela
deve aparecer como um sinal que vota em evento onde o 'vb' nao vota.

Construcao: mesma maquinaria do sinal de nivel -- z robusto com referencia rolante de
400 h e banda de guarda -- aplicada a kurtosis movel (1 h e 4 h) e ao crest factor
(1 h) de cada sonda. Max sobre as sondas. Voto continua >=2, agora sobre 5 ou 6 sinais.

Tambem responde: o portao de volatilidade (EXP10c) ainda ganha alguma coisa DEPOIS do
teto de duracao de 12 h, que e o ponto de operacao real daqui? O teto ja corta 72% das
horas; portao e teto podem estar mirando o mesmo tempo.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import indice_volatilidade, K_BASE, K_VIB
from auto_reset import trunca

JAN_KURT_H = [1, 4]
JAN_CREST_H = 1

NOTA_BUG = """Duas armadilhas numericas nesta feature, ambas descobertas medindo em vez
de supor -- e ambas produziriam silenciosamente a conclusao 'textura nao ajuda':

  1. pandas .rolling().kurt() devolve NaN para a janela inteira quando ha NaN no meio
     dela, mesmo com min_periods satisfeito. Mascarar a vibracao por operacao quente
     ANTES da janela (33% de NaN espalhado) deixava 0,04% de valores validos. Aqui a
     estatistica e calculada na serie densa e a mascara e aplicada depois, e a kurtosis
     e montada a mao a partir de medias moveis (que respeitam min_periods e pulam NaN).
  2. crest factor = pico/RMS sobre o sinal CRU mede o offset DC, nao a textura: as
     sondas ficam em ~18-33 um com desvio de ~0,07-6 um, entao o crest cru fica colado
     em 1,015 (p95 = 1,039). Sobre o sinal centrado por uma linha de base movel de 24 h
     ele abre para 1,62 (p95 = 2,61) -- ai sim mede forma. O EXP7 calcula textura sobre
     sinal de vibracao ja tratado; aqui a grade e de 2 min de uma amplitude ja agregada,
     entao centrar e obrigatorio."""

BASE_H = 24          # linha de base movel para centrar o sinal


def _mm(x, n):
    return x.rolling(n, min_periods=max(4, n // 2)).mean()


def textura(g, stable, falhas, jan_h, tipo):
    """Estatistica de forma da vibracao na janela, sobre o sinal CENTRADO. Ver NOTA_BUG."""
    n = int(jan_h * 30)
    V = g[C.VIBRATION_TAGS].astype("float64").interpolate(limit=5)     # serie densa
    D = V - V.rolling(int(BASE_H * 30), min_periods=int(BASE_H * 30) // 4).median()
    if tipo == "kurt":
        m = _mm(D, n)
        d = D - m
        v2 = _mm(d ** 2, n)
        F = _mm(d ** 4, n) / (v2 ** 2).replace(0, np.nan) - 3.0
    else:                                     # crest factor = pico / RMS na janela
        pico = D.abs().rolling(n, min_periods=n // 2).max()
        rms = _mm(D ** 2, n) ** 0.5
        F = pico / rms.replace(0, np.nan)
    F = F.where(stable, axis=0)
    Z = RO.z_rolante(F, stable, falhas, horas_base=400, guarda_h=24, phi=0.0)
    return Z.max(axis=1)


def alerta_n(out, mask, extras, k_base, k_vib, k_extra, voto=2):
    """Mesmo voto do detector, com sinais adicionais. extras = {nome: (serie, hl, thr)}"""
    idx = out.index
    def ew(s, hl):
        return s.ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    n = (DET._sustained(ew(out["t"], "1h"), DET.THR_FAM * k_base).astype(int)
         + DET._sustained(ew(out["p"], "1h"), DET.THR_FAM * k_base).astype(int)
         + DET._sustained(ew(out["sp"], "30min"), DET.THR_SPREAD * k_base).astype(int)
         + DET._sustained(ew(out["vb"], "30min"), 3.0 * k_vib).astype(int))
    for s, hl, thr in extras:
        n = n + DET._sustained(ew(s, hl), thr * k_extra).astype(int)
    return (n >= voto) & mask


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    stable = df["stable"].astype(bool)
    mask = mascara_pontuacao(df)
    tr = pd.Series(idx < CORTE, index=idx); te = ~tr

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)

    print("calculando textura (kurtosis 1h/4h, crest 1h) ...", flush=True)
    K1 = textura(g, stable, falhas, 1, "kurt")
    K4 = textura(g, stable, falhas, 4, "kurt")
    CR = textura(g, stable, falhas, JAN_CREST_H, "crest")
    for nome, s in [("kurt_1h", K1), ("kurt_4h", K4), ("crest_1h", CR)]:
        q = s[mask & tr]
        print(f"  z de {nome:9s}: p50={q.median():5.2f} p95={q.quantile(.95):6.2f} "
              f"p99={q.quantile(.99):7.2f} max={q.max():8.1f}", flush=True)

    def linha(al, nome):
        d = {"variante": nome}
        for tag, m, ev in [("tr", tr, falhas[falhas < CORTE]), ("te", te, falhas[falhas >= CORTE]),
                           ("tot", pd.Series(True, index=idx), falhas)]:
            am, qm = al[m], (mask & m)[m]
            x = A.avalia(am, ev, qm)
            d.update({f"{tag}_det": x["det"], f"{tag}_n": x["n_ev"], f"{tag}_fp": x["fp_mes"],
                      f"{tag}_h": x["h_fp_mes"], f"{tag}_eps": x["episodios"]})
        d["quais"] = ",".join(A.avalia(al, falhas, mask)["detectados"])
        return d

    linhas = [linha(base, "base (4 sinais)")]
    variantes = [("+kurt_1h", [(K1, "30min", 3.0)]), ("+kurt_4h", [(K4, "30min", 3.0)]),
                 ("+crest_1h", [(CR, "30min", 3.0)]),
                 ("+kurt_1h+crest", [(K1, "30min", 3.0), (CR, "30min", 3.0)]),
                 ("+kurt_1h+4h+crest", [(K1, "30min", 3.0), (K4, "30min", 3.0), (CR, "30min", 3.0)])]
    for nome, ex in variantes:
        for ke in (1.0, 1.7, 2.5):
            al = alerta_n(out, mask, ex, K_BASE, K_VIB, ke)
            linhas.append(linha(al, f"{nome} k_tx={ke}"))

    t = pd.DataFrame(linhas)
    t.to_csv("textura.csv", index=False)
    print(f"\n=== TEXTURA COMO SINAL EXTRA (voto >=2) ===")
    print(f"{'variante':>26} | {'treino':>7} {'h/mes':>7} | {'teste':>7} {'h/mes':>7} | "
          f"{'total':>7} {'eps':>4} {'h/mes':>7}")
    for _, r in t.iterrows():
        print(f"{r['variante']:>26} | {r.tr_det:>3d}/{r.tr_n:<3d} {r.tr_h:7.1f} | "
              f"{r.te_det:>3d}/{r.te_n:<3d} {r.te_h:7.1f} | {r.tot_det:>3d}/{r.tot_n:<3d} "
              f"{r.tot_eps:4d} {r.tot_h:7.1f}")

    print("\n=== PORTAO DE VOLATILIDADE EM CIMA DO TETO DE 12 h (ponto de operacao real) ===")
    vol = indice_volatilidade(g).reindex(idx)
    print(f"{'config':>34} {'det':>4} {'eps':>5} {'h/mes':>8}")
    for nome, al in [("base", base), ("base + teto 12h", trunca(base, 12))]:
        x = A.avalia(al, falhas, mask)
        print(f"{nome:>34} {x['det']:4d} {x['episodios']:5d} {x['h_fp_mes']:8.1f}")
    for v in (0.12, 0.15, 0.18, 0.22):
        al = trunca(base & ~(vol > v).fillna(False), 12)
        x = A.avalia(al, falhas, mask)
        print(f"{'base + vol>%.2f + teto 12h'%v:>34} {x['det']:4d} {x['episodios']:5d} "
              f"{x['h_fp_mes']:8.1f}")


main()
