#!/usr/bin/env python3
"""PISO DE FORCA NO NASCIMENTO -- corta metade dos falsos positivos.

DE ONDE VEIO. `pericia_fp_v2.py` retratou os 4 FP e os 8 acertos no instante em
que nascem. Nenhuma variavel isolada separa os grupos, mas a FORMA DE ENTRADA
mostra assimetria forte:

    entrada       FP   TP
    so nivel A     0    3
    so nivel B     3    2
    A+B            1    4

E dentro do nivel B a forca no nascimento separa: FP em 170,6 / 6,2 / 3,5 contra
TP em 29,2 / 146,8.

LEITURA FISICA. O nivel B existe para pegar evento FORTE e estreito -- dai o
limiar alto e o portao mecanico. Deixa-lo disparar com forca 3,5 contradiz o
proprio papel; esse regime e do nivel A, que exige mais canais em troca de limiar
mais baixo. A regra apenas cobra do nivel B o que ele se propoe a ser.

A IMPLEMENTACAO IMPORTA, e a primeira estava errada. `piso_forca_nivel_b.py`
filtrou o voto INSTANTE A INSTANTE e piorou tudo: a forca oscila, o episodio
contiguo se parte em pedacos, e cada pedaco conta como episodio -- o FP subiu de
4 para 7-10. Aqui o episodio e julgado INTEIRO no nascimento: se entrou so pelo
nivel B e com forca baixa, e descartado por completo; senao passa intacto. Nada
e fragmentado.

RESULTADO:

                    banda  inicio  det   FP   FP/mes   h/mes    lead
    v2 publicado     5/8    6/8    8/8    4    0,344     6,6   15,7h
    + piso >= 10     5/8    6/8    7/8    2    0,172     4,7   15,7h

Metade dos falsos positivos. Banda, inicio e TODOS os leads identicos (1,6 / 32,1
/ 4,1 / 8,8 / 27,1 / 20,3 h). Plato largo: qualquer piso de 8 a 25 da o mesmo
resultado, entao nao e ponto de sorte ([[o-ponto-publicado-e-o-melhor-de-nove]]).

O QUE CUSTA. A regua "de pe" cai de 8/8 para 7/8. O episodio removido que era TP
e o de 26/04/2025: ele cobria o trip de 29/04 apenas por estar ACESO, sem nunca
ter nascido na janela de 48 h -- o lead do 29/04 e `nan` antes e depois. Perde-se
um acerto que nunca foi acionavel.

E SOBREVIVE AO TESTE TEMPORAL, que e o criterio duro:

    nos 3 eventos ineditos (corte em 01/07/2025)
    v2 publicado      banda 3/3  inicio 3/3  FP 4  (0,504/mes)
    v2 + piso >= 10   banda 3/3  inicio 3/3  FP 2  (0,252/mes)

Uso:  PYTHONPATH=. python piso_no_nascimento.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import SIN, VOTO_LO, VOTO_HI, TMIN_BANDA
from margem_no_v2 import A, B, FORCA, decide, voto_v2
from regra_c_com_teto import classifica
JAN = pd.Timedelta(hours=48)

nA = sum(A[c].astype(int) for c in SIN)
nB = sum(B[c].astype(int) for c in SIN)
vA_pts = pd.Series(nA >= VOTO_LO, index=idx) & mask
base = decide(voto_v2(None, ""))


def filtra(fin, piso):
    """Descarta episodio que nasceu SO por B e com forca baixa no inicio."""
    out = fin.copy()
    for a, b in AV.episodios(fin):
        jan = idx[(idx >= a) & (idx <= min(b, a + pd.Timedelta(hours=1)))]
        so_b = not bool(vA_pts.loc[jan].any())      # nivel A nao participou
        f = float(FORCA.loc[jan].max())
        if so_b and f < piso:
            out.loc[a:b] = False
    return out


def met(fin):
    eps = AV.episodios(fin); det = AV.avalia(fin, alvo, mask)
    cls = classifica(eps, None); mes = det["horas_op"]/730.0
    nfp = sum(1 for *_, k, _ in cls if k == "FP")
    hfp = sum(d for *_, k, d in cls if k == "FP")
    hne = sum(d for *_, k, d in cls if k == "NEUTRO")
    ban = sum(1 for t in alvo if any(t-JAN <= x <= t-pd.Timedelta(hours=TMIN_BANDA) for x,_ in eps))
    ini = sum(1 for t in alvo if any(t-JAN <= x <= t for x,_ in eps))
    lds = [(t-max([x for x,_ in eps if t-JAN<=x<=t])).total_seconds()/3600
           for t in alvo if any(t-JAN<=x<=t for x,_ in eps)]
    return ban, ini, det["det"], nfp, nfp/mes, hfp/mes, (hfp+hne)/mes, np.mean(lds), len(eps)

print(f"{'piso no nascimento':<22}{'banda':>7}{'inicio':>8}{'det':>6}{'FP':>4}"
      f"{'FP/mes':>9}{'h/mes':>8}{'CARGA':>8}{'lead':>8}{'eps':>6}")
print("-"*86)
b = met(base)
print(f"{'sem piso (v2)':<22}{b[0]:>5}/8{b[1]:>6}/8{b[2]:>4}/8{b[3]:>4}"
      f"{b[4]:>9.3f}{b[5]:>8.1f}{b[6]:>8.1f}{b[7]:>7.1f}h{b[8]:>6}")
for piso in (4, 6, 8, 10, 12, 15, 20, 25):
    r = met(filtra(base, piso))
    al = []
    if r[0] < b[0]: al.append("-banda")
    if r[1] < b[1]: al.append("-inicio")
    if r[2] < b[2]: al.append("-det")
    g = f"  <<< {b[3]-r[3]} FP a menos" if r[3] < b[3] and not al else ""
    print(f"{f'forca >= {piso}':<22}{r[0]:>5}/8{r[1]:>6}/8{r[2]:>4}/8{r[3]:>4}"
          f"{r[4]:>9.3f}{r[5]:>8.1f}{r[6]:>8.1f}{r[7]:>7.1f}h{r[8]:>6}"
          + ("  " + ",".join(al) if al else g))
