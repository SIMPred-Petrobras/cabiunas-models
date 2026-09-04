"""Verifica os tres pontos da mensagem do Francisco.

(1) Ele cita "13 episodios de FP, 1,12 FP/mes, 39 h/mes" como sendo o nosso ponto.
    O nosso ponto PUBLICADO tem a porta de mancal (sp|vb). Sera que 1,12 e a
    configuracao sem a porta?
(2) Ele compara o nosso ponto de 8/8 contra o ponto de 6/8 dele na regua estrita --
    que e exatamente a comparacao sem nivel fixo que ele mesmo critica. Qual e a
    NOSSA fronteira na regua ESTRITA (regra A), nivel a nivel?
(3) Ele afirma que a fronteira de menor FP e dele "nas duas reguas". Testavel.

REGRA A (a dele, estrita): episodio e perdoado se ha parada real em
   [inicio do episodio, inicio + 48h]
REGRA C: perdoado se ha parada real em [inicio, fim + 48h]
CRU: nada e perdoado -- todo episodio nao-TP e FP.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo, EW, BASE, sel, T0
from publica_clearml import SIN, REFRAT_H, DUR_MIN, GRID, KAPPA, H_CUSUM, K
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h

paradas = paradas_reais_2h()
JAN = pd.Timedelta(hours=48)
jw = [(t-JAN, t) for t in alvo]
PAR = pd.DatetimeIndex(paradas.ini.sort_values())

def classifica(eps, regra):
    tp = fp = neu = 0; h_fp = 0.0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw):
            tp += 1; continue
        lim = (a if regra == "A" else b) + JAN
        if regra == "CRU":
            perdoado = False
        else:
            perdoado = bool(((PAR >= a) & (PAR <= lim)).any())
        if perdoado: neu += 1
        else:
            fp += 1; h_fp += (b-a).total_seconds()/3600
    return tp, fp, neu, h_fp

# ---------- (1) o ponto publicado, com e sem a porta de mancal ----------
print("(1) O NUMERO QUE ELE CITA -- qual configuracao e essa?")
print("="*88)
ON = partes(1.7, 2.2)
ns = sum(ON[c].astype(int) for c in SIN)
for nome, v in (("COM porta de mancal (nosso ponto publicado)",
                 pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])),
                ("SEM porta de mancal",
                 pd.Series(ns >= 2, index=idx) & mask)):
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    eps = AV.episodios(al)
    m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
    print(f"\n  {nome}")
    for regra in ("CRU", "A", "C"):
        tp, fp, neu, h = classifica(eps, regra)
        marca = "   <<< os 13 / 1,12 / 39,0 que ele cita" if (regra=="CRU" and fp==13) else ""
        print(f"    regra {regra:>3}: {tp}/8  FP={fp:2d} ({fp/meses:.3f}/mes)  "
              f"NEU={neu:2d}  {h/meses:5.2f} h/mes{marca}")

# ---------- (2)+(3) a nossa fronteira nas TRES reguas ----------
print("\n\n(2)+(3) A NOSSA FRONTEIRA, NIVEL A NIVEL, NAS TRES REGUAS")
print("="*88)
T = pd.read_csv("_tmp_fronteira_fp.csv")
g = pd.read_parquet("grade2min.parquet")
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)

linhas = []
for bl, sub in T.groupby("bl"):
    n_bl = int(pd.Timedelta(bl)/pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mk = (estavel & ~blk) & sel
    rst = ((~mk) | part).to_numpy()
    for (kb, kv), s2 in sub.groupby(["kb","kv"]):
        K_ = {"t":kb,"p":kb,"sp":kb,"vb":kv}; ONx = {}
        for c in SIN:
            thr = BASE[c]*K_[c]; E = EW[c].where(mk)
            deg = ((E>thr).astype(int).rolling(15,min_periods=15).sum()>=15)
            cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(),rst)>H_CUSUM, index=idx)
            ONx[c] = (deg|cu)&mk
        nsx = sum(ONx[c].astype(int) for c in SIN)
        vx = pd.Series(nsx>=2, index=idx)&mk&(ONx["sp"]|ONx["vb"])
        for r in s2.itertuples():
            al = pos(vx, nsx, int(r.rf), int(r.dm), False)
            eps = AV.episodios(al)
            if not eps: continue
            m = AV.avalia(al, alvo, mk); meses = m["horas_op"]/730.0
            row = dict(det=int(r.det), meses=meses)
            for regra in ("A","C"):
                tp, fp, neu, h = classifica(eps, regra)
                row[f"fp_{regra}"] = fp/meses; row[f"h_{regra}"] = h/meses
            linhas.append(row)
F = pd.DataFrame(linhas)

# fronteira dele, publicada (regra A, notebook 10 secoes 8 e 10)
DELE_A = {6: 0.87, 5: 0.43, 4: 0.34}
DELE_A_H = {6: 5.3, 5: 2.0, 4: 0.6}
print(f"{'nivel':>7} | {'REGRA A (a estrita, dele)':^34} | {'REGRA C':^20}")
print(f"{'':>7} | {'nosso FP':>9} {'dele FP':>8} {'nosso h':>7} {'dele h':>6} | {'nosso FP':>9} {'nosso h':>8}")
for n in (8,7,6,5,4):
    s = F[F.det == n]
    if s.empty: continue
    da = DELE_A.get(n); dh = DELE_A_H.get(n)
    da_s = f"{da:8.3f}" if da else "     n/a"
    dh_s = f"{dh:6.1f}" if dh else "   n/a"
    print(f"{n:>6}/8 | {s.fp_A.min():9.3f} {da_s} {s.h_A.min():7.2f} {dh_s} | "
          f"{s.fp_C.min():9.3f} {s.h_C.min():8.2f}")
print("\n  (fronteira dele em regra A: notebook 10, secoes 8 e 10 -- minimo entre as variantes)")

print("\n\nPARES (FP, horas) NO MESMO PONTO -- regra A, dentro do teto de 1 FP/mes")
print("=" * 88)
print(f"{'nivel':>7} {'menor FP':>10} {'h ai':>8} | {'menor h':>9} {'FP ai':>8} | {'configs no teto':>16}")
for n in (8, 7, 6, 5, 4):
    s2 = F[(F.det == n) & (F.fp_A <= 1.0)]
    if s2.empty:
        print(f"{n:>6}/8   nenhuma configuracao dentro do teto de 1 FP/mes em regra A")
        continue
    a = s2.sort_values("fp_A").iloc[0]
    b = s2.sort_values("h_A").iloc[0]
    print(f"{n:>6}/8 {a.fp_A:10.3f} {a.h_A:8.2f} | {b.h_A:9.2f} {b.fp_A:8.3f} | {len(s2):>10}/{len(F[F.det==n])}")
F.to_csv("_tmp_fronteira_AC.csv", index=False)
print("\n-> _tmp_fronteira_AC.csv")
