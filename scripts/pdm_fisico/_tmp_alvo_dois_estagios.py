"""Alvo de dois estagios: conta tambem o alarme de PRIMEIRO estagio de protecao
de maquina (PAL/TAH), nao so o intertravamento (PALL/TAHH/TRIP).

A TESE. O alvo de hoje conta 8 eventos = parada real + alarme de NIVEL (regex TRIP|
Mt.Alta|Mt.Bx). Isso e o estagio 2 da protecao. Mas a planta tem estagio 1 (PAL_6240339
"Pressao Bx. Header Oleo Lub.", TAH_6240305 "Temp.Alta Mancal Rad.LNA CP") para o mesmo
modo fisico. Quando o detector avisa, o operador intervem e a maquina para controlada
ANTES do trip, o evento nunca entra no alvo -- e o detector e contado como errado
exatamente quando funcionou.

Separa as tags em tres familias por SUJEITO FISICO:
  MAQUINA     -- oleo lubrificante, mancal, selo (a saude do TC-330.03A)
  SUPRIMENTO  -- gas combustivel, gas de selagem, linha de balanceamento (a montante)
  INSTRUMENTO -- falha de transmissor/termopar (qualidade de dado, nao de maquina)

Compara: alvo_hoje (nivel) x alvo_maquina (nivel + estagio 1 de maquina).
"""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, ".")

from plota_estilo_francisco import alarme, paradas_reais_2h
from pos_processamento import mask, idx, alvo as ALVO_HOJE
import avalia as AV
from verdade import carrega_alarmes

MAQUINA = {
    "PAL_6240339", "PALL_6240340", "PALL_6240309",      # oleo lubrificante
    "TAL_6240325", "TALL_6240325",                       # tanque de oleo
    "TAH_6240301", "TAH_6240303", "TAH_6240305", "TAH_6240307",   # mancais (estagio 1)
    "TAHH_6240303", "TAHH_6240305", "TAHH_6240307",      # mancais (estagio 2)
    "PDAH_6240305", "PDAHH6240305",                      # vazamento selo primario
}
SUPRIMENTO = {
    "PAL_6240315", "PDAL_6240302", "PI_6240319_AL", "PAH_6240319",
    "PI_6240307_AL", "PDT_6240301F", "PDT_6240317", "PDI_6240302_AL", "PAL_306_PERM",
}

alarmes = carrega_alarmes(0)
paradas = paradas_reais_2h()
al_series = alarme()
eps = AV.episodios(al_series)
m = AV.avalia(al_series, ALVO_HOJE, mask)
meses = m["horas_op"] / 730.0

T0 = ALVO_HOJE.min().normalize()  # mesma janela de avaliacao do alvo atual


def alvos_por_familia(tags: set) -> list:
    out = []
    for q in paradas.ini:
        jan = alarmes[(alarmes.ts >= q - pd.Timedelta(hours=1)) &
                      (alarmes.ts <= q + pd.Timedelta(minutes=30))]
        if len(jan) and jan["Tag Alarme"].isin(tags).any():
            out.append(q)
    # agrupa eventos a menos de 24h (mesma regra do verdade.py)
    ev, ult = [], None
    for q in sorted(out):
        if ult is None or (q - ult) >= pd.Timedelta(hours=24):
            ev.append(q)
        ult = q
    return [t for t in ev if t >= pd.Timestamp("2025-01-01", tz="UTC")]


alvo_maq = alvos_por_familia(MAQUINA)
alvo_sup = alvos_por_familia(SUPRIMENTO)

print("=" * 100)
print("QUANTOS EVENTOS CADA DEFINICAO DE ALVO PRODUZ (a partir de 2025-01-01)")
print("=" * 100)
print(f"  alvo HOJE (alarme de nivel):          {len(ALVO_HOJE)}")
print(f"  alvo MAQUINA (nivel + estagio 1):     {len(alvo_maq)}")
print(f"  alvo SUPRIMENTO (gas, a montante):    {len(alvo_sup)}")

novos = [t for t in alvo_maq if not any(abs((t - a).total_seconds()) < 24 * 3600 for a in ALVO_HOJE)]
print(f"\n  eventos NOVOS que o alvo de maquina traz: {len(novos)}")
for t in novos:
    jan = alarmes[(alarmes.ts >= t - pd.Timedelta(hours=1)) & (alarmes.ts <= t + pd.Timedelta(minutes=30))]
    tags = sorted(set(jan[jan["Tag Alarme"].isin(MAQUINA)]["Descrição Alarme"]))
    dur = paradas[paradas.ini == t].dur_h.iloc[0]
    print(f"    {t:%d/%m/%Y %H:%M}  parada de {dur:6.1f}h  {' | '.join(tags)}")

print("\n" + "=" * 100)
print("DESEMPENHO DO MESMO DETECTOR SOB CADA ALVO (sem mexer numa linha do detector)")
print("=" * 100)
for nome, av in (("HOJE (nivel, 8 eventos)", list(ALVO_HOJE)),
                 ("MAQUINA (nivel + estagio 1)", alvo_maq)):
    JAN = pd.Timedelta(hours=48)
    jw = [(t - JAN, t) for t in av]
    det = set()
    n_tp = n_out = 0
    h_out = 0.0
    for a, b in eps:
        hit = [t for t, (t0, t1) in zip(av, jw) if a <= t1 and b >= t0]
        if hit:
            n_tp += 1
            det.update(hit)
        else:
            n_out += 1
            h_out += (b - a).total_seconds() / 3600
    print(f"\n  {nome}")
    print(f"    deteccao: {len(det)}/{len(av)} eventos  |  episodios TP: {n_tp}")
    print(f"    episodios nao-TP: {n_out}  ({n_out/meses:.3f}/mes, {h_out/meses:.2f} h/mes)")
    perdidos = [t for t in av if t not in det]
    if perdidos:
        print(f"    nao detectados: {', '.join(f'{t:%d/%m/%Y}' for t in perdidos)}")

print("\n" + "=" * 100)
print("E OS QUE SOBRAM? classifica cada episodio nao-TP pela FAMILIA do alarme mais proximo")
print("=" * 100)
JAN = pd.Timedelta(hours=48)
jw = [(t - JAN, t) for t in alvo_maq]
J = pd.Timedelta(hours=6)
for a, b in eps:
    if any(a <= t1 and b >= t0 for t0, t1 in jw):
        continue
    jan = alarmes[(alarmes.ts >= a - J) & (alarmes.ts <= b + J)]
    fam = []
    if jan["Tag Alarme"].isin(MAQUINA).any():
        fam.append("MAQUINA")
    if jan["Tag Alarme"].isin(SUPRIMENTO).any():
        fam.append("SUPRIMENTO")
    if not fam:
        fam.append("SEM CONTEXTO")
    print(f"  {a:%d/%m/%Y %H:%M}  {(b-a).total_seconds()/3600:6.1f}h  ->  {'+'.join(fam)}")
