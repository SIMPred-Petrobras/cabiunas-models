"""A regra de ASSOCIACAO evento<->episodio e a mesma nos dois lados?

Nossa `avalia`: o alarme tem de estar DE PE em algum instante de [t-48h, t].
Dele:  `inside = [ep for ep in episodes if window_start <= ep <= evento]`
       -- o INICIO do episodio tem de cair dentro da janela.

Para episodio longo as duas divergem muito: um alerta que sobe 670 h antes do
trip e fica de pe ate ele conta pela nossa e NAO conta pela dele.
"""
import sys; sys.path.insert(0, ".")
import pandas as pd, avalia as AV
from pos_processamento import mask, idx, alvo
from plota_estilo_francisco import alarme

al = alarme()
eps = AV.episodios(al)
JAN = pd.Timedelta(hours=48)

print("OS 8 ALVOS SOB AS DUAS REGRAS DE ASSOCIACAO")
print("=" * 92)
print(f"{'evento':>12} {'inicio do episodio':>20} {'lead do inicio':>15} "
      f"{'nossa (de pe)':>14} {'dele (inicio na janela)':>24}")
n_nosso = n_dele = 0
for t in alvo:
    t0 = t - JAN
    de_pe = bool(al.loc[t0:t].any())                       # nossa regra
    inicio_na_janela = any(t0 <= a <= t for a, b in eps)   # regra dele
    cand = [(a, b) for a, b in eps if a <= t and b >= t0]
    ini = cand[0][0] if cand else None
    lead = f"{(t-ini).total_seconds()/3600:8.1f} h" if ini is not None else "        --"
    n_nosso += de_pe; n_dele += inicio_na_janela
    marca = "  <<< divergem" if de_pe != inicio_na_janela else ""
    print(f"{t:%d/%m/%Y} {str(ini)[:19]:>20} {lead:>15} "
          f"{('SIM' if de_pe else 'nao'):>14} {('SIM' if inicio_na_janela else 'nao'):>24}{marca}")
print(f"\n  pela NOSSA regra : {n_nosso}/8")
print(f"  pela regra DELE  : {n_dele}/8")
print(f"\n  divergem em {sum(1 for t in alvo if bool(al.loc[t-JAN:t].any()) != any(t-JAN <= a <= t for a,b in eps))} dos 8 eventos")
