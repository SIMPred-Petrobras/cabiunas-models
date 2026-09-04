"""Controle simetrico da rotulagem por familia: os TP tambem seriam rotulados
SUPRIMENTO sob a mesma regra? Se sim, a taxonomia e artefato de taxa de base."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from plota_estilo_francisco import alarme, paradas_reais_2h, classifica_regra_c
import avalia as AV
from verdade import carrega_alarmes

MAQUINA = {"PAL_6240339","PALL_6240340","PALL_6240309","TAL_6240325","TALL_6240325",
           "TAH_6240301","TAH_6240303","TAH_6240305","TAH_6240307",
           "TAHH_6240303","TAHH_6240305","TAHH_6240307","PDAH_6240305","PDAHH6240305"}
SUPRIMENTO = {"PAL_6240315","PDAL_6240302","PI_6240319_AL","PAH_6240319",
              "PI_6240307_AL","PDT_6240301F","PDT_6240317","PDI_6240302_AL","PAL_306_PERM"}

alarmes = carrega_alarmes(0)
classif = classifica_regra_c(AV.episodios(alarme()), paradas_reais_2h())
J = pd.Timedelta(hours=6)

print("CONTROLE: fracao de cada classe rotulada SUPRIMENTO pela regra [a-6h, b+6h]")
cont = {}
for a, b, c, lead in classif:
    jan = alarmes[(alarmes.ts >= a - J) & (alarmes.ts <= b + J)]
    sup = jan["Tag Alarme"].isin(SUPRIMENTO).any()
    cont.setdefault(c, [0, 0])
    cont[c][0] += int(sup); cont[c][1] += 1
for c, (n, t) in cont.items():
    print(f"  {c:>7}: {n}/{t} = {100*n/t:.0f}% rotulado SUPRIMENTO")

print("\nCONTROLE 2: janela APERTADA, ancorada so no INICIO do episodio [a-1h, a+1h]")
J2 = pd.Timedelta(hours=1)
cont = {}
for a, b, c, lead in classif:
    jan = alarmes[(alarmes.ts >= a - J2) & (alarmes.ts <= a + J2)]
    sup = jan["Tag Alarme"].isin(SUPRIMENTO).any()
    cont.setdefault(c, [0, 0])
    cont[c][0] += int(sup); cont[c][1] += 1
for c, (n, t) in cont.items():
    print(f"  {c:>7}: {n}/{t} = {100*n/t:.0f}% com alarme de suprimento em +-1h do inicio")

print("\nTAXA DE BASE: probabilidade de um instante qualquer ter alarme de suprimento perto")
sup_ts = pd.DatetimeIndex(alarmes[alarmes["Tag Alarme"].isin(SUPRIMENTO)].ts)
print(f"  alarmes de suprimento: {len(sup_ts)} em 16 meses = {len(sup_ts)/(16*30):.2f}/dia")
for h in (1, 6, 12):
    print(f"  P(pelo menos 1 em janela de +-{h}h) ~ {1 - pow(2.718, -len(sup_ts)/(16*30)*(2*h/24)):.2f}")
