"""O controle negativo que falta no Passo 2 do relatorio do Diego (04/09/2026).

Ele checou 42 episodios de FP assim: pico de |z| DENTRO do episodio contra um
baseline de 2 h imediatamente anterior, sobre o grupo de mancal (1 temperatura +
10 sondas de vibracao) + 2 de oleo. Resultado: 61,9% "confirmado real" (>=2
sensores com |z|>=3) e 90,5% "confirmado ou borderline". Dai a recomendacao de
nao suprimir mais.

O Passo 3 dele TEM controle negativo (23,8% contra 3,4% em 5.000 instantes
aleatorios = 6,9x). O Passo 2 nao tem. E o Passo 2 e o que sustenta a
recomendacao.

O problema: pico sobre uma janela de ate 9,65 h em ~13 sensores sao milhares de
sorteios. "max |z| >= 3 em >=2 sensores" pode ser o resultado ESPERADO sob
ruido. Este script mede o nulo na mesma maquina, com os mesmos sensores, em
janelas SORTEADAS com a mesma distribuicao de duracao dos episodios dele.
"""
import numpy as np, pandas as pd

RNG = np.random.default_rng(20260904)
N_SORTEIOS = 2000
BASELINE = pd.Timedelta("2h")
FOLGA = pd.Timedelta("3min")          # o baseline dele termina 3 min antes

# duracoes relatadas: 48 min a 9,65 h, mediana 1,2 h (Secao 6.1)
DUR_MIN, DUR_MED, DUR_MAX = 48/60, 1.2, 9.65

g = pd.read_parquet("grade2min.parquet")
TV = [c for c in g.columns if c.startswith("TV_")]
# "todo o grupo de mancal (temperatura e os 10 canais de vibracao) mais os dois
# sensores de oleo" -- a frase e ambigua quanto a quais temperaturas entram.
# Rodamos as duas leituras, porque o nulo cresce com o numero de sensores e a
# critica tem de valer na leitura mais favoravel a ele (a MENOR).
CONJUNTOS = {
    "estrito (14): TC382_03_A + T5_AVG + 10 TV + 2 oleo":
        ["TC382_03_A", "T5_AVG_A"] + TV + ["954005_624_PI_0308", "954005_624_PDIT_0305"],
    "amplo (18): + as 4 TI_030x de mancal":
        ["TC382_03_A", "T5_AVG_A", "954005_624_TI_0301", "954005_624_TI_0303",
         "954005_624_TI_0305", "954005_624_TI_0307"] + TV
        + ["954005_624_PI_0308", "954005_624_PDIT_0305"],
}
rodando = g["RUNNING_A"].fillna(0).to_numpy() > 0.5

passo_h = (g.index[1] - g.index[0]).total_seconds() / 3600
n_base = int(BASELINE / pd.Timedelta(hours=passo_h))
n_folga = int(FOLGA / pd.Timedelta(hours=passo_h)) or 1

# distribuicao de duracao log-uniforme entre min e max, ancorada na mediana
durs = np.exp(RNG.uniform(np.log(DUR_MIN), np.log(DUR_MAX), N_SORTEIOS))
durs *= DUR_MED / np.median(durs)
n_ep = np.maximum((durs / passo_h).astype(int), 1)

# nao sortear perto dos 8 trips: isso INFLARIA o nulo com sinal verdadeiro
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
longe = np.ones(len(g), dtype=bool)
for t in fal:
    longe &= ~((g.index >= t - pd.Timedelta("7d")) & (g.index <= t + pd.Timedelta("7d")))
print(f"exclusao de +-7 d dos {len(fal)} trips: sobra "
      f"{100*longe.mean():.1f}% da grade\n")

def roda(SENS):
    SENS = [c for c in SENS if c in g.columns]
    Xv = g[SENS].astype("float32").to_numpy()
    ok = np.flatnonzero(rodando & longe)
    ok = ok[(ok > n_base + n_folga) & (ok < len(g) - int(DUR_MAX / passo_h) - 1)]
    forte = mod = borda = fraco = 0
    n_ge3 = []
    for k in range(N_SORTEIOS):
        i0 = int(RNG.choice(ok)); i1 = i0 + int(n_ep[k])
        if i1 >= len(Xv) or not (rodando[i0:i1].all() and longe[i0:i1].all()):
            continue
        b = Xv[i0 - n_base - n_folga: i0 - n_folga]
        if np.isnan(b).all(axis=0).any():
            continue
        med = np.nanmedian(b, axis=0)
        mad = np.nanmedian(np.abs(b - med), axis=0) * 1.4826
        mad = np.where(mad < 1e-9, np.nan, mad)
        with np.errstate(invalid="ignore"):
            z = np.abs((Xv[i0:i1] - med) / mad)
        vivo = ~np.isnan(z).all(axis=0)
        pico = np.full(z.shape[1], np.nan)
        pico[vivo] = np.nanmax(z[:, vivo], axis=0)
        ge3 = int(np.nansum(pico >= 3)); ge10 = int(np.nansum(pico >= 10))
        n_ge3.append(ge3)
        if ge10 >= 2:   forte += 1
        elif ge3 >= 2:  mod += 1
        elif ge3 >= 1:  borda += 1
        else:           fraco += 1
    return forte, mod, borda, fraco, n_ge3, len(SENS)

DELE = [26.2, 35.7, 28.6, 9.5]
ROT = ["Confirmado forte  (|z|>=10 e >=2 sensores)",
       "Confirmado moderado (|z|>=3 e >=2 sensores)",
       "Borderline (so 1 sensor com |z|>=3)",
       "Fraco / sem explicacao fisica"]
for nome, sens in CONJUNTOS.items():
    forte, mod, borda, fraco, n_ge3, n_s = roda(sens)
    tot = forte + mod + borda + fraco
    print(f"CONJUNTO {nome}   [{tot} janelas sorteadas]")
    print("=" * 84)
    print(f"{'criterio':<44} {'nulo':>10} {'ele (42 FP)':>14} {'enriq.':>10}")
    print("-" * 84)
    for rot, nulo, dele in zip(ROT, [100*x/tot for x in (forte, mod, borda, fraco)], DELE):
        print(f"{rot:<44} {nulo:9.1f}% {dele:13.1f}% {dele/nulo if nulo else float('inf'):9.2f}x")
    print("-" * 84)
    for rot, nulo, dele in [("CONFIRMADO (forte+moderado)", 100*(forte+mod)/tot, 61.9),
                            ("COM SINAL (forte+mod+borderline)", 100*(forte+mod+borda)/tot, 90.5)]:
        print(f"{rot:<44} {nulo:9.1f}% {dele:13.1f}% {dele/nulo:9.2f}x")
    print(f"  sensores com pico |z|>=3 numa janela ALEATORIA: mediana "
          f"{np.median(n_ge3):.0f} de {n_s}\n")
