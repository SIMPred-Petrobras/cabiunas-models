#!/usr/bin/env python3
"""EXP 36 — DE QUAL NÍVEL DO VOTO (A OU B) VÊM OS FALSOS POSITIVOS E AS DETECÇÕES? (decomposição do detector publicado, 8 eventos)

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR. Descritivo: nenhum parâmetro é escolhido pelo resultado.

O VOTO (de `prever`): NÍVEL A = SENSÍVEL: >= 3 dos 4 canais acima do limiar baixo (base x k_nivel_a, com CUSUM); NÍVEL B = ESPECÍFICO: >= 2 dos 4 acima do limiar alto (k_nivel_b) E um deles mecânico (sp ou vb). voto = A ou B; depois escalada/refratário/duração.
O QUE SE FAZ (sinais de produção de 8 eventos; `ci.prever` como caixa-preta):
  (1) POR EPISÓDIO do detector publicado (os 16 episódios de 01/02/2025 a 21/04/2026): a fração dos seus instantes em que A está ligado, B está ligado, só A, só B, ambos; e o NÍVEL DO NASCIMENTO (os níveis ligados no primeiro instante do episódio); separados em FP e episódios de trip.
  (2) CONTRAFACTUAL: o detector inteiro (escalada, refratário, duração) com SÓ o nível A (voto_nivel_b desligado) e com SÓ o nível B (voto_nivel_a desligado): detecção de 8, FP, FP/mês, horas.
  (3) Os canais acesos no nascimento dos FP (a_t, a_p, a_sp, a_vb e b_*).
EXPECTATIVAS: a maioria dos FP nasce no nível A (o sensível) ~65%, e uma minoria só no B ~15%; o nível A sozinho reproduz quase todo o detector (detecção >= 6/8 e FP >= 8) ~70%; o nível B sozinho detecta <= 4/8 com FP <= 4 ~65%; os FP longos (> 100 h) são todos de nascimento A ~60%.
INTERPRETAÇÃO PREVISTA. Se os FP vêm do A, afrouxar ou apertar o A é a alavanca de FP (e custa detecção); se do B, o portão mecânico não está filtrando. Resultado só descritivo: a decisão sobre mexer em qualquer nível exige a prova prospectiva (o EXP 24 mostrou que os k estão no topo de um pico).
LIMITES. 16 episódios (9 FP); o nível B está encoberto quando o A já está ligado (a decomposição por episódio não prova causalidade; o contrafactual sim, dentro do mesmo código); sem rótulos na calibração: nenhuma.

Uso:  PYTHONPATH=src .venv/bin/python experiments/36_de_qual_nivel_vem_o_fp.py
"""
from __future__ import annotations
import contextlib, copy, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48)


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)

    def roda(d):
        with contextlib.redirect_stdout(io.StringIO()):
            return ci.prever({"detector": d}, proc)
    R = roda(det); mask = R["vigiado"].astype(bool)
    vA = (R["canais_nivel_a"] >= det["voto_nivel_a"]) & mask; vB = (R["canais_nivel_b"] >= det["voto_nivel_b"]) & mask & (R["b_sp"] | R["b_vb"])
    hv = float(vig.loc[INICIO8:].sum()) * 2 / 60 / 730
    r0 = avalia(R["is_anomaly"].astype(bool).reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8)
    print(f"PUBLICADO: {r0.det}/8, início {r0.inicio}, FP {r0.fp} ({r0.fp_mes:.2f}/mês), fora {r0.horas_alarme_fora:.0f} h, totais {r0.horas_alarme:.0f} h; {hv:.1f} meses de máquina vigiada", flush=True)
    print("\n(1) EPISÓDIOS DO PUBLICADO por nível (fração dos instantes com A, B, só A, só B, ambos) e nível do NASCIMENTO", flush=True); linhas = []
    for a, b in episodios_de_serie(R["is_anomaly"].astype(bool).loc[INICIO8:]):
        tr = [t for t in reg if a <= t and b >= t - JAN]; s = slice(a, b); n = len(vA.loc[s]); fa = float(vA.loc[s].mean()); fb = float(vB.loc[s].mean()); so_a = float((vA.loc[s] & ~vB.loc[s]).mean()); so_b = float((vB.loc[s] & ~vA.loc[s]).mean()); amb = float((vA.loc[s] & vB.loc[s]).mean())
        nasc = "A+B" if (vA.loc[a] and vB.loc[a]) else "só A" if vA.loc[a] else "só B" if vB.loc[a] else "nenhum (corte/refratário)"
        canais = "".join(c[0] if R.loc[a, f"a_{c}"] else "-" for c in ("t", "p", "sp", "vb")) + "/" + "".join(c[0] if R.loc[a, f"b_{c}"] else "-" for c in ("t", "p", "sp", "vb"))
        linhas.append(dict(inicio=f"{a:%Y-%m-%d %H:%M}", dur_h=round((b - a).total_seconds() / 3600 + 2 / 60, 1), tipo=("trip " + str(tr[0])[:10]) if tr else "FP", frac_A=round(fa, 2), frac_B=round(fb, 2), so_A=round(so_a, 2), so_B=round(so_b, 2), ambos=round(amb, 2), nasce_em=nasc, canais_A_B_no_nascimento=canais))
    T = pd.DataFrame(linhas); print(T.to_string(index=False), flush=True)
    for tipo in ("FP", "trip"):
        sub = T[T.tipo.str.startswith(tipo)]; print(f"   {tipo}: {len(sub)} episódios; nascimento: {sub.nasce_em.value_counts().to_dict()}; horas: {sub.dur_h.sum():.0f} h; fração média dos instantes só A {sub.so_A.mean():.2f}, só B {sub.so_B.mean():.2f}, ambos {sub.ambos.mean():.2f}", flush=True)
    fp = T[T.tipo == "FP"]; longos = fp[fp.dur_h > 100]; print(f"   FP longos (> 100 h): {len(longos)}; nascimento: {longos.nasce_em.value_counts().to_dict()}", flush=True)
    print("\n(2) CONTRAFACTUAL: o detector inteiro com um só nível", flush=True)
    for nome, ch, val in (("SÓ nível A (B desligado)", "voto_nivel_b", 99), ("SÓ nível B (A desligado)", "voto_nivel_a", 99)):
        d = copy.deepcopy(det); d[ch] = val; Rn = roda(d); r = avalia(Rn["is_anomaly"].astype(bool).reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8)
        print(f"   {nome}: {r.det}/8, início {r.inicio}, FP {r.fp} ({r.fp_mes:.2f}/mês), fora {r.horas_alarme_fora:.0f} h, totais {r.horas_alarme:.0f} h | por evento (detectado): { {k[:10]: x['det'] for k, x in r.por_evento.items()} }", flush=True)
    print(f"   (publicado: {r0.det}/8, FP {r0.fp}, totais {r0.horas_alarme:.0f} h | por evento: { {k[:10]: x['det'] for k, x in r0.por_evento.items()} })", flush=True)
    print("\n(3) FRAÇÃO DE TEMPO VIGIADO EM QUE CADA NÍVEL ESTÁ LIGADO (o nível, antes de refratário e duração)", flush=True)
    print(f"   A ligado {100 * vA.loc[INICIO8:].sum() / max(mask.loc[INICIO8:].sum(), 1):.1f}% | B ligado {100 * vB.loc[INICIO8:].sum() / max(mask.loc[INICIO8:].sum(), 1):.1f}% | só A {100 * (vA & ~vB).loc[INICIO8:].sum() / max(mask.loc[INICIO8:].sum(), 1):.1f}% | só B {100 * (vB & ~vA).loc[INICIO8:].sum() / max(mask.loc[INICIO8:].sum(), 1):.1f}% | ambos {100 * (vA & vB).loc[INICIO8:].sum() / max(mask.loc[INICIO8:].sum(), 1):.1f}%", flush=True)


if __name__ == "__main__":
    main()
