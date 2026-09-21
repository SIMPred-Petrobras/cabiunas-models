#!/usr/bin/env python3
"""SIMPred / Cabiúnas — troca os pickles dos bundles por JSON, sem retreinar.

POR QUÊ. O bundle guardava `RobustScaler` e `PCA` em pickle. Mesmo sendo
sklearn puro — e já é um avanço sobre a v1 da Transpetro, cujos pickles
referenciavam classes internas — pickle continua sendo **código fechado**:

  · não se lê nem se compara num diff; um `.pkl` é opaco na revisão;
  · exige a versão certa do sklearn (`InconsistentVersionWarning` ao abrir com
    1.7.1 um objeto salvo com 1.8.0);
  · executa código arbitrário ao carregar — `pickle.load` em arquivo de
    terceiro é execução remota, e o dashboard roda em outro time;
  · prende a integração a Python, quando o consumidor pode ser outra coisa.

O QUE ELES GUARDAVAM. Só quatro vetores por família — conferido, não suposto:
`with_centering=True`, `with_scaling=True`, `unit_variance=False`,
`whiten=False`. A aritmética inteira é:

    Xs  = (X - center) / scale               RobustScaler.transform
    Z   = Xs @ componentsᵀ - mean_proj       PCA.transform
    rec = Z @ components + mean              PCA.inverse_transform

São 56 números na temperatura e 84 na pressão.

A ORDEM É A DO SKLEARN. Ele centraliza DEPOIS de projetar (`_base.py::_transform`:
"Apply the centering after the projection"), então o bundle publica
`mean_proj = mean @ componentsᵀ` já calculado. Escrever `(Xs - mean) @ componentsᵀ`
é algebricamente igual e difere em ~6e-14 — barato aqui, caro numa reunião em que
o número do dashboard não bate com o nosso.

ESTE SCRIPT NÃO RETREINA. Lê os pickles que já existem e escreve os mesmos
números em `<fam>_transformacao.json`. Nenhum valor muda; o que muda é o
formato. Os `.pkl` ficam onde estão até a engenharia confirmar a troca — use
`--remover-pkl` depois disso.

    python3 migra_bundle_json.py                 # todos os bundles de ../modelos
    python3 migra_bundle_json.py --remover-pkl   # e apaga os .pkl ao final
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np

FAMILIAS = ("temperatura", "pressao")


def migra(bundle: Path, remover: bool) -> tuple[int, list[str]]:
    escritos, avisos = 0, []
    for fam in FAMILIAS:
        pkl_s, pkl_p = bundle / f"{fam}_scaler.pkl", bundle / f"{fam}_pca.pkl"
        if not (pkl_s.exists() and pkl_p.exists()):
            avisos.append(f"{bundle.name}/{fam}: pickles ausentes, pulando")
            continue
        with open(pkl_s, "rb") as fh:
            s = pickle.load(fh)
        with open(pkl_p, "rb") as fh:
            pc = pickle.load(fh)

        # A aritmética publicada só vale nestas condições. Se um bundle futuro
        # vier com whiten=True, o JSON estaria mentindo — melhor parar.
        if not (s.with_centering and s.with_scaling and not s.unit_variance):
            raise ValueError(f"{bundle.name}/{fam}: RobustScaler fora do padrão")
        if pc.whiten:
            raise ValueError(f"{bundle.name}/{fam}: PCA com whiten=True")

        norm = json.loads((bundle / "normalizacao.json").read_text(encoding="utf-8"))
        cols = norm[fam]["cols"]
        if len(cols) != len(s.center_):
            raise ValueError(f"{bundle.name}/{fam}: {len(cols)} colunas contra "
                             f"{len(s.center_)} no scaler")

        (bundle / f"{fam}_transformacao.json").write_text(json.dumps({
            "formato": "robustscaler+pca, explicito",
            "aritmetica": [
                "Xs = (X - center) / scale",
                "Z  = Xs @ components.T - mean_proj",
                "rec = Z @ components + mean",
                "ORDEM IMPORTA: o sklearn centraliza DEPOIS de projetar, e por isso",
                "mean_proj = mean @ components.T ja vem calculado. Escrever",
                "(Xs - mean) @ components.T e algebricamente igual mas difere em",
                "~6e-14 -- suficiente para uma discussao de numero divergente."
            ],
            "cols": cols,
            "center": [float(v) for v in s.center_],
            "scale": [float(v) for v in s.scale_],
            "mean": [float(v) for v in pc.mean_],
            "mean_proj": [float(v) for v in (np.reshape(pc.mean_, (1, -1)) @ pc.components_.T)[0]],
            "components": [[float(v) for v in linha] for linha in pc.components_],
        }, indent=2), encoding="utf-8")
        escritos += 1

        # Conferência: o JSON tem de reproduzir o pickle em float64 exato.
        d = json.loads((bundle / f"{fam}_transformacao.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, len(cols))) * 10.0
        alvo = s.transform(X)
        alvo = alvo - pc.inverse_transform(pc.transform(alvo))
        c = {k: np.asarray(d[k], dtype="float64")
             for k in ("center", "scale", "mean", "mean_proj", "components")}
        Xs = (X - c["center"]) / c["scale"]
        Z = Xs @ c["components"].T
        Z = Z - c["mean_proj"]
        meu = Xs - (Z @ c["components"] + c["mean"])
        if not np.array_equal(alvo, meu):
            raise ValueError(f"{bundle.name}/{fam}: JSON não reproduz o pickle "
                             f"(max|dif| {np.max(np.abs(alvo - meu)):.3e})")

        if remover:
            pkl_s.unlink(); pkl_p.unlink()
    return escritos, avisos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default=None, help="pasta modelos/ (padrão: ../modelos)")
    ap.add_argument("--remover-pkl", action="store_true",
                    help="apaga os .pkl depois de conferir o JSON")
    a = ap.parse_args()

    raiz = Path(a.modelos) if a.modelos else Path(__file__).resolve().parent.parent / "modelos"
    bundles = sorted(p for p in raiz.glob("model_*_PCA4SINAIS") if p.is_dir())
    if not bundles:
        print(f"nenhum bundle em {raiz}")
        return 1

    total, todos_avisos = 0, []
    for b in bundles:
        n, avisos = migra(b, a.remover_pkl)
        total += n
        todos_avisos += avisos
        print(f"  {b.name}  ->  {n} JSON  (conferido contra o pickle)")
    for w in todos_avisos:
        print(f"  AVISO: {w}")
    print(f"\n{len(bundles)} bundles, {total} transformações escritas."
          f"{'  .pkl removidos.' if a.remover_pkl else '  .pkl mantidos.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
