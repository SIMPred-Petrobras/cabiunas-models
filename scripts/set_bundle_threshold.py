#!/usr/bin/env python3
"""
set_bundle_threshold.py
Ajusta o threshold de alarme de bundles de produção JÁ deployados, sem retreinar e
sem acessar o ClearML. É o botão para o cliente/operador: "quero o alarme mais
conservador" vira uma mudança de `y` em `mu + y*sigma`.

Três formas de especificar o novo threshold (escolha uma):
  --std_mult Y   threshold = ewma_mean + Y * ewma_std   (régua em desvios-padrão)
  --scale  F     threshold = threshold_atual * F        (ex.: 1.10 = +10%)
  --abs    V     threshold = V                          (valor absoluto)

`ewma_mean`/`ewma_std` são gravados por `scripts/finalize_bundle.py`. Bundles antigos
que não os tenham só aceitam --scale/--abs (rode finalize_bundle de novo para liberar
--std_mult).

Toda alteração fica registrada em `production_alerting.threshold_history`, então dá
para auditar e voltar atrás.

⚠️ Subir o threshold reduz falso alarme MAS custa recall, e o custo não é linear: no
TC382_03_A/2025 o ponto calibrado equivale a y≈0,20 e y=3,0 zera a detecção
(ver scripts/sweep_threshold_mean_std.py). Meça antes de subir.

Uso:
    PYTHONPATH=. python scripts/set_bundle_threshold.py \
        production_bundles/TC382_03_A_inference_bundle.json --std_mult 0.5
    PYTHONPATH=. python scripts/set_bundle_threshold.py \
        production_bundles/*_inference_bundle.json --scale 1.15 --dry_run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys


def new_threshold(pa: dict, args) -> tuple[float, str]:
    """Retorna (novo_threshold, descrição da regra)."""
    if args.std_mult is not None:
        mu, sigma = pa.get("ewma_mean"), pa.get("ewma_std")
        if mu is None or sigma is None:
            raise SystemExit(
                "bundle sem 'ewma_mean'/'ewma_std' em production_alerting — não dá para usar "
                "--std_mult. Rode scripts/finalize_bundle.py (versão atual) para gravá-los, "
                "ou use --scale/--abs."
            )
        return float(mu) + args.std_mult * float(sigma), f"mean_std (y={args.std_mult:+.3f})"
    if args.scale is not None:
        return float(pa["ewma_abs_threshold"]) * args.scale, f"scale (x{args.scale:.3f})"
    return args.abs, "abs"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundles", nargs="+", help="caminhos dos *_inference_bundle.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--std_mult", type=float, help="threshold = ewma_mean + Y*ewma_std")
    g.add_argument("--scale", type=float, help="multiplica o threshold atual por F")
    g.add_argument("--abs", type=float, help="threshold absoluto")
    p.add_argument("--reason", default="", help="motivo, gravado no histórico")
    p.add_argument("--dry_run", action="store_true", help="mostra o efeito sem gravar")
    args = p.parse_args()

    if args.scale is not None and args.scale <= 0:
        raise SystemExit("--scale precisa ser > 0")

    for path in args.bundles:
        with open(path, encoding="utf-8") as f:
            bundle = json.load(f)
        pa = bundle.get("production_alerting")
        if pa is None:
            print(f"[skip] {path}: sem 'production_alerting' (rode finalize_bundle.py antes)")
            continue

        old = float(pa["ewma_abs_threshold"])
        thr, rule = new_threshold(pa, args)
        delta = (thr / old - 1.0) * 100.0 if old else float("nan")
        sigma = pa.get("ewma_std")
        y_txt = f"  y={(thr - pa['ewma_mean']) / sigma:+.3f}" if sigma else ""
        print(f"{bundle.get('sensor', path)}: {old:.6f} → {thr:.6f} ({delta:+.1f}%)"
              f"  [{rule}]{y_txt}")

        if args.dry_run:
            continue

        pa.setdefault("threshold_history", []).append({
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "from": old,
            "to": thr,
            "rule": rule,
            "reason": args.reason,
        })
        pa["ewma_abs_threshold"] = thr
        pa["thresh_rule"] = "mean_std" if args.std_mult is not None else "manual"
        if sigma:
            pa["std_mult"] = (thr - float(pa["ewma_mean"])) / float(sigma)
        # o threshold_q registrado deixa de valer quando o valor é sobrescrito à mão
        pa["threshold_q_stale"] = pa.pop("threshold_q", None)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print("\n(dry-run: nada foi gravado)", file=sys.stderr)


if __name__ == "__main__":
    main()
