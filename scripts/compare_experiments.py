#!/usr/bin/env python3
"""Compara dois experimentos (ex.: experimento_1_mascara_v3 vs
experimento_2_supressao_transiente) usando a mesma métrica de sempre —
BOM/PARCIAL/FRACO (scripts/analyze_failure_detection.py::analyze) — e a taxa
de anomalia/dia, para ver se uma mudança de pipeline reduziu falso positivo
sem derrubar a detecção real.

Uso:
    PYTHONPATH=. python scripts/compare_experiments.py \
        --exp1 resultados/experimento_1_mascara_v3 \
        --exp2 resultados/experimento_2_supressao_transiente \
        --label1 "exp1 (mascara v3)" --label2 "exp2 (+ supressao transiente_curto)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_failure_detection import analyze  # noqa: E402

EQUIPS = ["B-0302C", "B-24001B", "B-3403C", "B-402E", "B-4064A", "B-4703.24001B",
          "B-5401A", "B-5501B", "B-6511502A", "B-8801C", "B-8802B", "B-90001A"]

RANK = {"BOM": 3, "PARCIAL": 2, "FRACO": 1, "SEM_DADOS": 0}


def _suppressed(eq_dir: Path) -> int:
    for p in eq_dir.rglob("calibration_report.json"):
        return json.loads(p.read_text(encoding="utf-8")).get("suppressed_transient_episodes", 0) or 0
    return 0


def compare_mode(mode: str, exp1_root: Path, exp2_root: Path, label1: str, label2: str) -> list[str]:
    lines = [
        f"### {mode}",
        "",
        f"| Equip | {label1} | {label2} | rate/dia ({label1}) | rate/dia ({label2}) | Δrate | suprimidos (exp2) | Veredito |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for eq in EQUIPS:
        d1, d2 = exp1_root / eq, exp2_root / eq
        r1 = analyze(eq, d1) if d1.exists() else {"classe": "SEM_DADOS", "rate_day": None}
        r2 = analyze(eq, d2) if d2.exists() else {"classe": "SEM_DADOS", "rate_day": None}
        supp = _suppressed(d2) if d2.exists() else 0

        c1, c2 = r1["classe"], r2["classe"]
        rate1, rate2 = r1.get("rate_day"), r2.get("rate_day")
        drate = (rate2 - rate1) if isinstance(rate1, (int, float)) and isinstance(rate2, (int, float)) else None
        drate_s = f"{drate:+.2f}" if drate is not None else "—"

        if RANK.get(c2, 0) < RANK.get(c1, 0):
            veredito = "⚠️ PIOROU classe"
        elif RANK.get(c2, 0) > RANK.get(c1, 0):
            veredito = "✅ MELHOROU classe"
        elif drate is not None and drate < -0.01:
            veredito = "✅ menos ruído, classe igual"
        elif drate is not None and drate > 0.01:
            veredito = "⚠️ mais ruído, classe igual"
        else:
            veredito = "= igual"

        lines.append(
            f"| `{eq}` | {c1} | {c2} | {rate1 if rate1 is not None else '—'} | "
            f"{rate2 if rate2 is not None else '—'} | {drate_s} | {supp} | {veredito} |"
        )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1", required=True)
    ap.add_argument("--exp2", required=True)
    ap.add_argument("--label1", default="experimento_1")
    ap.add_argument("--label2", default="experimento_2")
    ap.add_argument("--out", default="analysis/COMPARACAO_experimento_1_vs_experimento_2.md")
    args = ap.parse_args()

    exp1_root, exp2_root = Path(args.exp1), Path(args.exp2)

    lines = [
        f"# Comparação: {args.label1} vs {args.label2}",
        "",
        f"- **{args.label1}:** `{exp1_root}`",
        f"- **{args.label2}:** `{exp2_root}`",
        "",
        "Classe: BOM (detectou em ±48h da falha) · PARCIAL (só precursor ≤10d) · "
        "FRACO (nada em ±10d). `rate/dia` = anomalias de ponto por dia (ruído).",
        "",
    ]
    lines += compare_mode("Uni_sensor", exp1_root / "Uni_sensor", exp2_root / "Uni_sensor", args.label1, args.label2)
    lines += compare_mode("Mult_sensor", exp1_root / "Mult_sensor", exp2_root / "Mult_sensor", args.label1, args.label2)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] {out_p}")


if __name__ == "__main__":
    main()
