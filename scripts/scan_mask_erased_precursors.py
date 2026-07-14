#!/usr/bin/env python3
"""Varredura: quantos equipamentos têm janelas perto da falha real onde o MAE
bruto (mae_seq) fica sustentado acima do threshold, mas a detecção final
(is_anom_point) fica zerada — sinal apagado pela máscara operacional, não por
falta de sinal real.

Achado motivador (B-3403C): ~30h antes da falha, corrente (sensor de status)
oscila violentamente (liga/transiente/desliga a cada 1-2min), e a máscara
decide por estado no MINUTO FINAL de cada janela — com o sensor piscando
tanto, quase toda sequência cai num instante "não-ligado" por acaso, apagando
uma detecção que deveria existir.

Método: para cada equipamento (uni/mult), dentro de ±10 dias da falha, agrupa
por bins de 6h e mede:
  - pct_mae_over: fração de sequências com mae_seq > threshold (sinal bruto)
  - pct_anom_seq: fração com is_anom_seq==1 (pós-máscara)
  - pct_on / pct_transiente / pct_off: composição do estado operacional
  - flicker: nº de transições de estado por hora (proxy de instabilidade)

Bin "apagado pela máscara" = pct_mae_over alto (sinal real) MAS pct_anom_seq
quase zero, MESMO com uma fração razoável do tempo em "on" (não é caso de
"máquina genuinamente desligada").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn1d_ae.pipeline import parse_failure_dates

UNI_DIR = Path("resultados/experimento_2_supressao_transiente/Uni_sensor")
MULT_DIR = Path("resultados/experimento_2_supressao_transiente/Mult_sensor")
OUT_MD = Path("analysis/MASCARA_APAGA_PRECURSOR_SCAN.md")

BIN = "2h"                  # fino o bastante p/ não diluir janelas curtas (~1h) de instabilidade
PCT_MAE_OVER_THR = 0.30     # ao menos 30% das sequências no bin cruzam o threshold
PCT_ANOM_SEQ_MAX = 0.05     # mas quase nada sobrevive à máscara
PCT_ON_MIN = 0.20           # e não é "genuinamente desligado" (tem uma fração relevante em on+transiente
                             # — "transiente" também é rejeitado pela máscara, mas é sinal presente,
                             # não ausência de operação; só off_longo/off_curto conta como "desligado")


def load_equip(eq_dir: Path) -> dict | None:
    seq_p = eq_dir / "csv" / "sequence_scores_all.csv"
    pt_p = eq_dir / "csv" / "point_anomalies_all.csv"
    calib_p = eq_dir / "csv" / "calibration_report.json"
    cfg_p = eq_dir / "csv" / "run_config.json"
    if not (seq_p.exists() and pt_p.exists() and calib_p.exists() and cfg_p.exists()):
        return None

    seq = pd.read_csv(seq_p)
    seq["t"] = pd.to_datetime(seq["seq_start_time"], errors="coerce")
    seq = seq.dropna(subset=["t"]).set_index("t").sort_index()
    if "is_anom_seq" not in seq.columns:
        return None

    # Guarda de qualidade (mesma do estudo de episódios): MAE degenerada
    # (IQR≈0, ex.: B-5401A) faz qualquer variação normal parecer "acima do
    # threshold" e produz falso achado de "máscara apagando sinal".
    q25, q75 = seq["mae_seq"].quantile([.25, .75])
    if (q75 - q25) <= 1e-12:
        print(f"[SKIP] {eq_dir.name}: mae_seq degenerada (IQR≈0) — excluído da varredura")
        return None

    pt = pd.read_csv(pt_p)
    pt["t"] = pd.to_datetime(pt.iloc[:, 0], errors="coerce")
    pt = pt.dropna(subset=["t"]).set_index("t").sort_index()
    state = pt["operational_state"] if "operational_state" in pt.columns else None

    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    fails = parse_failure_dates(cfg.get("FAILURE_DATE", ""))
    thr = float(calib.get("threshold", np.nan))
    return {"seq": seq, "state": state, "thr": thr, "fails": fails}


def scan_equip(eq: str, mode: str, data: dict) -> list[dict]:
    seq, state, thr, fails = data["seq"], data["state"], data["thr"], data["fails"]
    if not np.isfinite(thr) or not fails or seq.empty:
        return []

    findings = []
    for f in fails:
        # Datas de falha costumam ser só a data (00:00 de conveniência, sem hora
        # precisa) — alarga a janela pós-falha para +2d pra não perder eventos
        # que na verdade caem "antes" da hora real mas depois do timestamp nominal.
        t0, t1 = f - pd.Timedelta(days=10), f + pd.Timedelta(days=2)
        w = seq[(seq.index >= t0) & (seq.index <= t1)].copy()
        if w.empty:
            continue
        w["mae_over"] = w["mae_seq"] > thr

        if state is not None:
            st = state.reindex(w.index).ffill().fillna("on")
            flicker_all = state[(state.index >= t0) & (state.index <= t1)]
            n_transitions = int((flicker_all != flicker_all.shift()).sum())
            span_hours = max(1e-6, (t1 - t0).total_seconds() / 3600.0)
        else:
            st = pd.Series("on", index=w.index)
            n_transitions, span_hours = 0, 1.0

        w["is_on"] = st.eq("on").values
        w["is_transiente"] = st.eq("transiente").values
        # "presente" = on OU transiente (ambos rejeitados pela máscara hoje, mas
        # são sinal de operação real — só off_longo/off_curto é "genuinamente parado").
        w["is_present"] = st.isin(["on", "transiente"]).values

        bins = w.resample(BIN)
        agg = bins.agg(
            pct_mae_over=("mae_over", "mean"),
            pct_anom_seq=("is_anom_seq", "mean"),
            pct_on=("is_on", "mean"),
            pct_transiente=("is_transiente", "mean"),
            pct_present=("is_present", "mean"),
            n=("mae_seq", "size"),
        ).dropna()
        agg = agg[agg["n"] > 0]

        erased = agg[(agg["pct_mae_over"] >= PCT_MAE_OVER_THR) &
                     (agg["pct_anom_seq"] <= PCT_ANOM_SEQ_MAX) &
                     (agg["pct_present"] >= PCT_ON_MIN)]
        if len(erased):
            findings.append({
                "equip": eq, "mode": mode, "failure": str(f),
                "n_bins_erased": len(erased),
                "hours_erased": len(erased) * 6,
                "pct_mae_over_max": round(float(erased["pct_mae_over"].max()), 2),
                "pct_transiente_mean": round(float(erased["pct_transiente"].mean()), 2),
                "flicker_transitions_per_hour": round(n_transitions / span_hours, 2),
                "first_bin": str(erased.index.min()), "last_bin": str(erased.index.max()),
            })
    return findings


def main() -> None:
    all_findings = []
    for mode, root in (("uni", UNI_DIR), ("mult", MULT_DIR)):
        for eq_dir in sorted(root.iterdir()):
            if not (eq_dir.is_dir() and eq_dir.name.startswith("B-")):
                continue
            data = load_equip(eq_dir)
            if data is None:
                continue
            all_findings += scan_equip(eq_dir.name, mode, data)

    lines = [
        "# Varredura: máscara apagando precursor sustentado perto da falha",
        "",
        f"Critério: bins de {BIN} dentro de ±10d/+6h da falha com "
        f"`pct_mae_over>={PCT_MAE_OVER_THR}` (sinal bruto cruzando threshold) "
        f"E `pct_anom_seq<={PCT_ANOM_SEQ_MAX}` (quase nada sobrevive à máscara) "
        f"E `pct_on>={PCT_ON_MIN}` (não é caso de máquina genuinamente desligada).",
        "",
        f"**Equipamentos afetados: {len(set((f['equip'], f['mode']) for f in all_findings))} "
        f"ocorrência(s) (uni+mult) de {24} modelos avaliados.**",
        "",
        "| Equip | Modo | Falha | Bins apagados | Horas apagadas | %MAE-over máx | %transiente médio | flicker (transições/h) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in sorted(all_findings, key=lambda x: -x["hours_erased"]):
        lines.append(
            f"| `{f['equip']}` | {f['mode']} | {f['failure'][:10]} | {f['n_bins_erased']} | "
            f"{f['hours_erased']} | {f['pct_mae_over_max']} | {f['pct_transiente_mean']} | "
            f"{f['flicker_transitions_per_hour']} |"
        )
    lines.append("")
    if not all_findings:
        lines.append("_Nenhuma ocorrência encontrada com esse critério._")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] {OUT_MD}")


if __name__ == "__main__":
    main()
