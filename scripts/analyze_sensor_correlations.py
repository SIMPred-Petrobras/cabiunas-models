"""
analyze_sensor_correlations.py — Análise estatística de correlação entre sensores.

Fluxo por equipamento:
  1. Carrega feather via load_data_transpetro
  2. Filtra períodos operacionais (máscara ON) e exclui janela de falha
  3. Faz downsample para reduzir autocorrelação (~10 min de intervalo)
  4. Testa distribuição de cada sensor (skewness / kurtosis)
  5. Computa Pearson E Spearman; usa Spearman quando distribuição é não-normal
  6. Gera figuras: distribuições, heatmaps, ranking com target
  7. Salva JSON com SENSOR_GROUPS sugerido pronto para colar no config

Uso:
    PYTHONPATH=. python scripts/analyze_sensor_correlations.py \\
        --config configs/transpetro/B-8802B.json \\
        --target "Vibração Bomba LA" \\
        --output analysis/B-8802B

Flags:
    --threshold   |r| mínimo para incluir sensor no grupo (padrão: 0.30)
    --alpha       nível de significância p-value (padrão: 0.05)
    --step        passo do downsample; 0 = auto (~10 min) (padrão: 0)
    --no-mask     desativa filtro operacional (usa todos os dados)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Carregamento e limpeza
# ---------------------------------------------------------------------------

def _load_clean_data(
    cfg,
    target_sensor: str,
    apply_mask: bool,
    step: int,
) -> pd.DataFrame:
    from src.transpetro.io import load_data_transpetro
    from src.cnn1d_ae.preprocess import build_exclusion_mask
    from src.cnn1d_ae.scoring import build_operational_state

    df_alarm, _, df_raw, _ = load_data_transpetro(cfg)

    # Índice datetime
    df = df_raw.set_index(cfg.TIME_COL).sort_index()
    sensor_cols = [c for c in df.columns if c != cfg.TIME_COL]
    df = df[sensor_cols]

    # Exclui janela de falha
    if len(df_alarm) and "Data da Ocorrencia" in df_alarm.columns:
        excl = build_exclusion_mask(
            df.index,
            df_alarm["Data da Ocorrencia"].dropna(),
            cfg.EXCLUDE_MINUTES_AROUND_ALARM,
        )
        n_before = len(df)
        df = df.loc[~excl]
        print(f"  Excluídos {n_before - len(df):,} pontos da janela de falha "
              f"(±{cfg.EXCLUDE_MINUTES_AROUND_ALARM} min)")

    # Máscara operacional
    if apply_mask and cfg.ENABLE_OPERATIONAL_MASK and cfg.OPERATIONAL_REF_SENSOR:
        ref = cfg.OPERATIONAL_REF_SENSOR
        if ref in df.columns:
            state = build_operational_state(
                index=df.index,
                sensor_series=df[ref],
                off_value_quantile=cfg.OFF_VALUE_QUANTILE,
                off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
                off_long_min_hours=cfg.OFF_LONG_MIN_HOURS,
                transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
                transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
            )
            n_before = len(df)
            df = df.loc[state == "on"]
            removed = n_before - len(df)
            pct = 100 * removed / n_before
            print(f"  Máscara operacional ({ref}): removidos {removed:,} pontos "
                  f"({pct:.1f}%) — off/transiente")
        else:
            print(f"  [WARN] OPERATIONAL_REF_SENSOR='{ref}' não encontrado; máscara ignorada")

    # Drop colunas com NaN excessivo (> 50%)
    thresh = int(0.5 * len(df))
    df = df.dropna(axis=1, thresh=thresh)
    df = df.dropna()

    # Remove colunas com variância zero
    std_ok = df.std() > 1e-8
    dropped = std_ok[~std_ok].index.tolist()
    if dropped:
        print(f"  [WARN] Removidas por std≈0: {dropped}")
        df = df.loc[:, std_ok]

    # Verifica target
    if target_sensor not in df.columns:
        raise ValueError(
            f"target_sensor='{target_sensor}' não encontrado. "
            f"Disponíveis: {list(df.columns)}"
        )

    # Downsample para reduzir autocorrelação
    if step == 0:
        dt_s = df.index.to_series().diff().dt.total_seconds().median()
        step = max(1, int(round(600 / dt_s))) if np.isfinite(dt_s) and dt_s > 0 else 10
    n_before = len(df)
    df = df.iloc[::step]
    print(f"  Downsample 1:{step} → {len(df):,} amostras (era {n_before:,}, "
          f"~{step * (dt_s if 'dt_s' in dir() else 60) / 60:.0f} min/ponto)")

    return df


# ---------------------------------------------------------------------------
# Análise de distribuição
# ---------------------------------------------------------------------------

SKEW_THRESH = 1.0      # |skewness| acima disso → não-normal
KURT_THRESH = 3.0      # |excess kurtosis| acima disso → não-normal


def _check_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Skewness, kurtosis, classificação e teste D'Agostino-Pearson."""
    rows = []
    for col in df.columns:
        s = df[col].dropna().values
        skew = float(stats.skew(s))
        kurt = float(stats.kurtosis(s))          # excess kurtosis (normal=0)
        try:
            stat_dp, p_dp = stats.normaltest(s)
        except Exception:
            stat_dp, p_dp = np.nan, np.nan

        is_normal = abs(skew) < SKEW_THRESH and abs(kurt) < KURT_THRESH
        rows.append({
            "sensor": col,
            "n": len(s),
            "media": float(np.mean(s)),
            "mediana": float(np.median(s)),
            "std": float(np.std(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "skewness": round(skew, 4),
            "kurtosis_excess": round(kurt, 4),
            "normaltest_stat": round(float(stat_dp), 4) if np.isfinite(stat_dp) else np.nan,
            "normaltest_p": round(float(p_dp), 6) if np.isfinite(p_dp) else np.nan,
            "classificacao": "normal-ish" if is_normal else "nao-normal",
            "estatistica_recomendada": "pearson" if is_normal else "spearman",
        })
    return pd.DataFrame(rows).set_index("sensor")


# ---------------------------------------------------------------------------
# Correlações
# ---------------------------------------------------------------------------

def _compute_correlations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna (pearson_r, spearman_r, pearson_p, spearman_p) como DataFrames."""
    cols = df.columns.tolist()
    n = len(cols)
    pr = pd.DataFrame(np.nan, index=cols, columns=cols)
    pp = pd.DataFrame(np.nan, index=cols, columns=cols)
    sr = pd.DataFrame(np.nan, index=cols, columns=cols)
    sp = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                pr.loc[c1, c2] = sr.loc[c1, c2] = 1.0
                pp.loc[c1, c2] = sp.loc[c1, c2] = 0.0
                continue
            if i > j:
                pr.loc[c1, c2] = pr.loc[c2, c1]
                pp.loc[c1, c2] = pp.loc[c2, c1]
                sr.loc[c1, c2] = sr.loc[c2, c1]
                sp.loc[c1, c2] = sp.loc[c2, c1]
                continue
            x, y = df[c1].values, df[c2].values
            try:
                rp, pvp = stats.pearsonr(x, y)
                pr.loc[c1, c2] = rp; pp.loc[c1, c2] = pvp
            except Exception:
                pass
            try:
                rs, pvs = stats.spearmanr(x, y)
                sr.loc[c1, c2] = rs; sp.loc[c1, c2] = pvs
            except Exception:
                pass

    return pr, sr, pp, sp


def _build_target_ranking(
    target: str,
    dist_df: pd.DataFrame,
    pearson_r: pd.DataFrame,
    spearman_r: pd.DataFrame,
    pearson_p: pd.DataFrame,
    spearman_p: pd.DataFrame,
    threshold: float,
    alpha: float,
) -> pd.DataFrame:
    """Ranking de sensores por correlação com o target."""
    target_normal = dist_df.loc[target, "classificacao"] == "normal-ish"
    rows = []
    for s in dist_df.index:
        if s == target:
            continue
        s_normal = dist_df.loc[s, "classificacao"] == "normal-ish"
        use_pearson = target_normal and s_normal

        r_p = float(pearson_r.loc[s, target])
        p_p = float(pearson_p.loc[s, target])
        r_s = float(spearman_r.loc[s, target])
        p_s = float(spearman_p.loc[s, target])

        r_rec = r_s if not use_pearson else r_p
        p_rec = p_s if not use_pearson else p_p
        stat_used = "pearson" if use_pearson else "spearman"

        incluir = abs(r_rec) >= threshold and p_rec < alpha
        rows.append({
            "sensor": s,
            "r_pearson": round(r_p, 4),
            "p_pearson": round(p_p, 6),
            "r_spearman": round(r_s, 4),
            "p_spearman": round(p_s, 6),
            "r_recomendado": round(r_rec, 4),
            "estatistica_usada": stat_used,
            "|r|": round(abs(r_rec), 4),
            "p_recomendado": round(p_rec, 6),
            "significativo": p_rec < alpha,
            "incluir_no_grupo": incluir,
        })
    df_rank = pd.DataFrame(rows).sort_values("|r|", ascending=False)
    return df_rank.set_index("sensor")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_distributions(df: pd.DataFrame, dist_df: pd.DataFrame, out_dir: Path) -> None:
    cols = df.columns.tolist()
    ncols = min(4, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, cols):
        s = df[col].dropna().values
        row = dist_df.loc[col]
        color = "#2196F3" if row["classificacao"] == "normal-ish" else "#FF5722"
        ax.hist(s, bins=50, density=True, alpha=0.4, color=color, edgecolor="none")
        try:
            kde = stats.gaussian_kde(s)
            xs = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=color, lw=1.5)
        except Exception:
            pass
        ax.axvline(row["mediana"], color="black", lw=1, ls="--", alpha=0.7)
        ax.set_title(col, fontsize=8, pad=4)
        label = (f"skew={row['skewness']:.2f}\n"
                 f"kurt={row['kurtosis_excess']:.2f}\n"
                 f"{row['classificacao']}")
        ax.text(0.97, 0.95, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5,
                color=color, fontweight="bold")
        ax.tick_params(labelsize=6)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Distribuição dos sensores (azul=normal-ish | laranja=não-normal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'distributions.png'}")


def _plot_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    out_dir: Path,
    filename: str,
    target: Optional[str] = None,
) -> None:
    if target and target in corr_df.columns:
        order = [target] + [c for c in corr_df.columns
                            if c != target and c in corr_df.index]
        corr_df = corr_df.loc[order, order]

    size = max(6, len(corr_df) * 0.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    mask = np.zeros_like(corr_df.values, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr_df.astype(float),
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        linewidths=0.4,
        annot_kws={"size": 7},
        mask=mask,
        square=True,
    )
    if target:
        ax.axhline(1, color="gold", lw=2)
        ax.axvline(1, color="gold", lw=2)
    ax.set_title(title, fontsize=10, pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / filename}")


def _plot_correlation_bars(
    rank_df: pd.DataFrame,
    target: str,
    threshold: float,
    alpha: float,
    out_dir: Path,
) -> None:
    df = rank_df.sort_values("r_spearman", key=abs, ascending=False)
    sensors = df.index.tolist()
    x = np.arange(len(sensors))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(sensors) * 0.9), 5))

    bars_p = ax.bar(x - w / 2, df["r_pearson"], w,
                    label="Pearson", color="#2196F3", alpha=0.75, edgecolor="white")
    bars_s = ax.bar(x + w / 2, df["r_spearman"], w,
                    label="Spearman", color="#FF5722", alpha=0.75, edgecolor="white")

    # Marca não-significativo com hachura
    for bar, p_val in zip(bars_p, df["p_pearson"]):
        if p_val >= alpha:
            bar.set_hatch("///"); bar.set_alpha(0.35)
    for bar, p_val in zip(bars_s, df["p_spearman"]):
        if p_val >= alpha:
            bar.set_hatch("///"); bar.set_alpha(0.35)

    ax.axhline(threshold, color="green", lw=1.2, ls="--",
               label=f"threshold={threshold}")
    ax.axhline(-threshold, color="green", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.6)

    # Destaca sensores que entram no grupo
    for i, s in enumerate(sensors):
        if rank_df.loc[s, "incluir_no_grupo"]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="green", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Correlação com target")
    ax.set_title(f"Correlação com '{target}'\n"
                 f"(fundo verde = selecionado para o grupo | hachura = p ≥ {alpha})",
                 fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'correlation_bars.png'}")


# ---------------------------------------------------------------------------
# JSON de saída
# ---------------------------------------------------------------------------

def _build_sensor_groups_json(
    equip_id: str,
    target: str,
    rank_df: pd.DataFrame,
) -> dict:
    selected = [s for s in rank_df.index if rank_df.loc[s, "incluir_no_grupo"]]
    group = {
        "name": equip_id,
        "sensors": [target] + [s for s in selected if s != target],
        "target_sensor": target,
    }
    return {"SENSOR_GROUPS": [group]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Análise estatística de correlação entre sensores Transpetro."
    )
    parser.add_argument("--config", required=True,
                        help="Caminho para o JSON de configuração do equipamento.")
    parser.add_argument("--target", required=True,
                        help="Nome do sensor-alvo (o que falhou).")
    parser.add_argument("--output", required=True,
                        help="Diretório de saída para figuras e CSVs.")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="|r| mínimo para incluir sensor no grupo (padrão: 0.30).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Nível de significância p-value (padrão: 0.05).")
    parser.add_argument("--step", type=int, default=0,
                        help="Passo do downsample; 0=auto (~10 min) (padrão: 0).")
    parser.add_argument("--no-mask", action="store_true",
                        help="Desativa filtro operacional.")
    args = parser.parse_args()

    from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict
    cfg = PipelineConfig()
    cfg = update_cfg_from_dict(cfg, json.loads(Path(args.config).read_text()))

    equip_id = cfg.EQUIPMENT_ID or Path(args.config).stem
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Equipamento : {equip_id}")
    print(f"Target      : {args.target}")
    print(f"Threshold   : |r| ≥ {args.threshold}  p < {args.alpha}")
    print(f"{'='*60}")

    # 1. Carrega e limpa
    print("\n[1/5] Carregando e filtrando dados...")
    df = _load_clean_data(cfg, args.target, not args.no_mask, args.step)
    print(f"  Shape final: {df.shape}  |  sensores: {list(df.columns)}")

    # 2. Distribuição
    print("\n[2/5] Analisando distribuições...")
    dist_df = _check_distributions(df)
    n_normal = (dist_df["classificacao"] == "normal-ish").sum()
    print(f"  {n_normal}/{len(dist_df)} sensores classificados como normal-ish")
    print(f"  Target '{args.target}': {dist_df.loc[args.target, 'classificacao']} "
          f"(skew={dist_df.loc[args.target, 'skewness']:.3f}, "
          f"kurt={dist_df.loc[args.target, 'kurtosis_excess']:.3f})")

    dist_df.to_csv(out_dir / "distribution_summary.csv")
    print(f"  → {out_dir / 'distribution_summary.csv'}")

    # 3. Correlações
    print("\n[3/5] Calculando correlações (Pearson + Spearman)...")
    pearson_r, spearman_r, pearson_p, spearman_p = _compute_correlations(df)
    pearson_r.to_csv(out_dir / "correlation_pearson.csv")
    spearman_r.to_csv(out_dir / "correlation_spearman.csv")

    rank_df = _build_target_ranking(
        args.target, dist_df, pearson_r, spearman_r, pearson_p, spearman_p,
        args.threshold, args.alpha,
    )
    rank_df.to_csv(out_dir / "correlation_with_target.csv")
    print(f"  → {out_dir / 'correlation_with_target.csv'}")

    n_sel = rank_df["incluir_no_grupo"].sum()
    print(f"\n  Sensores selecionados (|r|≥{args.threshold} e p<{args.alpha}): "
          f"{n_sel}/{len(rank_df)}")
    for s, row in rank_df[rank_df["incluir_no_grupo"]].iterrows():
        print(f"    ✓ {s:40s}  r={row['r_recomendado']:+.3f}  "
              f"({row['estatistica_usada']})  p={row['p_recomendado']:.2e}")
    not_sel = rank_df[~rank_df["incluir_no_grupo"]]
    if len(not_sel):
        print(f"  Sensores excluídos:")
        for s, row in not_sel.iterrows():
            print(f"    ✗ {s:40s}  r={row['r_recomendado']:+.3f}  "
                  f"p={row['p_recomendado']:.2e}")

    # 4. Plots
    print("\n[4/5] Gerando figuras...")
    _plot_distributions(df, dist_df, out_dir)
    _plot_heatmap(pearson_r, f"Pearson — {equip_id}", out_dir,
                  "heatmap_pearson.png", target=args.target)
    _plot_heatmap(spearman_r, f"Spearman — {equip_id}", out_dir,
                  "heatmap_spearman.png", target=args.target)
    _plot_correlation_bars(rank_df, args.target, args.threshold, args.alpha, out_dir)

    # 5. JSON sugerido
    print("\n[5/5] Gerando SENSOR_GROUPS sugerido...")
    sg = _build_sensor_groups_json(equip_id, args.target, rank_df)
    sg_path = out_dir / "sensor_groups_suggested.json"
    sg_path.write_text(json.dumps(sg, indent=2, ensure_ascii=False))
    print(f"  → {sg_path}")
    print(f"\n  Cole em configs/transpetro/{equip_id}.json:")
    print(json.dumps(sg, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Concluído. Resultados em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
