# Triagem automática de episódios — glitch vs regime vs precursor vs indefinido

Classifica cada episódio (`is_anom_point` contínuo) por mecanismo físico provável,
usando o sinal bruto ao redor (baseline antes/depois, salto máximo dentro do
episódio, estado operacional, se há parada logo depois). Critérios:

- **glitch_sensor**: salto ≥8x o passo típico do sinal E ≥20% do episódio em estado off/transiente.
- **mudanca_regime**: sinal se estabiliza num patamar novo (|baseline_depois−antes| ≥4x o passo típico) sem ser glitch.
- **precursor_parada**: episódio ≥30min, sem glitch/regime, seguido de parada em até 6h.
- **sustentado_sem_causa**: ≥30min sem nenhum dos padrões acima — revisão manual.
- **transiente_curto**: sobra (curto/fraco, sem padrão claro).

### Uni_sensor (univariado)

| Classe | far | near_10d | near_48h | dur mediana (min) | peak_ratio mediana |
|---|---|---|---|---|---|
| glitch_sensor | 204 | 7 | 0 | 20.0 | 1.59 |
| mudanca_regime | 290 | 15 | 2 | 27.0 | 1.44 |
| precursor_parada | 67 | 4 | 2 | 57.0 | 1.58 |
| sustentado_sem_causa | 46 | 3 | 2 | 75.0 | 1.61 |
| transiente_curto | 174 | 0 | 1 | 4.0 | 1.15 |

### Mult_sensor (multivariado, canal-alvo)

| Classe | far | near_10d | near_48h | dur mediana (min) | peak_ratio mediana |
|---|---|---|---|---|---|
| glitch_sensor | 283 | 11 | 7 | 80.0 | 1.57 |
| mudanca_regime | 425 | 22 | 5 | 70.5 | 1.37 |
| precursor_parada | 185 | 5 | 1 | 80.0 | 1.44 |
| sustentado_sem_causa | 117 | 4 | 2 | 95.0 | 1.35 |
| transiente_curto | 28 | 1 | 0 | 8.0 | 1.15 |

### Pooled (uni + mult)

| Classe | far | near_10d | near_48h | dur mediana (min) | peak_ratio mediana |
|---|---|---|---|---|---|
| glitch_sensor | 487 | 18 | 7 | 55.0 | 1.58 |
| mudanca_regime | 715 | 37 | 7 | 59.0 | 1.38 |
| precursor_parada | 252 | 9 | 3 | 75.0 | 1.46 |
| sustentado_sem_causa | 163 | 7 | 4 | 90.5 | 1.41 |
| transiente_curto | 202 | 1 | 1 | 4.5 | 1.15 |
