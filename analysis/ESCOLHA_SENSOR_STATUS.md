# Escolha do sensor de status (ligado/desligado) por equipamento

Baseado na doc do time PdM (DOC/): para bombas o melhor indicador de parada é a
**pressão de descarga** (bomba parada não gera pressão). Aqui comparamos candidatos por:

- **%parado** = fração do tempo abaixo do limiar operacional estimado.
- **preserva** = dos pontos com sinal-alvo alto (vibração/temp no p90), quantos têm o
  candidato em operação. **Alto = o sensor NÃO apaga o sinal da falha** (é o que queremos).

| Equip | Alvo | Candidato | %parado | preserva sinal |
|---|---|---|---|---|
| `B-0302C` | Vibração Mancal Bomba LA | Corrente Media Motor (corrente) ⭐ | 85.4% | 12.1% |
| `B-0302C` |  | Potência Ativa (potencia_ativa) | 89.0% | 12.1% |
| `B-24001B` | VIBRAÇÃO DO MANCAL BOMBA LNA | PRESSÃO NA DESCARGA DA BOMBA (descarga) ⭐ | 8.5% | 99.9% |
| `B-24001B` |  | SENSOR RADIAL VELOCIDADE (velocidade) | 100.0% | 0.0% |
| `B-3403C` | Vibração | Corrente (corrente) ⭐ | 54.5% | 100.0% |
| `B-3403C` |  | Press Desc (descarga) | 39.5% | 99.4% |
| `B-402E` | Corrente | Corrente (corrente) ⭐ | 69.3% | 100.0% |
| `B-402E` |  | Vazão (vazao) | 69.2% | 99.9% |
| `B-402E` |  | Pressão Descarga (descarga) | 69.3% | 99.9% |
| `B-4064A` | Vibração Bomba LNA | — | — | (dados/alvo indisponível) |
| `B-4703.24001B` | Vibração Motor LNA X | Vazão (vazao) ⭐ | 11.5% | 99.8% |
| `B-4703.24001B` |  | Corrente (corrente) | 76.3% | 99.5% |
| `B-5401A` | Corrente | Indicador de Velocidade (velocidade) ⭐ | 54.3% | 100.0% |
| `B-5401A` |  | Corrente (corrente) | 56.6% | 100.0% |
| `B-5401A` |  | Pressão de Descarga (descarga) | 25.7% | 99.9% |
| `B-5501B` | Temperatura Mancal Bomba LA | Pressão Descarga (descarga) ⭐ | 38.2% | 99.6% |
| `B-5501B` |  | Vazão (vazao) | 16.5% | 99.2% |
| `B-5501B` |  | Corrente (corrente) | 52.8% | 98.6% |
| `B-6511502A` | VIB. MANCAL RADIAL BB LA 0° VE-50C | PRESSÃO DESCARGA (descarga) ⭐ | 65.4% | 100.0% |
| `B-6511502A` |  | CORRENTE ELÉTRICA DO MOTOR (corrente) | 65.8% | 100.0% |
| `B-6511502A` |  | VELOCIDADE BB KE-50 (velocidade) | 65.9% | 100.0% |
| `B-6511502A` |  | POTÊNCIA ATIVA MOTOR BOMBA (potencia_ativa) | 65.9% | 100.0% |
| `B-6511502A` |  | VAZÃO DESCARGA (vazao) | 65.9% | 99.8% |
| `B-8801C` | Vibração Bomba LA | Corrente (corrente) ⭐ | 88.6% | 100.0% |
| `B-8802B` | Vibração Bomba LA | — | — | (dados/alvo indisponível) |
| `B-90001A` | Vibração Bomba LA X | — | — | (dados/alvo indisponível) |

## Decisão (sensor de status por equipamento)

Regra: escolher o candidato que **melhor preserva o sinal** (⭐). Para bombas, tende a
ser a pressão de descarga; corrente fica como fallback (e falha onde está zerada).

```json
{
  "B-0302C": {
    "sensor": "Corrente Media Motor",
    "op_thr": 13.0,
    "preserva": 12.1
  },
  "B-24001B": {
    "sensor": "PRESSÃO NA DESCARGA DA BOMBA",
    "op_thr": 27.54,
    "preserva": 99.9
  },
  "B-3403C": {
    "sensor": "Corrente",
    "op_thr": 57.5,
    "preserva": 100.0
  },
  "B-402E": {
    "sensor": "Corrente",
    "op_thr": 177.69,
    "preserva": 100.0
  },
  "B-4703.24001B": {
    "sensor": "Vazão",
    "op_thr": 254.02,
    "preserva": 99.8
  },
  "B-5401A": {
    "sensor": "Indicador de Velocidade",
    "op_thr": 1435.38,
    "preserva": 100.0
  },
  "B-5501B": {
    "sensor": "Pressão Descarga",
    "op_thr": 18.43,
    "preserva": 99.6
  },
  "B-6511502A": {
    "sensor": "PRESSÃO DESCARGA",
    "op_thr": 41.05,
    "preserva": 100.0
  },
  "B-8801C": {
    "sensor": "Corrente",
    "op_thr": 52.01,
    "preserva": 100.0
  }
}
```
