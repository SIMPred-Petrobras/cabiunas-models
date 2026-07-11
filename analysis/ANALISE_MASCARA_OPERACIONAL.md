# Análise da máscara operacional — diagnóstico e recomendação

**Contexto:** a máscara zera qualquer anomalia fora do estado `on`. O estado é
definido pelo sensor `OPERATIONAL_REF_SENSOR`: fica `off` quando `ref ≤ OFF_ABS_THRESHOLD`
(hoje **fixo = 5.0** para todos). Onde a corrente opera em faixa baixa, 5.0 marca a
máquina como desligada a maior parte do tempo e apaga o sinal da falha.

## Tabela por equipamento

| Equip | Ref | Mediana | q75 | %OFF@5.0 | Limiar sugerido | %OFF sugerido | Máquina LIGADA perto da falha? |
|---|---|---|---|---|---|---|---|
| `B-0302C` | Corrente Media Motor | 0.0 | 0.0 | **85.0%** | 3.9 | 85.0% | 0.0% on (med=0.0) |
| `B-24001B` | PRESSÃO NA DESCARGA DA BOMBA | 42.8 | 52.4 | **1.5%** | 8.26 | 1.7% | 95.8% on (med=57.8) |
| `B-3403C` | Corrente | 0.0 | 110.0 | **54.5%** | 17.25 | 54.5% | 30.9% on (med=0.0) |
| `B-402E` | Corrente | 0.0 | 315.8 | **69.3%** | 53.31 | 69.3% | 8.2% on (med=0.0) |
| `B-4064A` | Corrente | — | — | — | — | — | ref 'Corrente' indisponível |
| `B-4703.24001B` | Corrente | 0.0 | 0.0 | **76.2%** | 11.21 | 76.2% | 0.3% on (med=0.0) |
| `B-5401A` | Corrente | 0.8 | 165.1 | **54.2%** | 28.27 | 54.3% | 63.0% on (med=129.5); 29.5% on (med=0.0) |
| `B-5501B` | Corrente | 0.7 | 1.3 | **88.0%** | 0.22 | 36.7% | 0.0% on (med=0.6); 0.0% on (med=0.6); 0.0% on (med=0.6) |
| `B-6511502A` | CORRENTE ELÉTRICA DO MOTOR | 0.0 | 214.2 | **65.3%** | 32.9 | 65.4% | 35.3% on (med=0.0) |
| `B-8801C` | Corrente | 0.0 | 0.0 | **88.6%** | 15.6 | 88.6% | 19.9% on (med=0.0) |
| `B-8802B` | None | — | — | — | — | — | máscara desativada |
| `B-90001A` | Pressão Descarga | — | — | — | — | — | ref 'Pressão Descarga' indisponível |

## Leitura

- **`%OFF@5.0`** = fração do tempo que o limiar atual marca como desligado.
- **`Máquina LIGADA perto da falha?`** = % dos pontos com `ref > 5` na janela −10d..+2d da falha.
  Se está alto, a máquina operava no período da falha e a máscara está **apagando sinal válido**.

## Recomendação por grupo

- **Recalibrar limiar (máscara atrapalha):** `B-5401A`
  → máquina operando perto da falha, mas limiar 5.0 corta demais. Usar limiar sugerido.
- **Máquina genuinamente intermitente:** `B-0302C`, `B-3403C`, `B-402E`, `B-4703.24001B`, `B-5501B`, `B-6511502A`, `B-8801C`
  → fica realmente desligada a maior parte do tempo; o sinal só existe nas janelas `on` (curtas).
  Recalibrar ajuda pouco; considerar avaliar só trechos operacionais e/ou relaxar a regra de ponto.
- **Máscara já ok (máquina quase sempre ligada):** `B-24001B`

## Decisão adotada

Trocar `OFF_ABS_THRESHOLD` fixo=5.0 pelo **limiar sugerido por equipamento** (≈15% da
mediana operacional), que se adapta à faixa real de cada corrente. Para os intermitentes,
isso ao menos deixa de marcar como `off` os trechos de operação real. Equipamentos já ok
permanecem praticamente inalterados.
