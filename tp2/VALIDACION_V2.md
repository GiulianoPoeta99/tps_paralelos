# Checklist de validacion v2

Este checklist valida la migracion a `int`, la consistencia de checksum y el comportamiento de performance en la grilla pedida.

## 1) Corridas recomendadas

- `c=512`, `workers=1,4,10`
- `c=1024`, `workers=4,10`
- Semilla fija: `--seed 2026`
- Perfil recomendado: `--perfil cubo`

## 2) Comandos base

```bash
python benchmark.py --c 512 --workers-values 1,4,10 --seed 2026 --perfil cubo --max-val 256 --output resultados_v2_c512.csv --stream-output
python benchmark.py --c 1024 --workers-values 4,10 --seed 2026 --perfil cubo --max-val 256 --output resultados_v2_c1024.csv --stream-output
```

Si no queres regenerar informe Markdown en cada corrida:

```bash
python benchmark.py --c 512 --workers-values 1,4,10 --seed 2026 --perfil cubo --max-val 256 --output resultados_v2_c512.csv --stream-output --no-informe
```

## 3) Validaciones obligatorias

- [ ] Para cada combinacion de `c` y `workers`, el estado de cada metodo es `ok`.
- [ ] `checksum_ab == checksum_btat` para cada fila (propiedad `(AB)^T = B^T A^T`).
- [ ] Checksums iguales entre metodos para la misma combinacion de entrada.
- [ ] `numba` ejecutado con `workers=4` y `workers=10` en modo `--parallel`.

## 4) Comparacion con v1

- [ ] Comparar tiempos de `threading`, `ThreadPoolExecutor` y `multiprocessing` entre `v1` y `v2`.
- [ ] Confirmar reduccion de overhead en cargas grandes (especialmente por remover `Fraction` y granularidad por filas).
- [ ] Marcar en el informe los mejores resultados.
- [ ] Si hay dos resultados mejores muy cercanos, marcar ambos como mejores.

## 5) Nota de entorno

Si `numba` no esta instalado en el entorno, ese metodo fallara hasta instalar dependencia:

```bash
pip install numba
```
