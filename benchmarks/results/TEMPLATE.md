# Resultados de Benchmark — Regresión Simbólica Evolutiva — Fase 16

## Entorno

| Campo | Valor |
|---|---|
| Fecha de medición | 26/11/2026 |
| Commit base previo a Fase 16 | `07dfc74` (confirmar con `git rev-parse HEAD` antes del commit de esta fase) |
| CPU model | 12th Gen Intel(R) Core(TM) i7-1255U |
| Kernel | `6.8.0-151-generic` |
| CPU governor | `powersave` |
| NumCPU | 12 procesadores lógicos detectados por Go |
| GOMAXPROCS | 12 (`runtime.GOMAXPROCS(0)`) |
| Go version | `go1.24.4` |
| GOOS/GOARCH | `linux/amd64` |

Las mediciones se realizaron en el mismo equipo y con el mismo governor. No se modificó automáticamente la configuración de energía durante los experimentos.

## Configuración experimental

- Dataset: `benchmarks/datasets/benchmark_quadratic_625pts.csv`.
- Función objetivo: \(f(x)=x^2+3x+2\), con \(x\in[1,20]\), 625 puntos.
- División: 500 puntos de entrenamiento (80 %) y 125 de validación (20 %).
- Perfil base: `config.BENCHMARKConfig()`.
- Population: 1.000 individuos.
- Generations: 100.
- EliteCount: 50.
- Lambda: 0.01.
- Seed GA: 42.
- Seed del generador: 43 (`Seed + 1`, misma convención que el CLI).
- TargetMSEEnabled: `false`.
- StagnationLimit: `0`.
- Workers formales: 1, 2, 4 y 8.
- Repeticiones: un warmup por configuración y cinco mediciones formales, con orden alternado.

El benchmark de Go mide exclusivamente `evol.Run()`. La construcción de `Evolution`, la carga del CSV y la división del dataset quedan fuera del timer.

### Evaluaciones consideradas para Meval/s

| Etapa | Evaluaciones de puntos |
|---|---:|
| Población inicial: 1.000 × 500 | 500.000 |
| Descendencia: 100 × 950 × 500 | 47.500.000 |
| Validación final: 1 × 125 | 125 |
| **Total por `Run()`** | **48.000.125** |

`Meval/s` se calcula como `48.000.125 / tiempo_de_Run_en_segundos / 1.000.000`.

---

## Tabla 1 — Go benchmarks (`evol.Run()`)

Cinco observaciones por configuración (`-benchtime=1x -count=1 -benchmem`), precedidas por un warmup. La desviación estándar (σ) de esta tabla es la desviación estándar muestral (`n−1`). `B/op` y `allocs/op` son medianas de las cinco observaciones.

| Workers | Mediana (ns/op) | Mínimo (ns/op) | Máximo (ns/op) | σ (ns/op) | Meval/s | B/op (mediana) | allocs/op (mediana) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.505.893.270 | 2.282.162.318 | 3.354.351.742 | 419.347.532 | 19,15 | 254.874.328 | 5.148.532 | 1,00× | — |
| 2 | 1.799.792.855 | 1.521.268.505 | 2.409.081.046 | 338.371.589 | 26,67 | 254.931.904 | 5.148.973 | 1,39× | 69,6 % |
| 4 | 1.243.918.269 | 1.053.030.133 | 1.592.561.182 | 196.253.778 | 38,59 | 255.030.640 | 5.149.585 | 2,01× | 50,4 % |
| 8 | 972.471.664 | 766.315.265 | 1.227.034.964 | 192.580.462 | 49,36 | 255.191.272 | 5.150.732 | 2,58× | 32,2 % |

```text
Speedup(n)    = Mediana(T1) / Mediana(Tn)
Efficiency(n) = Speedup(n) / n
```

### Experimento complementario — Workers=12

Workers=12 se ejecutó por separado, con un warmup y cinco repeticiones. No forma parte del conjunto formal alternado 1/2/4/8.

| Métrica | Resultado |
|---|---:|
| Mediana (`ns/op`) | 679.223.242 |
| Mínimo (`ns/op`) | 665.330.068 |
| Máximo (`ns/op`) | 688.583.805 |
| σ muestral (`ns/op`) | 8.667.609 |
| Meval/s | 70,67 |
| B/op (mediana) | 255.090.480 |
| allocs/op (mediana) | 5.151.550 |
| Speedup vs. Workers=1 | 3,69× |
| Efficiency vs. Workers=1 | 30,7 % |

Frente a Workers=8, Workers=12 obtuvo aproximadamente 43,2 % más throughput y redujo la mediana del tiempo de `Run()` en aproximadamente 30,2 %. Esta comparación es complementaria: las mediciones de Workers=12 se realizaron en una tanda separada y no intercalada con Workers=8.

---

## Tabla 2 — CLI end-to-end (proceso completo)

El alcance incluye carga del CSV, split, construcción de la evolución, `Run()` e impresión del resultado. Las estadísticas de `real` se calcularon sobre las cinco corridas; σ es la desviación estándar muestral. Las columnas de `user`, `sys`, cores y CPU utilization muestran la mediana de las cinco observaciones.

| Workers | Mediana real (s) | Mínimo (s) | Máximo (s) | σ (s) | user (s), mediana | sys (s), mediana | Cores used, mediana | CPU util %, mediana |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,16 | 2,15 | 2,37 | 0,116 | 2,39 | 0,11 | 1,153 | 9,61 % |
| 2 | 1,50 | 1,38 | 1,69 | 0,132 | 3,02 | 0,09 | 2,074 | 17,29 % |
| 4 | 1,07 | 1,02 | 1,27 | 0,108 | 3,85 | 0,07 | 3,697 | 30,81 % |
| 8 | 0,77 | 0,77 | 0,99 | 0,097 | 4,64 | 0,09 | 6,104 | 50,87 % |

```text
CPU cores used = (user + sys) / real
CPU util %     = (user + sys) / real / GOMAXPROCS * 100
```

Speedup y efficiency calculados exclusivamente con la mediana `real` del CLI:

| Workers | Speedup CLI | Efficiency CLI |
|---:|---:|---:|
| 1 | 1,00× | — |
| 2 | 1,44× | 72,0 % |
| 4 | 2,02× | 50,5 % |
| 8 | 2,81× | 35,1 % |

**Incidencia del CSV CLI (resuelta):** la primera ejecución generó un CSV mal formado porque la configuración regional hizo que `awk` escribiera coma decimal en `cpu_cores_used` y `cpu_util_pct`. El archivo se reconstruyó a partir de los `time.txt` originales, sin repetir las mediciones. Luego se validó que las 20 filas de datos tengan exactamente 8 columnas. Además, `run_benchmark.sh` se corrigió para utilizar `LC_ALL=C` en los dos cálculos `awk`, evitando que el problema se repita en futuras ejecuciones.

---

## Análisis

### 1. ¿Cuánto mejora el rendimiento al pasar de 1 a 2 workers?

En Go benchmarks, el speedup mediano fue 1,39× (efficiency 69,6 %). En CLI, el speedup fue 1,44× (efficiency 72,0 %). El paralelismo reduce el tiempo, aunque el aumento no es lineal.

### 2. ¿Qué ocurre al pasar a 4?

Go obtuvo 2,01× respecto de Workers=1 y CLI 2,02×. La mejora sigue siendo importante y la eficiencia cae aproximadamente al 50 %.

### 3. ¿Qué ocurre al pasar a 8?

Go obtuvo 2,58× y CLI 2,81× respecto de Workers=1. La CPU utilizó una mediana de aproximadamente 6,1 cores durante el proceso CLI y el CPU utilization normalizado fue 50,87 % de GOMAXPROCS=12.

### 4. ¿En qué punto deja de ser rentable agregar workers?

No se observó una pérdida de rendimiento al pasar de 1 a 8 en el conjunto formal. El experimento complementario de Workers=12 también mejoró respecto de Workers=8: el throughput de `Run()` aumentó aproximadamente 43,2 %. Por lo tanto, dentro de las configuraciones medidas no se identificó un punto en el que agregar workers dejara de mejorar el tiempo.

Sin embargo, Workers=12 se midió en una tanda separada, no alternada con Workers=8. La mejora debe considerarse un indicio sólido, pero no una comparación con el mismo control de orden que el conjunto formal. Para una decisión de configuración por defecto convendría hacer una prueba adicional intercalando Workers=8 y Workers=12.

### 5. ¿Qué limitaciones tiene Workers=1 como baseline?

Workers=1 utiliza la infraestructura completa del pool (canales y una goroutine worker). No es una implementación secuencial sin pool. Esta fase no permite cuantificar el overhead del pool respecto de una ruta secuencial alternativa porque esa implementación no existe y no se agregó.

### 6. ¿Qué configuración es óptima para el hardware objetivo?

Entre las configuraciones formales, Workers=8 fue la más rápida. En el experimento complementario, Workers=12 obtuvo el mejor tiempo y throughput observados. Para este workload y este i7-1255U, Workers=12 es el mejor resultado medido hasta ahora, pero se recomienda confirmar la comparación 8/12 con rondas intercaladas antes de convertir ese valor en una recomendación general o en un cambio del perfil.

### Fracción paralelizable estimada (Ley de Amdahl inversa)

Se estimó con `p ≈ (1/Speedup(n) − 1) / (1/n − 1)`. Los resultados son indicativos:

| Fuente | Workers | Speedup | Fracción paralelizable estimada |
|---|---:|---:|---:|
| Go `Run()` | 2 | 1,39× | 56,4 % |
| Go `Run()` | 4 | 2,01× | 67,1 % |
| Go `Run()` | 8 | 2,58× | 69,9 % |
| Go `Run()` complementario | 12 | 3,69× | 79,5 % |
| CLI end-to-end | 2 | 1,44× | 61,1 % |
| CLI end-to-end | 4 | 2,02× | 67,3 % |
| CLI end-to-end | 8 | 2,81× | 73,5 % |

Estas estimaciones no deben interpretarse como una propiedad fija del algoritmo: dependen del workload, del entorno, del overhead del pool y de las condiciones de medición. La comparación CLI incluye el costo del proceso completo; la de Go mide `Run()`.

### Variabilidad y limitaciones

- Las cinco mediciones de Go se ejecutaron con `-benchtime=1x`; las desviaciones muestran variación apreciable en el conjunto formal (coeficientes de variación aproximados entre 15 % y 20 %). Las medianas son más representativas que una corrida individual.
- El governor registrado fue `powersave`. La frecuencia dinámica y las condiciones térmicas pueden influir en los tiempos; no se modificó el governor durante el experimento.
- `B/op` representa bytes asignados por operación de benchmark, no memoria residente máxima. `allocs/op` representa cantidad acumulada de asignaciones.
- No se debe comparar directamente el tiempo absoluto de Go benchmarks con el tiempo absoluto del CLI: sus alcances temporales son distintos.
- Workers=12 es complementario y no pertenece a las cinco rondas formales de 1/2/4/8.
- El CSV CLI original se reconstruyó correctamente desde los `time.txt` y se validó (20 filas de datos, 8 columnas por fila). El script quedó corregido con `LC_ALL=C` para futuras ejecuciones.

## Conclusiones

El paralelismo mejora claramente el rendimiento de la regresión simbólica en el workload cuadrático de 625 puntos. En el conjunto formal, el mejor resultado fue Workers=8: `Run()` alcanzó una mediana de aproximadamente 0,972 s y 49,36 Meval/s, mientras que el CLI obtuvo una mediana de 0,77 s end-to-end.

El experimento complementario de Workers=12 alcanzó una mediana de 0,679 s y 70,67 Meval/s en `Run()`, superando a Workers=8. En el equipo evaluado, la evidencia sugiere que el workload todavía aprovecha más paralelismo hasta 12 workers. La configuración óptima observada fue 12 workers para este experimento, con la salvedad de que la comparación 8/12 debe confirmarse mediante rondas alternadas.

Workers=1 debe entenderse como baseline del pool de workers, no como ejecución secuencial pura. La fase no mide el overhead del pool frente a una implementación sin pool.

Las mediciones se consideran completadas. Antes de cerrar y versionar los resultados, se debe corregir/regenerar el CSV CLI con formato numérico independiente de la configuración regional y revisar el estado de Git para no incluir datasets ajenos a esta fase.
