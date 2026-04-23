# Estándar de estructura para el informe

Estructura basada en papers de ciencias de la computación (ACM / IEEE).

---

## 1. Título (Title)

- Conciso, informativo.
- Refleja claramente el problema y/o contribución principal.

## 2. Resumen (Abstract)

- 150–250 palabras típicamente.
- Contiene: problema, enfoque, resultados principales.
- Debe poder leerse de forma independiente.

## 3. Introducción (Introduction)

- Contexto y motivación.
- Definición del problema.
- Por qué es relevante.
- Resumen de contribuciones (bullet points frecuentes).
- Opcionalmente: ejemplo ilustrativo, estructura del paper.

## 4. Trabajo Relacionado (Related Work)

- Revisión de literatura previa.
- Comparación con enfoques existentes.
- Posicionamiento de la contribución.

## 5. Preliminares / Background (opcional)

- Definiciones, notación, conceptos base.
- Muy común en: teoría, criptografía, machine learning formal.

## 6. Metodología / Enfoque (Method / Approach / Model)

- Núcleo técnico del paper.
- Puede incluir:
  - Modelos matemáticos
  - Algoritmos
  - Arquitecturas
- En ML: descripción del modelo, funciones de pérdida, pipeline.

## 7. Implementación o Diseño del Sistema (opcional)

- Detalles prácticos: arquitectura de software, decisiones de ingeniería.
- Más común en sistemas o software engineering.

## 8. Experimentos / Evaluación (Experiments / Evaluation)

- Cómo se valida la propuesta.
- Incluye: datasets, métricas, configuración experimental.
- Resultados: tablas, gráficos.
- Comparaciones con baselines.

## 9. Resultados (Results)

- A veces separado de "Experiments".
- Presentación clara de outputs sin interpretación profunda.

## 10. Discusión (Discussion)

- Interpretación de resultados.
- Limitaciones.
- Implicaciones.

## 11. Conclusión (Conclusion)

- Resumen de contribuciones.
- Qué se logró.
- Trabajo futuro (future work).

## 12. Referencias (References / Bibliography)

- Citaciones formateadas (BibTeX en la mayoría de casos).

## 13. Apéndices (Appendix) (opcional)

- Pruebas matemáticas.
- Detalles experimentales.
- Material suplementario.

---

## Variantes según subárea

| Subárea | Secciones típicas adicionales |
|---|---|
| **Machine Learning** | Dataset section explícita, ablation studies, hyperparameters |
| **Sistemas** | System design, deployment details, performance benchmarks |
| **Teoría** | Theorems + proofs, lemmas, formal models |

---

## Estructura mínima (para implementación rápida)

1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiments
6. Conclusion
7. References
