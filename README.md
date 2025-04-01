**Análisis de Satisfacción Laboral de Empleados**

Este proyecto tiene como objetivo explorar y analizar los factores que influyen en la satisfacción laboral de los empleados dentro de una organización. El análisis incluye la exploración de datos, transformación, visualización y extracción de conclusiones con el fin de proporcionar información clave sobre la rotación de personal y otros aspectos importantes como el salario, el teletrabajo, y el género.
Además, se han realizado análisis estadísticos y pruebas de A/B testing para determinar si existen diferencias significativas en las variables analizadas.

**Objetivo del Proyecto**

A través de un análisis exhaustivo de los datos, se busca responder a varias preguntas clave relacionadas con la satisfacción laboral y cómo diferentes factores impactan la rotación de empleados, salarios, género y otros aspectos del ambiente de trabajo.

**Metodología**

*Proceso ETL (Extract, Transform, Load)*
- Extracción (Extract): Los datos se obtuvieron de diversas fuentes, como bases de datos internas, archivos CSV y otras fuentes externas.

- Transformación (Transform): Los datos fueron limpiados y transformados para su análisis, incluyendo el manejo de valores faltantes, la creación de nuevas variables y la conversión de tipos de datos.

- Carga (Load): Los datos transformados fueron cargados en el entorno de análisis para su visualización y modelado.

*Análisis Exploratorio de Datos (EDA)*
Durante el análisis exploratorio de datos (EDA), se utilizaron diversas técnicas estadísticas y visuales para obtener una comprensión clara de los datos y las relaciones entre las diferentes variables.

**Preguntas Respondidas en el Análisis**
Este análisis tiene como objetivo responder a las siguientes preguntas clave:
- Rotación según grupo de satisfacción: ¿La rotación de empleados varía según el nivel de satisfacción general?
- Distribución de teletrabajo: ¿Influye el teletrabajo en la rotación de empleados?
- Rotación según la edad: ¿Hay alguna relación entre la edad de los empleados y su propensión a dejar la empresa?
- Rotación según el puesto de trabajo: ¿Existen diferencias en la rotación según el puesto de trabajo de los empleados?
- Relación entre nivel de satisfacción y teletrabajo: ¿Los empleados que teletrabajan están más satisfechos en general?
- Distribución del género: ¿La muestra de empleados está equilibrada entre géneros?
- Relación entre salario y género: ¿Existen diferencias en la distribución salarial entre géneros?
- Relación entre rotación y género: ¿Hay diferencias de género en la rotación de los empleados?
- Distribución de la variable satisfacción: ¿Cuántos empleados se encuentran en cada nivel de satisfacción?
- Relación entre nivel de satisfacción y salario: ¿Existe alguna relación entre el nivel de satisfacción y el salario de los empleados?
- Relación entre salario y rotación: ¿El salario tiene un impacto en la probabilidad de rotación de los empleados?
- Relación entre nivel de satisfacción y puesto de trabajo: ¿Existen puestos de trabajo con un promedio de satisfacción más alto?
- Relación entre satisfacción y género: ¿Existen diferencias de satisfacción laboral entre géneros?
- Relación entre salario y puesto de trabajo: ¿Cuál es el salario promedio en cada puesto de trabajo dentro de la empresa?
- Relación de nivel de satisfacción general y satisfacción en relaciones interpersonales: ¿Hay alguna relación entre la satisfacción general y la satisfacción con las relaciones interpersonales en el trabajo?

**Resultados Principales**
- Se observó que la rotación de empleados varía considerablemente según su nivel de satisfacción. Los empleados más insatisfechos tienen una mayor probabilidad de dejar la empresa.
- No se encontraron diferencias significativas en la rotación de empleados en función del género.
- El teletrabajo no mostró una relación directa con la satisfacción general, aunque algunos empleados que teletrabajan parecen más satisfechos con su equilibrio entre trabajo y vida personal.
- La edad no parece ser un factor determinante en la rotación, pero sí se identificaron ciertos puestos con una rotación más alta.
- El salario tiene una relación significativa con la rotación, mostrando que los empleados con salarios más bajos tienen más probabilidades de dejar la empresa.
- La satisfacción interpersonal en el trabajo es un factor importante en la satisfacción general de los empleados, y algunos puestos presentan un nivel de satisfacción notablemente más alto que otros.

**Visualización**
Se utilizaron varias visualizaciones como boxplots, gráficos de barras e histogramas, para identificar patrones y diferencias significativas entre las variables. Estas visualizaciones ayudan a interpretar los datos de manera más clara y facilitar la toma de decisiones.

**Conclusión**
Este análisis proporciona una visión integral sobre cómo diversos factores, como la satisfacción laboral, el género, el teletrabajo y el salario, impactan la rotación de empleados. Los resultados pueden ser útiles para tomar decisiones estratégicas sobre la gestión del talento, las políticas de retención y la mejora del ambiente de trabajo.

**Herramientas Utilizadas**
Python: Para el análisis de datos y visualización (librerías como pandas, seaborn, matplotlib).
Jupyter Notebooks: Para la documentación y ejecución del análisis de manera interactiva.
Pruebas Estadísticas: Se realizaron pruebas de A/B testing para validar la significancia de las diferencias observadas.
