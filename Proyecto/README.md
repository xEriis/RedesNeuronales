# Proyecto Final: Red Neuronal para identificar falsificaciones en firmas
## Alumnas:
- Martínez Marcelo Ingrid Aylen - 320179884
- Pérez Evaristo Eris - 320211162
- Ramírez Venegas Alexa Paola - 320111677

## Estructura del proyecto
- En classroom se subió el notebook del proyecto.

- Sobre el punto extra:
    + Se necesita haber clonado el repositorio.
    + Haber creado el environment 
    ```
    conda env create -f environment.yml
    ```
    + Una vez creado, activar con: 
    ```
    conda activate proyecto
    ```
    + Ejecutar 
    ```
    pip install -r requirements.txt
    ```

    (Nota: Si no tienes cuenta en streamlit tendrás que crear una con correo)

    + Ejecutar la aplicación a la altura de ./Proyecto:
    ```
    streamlit run app.p
    ```

### Observaciones
No es necesario correr el notebook que se encuentra en el repositorio ya que adjuntamos este en el google colab con los con un enlace para descargar los mismos pesos que se encuentran aqui en el repositorio.
Durante la ejecucion de los notebooks se agregaron notas a las celdas de train y metricas ya que no es necesario ejecutarlas ya que se dejaron ahi los resultados y además, correrlas sin el GPU puede tardar bastante (incluso con GPU).

### Entregables:
- Notebook en google colab con todos los detalles de nuestra red neuronal (en el mismo notebook se agrega una celda que conecta con los pesos del mejor modelo)
- app.py desarrollada en Streamlit (las instrucciones de ejecución se detallan en este mismo README)
- Reporte_Proyecto_Redes_Neuronales.pdf reporte de nuestro proyecto final de la materia
- mejor_modelo_entrenado.pt contiene el modelo resultante del entrenamiento realizado en nuestro notebook (incluyendo los pesos y sesgos aprendidos). Este lo agregamos en el repositorio pues la app hace uso de este, además de que tanto el notebook como la app cargan este modelo ya sea del repo o de drive.
