import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Identificador de firmas falsas",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded" # Mostrar siempre para mover el threshold
)

# Cache del modelo para cargar solo una vez
@st.cache_resource
# Cargar modelo
def load_model():
    # Copiamos la definición de nuestra red (la misma del archivo .ipynb)
    class SiamesCNN_RN18(nn.Module):
        def __init__(self):
            super(SiamesCNN_RN18, self).__init__()
            
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

            #Capa final de nuestro modelo, sustituyendo la capa final del modelo de resnet18 preentrenado
            #Embedding para extraer características, completamente conectada
            self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                #regularización con dropout
                nn.Dropout(0.5), 
                
                nn.Linear(512, 256),
                nn.ReLU(),
                
                #Salida con embedding de 128 dimensiones
                nn.Linear(256, 128)
            )
        
        def forward_once(self, out):
            out = self.backbone(out)
            out = self.fc(out)
            out = out / out.norm(p=2, dim=1, keepdim=True) # Normalización L2 para la función de pérdida
            return out 
        
        def forward(self, img1, img2):
            y1 = self.forward_once(img1)
            y2 = self.forward_once(img2)
            return y1, y2

    # Instancia de la red: nuestro modelo
    modelo = SiamesCNN_RN18()

    # Dispositivo en el que se va a ejecutar 'CPU' si no se cuenta con GPU
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Cargar pesos de nuestro mejor entrenamiento en el modelo
        estado = torch.load('./mejor_modelo_entrenado.pt', 
                          map_location=dispositivo,
                          weights_only=True)
        # Usando nn.Module
        modelo.load_state_dict(estado) # Pasamos los valores al modelo
        modelo.to(dispositivo) # Seleccionamos el dispositivo
        modelo.eval()  # 
        return modelo, dispositivo
    except Exception as e:
        st.error(f"Error: {e}")
        return None, dispositivo

# Transformaciones para las imágenes de entrada de acuerdo a la tranformacion que hacemos en el modelo
def get_transformaciones():
    tamaño = (128, 256)  # Mismo tamaño que se usó en train, test, y val
    
    return transforms.Compose([
        transforms.Resize(tamaño),
        transforms.ToTensor(),
    ])

# Se usa para calcular las predicciones (al igual que lo hicimos el en .ipynb)
def calcular_distancia(emb1, emb2):
    distancia = nn.functional.pairwise_distance(emb1, emb2)
    return distancia.item()

# Función para calcular confianza
def calcular_confianza(distancia, threshold):
    """
    Calcula confianza de 0 a 1 basada en la distancia y threshold.
    
    Args:
        distancia: Distancia entre embeddings
        threshold: Umbral de decisión
    
    Returns:
        Confianza entre 0 y 1
    """
    temp = threshold / 2
    logit = -(distancia - threshold) / temp
    confianza = 1 / (1 + np.exp(-logit))
    return confianza

# Interfaz de usuario
def main():
    # Título y descripción
    st.title("🔍 Identificador de firmas falsas mediante Redes Neuronales Siamesas 🔎")
    st.markdown("""
    Esta aplicación utiliza una **Red Neuronal Siamesa** y se usa un modelo pre-entrenado ResNet18 para detectar si una firma es verdadera o es un posible fraude 
    mediante la comparación con una firma de referencia original.
    
    ### ¿Cómo usar?
    1. Sube una **firma de referencia** (firma original)
    2. Sube una **firma a verificar** (firma que quieres verificar)
    3. El modelo calculará la similitud entre ambas firmas
    4. Obtienes el nivel de **confianza**, es decir se va a mostrar **que tan probable es que la firma a verificar sea confiable**(de la misma persona).
    """)
    
    st.divider()
    
    # Cargar modelo
    with st.spinner("Cargando modelo..."):
        model, dispositivo = load_model()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que './mejor_modelo_entrenado.pt' esté en el directorio.")
        return
    
    # Ajuste del threshold
    with st.sidebar:
        st.header("⚙️ Configuración")
        threshold = st.slider(
            "Umbral de decisión",
            min_value=0.1,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Distancia máxima para considerar una firma como genuina"
        )
        st.info(f"📏 Threshold actual: {threshold}")
        st.markdown("""
        **Menor threshold**: Más estricto, **menos falsos positivos** (rechazarás firmas originales si no se parecen mucho).\n
        **Mayor threshold**: Más permisivo, **más falsos positivos** (puedes aceptar firmas falsas que se parezcan parcialmente).
        """)
    
    # Inicializar variables para evitar errores
    distancia = None
    is_real = None
    # Crear dos columnas para las imágenes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🫆 Firma de Referencia")
        st.caption("Sube la firma autentica y/o original")
        reference_file = st.file_uploader(
            "Selecciona la firma de referencia",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="reference",
            help="Esta debe ser una firma original verificada"
        )
        
        if reference_file is not None:
            reference_image = Image.open(reference_file).convert('RGB')
            st.image(reference_image, caption="Firma de Referencia original", use_column_width=True)
            #st.image(reference_image, caption="Firma de Referencia original", use_container_width=True)
    
    with col2:
        st.subheader("📥 Firma a Verificar")
        st.caption("Sube la firma que deseas validar")
        test_file = st.file_uploader(
            "Selecciona la firma a verificar",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="test",
            help="Esta es la firma que se comparará con la referencia"
        )
        
        if test_file is not None:
            test_image = Image.open(test_file).convert('RGB')
            st.image(test_image, caption="Firma a Verificar", use_column_width=True)

    
    # Realizar predicción si ambas imágenes están cargadas
    if reference_file is not None and test_file is not None:
        st.divider()
        
        col_result1, col_result2 = st.columns([2, 2])
        
        with col_result1:
            st.subheader("👩🏻‍💻 Resultado")
            
            with st.spinner("Comparando firmas..."):
                try:
                    # 1. Preprocesar imágenes, es decir, redimensionar los tamaños
                    transform = get_transformaciones()
                    
                    # Convertir a tensores y añadir dimensión batch (como se hizo con la prueba)
                    tensor_ref = transform(reference_image).unsqueeze(0).to(dispositivo)
                    tensor_ver = transform(test_image).unsqueeze(0).to(dispositivo)
                    
                    # Obtener embeddings (sin gradientes)
                    with torch.no_grad():
                        emb1, emb2 = model(tensor_ref, tensor_ver)
                
                    # Calcular distancia euclidiana
                    distancia = torch.nn.functional.pairwise_distance(emb1, emb2).item()
                    
                    # Determinar si es genuina
                    is_real = distancia < threshold
                    
                    # Calcular confianza
                    confianza = calcular_confianza(distancia, threshold)

                    with col_result1:
                        if is_real:
                            st.success("### ✅ Firma válida")
                            st.markdown("La firma coincide con la referencia.")
                        else:
                            st.error("### ❌ Firma fake")
                            st.markdown("Posible falsificación - no coincide con la referencia.")
                
                    with col_result2:
                        st.metric(
                            label="Confianza de que la firma es de la misma persona",
                            value=f"{confianza*100:.1f}%",
                            help="Probabilidad de que sea genuina"
                        )
                        # Barra de progreso visual
                        color_rosa = "#8E3D65"
                        st.markdown(f"""<style>.stProgress > div > div > div > div {{background-color: {color_rosa};}}
                            </style>
                            """, unsafe_allow_html=True)
                        st.progress(confianza)
                    
    
                    
                    # Interpretación detallada
                    st.divider()
                    st.subheader("Desglose de resultados 🌟")
                    
                    # Crear métricas visuales
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown("**Distancia:**")
                        st.code(f"{distancia:.4f}")
                    with cols[1]:
                        st.markdown("**Umbral:**")
                        st.code(f"{threshold:.2f}")
                    with cols[2]:
                        st.markdown("**Diferencia:**")
                        diff = abs(distancia - threshold)
                        st.code(f"{diff:.4f}")
                    with cols[3]:
                        st.markdown("**Estado:**")
                        if is_real:
                            st.success("DENTRO")
                        else:
                            st.error("FUERA")
                    
                    # Guía de interpretación
                    st.markdown("---")
                    col_guide1, col_guide2 = st.columns(2)
                    
                    with col_guide1:
                        st.markdown("**Guía de distancia:**")
                        if distancia < threshold * 0.3:
                            st.success("**Muy similar** - Probablemente misma persona")
                        elif distancia < threshold * 0.7:
                            st.info("**Similar** - Coincidencia probable")
                        elif distancia < threshold:
                            st.warning("**Poco similar** - Diferencias notables")
                        elif distancia < threshold * 1.3:
                            st.error("**Diferente** - Posible falsificación")
                        else:
                            st.error("**Muy diferente** - Falsificación probable")
                    
                    
                except Exception as e:
                    st.error(f"Error al procesar las imágenes: {e}")
                    st.error("Posible causa: Tamaño de imagen incompatible o modelo corrupto")
        
    else:
        st.info("👆 Sube ambas imágenes determinar el resultado")
    
    # Información
    st.divider()
    with st.expander("ℹ️ Información sobre el modelo"):
        st.markdown("""
        ### Características del Modelo
        - **Arquitectura**: Red Neuronal Siamesa con ResNet18
        - **Modelo pre-entrenado**: ResNet18
        - **Función de pérdida**: ContrastiveLoss
        - **Optimizador**: SGD con momento
        - **Regularización**: Drop-out
        - **CNN**: Las redes Siamesas usan Redes Neuronales Convolucionales como subredes principales.
        - **Input**: Dos imágenes de firmas (128x256 píxeles)
        
        ### ¿Cómo funciona?
        1. El modelo extrae características (embeddings) de ambas firmas
        2. Calcula la distancia euclidiana entre los embeddings
        3. Compara la distancia con un threshold para determinar si son similares
        4. Menor distancia = Mayor similitud
        
        ### Consideraciones
        - La calidad de las imágenes afecta la precisión
        - Se recomienda usar imágenes con fondo blanco y buena resolución
        - El threshold puede ajustarse según tus necesidades de seguridad
        """)
    
    with st.expander("🎯 Ajustando el Threshold"):
        st.markdown("""
        El **threshold** (umbral) determina qué tan estricto es el sistema:
        
        - **Threshold bajo (0.1 - 0.5)**: 
          - Más estricto
          - Menos falsos positivos (menos firmas falsas aceptadas)
          - Más falsos negativos (más firmas genuinas rechazadas)
          
        - **Threshold medio (0.5 - 1.0)**:
          - Balance entre seguridad y usabilidad
          - Recomendado para la mayoría de casos
          
        
        **Recomendación**: Empieza con 0.3-0.35 y ajusta según tus resultados.
        """)
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Desarrollado por estudiantes de la FC-UNAM :)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
