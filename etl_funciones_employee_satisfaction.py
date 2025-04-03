# ETL - FUNCIONES - PROYECTO 3 - TEAM 1 - OPTIMIZACIÓN DEL TALENTO

# =========================
# IMPORTACIÓN DE LIBRERÍAS
# =========================
# Tratamiento de datos:
import pandas as pd
import numpy as np
# Visualización:
import matplotlib.pyplot as plt
import seaborn as sns
# Evaluar linealidad de las relaciones entre las variables y la distribución de las variables:
import scipy.stats as st
import scipy.stats as stats
from scipy.stats import shapiro, poisson, chisquare, expon, kstest
# Librería para realizar el A/B Testing:
from statsmodels.stats.proportion import proportions_ztest
# Librerias para imputar nulos:
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Importar librerías para conectar con MySQL:
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine  
import pymysql
# Importar librerías para transformar datos de categóricos a numéricos y viceversa:
from sklearn.preprocessing import LabelEncoder
# Configuración: Para poder visualizar todas las columnas de los DataFrames
pd.set_option('display.max_columns', None) 
# Gestión de los warnings:
import warnings
warnings.filterwarnings("ignore")

# ===========================================
# Configuración de la base de datos para SQL
# ===========================================
host = 'localhost'
user = 'root'
password = 'AdalabAlumnas'
database = 'bd_proyecto3'

# ===================================================== 
# EXTRACCIÓN: Obtener los datos desde los archivos CSV
# =====================================================
def extract_data(file_path):
    print("----------------------------------------------------------------")
    print(f"Extrayendo datos desde {file_path}...")
    df = pd.read_csv(file_path)
    print(df.head())
    print("----------------------------------------------------------------")
    return df

# ======================================
# EXPLORACIÓN: Exploración de los datos
# ======================================
def explore_data(df):
    print("----------------------------------------------------------------")
    print("Explorando los datos...\n")
    print('Aplicando ".info()"', df.info())
    print("----------------------------------------------------------------")
    print('Aplicando ".shape"', df.shape)
    print("----------------------------------------------------------------")
    print('Aplicando ".columns"', df.columns)
    print("----------------------------------------------------------------")
    print('Aplicando ".describe()"', df.describe())
    print("----------------------------------------------------------------")
    print('Aplicando ".isnull().sum()"', df.isnull().sum())
    print("----------------------------------------------------------------")
    print('Aplicando ".isnull().mean()*100).round(2).sort_values(ascending=False)"', (df.isnull().mean()*100).round(2).sort_values(ascending=False))
    print("----------------------------------------------------------------")
    print('Aplicando .duplicated().sum()"', df.duplicated().sum())
    print("----------------------------------------------------------------")
    return df

# ================================================
# TRANSFORMACIÓN: Limpiar y transformar los datos
# ================================================

# FUNCIÓN 1 - Transformar los registros a minúsculas, cuando proceda (strings con letras):
def transform_lower_rows(df):
    columns_object = df.select_dtypes(include=['object']).columns # Seleccionar las columnas str.
    df[columns_object] = df[columns_object].apply(lambda x: x.str.lower()) # aplicar la transformación a minúsculas.
    return df

# FUNCIÓN 2 - Transformar los nombres de las columnas:
def transform_column_name(column):
    # Dividir por mayúsculas o caracteres no alfabéticos y unir con "_":
    words = ''.join(['_' + c if c.isupper() and i != 0 else c for i, c in enumerate(column)]).split('_')
    words = [word.capitalize() for word in words if word]  # Poner en mayúscula la primera letra de cada palabra.
    return '_'.join(words)  # Unir con "_".

# FUNCIÓN 3 - Transformar variables categórica seleccionadas a numéricas:
def transform_numeric(df, col_categoric_numeric):
    for col in col_categoric_numeric: # iterar por las columnas que queremos transformar a numéricas.
        if col in df.columns: # verificar si la columna existe en el df.
            try:
                df[col] = df[col].astype(str).str.replace(',0$', '', regex=True) # eliminar ',0$'.
                df[col] = df[col].str.replace('$', '') # eliminar '$'.
                df[col] = df[col].str.replace("'", '') # quitar las comillas.
                df[col] = df[col].replace('Not Available', np.nan) # reemplazar 'Not Available' por un nulo 'nan'.
                df[col] = df[col].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x) # reemplar ',' por '.'
                #df[col] = pd.to_numeric(df[col], error='coerce') # convertir a float.
                df[col] = df[col].astype(float)
            except ValueError:
                # Manejar el error convirtiendo los valores no numéricos a NaN
                df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))  #errors='coerce' para versiones antiguas
                #df[col] = df[col].astype(float) #Intenta convertir a float luego de haber manejado los errores
    return df

# FUNCIÓN 4 - Transformar variables numéricas seleccionadas a categóricas:
def transform_categoric(df, col_numeric_categoric):
    for col in col_numeric_categoric: # iterar por las columnas que queremos transformar a categóricas.
        if col in df.columns: # verificar si la columna existe en el df.
            df[col] = df[col].astype('object') # convertir a string.
    return df

# FUNCIÓN 5 - Reemplazar los números escritos a cifras:
def convertir_a_numero(valor):
    # Creamos un diccionario para mapear palabras a números:
    numeros_escritos = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 
                    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 
                    'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 
                    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    if valor.isdigit(): # si ya es un número, devolverlo como int.
        return int(valor)
    else:
        try:
            partes = valor.split('-') # Dividimos el número escrito en decenas y unidades.
            if len(partes) == 2:
                decenas = numeros_escritos.get(partes[0].lower(), 0)
                unidades = numeros_escritos.get(partes[1].lower(), 0)
                return decenas + unidades # sumamos las cifras obtenidas.
            else:
                return numeros_escritos.get(valor.lower(), 0)
        except:
            return None

# FUNCIÓN 6 - Reemplazar valores negativos por nulos 'nan':
def transform_negative_values(df, column):
    df[column] = np.where(df[column] < 0, np.nan, df[column])
    return df

# FUNCIÓN 7 - Reemplazar valores con símbolos por espacios:
def transform_symbols(df, column):
    df[column] = df[column].str.replace(r"[_-]", " ", regex=True)
    return df

# FUNCIÓN 8 - Seleccionar el número de decimales de los float:
def transform_round_float(df, column, decimals):
    df[column] = df[column].round(decimals)
    return df

# FUNCIÓN PRINCIPAL DE TRANSFORMACIÓN DE DATOS:

import numpy as np
import pandas as pd

def transform_data(df, rename_dict=None, column_list_negative=None, columns_to_numeric=None, columns_to_category=None):
    """
    Función que realiza varias transformaciones en el DataFrame:
    1. Renombrar columnas.
    2. Convertir todos los strings a minúsculas.
    3. Reemplazar valores negativos por NaN en las columnas especificadas.
    4. Convertir columnas a tipo numérico o categoría."""
    
    # 1. Renombrar columnas (si se pasa el diccionario)
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # 2. Convertir todas las cadenas de texto a minúsculas
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
    # 3. Reemplazar valores negativos por NaN en las columnas indicadas
    if column_list_negative:
        for col in column_list_negative:
            df[col] = df[col].where(df[col] >= 0, np.nan)
    
    # 4. Convertir las columnas a tipo numérico o de categoría
    if columns_to_numeric:
        for col in columns_to_numeric:
            df = df.apply(pd.to_numeric, errors='coerce')
    
    if columns_to_category:
        for col in columns_to_category:
            df[col] = df[col].astype('category')
    
    return df


# ==================================
# TRANSFORMACIÓN DE LOS DATOS NULOS
# ==================================

# FUNCIÓN 9 - Imputar la Moda en nulos (categóricas con bajo % nulos con categoría dominante):
def nulls_impute_mode(df, column):
    for col in column:
        moda = df[col].mode()[0]
        df[col] = df[col].fillna(moda)
    return df

# FUNCIÓN 10 - Imputar Nueva Categoría 'unknown' (categóricas con bajo % nulos sin categoría dominante):
def nulls_impute_newcategory(df, column):
    df[column] = df[column].fillna('unknown')
    return df

# FUNCIÓN 11 - Transformar datos categóricos a numéricos, imputar y reconvertir a categorías originales:
def transform_labelencoder(df, columns):
    le_dict = {}  # Diccionario para guardar los LabelEncoders de cada columna
    for column in columns:
        le = LabelEncoder()
        col_encoded = f"{column}_encoded"  # Nombre de la columna codificada.
        col_imputed = f"{col_encoded}_imputed"  # Nombre de la columna imputada.
        # Codificar la columna categórica:
        df[col_encoded] = le.fit_transform(df[column].fillna('missing'))
        le_dict[column] = le  # Guardamos el LabelEncoder para la reconversión.
    # Aplicar KNN Imputer:
    imputer = KNNImputer(n_neighbors=2)
    df[[f"{col}_encoded_imputed" for col in columns]] = imputer.fit_transform(df[[f"{col}_encoded" for col in columns]])
    # Reconvertir los valores imputados a las categorías originales:
    for column in columns:
        col_encoded = f"{column}_encoded"
        col_imputed = f"{col_encoded}_imputed"
        df[column] = le_dict[column].inverse_transform(df[col_imputed].round().astype(int))  # Convertir a categoría
    # Eliminar columnas residuales (_encoded y _imputed):
    df.drop(columns=[f"{col}_encoded" for col in columns] + [f"{col}_encoded_imputed" for col in columns], inplace=True)
    return df

# FUNCIÓN 12 - Aplicar las Técnicas Avanzadas de Imputación:
def nulls_knn_iterative_imputer(df, columns):
    imputer_iterative = IterativeImputer(max_iter=20, random_state=42)
    imputer_knn = KNNImputer(n_neighbors=2)
    for column in columns:
        # Aplicar Iterative Imputer:
        col_iterative = f"{column}_iterative"
        df[col_iterative] = imputer_iterative.fit_transform(df[[column]])
        # Aplicar KNN Imputer:
        col_knn = f"{column}_knn"
        df[col_knn] = imputer_knn.fit_transform(df[[column]])
        # Mostrar estadísticas de comparación:
        print(f"Comparación de imputación para {column}:\n")
        print(df[[column, col_iterative, col_knn]].describe().T)
        print("-" * 50)
    # Eliminar las columnas originales y las imputadas con Iterative Imputer:
    df.drop(columns=[col for column in columns for col in [column, f"{column}_iterative"]], inplace=True)
    # Renombrar las columnas KNN con su nombre original:
    df.rename(columns={f"{column}_knn": column for column in columns}, inplace=True)
    return df

# FUNCIÓN 13 - Imputar la Mediana:
def nulls_impute_median(df, column):
    for col in column:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        return df
    
# FUNCIÓN PRINCIPAL DE TRANSFORMACIÓN DE NULOS:
def transform_nulls(df, imputations=None, plot_histograms=True):
    # Listar los nulos por porcentaje de mayor a menor:
    nulos = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    nulos = nulos[nulos > 0]
    # Lo transformamos en un DataFrame:
    nulos = nulos.to_frame(name='perc_nulos').reset_index().rename(columns={'index': 'var'})
    print(nulos)
    print("--------------------------------------------------------------------")
    
    # SELECCIÓN DE VARIABLES CATEGÓRICAS CON NULOS:
    columnas_object = df.select_dtypes(include=['object']).columns
    columnas_nulos = nulos['var'].to_list()
    columnas_object_nulos = columnas_object.intersection(columnas_nulos)
    
    for col in columnas_object_nulos:
        print(f"La distribución de las categorías para la columna {col}")
        print((df[col].value_counts() / df.shape[0]) * 100)  
        print("--------------------------------------------------------------------")

    # SELECCIÓN DE VARIABLES NUMÉRICAS CON NULOS:
    columnas_number = df.select_dtypes(include=['number']).columns
    columnas_number_nulos = columnas_number.intersection(columnas_nulos)
    
    if plot_histograms:
        for col in list(columnas_number_nulos):
            plt.figure(figsize=(8, 5))
            plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
            plt.title(f'Histograma de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.show()

    # Imputación de nulos según las opciones proporcionadas
    if imputations:
        for col, method in imputations.items():
            if method == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif method == 'unknown':
                df[col] = df[col].fillna('unknown')
            elif method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=2)
                df[[col]] = imputer.fit_transform(df[[col]])

    print("--------------------------------------------------------------------")
    return df


# ========================
# ELIMINACIÓN DE COLUMNAS
# ========================

# FUNCIÓN PRINCIPAL PARA ELIMINAR COLUMNAS:
def remove_column(df, col_remove):
    df.drop(columns = col_remove, inplace = True)
    return df
    print("Eliminando las columnas seleccionadas ", df.columns)
    print("--------------------------------------------------------------------")


# =======================================================
# DISEÑO BBDD: Crear la base de datos y cargar los datos
# =======================================================
def create_db():
    # Conectar a MySQL usando pymysql:
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password)
    # Crear un cursor:
    cursor = connection.cursor()
    # Crear una base de datos si no existe:
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
    print("Base de Datos creada exitosamente en MySQL.")
    # Cerrar la conexión:
    connection.close()

def load_data(table_name, data):
    print(f"Cargando datos en la tabla {table_name}...")
    # Crear conexión a MySQL usando SQLAlchemy:
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
    # Volvemos a conectarnos con SQL para crear las tablas:
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    mycursor = cnx.cursor()
    # Dividimos el DataFrame con las tablas y sus columnas correspondientes:
    datos_education_level = data[['Id_Employee', 'Education', 'Education_Field']] 
    datos_job_category = data[['Id_Employee', 'Job_Level', 'Job_Role', 'Stock_Option_Level', 'Performance_Rating']] 
    datos_salary = data[['Id_Employee', 'Daily_Rate', 'Hourly_Rate', 'Monthly_Income', 'Monthly_Rate', 'Percent_Salary_Hike']]
    datos_logistics = data[['Id_Employee', 'Business_Travel', 'Distance_From_Home', 'Over_Time', 'Remote_Work']]
    datos_personal = data[['Id_Employee', 'Age', 'Date_Birth', 'Gender', 'Marital_Status', 'Remote_Work',  'Relationship_Satisfaction', 'Work_Life_Balance']]
    datos_company_work = data[['Id_Employee', 'Attrition', 'Environment_Satisfaction', 'Job_Involvement', 'Job_Satisfaction', 'Num_Companies_Worked', 
                                'Over_Time', 'Performance_Rating', 'Relationship_Satisfaction', 'Total_Working_Years', 'Training_Times_Last_Year', 
                                'Years_At_Company', 'Years_Since_Last_Promotion']]
    # Insertar datos desde el DataFrame en MySQL:
    datos_education_level.to_sql('education_level', con=engine, if_exists='append', index=False)
    datos_job_category.to_sql('job_category', con=engine, if_exists='append', index=False)
    datos_salary.to_sql('salary', con=engine, if_exists='append', index=False)
    datos_logistics.to_sql('logistics', con=engine, if_exists='append', index=False)
    datos_personal.to_sql('personal', con=engine, if_exists='append', index=False)
    datos_company_work.to_sql('company_work', con=engine, if_exists='append', index=False)
    print("Datos insertados exitosamente en sus correspondientes tablas.")
    cnx.close()

# ============
# A/B TESTING
# ============

# FUNCIÓN 14 - Asignar grupo:
def asignar_grupo_variable(valor, umbral):
    return 'A' if valor >= umbral else 'B'

def ab_testing(df, columna_satisfaccion, umbral=3):
    df['Grupo'] = df[columna_satisfaccion].apply(asignar_grupo_variable, umbral=umbral)

# FUNCIÓN 15 - Calcular la tasa de rotación de cada grupo:
def calcular_tasa_rotacion(df, columna_grupo, grupo):
    grupo_df = df[df[columna_grupo] == grupo]
    empleados_que_dejaron = grupo_df['Attrition'].apply(lambda x: 1 if x == 'yes' else 0).sum()
    total_empleados = len(grupo_df)
    return (empleados_que_dejaron / total_empleados) * 100

# FUNCIÓN PRINCIPAL:
def ab_testing(df, columna_satisfaccion, columna_attrition, umbral=3):
    df['Grupo'] = df[columna_satisfaccion].apply(asignar_grupo_variable, umbral=umbral)
    
    # Calcula las tasas de rotación de A y B
    tasa_rotacion_A = calcular_tasa_rotacion(df, 'Grupo', 'A')
    tasa_rotacion_B = calcular_tasa_rotacion(df, 'Grupo', 'B')

    # Realizar prueba de proporciones (Z-test)
    rotacion_A = df[(df['Grupo'] == 'A') & (df[columna_attrition] == 'yes')].shape[0]
    rotacion_B = df[(df['Grupo'] == 'B') & (df[columna_attrition] == 'yes')].shape[0]

    tamaño_muestra_A = df[df['Grupo'] == 'A'].shape[0]
    tamaño_muestra_B = df[df['Grupo'] == 'B'].shape[0]
    
    counts = [rotacion_A, rotacion_B]
    nobs = [tamaño_muestra_A, tamaño_muestra_B]

    stat, p_value = proportions_ztest(counts, nobs)

    print(f'Estadístico Z: {stat}')
    print(f'Valor p: {p_value}')
    
    # Interpretar el p-valor
    if p_value < 0.05:
        conclusion = "Rechazar H0: Hay diferencia significativa entre los grupos."
    else:
        conclusion = "No se puede rechazar H0: No hay diferencia significativa entre los grupos."

    print(f'Conclusión: {conclusion}')
    return df

# =====================
# Proceso ETL completo
# =====================
def etl_process(file_path, col_remove, table_names):
    # Extraer datos:
    df = extract_data(file_path)

    # Explorar los datos:
    explore_data(df)

    # Transformar datos:
    df = transform_data(df)

    # Transformar nulos:
    df = transform_nulls(df)

    # Crear una copia del data frame antes de eliminar columnas:
    df2 = df.copy()

    # Eliminar columnas:
    df2 = remove_column(df2, col_remove)

    # Guardar archivo nuevo .csv
    df2.to_csv("df_final.csv", index=False)

    # Crear la base de datos en SQL:
    create_db()

    # Cargar los datos en la base de datos en SQL:
    datos = pd.read_csv("df_final.csv")
    load_data(table_names, datos)

    # Realizar A/B Testing a los datos:
    ab_testing(df2)

# Uso de la función con parámetros configurables:
etl_process("HR_RAW_DATA.csv", 
            col_remove=['Over_18', 'Number_Children', 'Employee_Count', 'Salary', 'Role_Departament', 
                        'Standard_Hours', 'Years_In_Current_Role', 'Department', 'Same_As_Monthly_Income'], 
            table_names=['education_level', 'job_category', 'salary', 'logistics', 'personal', 'company_work'])
