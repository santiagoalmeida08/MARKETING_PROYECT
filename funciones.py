import numpy as np
import pandas as pd
import sqlite3 as sql
import openpyxl
import funciones as fn

####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors



def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)

######funciones ######

#2. funcion recomendación por popularidad 
def mejores_peliculas_por_año(w):
    lista = []
    for año_pel in w['anio_pel'].unique(): 
        mejores_peliculas = w[w['anio_pel'] == año_pel].sort_values(by='pond', ascending=False).head(1)
        lista.append(mejores_peliculas)
    return pd.concat(lista)

#3. Funcion para preprocesar los generos de las peliculas y escalar el año de lanzamiento

def pre_KNN_1producto():
    conn = sql.connect('data_marketing//db_movies') # identifica bases de datos
    cur = conn.cursor() # permite e]jecutar comandos SQL

    pelicula=pd.read_sql('select * from movie_final2', conn )

    pelicula.info()
    pelicula['anio_pel']=pelicula.anio_pel.astype('int')
    pelicula.info()

    #convertir a dummies variable genero
    generos = set()
    for i in pelicula['genres'].str.split('|'):
        generos.update(i)

    #cuantos generos hay y cuales son?
    num_generos = len(generos)
    print(f'Hay {num_generos} generos en la base de datos')

    #convertir la variable genero en una lista de generos 
    pelicula['gen_list'] = pelicula['genres'].str.split('|')

    #obtener las categorias unicas de genero
    gen_unique = set()
    for i in pelicula['gen_list']:
        gen_unique.update(i)
        
    #Convertir a dummies y agregar al dataframe original
    for i in gen_unique:
        pelicula[i] = pelicula['gen_list'].apply(lambda x: 1 if i in x else 0)


    #eliminar la columna gen_list
    pelicula.drop(columns=['gen_list'], inplace=True)

    pelicula2 = pelicula.drop(columns = 'genres',axis=1)

    basemod = pelicula2.copy()

    #Eliminar columnas innecesarias
    basemod2 = basemod.copy()

    base_unique=basemod2.drop(columns=['movieId','pelicula'])

    #Escalamos el año de lanzamiento de la pelicula
    sc=MinMaxScaler()
    base_unique['anio_pel']= sc.fit_transform(base_unique[['anio_pel']])
    
    return base_unique,pelicula2

