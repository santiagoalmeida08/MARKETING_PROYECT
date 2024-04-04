import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler




def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)

######funciones ######
#1. Funcion sistema de recomendación por popularidad

def mejores_peliculas_por_mes(m):
    lista = []
    for mes in m['mes_clf'].unique(): 
        mejores_peliculas = m[m['mes_clf'] == mes].sort_values(by='pond', ascending=False).head(5)
        lista.append(mejores_peliculas)
    return pd.concat(lista)
#2. funcion recomendación por popularidad 
def mejores_peliculas_por_año(w):
    lista = []
    for año_pel in w['anio_pel'].unique(): 
        mejores_peliculas = w[w['anio_pel'] == año_pel].sort_values(by='pond', ascending=False).head(1)
        lista.append(mejores_peliculas)
    return pd.concat(lista)

#1. Sistema de recomendacion basado en 1 solo producto
def proproc_sistema_1soloproducto():
    
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

    pd.set_option('display.max_columns', None)

    #Obstervar dataframe con generos dummies 
    pelicula.sample(10)

    pelicula2 = pelicula.drop(columns = 'genres',axis=1)

    pelicula2[pelicula2['movieId']== 5]

    basemod = pelicula2.copy()

    #eliminar columnas innecesarias
    basemod2 = basemod.copy()


    base_unique=basemod2.drop(columns=['movieId','pelicula'])

    sc=MinMaxScaler()
    base_unique['anio_pel']= sc.fit_transform(base_unique[['anio_pel']])
    base_unique
    return base_unique, pelicula2 
