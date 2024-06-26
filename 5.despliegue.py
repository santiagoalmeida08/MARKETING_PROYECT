 #En este script de explica inicialmente de forma textual cual podria ser la mejor opcion de realizar el despliegue del modelo
# y adicionalmente se realiza la implementacion del modelo de recomendacion basado en contenido acorde al perfil de usuario

#1. Importar paquetes
#2. Descripción del despliegue 
#3. Implementación del modelo de recomendación basado en contenido	

#1. Importar paquetes
import numpy as np
import pandas as pd
import sqlite3 as sql
import funciones as fn ## para procesamiento
import openpyxl
#Paquete para sistema basado en contenido 
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

"""2. Descripción del despliegue : --considerar prediccion y entrenamiento 
-Para los sistemas de recomendación basados en el perfil del usuario se realizara una predicción a demanda de los usuarios, es decir,
cuando un usuario termine de ver una pelicula , se le recomendará una lista de peliculas que le puedan interesar; por lo tanto, el entrenamiento
del modelo se realizara diariamente para asi tener actualizado el perfil de los usuarios y poder realizar las recomendaciones de forma eficiente.

- Los sistemas basados en popularidad se actualizaran y se entrenaran  mensualmente, ya que se basan en la popularidad de las peliculas y no en el perfil
de los usuarios.Una vez se tengan las listas del top 5 o 10 de las peliculas se compartirá la información a los desarrodores front-end para que puedan hacer 
el respectivo diseño y modificaciones en el interfaz de la aplicación.

-Definir tipo de despliegue a abordar con : 
1.Frecuencia de entrenamiento
2.Frecuencia de predicción
Cuando se tenga las predicciones se debe:

-"""

#3. Implementación del modelo de recomendación basado en contenido
#3.1 Preprocesamiento de datos

def preprocesar():

    #### conectar_base_de_Datos#################
    conn = sql.connect('C:\\Users\\Usuario\\Desktop\\Analitica3\\MARKETING_PROYECT\\MARKETING_PROYECT\\data_marketing\\db_movies') # identifica bases de datos
    cur = conn.cursor()
    

    ######## Aplicar preprocesamiento 
    fn.ejecutar_sql('C:\\Users\\Usuario\\Desktop\\Analitica3\\MARKETING_PROYECT\\MARKETING_PROYECT\\2.preprocesamiento.sql',conn)

    ##### llevar datos que cambian constantemente a python ######
    pelicula=pd.read_sql('select * from movie_final2', conn )
    ratings=pd.read_sql('select * from rating_final', conn)
    final=pd.read_sql('select * from final_table', conn)
    user=pd.read_sql('select distinct (user_id) as user_id from final_table',conn)

    
    #### transformación de datos crudos - Preprocesamiento ################
    
    pelicula['anio_pel']=pelicula.anio_pel.astype('int')

    #convertir la variable genero en una lista de generos 
    pelicula['gen_list'] = pelicula['genres'].str.split('|')
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

    basemod = pelicula2.copy()

    #Eliminar columnas innecesarias
    basemod2 = basemod.copy()

    base_unique=basemod2.drop(columns=['movieId','pelicula'])

    #Escalamos el año de lanzamiento de la pelicula
    sc=MinMaxScaler()
    base_unique['anio_pel']= sc.fit_transform(base_unique[['anio_pel']])

    return base_unique,pelicula, conn, cur


################################################################################################
###############Función para entrenar modelo por cada usuario ###################################
###############Basado en contenido todo lo visto por el usuario Knn#############################

#3.2 Función para recomendar peliculas a un usuario en especifico
def recomendar(user_id):
    
    base_unique, pelicula, conn, cur= preprocesar()
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select *from rating_final where user_id=:user',conn, params={'user':user_id,})
    
    ###convertir ratings del usuario a array
    mov_view=ratings['movie_id'].to_numpy()
    
    # Agregamos el movir_id y el nombre de la pelicula a la base de datos con dummies
    base_unique[['movie_id','pelicula']]=pelicula[['movieId','pelicula']]
    
    ### filtrar peliculas calificadas por el usuario
    mov_v2=base_unique[base_unique['movie_id'].isin(mov_view)]
    
    ## eliminar columnas de nombre y movie_id
    mov_v2=mov_v2.drop(columns=['movie_id','pelicula'])
    mov_v2["indice"]=1 
    
    ##centroide o perfil del usuario
    centroide=mov_v2.groupby("indice").mean()
    
    
    ### filtrar peliculas que no ha visto el usuario 
    mov_nv=base_unique[~base_unique['movie_id'].isin(mov_view)]
    ## eliminbar nombre e isbn
    mov_nv=mov_nv.drop(columns=['movie_id','pelicula'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(mov_nv)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_b=pelicula.loc[ids][['pelicula','movieId']]
    #leidos=pelicula[pelicula['movieId'].isin(mov_v2)][['pelicula','movieId']] 
    return recomend_b

##### Generar recomendaciones para usuario lista de usuarios ####
##### No se hace para todos porque es muy pesado #############

#3.3 Función para generar recomendaciones para una lista de usuarios
def main(list_user):
    
    recomendaciones_todos=pd.DataFrame()
    for user_id in list_user:
            
        recomendaciones=recomendar(user_id)
        recomendaciones["user_id"]=user_id
        recomendaciones.reset_index(inplace=True,drop=True)
        
        recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])
        
    #RUTAS COMPUTADOR SANTIAGO  
    recomendaciones_todos.to_excel('C:\\Users\\Usuario\\Desktop\\Analitica3\\MARKETING_PROYECT\\MARKETING_PROYECT\\salidas\\recomendaciones.xlsx')
    recomendaciones_todos.to_csv('C:\\Users\\Usuario\\Desktop\\Analitica3\\MARKETING_PROYECT\\MARKETING_PROYECT\\salidas\\recomendaciones.csv')


if __name__=="__main__":
    list_user=[6,100,350,120 ]
    main(list_user)
    

import sys
sys.executable