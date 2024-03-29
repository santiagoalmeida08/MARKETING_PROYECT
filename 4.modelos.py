import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib


conn = sql.connect('data_marketing//db_movies') # identifica bases de datos
cur = conn.cursor() # permite e]jecutar comandos SQL


#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

######################################################################
################## 1. sistemas basados en popularidad ###############
#####################################################################


##### recomendaciones basado en popularidad ######

#### 10 peliculas con mejores calificación 
#MIRAR SI ES IMPORTANTE QUE SEAN LAS MAS VISTAS 

pd.read_sql("""select pelicula, 
            avg(rating) as avg_rat,
            count(*) as vistas
            from full_table
            group by pelicula
            order by avg_rat desc
            limit 10
            """, conn)


###Libros mas leidos con su promedio de calificación####

pd.read_sql("""select pelicula, 
            avg(rating) as avg_rat,
            count(*) as vistas
            from full_table
            group by pelicula
            order by vistas desc
            """, conn)

# las  peliculas mejores calificadas segun el año de lanzamiento de la pelicula###
pd.read_sql("""select anio_pel, pelicula, 
            avg(rating) as avg_rat,
            count(rating) as rat_numb,
            count(*) as vistas
            from full_table
            group by  anio_pel, pelicula
            order by anio_pel desc, avg_rat desc limit 20
            """, conn)




#######################################################################
######## 2.1 Sistema de recomendación basado en contenido un solo producto - KNN########
#######################################################################

pelicula=pd.read_sql('select * from final_table', conn )
pelicula.info()
pelicula['anio_pel']=pelicula.anio_pel.astype('int')
pelicula.info()


##### escalar para que año esté en el mismo rango ###

sc=MinMaxScaler()
pelicula[["year_sc"]]=sc.fit_transform(pelicula[['anio_pel']])


## eliminar variables que no se van a utilizar ###
"""Las columnas que no se van a usar son:
-user id -movie id y movieId  -rating  -mes y año de calificacion -pelicula y title """

"""Se usaran solos las columnas genero y año de la pelicula"""

pelicula_dum1=pelicula.drop(columns=['user_id','movie_id','rating','mes_clf', 'anio_clf', 'movieId', 'title','pelicula', 'anio_pel'])

#convertir a dummies 
pelicula_dum1['genres'].nunique()

col_dum=['genres'] #columnas que se van a convertir a dummies
pelicula_dum2= pd.get_dummies(pelicula_dum1, columns=col_dum)
pelicula_dum2.shape

##### ### entrenar modelo #####

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similar324e-06	3.336112e-01	3.336665e-01	3.336665e-es)
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(pelicula_dum2)
dist, idlist = model.kneighbors(pelicula_dum2)


distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(libro)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde




def MovieRecommender(movie_name = list(pelicula['pelicula'].value_counts().index)):
    movie_list_name = []
    movie_id = pelicula[pelicula['pelicula'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        movie_list_name.append(pelicula.loc[newid].pelicula)
    return movie_list_name

print(interact(MovieRecommender))
