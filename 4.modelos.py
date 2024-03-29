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
-user id -movie id  -rating  -mes y año de calificacion -pelicula"""



books_dum1=books.drop(columns=['isbn','i_url','year_pub','book_title'])

