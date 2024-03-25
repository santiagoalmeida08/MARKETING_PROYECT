import sqlite3 as sql
import pandas as pd
import seaborn as sns

# Conectarse a la base de datos 
conn = sql.connect('data_marketing//db_movies') # identifica bases de datos
cur = conn.cursor() # permite ejecutar comandos SQL

#Verificación de conexión
cur.execute('SELECT name FROM sqlite_master WHERE type="table"') # selecciona las tablas de la base de datos
cur.fetchall() # observamos las tablas que tiene la base de datos

# Ejecutar consultas con pandas #

ratings = pd.read_sql('SELECT * FROM ratings', conn) # contiene las calificaciones que los usuarios dieron a las películas
movie = pd.read_sql('SELECT * FROM movies', conn) # contiene información de las películas

# Observar contenido de las tablas y hacer consideraciones para preprocesamiento 

ratings.info()
ratings.sample(5)  #Cambiar formato timestamp, poner todas letras de variables en minúsculas

movie.info()
movie.sample(5) # separar el año de la película del nombre , poner nombre de variables y valores en minúsculas
                # Genero ??? 

# 1. Explorar la tabla ratings

#¿Cuales son las calificaciones mas frecuentes que los usuarios dan a las películas?

r = pd.read_sql(""" SELECT rating as calificacion, 
                    count(*) as conteo
                    FROM ratings    
                    GROUP BY rating 
                    ORDER BY conteo DESC""",conn)


sns.barplot(x='calificacion', y='conteo', data=r, color='orange')

#¿Cuantas peliculas ha visto cada usuario? #POR MES  --- SE NECESITA CONVERSION DE TIMESTAMP EN PREPROCESAMIENTO

p = pd.read_sql(""" SELECT userId as usuario, count(*) as conteo
                    FROM ratings
                    GROUP BY usuario
                    ORDER BY conteo DESC""",conn)


# ¿Cuales han sido la película con mayor calificadas?

mc = pd.read_sql(""" SELECT movieID as pelicula, rating as calificacion,
                     count(*) as conteo
                     FROM ratings
                     WHERE calificacion = 5.0
                     GROUP BY calificacion
                     ORDER BY conteo DESC""",conn)

