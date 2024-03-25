import sqlite3 as sql
import pandas as pd
import seaborn as sns
import funciones as fn

# Conectarse a la base de datos 
conn = sql.connect('data_marketing//db_movies') # identifica bases de datos
cur = conn.cursor() # permite ejecutar comandos SQL

#Verificación de conexión
cur.execute('SELECT name FROM sqlite_master WHERE type="table"') # selecciona las tablas de la base de datos
cur.fetchall() # observamos las tablas que tiene la base de datos

# Ejecutar consultas con pandas #

ratings = pd.read_sql('SELECT * FROM ratings', conn) # contiene las calificaciones que los usuarios dieron a las películas
movie = pd.read_sql('SELECT * FROM movies', conn)# contiene información de las películas
#ratings2 = pd.read_sql('SELECT * FROM ratings2', conn) 

# Observar contenido de las tablas y hacer consideraciones para preprocesamiento 

ratings.info()
ratings.sample(5)  #Cambiar formato timestamp, poner todas letras de variables en minúsculas

movie.info()
movie.sample(5) # separar el año de la película del nombre , poner nombre de variables y valores en minúsculas
                # Genero ??? 


# Convertir el timestamp a formato fecha en la tabla ratings

p = ratings.copy()
p['fecha'] = pd.to_datetime(p['timestamp'], unit='s') # convertir timestamp a fecha 

fn.ejecutar_sql('2.preprocesamiento.sql',conn) # ejecutar script de preprocesamiento

ratings_alter.info()




# 1. Explorar la tabla ratings

## CONVERTIR EL TIMESTAMP A FORMATO FECHA DE LA TABLA RATINGS Y DESPUES REALIZAR FILTROS DE DATOS##

#¿Cuantas películas ha visto cada usuario?#POR MES  --- SE NECESITA CONVERSION DE TIMESTAMP EN PREPROCESAMIENTO

s = pd.read_sql(""" SELECT userId as usuario, count(*) as numero_peliculas
                    FROM ratings
                    GROUP BY usuario
                    ORDER BY numero_peliculas DESC""",conn)
s.describe()

sns.histplot(s['numero_peliculas'], color='orange', bins=70 )

# Se eliminaran a los usuarios que han visto mas de 1000 peliculas

s2 = pd.read_sql(""" SELECT userId, count(*) num_peliculas
                FROM ratings
                GROUP BY userId
                HAVING num_peliculas < 1000
                ORDER BY num_peliculas""",conn)

sns.histplot(s2['num_peliculas'], color='orange', bins=70 )

#¿Cuales son las calificaciones mas frecuentes que los usuarios dan a las películas?

r = pd.read_sql(""" SELECT rating as calificacion, 
                    count(*) as conteo
                    FROM ratings    
                    GROUP BY rating 
                    ORDER BY conteo DESC""",conn)


sns.barplot(x='calificacion', y='conteo', data=r, color='orange')

# ¿Cuales han sido las películas con mayor calificación?

mc = pd.read_sql(""" SELECT movieId as pelicula, rating as calificacion,
                     count(*) as conteo
                     FROM ratings
                     GROUP BY calificacion
                     ORDER BY conteo ASC""",conn)

sns.histplot(mc[''], color='orange', bins=70)   

# Por cuantos usuarios ha sido calificada cada película?

f = pd.read_sql("""SELECT movieId as pelicula, count(*) as conteo
               FROM ratings 
               GROUP BY pelicula
               ORDER BY conteo DESC""",conn)

sns.histplot(f['pelicula'], color='orange', bins=70)
