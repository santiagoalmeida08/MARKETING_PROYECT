import sqlite3 as sql
import pandas as pd
import seaborn as sns
import funciones as fn

# Conectarse a la base de datos 

conn = sql.connect('data_marketing//db_movies') # identifica bases de datos
cur = conn.cursor() # permite e]jecutar comandos SQL


#Verificación de conexión
cur.execute('SELECT name FROM sqlite_master WHERE type="table"') # selecciona las tablas de la base de datos
cur.fetchall() # observamos las tablas que tiene la base de datos


# Ejecutar consultas con pandas para observar el contenido de las tablas #

ratings = pd.read_sql('SELECT * FROM ratings', conn) # contiene las calificaciones que los usuarios dieron a las películas
movie = pd.read_sql('SELECT * FROM movies', conn)# contiene información de las películas
#usua = pd.read_sql('SELECT * FROM usuarios_sel', conn)

# Observar contenido de las tablas y hacer consideraciones para preprocesamiento 

ratings.info()
ratings.sample(5)  #Cambiar formato timestamp, poner todas letras de variables en minúsculas

movie.info()
movie.sample(5) # separar el año de la película del nombre , poner nombre de variables y valores en minúsculas
                # Genero ??? 


# Convertir el timestamp a formato fecha en la tabla ratings
#Ejecutamos el script de preprocesamiento en esta etapa para cambiar el formato de timestamp a fecha y hacer la exploración de datos

fn.ejecutar_sql('2.preprocesamiento.sql',conn) # ejecutar script de preprocesamiento

ratings_alter = pd.read_sql('SELECT * FROM ratings_alter', conn)
ratings_alter.info()

# 1. Exploración de datos tabla ratings_alter

#¿Cuantas películas ha visto cada usuario?

s = pd.read_sql(""" SELECT userid as usuario, count(*) as numero_peliculas
                    FROM ratings_alter
                    GROUP BY usuario
                    ORDER BY numero_peliculas DESC""",conn)
s.sample(10)

sns.histplot(s['numero_peliculas'], color='orange', bins=70 )

# Se eliminaran a los usuarios que han visto menos de 1000 peliculas y mas de 10

s2 = pd.read_sql(""" SELECT userid, count(*) AS num_peliculas
                FROM ratings_alter
                GROUP BY userid
                HAVING num_peliculas < 1000 and num_peliculas > 10
                ORDER BY num_peliculas""",conn)

sns.histplot(s2['num_peliculas'], color='orange', bins=70 )

# cuantas peliculas han sido vistas mensualmente ?

sr = pd.read_sql(""" SELECT mes, count(*) as peliculas_vistas
                FROM ratings_alter
                GROUP BY mes
                ORDER BY peliculas_vistas DESC""",conn) 


#¿Cuales son las calificaciones mas frecuentes que los usuarios dan a las películas?

r = pd.read_sql(""" SELECT rating as calificacion, 
                    count(*) as conteo
                    FROM ratings_alter    
                    GROUP BY rating 
                    ORDER BY conteo DESC""",conn)


sns.barplot(x='calificacion', y='conteo', data=r, color='orange')


# Por cuantos usuarios ha sido calificada cada película?

f = pd.read_sql("""SELECT movieId as pelicula, count(*) as calificaciones
               FROM ratings_alter
               GROUP BY pelicula
               ORDER BY calificaciones DESC""",conn)

sns.histplot(f['calificaciones'], color='orange', bins=70)

f.describe() # Se eliminan las peliculas con mas de 7 calificaciones y menos de 150    



f2 = pd.read_sql("""SELECT movieid, count(*) as calificaciones
                    FROM ratings_alter
                    GROUP BY movieid
                    HAVING calificaciones >= 7 and calificaciones <= 150
                    ORDER BY calificaciones DESC""",conn)

f2.describe()

sns.histplot(f2['calificaciones'], color='orange', bins=70)



#Exploración de datos tabla movies

movie.info()

#Separar el año de la película del nombre

ll = pd.read_sql("""
                     SELECT *,  SUBSTRING(
                                        title, -5, 4) AS anio,
                    SUBSTRING(title, 1, LENGTH(title)-6) AS pelicula 
                    FROM movies""", conn)

ll.head(200)

tabla= pd.read_sql("""SELECT * FROM movie_final
            WHERE anio_pel GLOB '*[0-9]*' 
            """, conn) # SELECCIONAR SOLO LAS FILAS QUE TIENEN CARACTERES NUMERICOS


# Cuantas peliculas tiene cada genero?

g1 = pd.read_sql("""SELECT genres AS genero,count(*) AS num_peliculas
                    FROM movies_sel
                    GROUP BY genero
                    ORDER BY num_peliculas DESC""",conn)###


g2 = pd.read_sql("""SELECT genres AS genero,count(*) AS num_peliculas
                    FROM movies_sel
                    GROUP BY genero
                    HAVING num_peliculas >= 20
                    ORDER BY num_peliculas DESC""",conn)###



gen = pd.read_sql('SELECT * FROM gen', conn)


#Observamos la tabla final

final_table = pd.read_sql('SELECT * FROM final_table', conn)
final_table.sample(3)

