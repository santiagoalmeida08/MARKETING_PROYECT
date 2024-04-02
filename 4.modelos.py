import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
import funciones as fn


from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import train_test_split

conn = sql.connect('data_marketing//db_movies') # identifica bases de datos
cur = conn.cursor() # permite e]jecutar comandos SQL


#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

######################################################################
################## 1. sistemas basados en popularidad ###############
#####################################################################


##### recomendaciones basado en popularidad ######

# las peliculas 10 mejores calificadas DE LA PLATAFORMA  

q = pd.read_sql("""select pelicula,
            avg(rating) as avg_rat,
            count(*) as vistas
            from final_table
            group by  pelicula
            order by avg_rat desc 
            """, conn)

q['pond'] = q['avg_rat']*q['vistas']

q.sort_values(by=['pond'], ascending=False).head(10)# Se tiene una nueva columna que es pond en la cual se balancea el rating con las vistas y obetenr un nuevo puntaje 


# 5 peliculas mejor calificadas del mes

m = pd.read_sql("""select mes_clf, pelicula,
            avg(rating) as avg_rat,
            count(*) as vistas
            from final_table
            group by mes_clf, pelicula
            order by mes_clf, avg_rat desc
            """, conn)

m['pond'] = m['avg_rat']*m['vistas']


mejores_peliculas = fn.mejores_peliculas_por_mes(m)


# las  peliculas mejores calificadas segun el año de lanzamiento de la pelicula###
w = pd.read_sql("""select anio_pel, pelicula, 
            avg(rating) as avg_rat,
            count(rating) as rat_numb,
            count(*) as vistas
            from final_table
            group by  anio_pel, pelicula
            
            """, conn)

w['pond'] = w['avg_rat']*w['vistas']
w.sort_values(by=['pond'], ascending=False)


mejores_peliculas_año = fn.mejores_peliculas_por_año(w)


#######################################################################
######## 2.1 Sistema de recomendación basado en contenido un solo producto - KNN########
#######################################################################


pelicula=pd.read_sql('select * from movie_final2', conn )
pelicula.info()
pelicula['anio_pel']=pelicula.anio_pel.astype('int')
pelicula.info()


##### escalar para que año esté en el mismo rango ###

sc=MinMaxScaler()
pelicula[["year_sc"]]=sc.fit_transform(pelicula[['anio_pel']])#año escalado


## eliminar variables que no se van a utilizar ###
"""Las columnas que no se van a usar son:
- movieId   -pelicula y anio_pel """

"""Se usaran solos las columnas genero y año de la pelicula"""

pelicula_dum1=pelicula.drop(columns=['movieId','pelicula', 'anio_pel'])

#convertir a dummies 
pelicula_dum1['genres'].nunique()

col_dum=['genres'] #columnas que se van a convertir a dummies
pelicula_dum2= pd.get_dummies(pelicula_dum1, columns=col_dum)
pelicula_dum2.shape

##### ### entrenar modelo #####

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similares)
model = neighbors.NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(pelicula_dum2)
dist, idlist = model.kneighbors(pelicula_dum2)


distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(pelicula)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde



pelicula2=pelicula.copy()

def MovieRecommender(movie_name = list(pelicula['pelicula'].value_counts().index)):
    movie_list_name = []
    movie_id = pelicula[pelicula['pelicula'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        movie_list_name.append(pelicula.loc[newid].pelicula)
    return movie_list_name

print(interact(MovieRecommender))




















#######################################################################
######## 2.1 Sistema de recomendación basado en contenido un solo producto - KNN########
#######################################################################


pelicula=pd.read_sql('select * from final_table', conn )

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

basemod = pelicula2.copy()
##### escalar para que año esté en el mismo rango ###


basemod=basemod.drop(columns=['movie_id','user_id','rating','mes_clf','anio_clf','pelicula'])
basemod.info()

sc=MinMaxScaler()
basemod = sc.fit_transform(basemod)




## eliminar variables que no se van a utilizar ###
"""Las columnas que no se van a usar son:
- movieid,user_id,rating,mest_clf,anio_clf,pelicula"""

"""Se usaran solos las columnas genero y año de la pelicula"""



##### ### entrenar modelo #####

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similares)
model = neighbors.NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(basemod)
dist, idlist = model.kneighbors(basemod)


distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(pelicula)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde


pelicula2['pelicula'].sample(5)

def MovieRecommender(movie_name = list(pelicula2['pelicula'].value_counts().index)):
    movie_list_name = []
    movie_id = pelicula2[pelicula2['pelicula'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        movie_list_name.append(pelicula2.loc[newid].pelicula)
    return movie_list_name

print(interact(MovieRecommender))

