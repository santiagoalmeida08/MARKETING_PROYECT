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

pelicula=pd.read_sql('select * from movie_final', conn )
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

#### convertir a dummies

books_dum1['book_author'].nunique()
books_dum1['publisher'].nunique()

col_dum=['book_author','publisher']
books_dum2=pd.get_dummies(books_dum1,columns=col_dum)
books_dum2.shape

joblib.dump(books_dum2,"salidas\\books_dum2.joblib") ### para utilizar en segundos modelos



###### libros recomendadas ejemplo para un libro#####

libro='The Testament'
ind_libro=books[books['book_title']==libro].index.values.astype(int)[0]
similar_books=books_dum2.corrwith(books_dum2.iloc[ind_libro,:],axis=1)
similar_books=similar_books.sort_values(ascending=False)
top_similar_books=similar_books.to_frame(name="correlación").iloc[0:11,] ### el 11 es número de libros recomendados
top_similar_books['book_title']=books["book_title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    


#### libros recomendados ejemplo para visualización todos los libros

def recomendacion(libro = list(books['book_title'])):
     
    ind_libro=books[books['book_title']==libro].index.values.astype(int)[0]   #### obtener indice de libro seleccionado de lista
    similar_books = books_dum2.corrwith(books_dum2.iloc[ind_libro,:],axis=1) ## correlación entre libro seleccionado y todos los otros
    similar_books = similar_books.sort_values(ascending=False) #### ordenar correlaciones
    top_similar_books=similar_books.to_frame(name="correlación").iloc[0:11,] ### el 11 es número de libros recomendados
    top_similar_books['book_title']=books["book_title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    
    return top_similar_books


print(interact(recomendacion))


##############################################################################################
#### 2.1 Sistema de recomendación basado en contenido KNN un solo producto visto #################
##############################################################################################

##### ### entrenar modelo #####

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similar324e-06	3.336112e-01	3.336665e-01	3.336665e-es)
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(books_dum2)
dist, idlist = model.kneighbors(books_dum2)


distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(libro)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde


####ejemplo para un libro
book_list_name = []
book_name='2nd Chance'
book_id = books[books['book_title'] == book_name].index ### extraer el indice del libro
book_id = book_id[0] ## si encuentra varios solo guarde uno

for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].book_title) ### agrega el nombre de cada una de los id recomendados

book_list_name




def BookRecommender(book_name = list(books['book_title'].value_counts().index)):
    book_list_name = []
    book_id = books[books['book_title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].book_title)
    return book_list_name


print(interact(BookRecommender))
