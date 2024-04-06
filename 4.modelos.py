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

q['pond'] = q['avg_rat']*(q['vistas']/q['vistas'].max())

q.sort_values(by=['pond'], ascending=False).head(10)# Se tiene una nueva columna que es pond en la cual se balancea el rating con las vistas y obetener un nuevo puntaje 
q



# 5 peliculas mejor calificadas del mes

def Top_5_mejor_calificadas_del_mes(mes):
    m = pd.read_sql(f"""select mes_clf, pelicula,
            avg(rating) as avg_rat,
            count(*) as vistas
            from final_table
            where mes_clf = {mes}
            group by mes_clf, pelicula
            order by avg_rat desc
            """, conn)
    m['pond'] = m['avg_rat']*(m['vistas']/m['vistas'].max())
    return m.sort_values(by=['pond'], ascending=False).head(5)

print(interact(Top_5_mejor_calificadas_del_mes,mes=(1,12)))


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


####################################################################################
######## 2.1 Sistema de recomendación basado en contenido un solo producto - KNN ########
####################################################################################


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

basemod = pelicula2.copy()

#Eliminar columnas innecesarias
basemod2 = basemod.copy()

base_unique=basemod2.drop(columns=['movieId','pelicula'])

#Escalamos el año de lanzamiento de la pelicula
sc=MinMaxScaler()
base_unique['anio_pel']= sc.fit_transform(base_unique[['anio_pel']])
base_unique

#Exportamos la base de datos con dummies y escalada para poder utilizarla en otros modelos
joblib.dump(base_unique,"salidas\\base_unique.joblib") 

# Train Modelo de recomendacion  a travez de KNN (5 peliculas mas similares)

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similares)
model = neighbors.NearestNeighbors(n_neighbors=5, metric='cosine') # definimos las peliculas a recomendar y la metrica para medir las distancias 
model.fit(base_unique)
dist, idlist = model.kneighbors(base_unique)


distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(pelicula)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde


def Top_5_peliculas_similares(movie_name = list(pelicula2['pelicula'].value_counts().index)):
    movie_list_name = []
    movie_id = pelicula2[pelicula2['pelicula'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:  
        if newid == movie_id:
            continue # si es el mismo no lo recomienda
        movie_list_name.append(pelicula2.loc[newid].pelicula)
    return list(set(movie_list_name)) 

print(interact(Top_5_peliculas_similares))


#######################################################################
#### 3 Sistema de recomendación basado en contenido KNN #################
#### Generando un perfil al usuario              #######################
#######################################################################

# Base de datos con dummies y escalada

dum1 = joblib.load("salidas\\base_unique.joblib") 
dum1

#   Seleccionamos el usuario

rat = pd.read_sql('select * from rating_final', conn) # se selecciona la tabla que contiene los ratings de los usuarios
pel = pd.read_sql('select * from movie_final2', conn) # se selecciona la tabla que contiene informacion de las peliculas

user = pd.read_sql('select distinct (user_id) as user_id from rating_final',conn)

def Recomendacion_segun_perfil_usuario(user_id=list(user['user_id'].value_counts().index)):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select *from rating_final where user_id=:user',conn, params={'user':user_id,})
    
    ###convertir ratings del usuario a array
    mov_view=ratings['movie_id'].to_numpy()
    
    # Agregamos el movir_id y el nombre de la pelicula a la base de datos con dummies
    dum1[['movie_id','pelicula']]=pel[['movieId','pelicula']]
    
    ### filtrar peliculas calificadas por el usuario
    mov_v2=dum1[dum1['movie_id'].isin(mov_view)]
    
    ## eliminar columnas de nombre y movie_id
    mov_v2=mov_v2.drop(columns=['movie_id','pelicula'])
    mov_v2["indice"]=1 
    
    ##centroide o perfil del usuario
    centroide=mov_v2.groupby("indice").mean()
    
    
    ### filtrar peliculas que no ha visto el usuario 
    mov_nv=dum1[~dum1['movie_id'].isin(mov_view)]
    ## eliminbar nombre e isbn
    mov_nv=mov_nv.drop(columns=['movie_id','pelicula'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(mov_nv)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_b=pel.loc[ids][['pelicula','movieId']]
    leidos=pel[pel['movieId'].isin(mov_v2)][['pelicula','movieId']]
    
    return recomend_b


print(interact(Recomendacion_segun_perfil_usuario))


############################################################################
#####4. Sistema de recomendación filtro colaborativo #####
############################################################################

### datos originales en pandas
## knn solo sirve para calificaciones explicitas
ratings=pd.read_sql('select * from rating_final', conn)
ratings['rating'].value_counts()

#Lectura de datos con surprise

reader = Reader(rating_scale=(0, 5)) ### la escala de la calificación 0-5

#Lectura dataset con el orden especificado 
data   = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)

from surprise import SVD
#####Existen varios modelos 
models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline(),SVD()] 
results = {}

###knnBasiscs: calcula el rating ponderando por distancia con usuario/Items
###KnnWith means: en la ponderación se resta la media del rating, y al final se suma la media general
####KnnwithZscores: estandariza el rating restando media y dividiendo por desviación 
####Knnbaseline: calculan el desvío de cada calificación con respecto al promedio y con base en esos calculan la ponderación


#### for para probar varios modelos ##########
model=models[1]

for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1) # se usan medidas de regresion ya que se esta prediciendo una variable numerica
    
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result


performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')


#el modelo de knns.KNNWithMeans el RMSE no varia mucho comparado con los demas modelos y ademas el tiempo es el menor por esto se selecciona este modelo
###################se escoge el mejor knn withmeans#########################
param_grid = { 'sim_options' : {'name': ['msd','cosine'], \
                                'min_support': [10,5], \
                                'user_based': [False, True]}
             }# si no hay minimo 5 personas que lo calificaron no recomiende el producto 

## min support es la cantidad de items o usuarios que necesita para calcular recomendación
## name medidas de distancia

### se afina si es basado en usuario o basado en ítem

gridsearchKNNWithMeans = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], \
                                      cv=2, n_jobs=-1)# el cv y el n_jobs puede variar
                                    
gridsearchKNNWithMeans.fit(data)


gridsearchKNNWithMeans.best_params["rmse"]
gridsearchKNNWithMeans.best_score["rmse"]
gs_model=gridsearchKNNWithMeans.best_estimator['rmse'] ### mejor estimador de gridsearch


################# Entrenar con todos los datos y Realizar predicciones con el modelo afinado

trainset = data.build_full_trainset() ### esta función convierte todos los datos en entrnamiento, las funciones anteriores dividen  en entrenamiento y evaluación
model=gs_model.fit(trainset) ## se reentrena sobre todos los datos posibles (sin dividir)



predset = trainset.build_anti_testset() ### crea una tabla con todos los usuarios y las peliculas que no han visto
#### en la columna de rating pone el promedio de todos los rating, en caso de que no pueda calcularlo para un item-usuario
len(predset)

predictions = gs_model.test(predset) ### función muy pesada, hace las predicciones de rating para todos las peliculas que no hay leido un usuario
### la funcion test recibe un test set constriuido con build_test method, o el que genera crosvalidate

####### la predicción se puede hacer para un libro puntual
#model.predict(uid=269397, iid='0446353205',r_ui='') ### uid debía estar en número e isb en comillas

predictions_df = pd.DataFrame(predictions) ### esta tabla se puede llevar a una base donde estarán todas las predicciones
predictions_df.shape
predictions_df.head()
predictions_df['r_ui'].unique() ### promedio de ratings
predictions_df.sort_values(by='est',ascending=False)


##### funcion para recomendar los 10 peliculas con mejores predicciones y llevar base de datos para consultar resto de información
user = pd.read_sql('select distinct (user_id) as user_id from rating_final',conn)
user
def recom_colaborativas(user_id=list(user['user_id'].value_counts().index),n_recomend=10):
    
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace")
    
    recomendados=pd.read_sql('''select a.*, b.pelicula 
                             from reco a left join movie_final2 b
                             on a.iid=b.movieId ''', conn)

    return(recomendados)


 
recom_colaborativas(user_id=609,n_recomend=10)


print(interact(recom_colaborativas))







































