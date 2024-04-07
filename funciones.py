import numpy as np
import pandas as pd
import sqlite3 as sql
import openpyxl


####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors



def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)

######funciones ######

#2. funcion recomendación por popularidad 
def mejores_peliculas_por_año(w):
    lista = []
    for año_pel in w['anio_pel'].unique(): 
        mejores_peliculas = w[w['anio_pel'] == año_pel].sort_values(by='pond', ascending=False).head(1)
        lista.append(mejores_peliculas)
    return pd.concat(lista)
