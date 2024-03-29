
def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)

######funciones ######
#1. Funcion sistema de recomendación por popularidad

def mejores_peliculas_por_mes(m):
    lista = []
    for mes in m['mes_clf'].unique(): 
        mejores_peliculas = m[m['mes_clf'] == mes].sort_values(by='pond', ascending=False).head(5)
        lista.append(mejores_peliculas)
    return pd.concat(lista)
