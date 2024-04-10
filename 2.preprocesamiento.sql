
-- PREPROCESAMIENTO BASE RATINGS

--1. Cambiar formato de timestamp a date

DROP TABLE IF EXISTS ratings_alter;
CREATE TABLE ratings_alter AS SELECT * FROM ratings;
ALTER TABLE ratings_alter ADD fecha DATETIME;
UPDATE ratings_alter SET fecha = DATETIME(ratings_alter.timestamp, 'unixepoch');

--2. SEPARAR EL MES DE LA FECHA  
ALTER TABLE ratings_alter ADD mes INT;
ALTER TABLE ratings_alter ADD anio INT;
UPDATE ratings_alter SET mes = strftime('%m', ratings_alter.fecha);
UPDATE ratings_alter SET anio = strftime('%Y', ratings_alter.fecha);

--3-BORRAR COLUMNAS INNECESARIAS TIMESTAMP Y FECHA
ALTER TABLE ratings_alter DROP COLUMN timestamp;
ALTER TABLE ratings_alter DROP COLUMN fecha;

                    -- FILTROS DE TABLA RATINGS--

--4. CREAR TABLA CON USUARIOS QUE HAN VISTO ENTRE 10 Y 1000 PELICULAS

DROP TABLE IF EXISTS usuarios_sel;
CREATE TABLE usuarios_sel AS SELECT userid AS usuario, count(*) AS num_peliculas
FROM ratings_alter
GROUP BY usuario
HAVING num_peliculas < 1000 and num_peliculas > 10
ORDER BY num_peliculas;

--5. CREAR TABLA CON PELICULAS QUE HAYAN SIDO CALIFICADAS MAS DE 20 VECES Y MENOS DE 150
DROP TABLE IF EXISTS peliculas_sel;
CREATE TABLE peliculas_sel AS SELECT movieid, count(*) AS calificaciones
FROM ratings_alter
GROUP BY movieid
HAVING calificaciones >= 7 and calificaciones <= 150
ORDER BY calificaciones DESC;

--CREAR TABLA FILTRADA DE RATINGS  --

DROP TABLE IF EXISTS rating_final;
CREATE TABLE rating_final AS 
SELECT ratings_alter.userid AS user_id, ratings_alter.movieId AS movie_id,rating,ratings_alter.mes AS mes_clf,ratings_alter.anio AS anio_clf
FROM ratings_alter 
INNER JOIN usuarios_sel ON ratings_alter.userid = usuarios_sel.usuario
INNER JOIN peliculas_sel ON ratings_alter.movieId = peliculas_sel.movieid;

-- PREPROCESAMIENTO BASE MOVIES

--1. SEPARAR EL AÑO DE LA PELICULA
DROP TABLE IF EXISTS movies_sel;
CREATE TABLE movies_sel AS
SELECT *,  SUBSTRING(
                    title, -5,4) AS anio_pel,
                    SUBSTRING(title, 1, LENGTH(title)-6) AS pelicula 
                    FROM movies;
                    

--- SELECCIONAR SOLO LAS FILAS QUE EL AÑO TENGA UN VALOR NUMERICO YA QUE HAY PELICULAS QUE NO TENIAN ESTA INFORMACION
DROP TABLE IF EXISTS movie_final2;
CREATE TABLE movie_final2 AS
SELECT movies_sel.*
from movies_sel
WHERE anio_pel GLOB '*[0-9]*';

DELETE FROM movie_final2
WHERE anio_pel LIKE '%)'; --Eliminar años que tengan parentesis

--BORRAR COLUMNAS INNECESARIAS

ALTER TABLE movie_final2 DROP COLUMN title;

---JUNTAR TABLAS---

DROP TABLE IF EXISTS final_table;
CREATE TABLE final_table AS
SELECT rating_final.user_id,rating_final.rating,rating_final.mes_clf,rating_final.anio_clf,rating_final.movie_id,
        movie_final2.genres,movie_final2.anio_pel,movie_final2.pelicula
FROM rating_final
INNER JOIN movie_final2 ON rating_final.movie_id = movie_final2.movieId;

