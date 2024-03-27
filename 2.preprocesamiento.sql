--PREPROSESAMIENTO

-- BASE RATINGS

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

--4. CREAR TABLA CON USUARIOS QUE HAN VISTO MENOS DE 1000 PELICULAS

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
HAVING calificaciones >= 20 and calificaciones <= 150
ORDER BY calificaciones DESC;

--7. SE DEBERIA HACER UN FILTRO DE NUMERO DE PELICULAS VISTAS POR AÑO?

--CREAR TABLA FILTRADA DE RATINGS  --

DROP TABLE IF EXISTS rating_final;
CREATE TABLE rating_final AS 
SELECT ratings_alter.userid AS user_id, ratings_alter.movieId AS movie_id,rating,mes,anio
FROM ratings_alter 
INNER JOIN usuarios_sel ON ratings_alter.userid = usuarios_sel.usuario
INNER JOIN peliculas_sel ON ratings_alter.movieId = peliculas_sel.movieid;

-- CREAR TABLA FILTRADA DE MOVIES_ID

---JUNTAR TABLAS---