--PREPROSESAMIENTO

-- BASE RATINGS

--1. Cambiar formato de timestamp a date

DROP TABLE IF EXISTS ratings_alter;
CREATE TABLE ratings_alter AS SELECT * FROM ratings;
ALTER TABLE ratings_alter ADD fecha DATETIME;
UPDATE ratings_alter SET fecha = DATETIME(ratings_alter.timestamp, 'unixepoch');

--2. SEPARAR EL MES DE LA FECHA  
ALTER TABLE ratings_alter ADD mes INT;
UPDATE ratings_alter SET mes = strftime('%m', ratings_alter.fecha);

--3-BORRAR COLUMNAS INNECESARIAS TIMESTAMP Y FECHA
ALTER TABLE ratings_alter DROP COLUMN timestamp;
ALTER TABLE ratings_alter DROP COLUMN fecha;

