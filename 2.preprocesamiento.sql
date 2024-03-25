--PREPROSESAMIENTO

-- BASE RATINGS
--1. Cambiar formato de timestamp a date

DROP TABLE IF EXISTS ratings_alter;
CREATE TABLE ratings_alter AS 
SELECT * FROM ratings;
ALTER TABLE ratings_alter ADD  fecha DATE;
UPDATE rating_alter SET fecha = date(timestamp, 'unixepoch');
ALTER TABLE ratings_alter DROP timestamp;





