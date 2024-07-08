

-- Step 1: Check continent counts
SELECT Continent, COUNT(*) 
FROM wine_schema.consowine
GROUP BY Continent;

-- Step 2: Create a new dimension table for the continent dimension
CREATE TABLE IF NOT EXISTS wine_schema.continent (
    continent_id INT AUTO_INCREMENT,
    Continent VARCHAR(255),
    PRIMARY KEY (continent_id)
);

-- Step 3: Populate the continent table with distinct continent values from consowine
INSERT INTO wine_schema.continent (Continent)
SELECT DISTINCT Continent 
FROM wine_schema.consowine;

-- Step 4: Verify the continent table is populated correctly
SELECT * 
FROM wine_schema.continent;

-- Step 5: Alter the consowine table to add the continent_id column
ALTER TABLE wine_schema.consowine 
ADD COLUMN continent_id INT AFTER Continent;

-- Optional Step 6: Set up the foreign key reference
ALTER TABLE wine_schema.consowine 
ADD CONSTRAINT continent_fk FOREIGN KEY (continent_id) REFERENCES wine_schema.continent (continent_id);

-- Step 7: Verify the continent_id column has been added
SELECT * 
FROM wine_schema.consowine 
LIMIT 10;

SET SQL_SAFE_UPDATES = 0;

-- Step 8: Populate the continent_id column using the dimension table we created
UPDATE wine_schema.consowine AS cw
JOIN wine_schema.continent AS c
ON cw.Continent = c.Continent
SET cw.continent_id = c.continent_id;

SET SQL_SAFE_UPDATES = 1;

-- Step 9: Verify the continent_id column is populated
SELECT * 
FROM wine_schema.consowine 
LIMIT 200;

-- Optional: Check for any rows where the continent_id is still NULL
SELECT * 
FROM wine_schema.consowine 
WHERE continent_id IS NULL;

-- lets drop the original column now
alter table wine_schema.consowine drop column Continent;

-- check everything is as expected
select * from wine_schema.consowine limit 10;


SELECT
    TABLE_NAME,
    COLUMN_NAME,
    CONSTRAINT_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM
    INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE
	REFERENCED_TABLE_SCHEMA = 'wine_schema'
    AND REFERENCED_TABLE_NAME = 'continent';


