ALTER TABLE consowine RENAME COLUMN `Region/Country` TO Country;

-- Step 1: Check continent counts
SELECT Country, COUNT(*)
FROM wine_schema.consowine
GROUP BY 1;

-- Step 2: Create a new dimension table for the country dimension
CREATE TABLE IF NOT EXISTS wine_schema.country (
    country_id INT AUTO_INCREMENT,
    Country VARCHAR(255),
    PRIMARY KEY (country_id)
);

-- Step 3: Populate the country table with distinct continent values from consowine
INSERT INTO wine_schema.country (Country)
SELECT DISTINCT Country
FROM wine_schema.consowine;

-- Step 4: Verify the continent table is populated correctly
SELECT * 
FROM wine_schema.country;

-- Step 5: Alter the consowine table to add the continent_id column
ALTER TABLE wine_schema.consowine 
ADD COLUMN country_id INT AFTER Country;

-- Optional Step 6: Set up the foreign key reference
ALTER TABLE wine_schema.consowine 
ADD CONSTRAINT country_fk FOREIGN KEY (country_id) REFERENCES wine_schema.country (country_id);

-- Step 7: Verify the continent_id column has been added
SELECT * 
FROM wine_schema.consowine 
LIMIT 10;


-- Disable safe update mode
SET SQL_SAFE_UPDATES = 0;

-- Update the country_id column
UPDATE wine_schema.consowine AS cw
JOIN wine_schema.country AS c
ON cw.Country = c.Country
SET cw.country_id = c.country_id;

-- Re-enable safe update mode
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
alter table wine_schema.consowine drop column Country;

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
    AND REFERENCED_TABLE_NAME = 'country';
