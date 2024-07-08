

-- Step 1: Check continent counts
SELECT Dept_Name, Dept_Name, COUNT(*) 
FROM wine_schema.regions_prod
GROUP BY 1;

-- Step 2: Create a new dimension table for the continent dimension
CREATE TABLE IF NOT EXISTS wine_schema.departement (
    departement_id INT AUTO_INCREMENT,
    Dept_Code VARCHAR(255),
    Dept_Name VARCHAR(255),
    PRIMARY KEY (departement_id)
);

-- Step 3: Populate the department table with distinct department values from consowine
INSERT INTO wine_schema.departement (Dept_Code, Dept_Name)
SELECT DISTINCT Dept_Code, Dept_Name 
FROM wine_schema.regions_prod;

-- Step 4: Verify the continent table is populated correctly
SELECT * 
FROM wine_schema.departement;

-- Step 5: Alter the consowine table to add the continent_id column
ALTER TABLE wine_schema.regions_prod 
ADD COLUMN departement_id INT AFTER Dept_Code;

-- Optional Step 6: Set up the foreign key reference
ALTER TABLE wine_schema.regions_prod 
ADD CONSTRAINT departement_fk FOREIGN KEY (departement_id) REFERENCES wine_schema.departement (departement_id);

-- Step 7: Verify the continent_id column has been added
SELECT * 
FROM wine_schema.regions_prod 
LIMIT 10;

SET SQL_SAFE_UPDATES = 0;

-- Step 8: Populate the continent_id column using the dimension table we created
UPDATE wine_schema.regions_prod AS rp
JOIN wine_schema.departement AS d 
ON rp.Dept_Code = d.Dept_Code AND rp.Dept_Name = d.Dept_Name
SET rp.departement_id = d.departement_id;

SET SQL_SAFE_UPDATES = 1;

-- Step 9: Verify the continent_id column is populated
SELECT * 
FROM wine_schema.regions_prod
LIMIT 200;

-- Optional: Check for any rows where the continent_id is still NULL
SELECT * 
FROM wine_schema.regions_prod
WHERE departement_id IS NULL;

-- lets drop the original column now
alter table wine_schema.regions_prod drop column Dept_Name;
alter table wine_schema.regions_prod drop column Dept_Code; 
-- check everything is as expected
select * from wine_schema.regions_prod limit 10;


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
    AND REFERENCED_TABLE_NAME = 'departement';
