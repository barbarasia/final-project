-- Average yearly wine consumption per continent
SELECT 
    cont.Continent, 
    AVG(cw.Quantity) AS Avg_Consumption
FROM 
    wine_schema.consowine AS cw
JOIN 
    wine_schema.continent AS cont
ON 
    cw.continent_id = cont.continent_id
WHERE 
    cw.Variable = 'Consumption'
GROUP BY 
    cont.Continent
ORDER BY 
    Avg_Consumption DESC;

-- Wine production vs consumption by country for 2020 top 10
SELECT 
    prod_c.Country AS Country, 
    prod.Quantity AS Production_Quantity, 
    cons.Quantity AS Consumption_Quantity
FROM 
    (SELECT 
         country_id, 
         Quantity
     FROM 
         wine_schema.consowine
     WHERE 
         Variable = 'Production'
         AND Year = 2020) AS prod
JOIN 
    wine_schema.country AS prod_c
ON 
    prod.country_id = prod_c.country_id
JOIN 
    (SELECT 
         country_id, 
         Quantity
     FROM 
         wine_schema.consowine
     WHERE 
         Variable = 'Consumption'
         AND Year = 2020) AS cons
ON 
    prod.country_id = cons.country_id
JOIN 
    wine_schema.country AS cons_c
ON 
    cons.country_id = cons_c.country_id
ORDER BY 
    prod.Quantity DESC
    limit 10;
    
    
-- Top alcohol type by spending
SELECT 
    alcohol_type AS Alcohol_Type,
    SUM(Value) AS Total_Spending
FROM 
    wine_schema.alcohol_spending
GROUP BY 
    alcohol_type
ORDER BY 
    Total_Spending DESC
LIMIT 10; 
    


-- Calculate the total production of wine for each country over the last 20 years and show the top 10
SELECT
    c.country_id,
    ct.Country,
    SUM(c.Production) AS Total_Production_Last_20_Years
FROM
    (SELECT country_id, Year, Quantity AS Production 
     FROM wine_schema.consowine 
     WHERE Variable = 'Production' AND Year >= YEAR(CURDATE()) - 20) AS c
JOIN
    wine_schema.country AS ct ON c.country_id = ct.country_id
GROUP BY
    c.country_id, ct.Country
ORDER BY
    Total_Production_Last_20_Years DESC
LIMIT 10;
    
    
-- Find the most frequent occurrences in the Category_5 column from API
SELECT
    Category_5,
    COUNT(*) AS Occurrence
FROM
    wine_schema.cleaned_api
GROUP BY
    Category_5
ORDER BY
    Occurrence DESC
LIMIT 10; 
    
    