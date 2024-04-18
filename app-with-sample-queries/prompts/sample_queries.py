SAMPLE_QUERY_ONE = """
Show me the total number of accounts and total deposit amount by cost center in the descending order of deposit amounts.  
Show only 10 cost centers. Make sure to format the amounts and counts properly.

Your response should be


SELECT TOP 10
    a.COST_CENTR_ID AS Cost_Center,
    COUNT(DISTINCT a.ACCT_ID) AS Total_Accounts,
    FORMAT(SUM(d.DP_CUR_BAL), 'C', 'en-US') AS Total_Deposit_Amount
FROM 
    trg.ACCOUNT a
LEFT JOIN 
    trg.DEPOSIT d ON a.ACCT_ID = d.ACCT_ID
GROUP BY 
    a.COST_CENTR_ID
ORDER BY 
    SUM(d.DP_CUR_BAL) DESC;

"""

EXPECTED_OUTPUT_ONE = """
Cost_Center
 
	
Total_Accounts
 
	
Total_Deposit_Amount
 
 
1139	1326	$54,612,879,322.10
4211	1069	$10,837,259,600.71
2130	12124	$8,550,085,336.73
2131	10583	$8,380,054,765.56
3293	5	$7,277,773,132.41
2190	557	$7,261,036,949.67
4450	42	$6,969,893,826.84
4451	1	$6,929,702,436.00
2134	6931	$6,651,075,176.97
4706	859	$4,416,045,505.62
"""

SAMPLE_QUERY_TWO = """
Show me the number of loan accounts and total loan amount by cost center. 
Show me the top 20 cost centers with the highest loan amounts. 
Order the output by descending loan amount

Your response should be


SELECT TOP 20
    a.COST_CENTR_ID AS Cost_Center,
    COUNT(DISTINCT l.ACCT_ID) AS Number_of_Loan_Accounts,
    FORMAT(SUM(l.LN_ACCT_TOTAL_OWE), 'C', 'en-US') AS Total_Loan_Amount
FROM 
    trg.ACCOUNT a
JOIN 
    trg.LOAN l ON a.ACCT_ID = l.ACCT_ID
GROUP BY 
    a.COST_CENTR_ID
ORDER BY 
    SUM(l.LN_ACCT_TOTAL_OWE) DESC;
"""

EXPECTED_OUTPUT_TWO = """
Cost_Center
 
	
Number_of_Loan_Accounts
 
	
Total_Loan_Amount
 
 
6313	1477540	$467,294,006,549.87
6117	113202	$27,058,895,280.00
6318	18195	$7,512,108,079.40
6509	11340	$3,990,665,219.00
6513	4999	$3,026,231,191.84
3134	30	$1,415,143,952.80
4206	67	$1,140,962,275.44
6510	4075	$1,046,732,037.00
6511	2676	$941,331,454.00
4707	30	$209,000,000.00
4709	15	$136,111,111.08
6508	466	$109,794,782.00
4500	22	$72,334,000.00
1135	676	$65,066,912.31
4710	13	$31,500,000.00
0	24	$26,126,237.00
2330	1226	$16,768,911.69
4600	95	$16,602,957.37
2131	599	$13,998,292.73
3232	889	$13,062,993.11
"""

SAMPLE_QUERY_THREE = """
Show me 20 distinct customers who have the highest loan amounts or deposit amounts.

Your response should be


WITH CustomerBalances AS (
    SELECT
        p.PRTY_ID AS Customer_ID,
        p.PRTY_NM AS Customer_Name,
        COALESCE(SUM(l.LN_ACCT_TOTAL_OWE), 0) AS Total_Loan_Amount,
        COALESCE(SUM(d.DP_CUR_BAL), 0) AS Total_Deposit_Amount
    FROM 
        trg.PARTY p
    LEFT JOIN 
        trg.PARTY_ACCOUNT pa ON p.PRTY_ID = pa.PRTY_ID
    LEFT JOIN 
        trg.LOAN l ON pa.ACCT_ID = l.ACCT_ID
    LEFT JOIN 
        trg.DEPOSIT d ON pa.ACCT_ID = d.ACCT_ID
    GROUP BY 
        p.PRTY_ID, p.PRTY_NM
),
RankedCustomers AS (
    SELECT 
        Customer_ID,
        Customer_Name,
        Total_Loan_Amount,
        Total_Deposit_Amount,
        ROW_NUMBER() OVER (ORDER BY Total_Loan_Amount DESC, Total_Deposit_Amount DESC) AS rn
    FROM 
        CustomerBalances
)
SELECT TOP 20
    Customer_ID,
    Customer_Name,
    FORMAT(Total_Loan_Amount, 'C', 'en-US') AS Formatted_Loan_Amount,
    FORMAT(Total_Deposit_Amount, 'C', 'en-US') AS Formatted_Deposit_Amount
FROM 
    RankedCustomers
WHERE 
    rn <= 20
ORDER BY 
    rn;
"""

EXPECTED_OUTPUT_THREE = """
Customer_ID
 
	
Customer_Name
 
	
Formatted_Loan_Amount
 
	
Formatted_Deposit_Amount
 
 
278222	ERICKA LEMASTER	$5,919,727,083.45	$132,016,462.97
277593	JOHN ELDEAN	$3,792,561,300.73	$189,506,451.38
278763	MONIKA SUAREZ	$2,449,267,782.62	$119,032,134.75
278264	JOSH ORMISTON	$2,392,322,684.75	$592,987.10
278800	MARIA ENGLERT	$2,096,826,066.78	$8,073.88
279222	BARRY LIEBERMAN	$1,929,877,972.71	$52,231,008.67
278777	ASHAN PERERA	$1,492,257,321.39	$49,222,288.43
278267	TRACY GAYLORD	$1,458,998,809.47	$41,747,936.37
278089	BRENT EDGECUMBE	$1,344,235,666.34	$750,202.54
279223	JAMES PETTY	$1,215,282,080.22	$53,623,945.70
279224	CHRIS WILLIAMS	$1,190,859,598.51	$21,133,291.97
278332	LUCY RAY	$1,014,442,389.75	$32,722,139.01
278266	THOMAS PERROTT	$818,427,457.82	$5,740,292.84
277908	TYLER PETERSON	$721,670,284.27	$0.00
278164	MARK ROBERTS	$700,179,947.19	$3,503,432,140.28
277750	TOMAS FACH	$690,418,606.93	$55,656,961.19
278313	DAVID BERNARD	$685,437,607.95	$3,009,527,687.31
278673	LISA ALBERTI	$674,832,647.53	$35,236,679.52
278670	JEFFREY FORSYTHE	$653,604,524.85	$9,559,943.81
277687	ALAN POTHAST	$563,778,986.81	$0.00
"""