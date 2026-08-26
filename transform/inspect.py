import duckdb

path = "../data/original_data/oecd_regional_2016/RWB-2016-1-EN-20161128T112241.csv"
path = "../data/original_data/oecd_regional_2016/REGION_DEMOGR-2016-1-EN-20161128T111326.csv"

path = "../data/original_data/oecd_regional_2016/REGION_LABOUR-2016-1-EN-20161128T112125.csv"
path = "../data/original_data/oecd_regional_2016/REGION_ECONOM-2016-1-EN-20161128T111738.csv"
path = "../data/original_data/oecd_regional_2016/REGION_SOCIAL-2016-1-EN-20161128T112210.csv"


path = "../data/original_data/oecd_regional_2016/REGION_INNOVATION-2016-1-EN-20161128T112220.csv"



print(duckdb.sql(f"describe select * from read_csv_auto('{path}')"))
print(duckdb.sql(f"select * from read_csv_auto('{path}') limit 5"))
print(duckdb.sql(f"""
    select distinct "Indicator", "MEAS"
    from read_csv_auto('{path}')
    order by "Indicator", "MEAS"
"""))