select *, current_timestamp as ingested_at
from read_csv_auto('{{ var("raw_data_dir") }}/REGION_LABOUR-2016-1-EN-20161128T112125.csv', header = true)