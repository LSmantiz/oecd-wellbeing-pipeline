select *, current_timestamp as ingested_at
from read_csv_auto('{{ var("raw_data_dir") }}/REGION_SOCIAL-2016-1-EN-20161128T112210.csv', header = true)