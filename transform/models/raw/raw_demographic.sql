select *, current_timestamp as ingested_at
from read_csv_auto('{{ var("raw_data_dir") }}/REGION_DEMOGR-2016-1-EN-20161128T111326.csv', header = true)