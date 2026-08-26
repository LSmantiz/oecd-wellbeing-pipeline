select *, current_timestamp as ingested_at
from read_csv_auto(
    '{{ var("raw_data_dir") }}/REGION_INNOVATION-2016-1-EN-20161128T112220.csv',
    header = true,
    sample_size = -1,
    types = {'TL': 'VARCHAR'}
)