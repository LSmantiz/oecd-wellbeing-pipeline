select
    reg_id,
    indicator,
    year,
    source,
    (value - avg(value) over (partition by indicator))
        / stddev_samp(value) over (partition by indicator) as value
from {{ ref('mart_analysis_long') }}