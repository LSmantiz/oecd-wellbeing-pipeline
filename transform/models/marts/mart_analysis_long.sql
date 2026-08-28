with resolved as (
    select * from {{ ref('mart_indicators_resolved') }}
),

region_count as (
    select count(distinct reg_id) as n_regions from resolved
),

complete_indicators as (
    select indicator
    from resolved
    group by indicator
    having count(distinct reg_id) = (select n_regions from region_count)
)

select r.reg_id, r.indicator, r.year, r.source, r.value
from resolved r
join complete_indicators c using (indicator)