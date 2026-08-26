with source as (
    select
        cast("REG_ID" as varchar) as reg_id,
        cast("Region" as varchar) as region,
        cast("VAR"    as varchar) as var,
        cast("SEX"    as varchar) as sex,
        cast("POS"    as varchar) as pos,
        cast("TL"     as varchar) as tl,
        cast("TIME"   as integer) as year,
        cast("Value"  as double)  as value
    from {{ ref('raw_labour') }}
),

filtered as (
    select *
    from source
    where pos = 'ALL'
      and left(tl, 1) != '1'
)


select
    reg_id,
    region,
    var || '_' || sex as indicator,
    year,
    value,
    'labour' as source
from filtered