with source as (
    select
        cast("REG_ID" as varchar) as reg_id,
        cast("Region" as varchar) as region,
        cast("VAR"    as varchar) as var,
        cast("POS"    as varchar) as pos,
        cast("TL"     as varchar) as tl,
        cast("TIME"   as integer) as year,
        cast("Value"  as double)  as value
    from {{ ref('raw_social') }}
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
    var as indicator,
    year,
    value,
    'social' as source
from filtered