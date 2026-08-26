with source as (
    select
        cast("REG_ID" as varchar) as reg_id,
        cast("Region" as varchar) as region,
        cast("VAR"    as varchar) as var,
        cast("MEAS"   as varchar) as meas,
        cast("SERIES" as varchar) as series,
        cast("POS"    as varchar) as pos,
        cast("TL"     as varchar) as tl,
        cast("TIME"   as integer) as year,
        cast("Value"  as double)  as value
    from {{ ref('raw_economic') }}
),

filtered as (
    select *
    from source
    where pos = 'ALL'
      and left(tl, 1) != '1'
      and series = 'SNA_2008'
      and meas in (select meas from {{ ref('economic_measures') }})
)

select
    reg_id,
    region,
    var || '_' || meas as indicator,
    year,
    value,
    'economic' as source
from filtered