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
),

preferred_level as (
    select *
    from filtered
    qualify row_number() over (
        partition by reg_id, var, year
        order by case when tl = '2' then 1 else 2 end, tl
    ) = 1
)

select
    reg_id,
    region,
    var || '_' || sex as indicator,
    year,
    value,
    'labour' as source
from preferred_level