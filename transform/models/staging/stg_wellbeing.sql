with source as (
    select
        cast("REG_ID"  as varchar) as reg_id,
        cast("Regions" as varchar) as region,
        cast("IND"     as varchar) as indicator,
        cast("MEAS"    as varchar) as measure,
        cast("TIME"    as integer) as year,
        cast("Value"   as double)  as value
    from {{ ref('raw_wellbeing') }}
),

filtered as (
    select *
    from source
    where reg_id not in (select reg_id from {{ ref('excluded_reg_ids') }})
      and indicator not in (select indicator from {{ ref('dropped_indicators') }})
      and measure = 'VALUE'
),

latest as (
    select *
    from filtered
    qualify row_number() over (
        partition by reg_id, indicator, year
        order by value
    ) = 1
)

select reg_id, region, indicator, year, value, 'wellbeing' as source
from latest