with recent as (
    select *
    from {{ ref('int_indicators_long') }}
    where year >= {{ var('min_year') }}
),

latest as (
    select *
    from recent
    qualify year = max(year) over (partition by reg_id, indicator, source)
)

select * from latest