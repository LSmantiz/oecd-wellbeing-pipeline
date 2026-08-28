with anchor_regions as (
    select reg_id
    from {{ ref('mart_indicators_latest') }}
    where source = 'wellbeing'
    group by reg_id
    having count(distinct indicator) = (
        select count(distinct indicator)
        from {{ ref('mart_indicators_latest') }}
        where source = 'wellbeing'
    )
),

restricted as (
    select l.*
    from {{ ref('mart_indicators_latest') }} l
    join anchor_regions a using (reg_id)
),

resolved as (
    select r.*
    from restricted r
    left join {{ ref('indicator_precedence') }} p using (indicator)
    where p.indicator is null or r.source = p.winning_source
),

corrected as (
    select
        r.reg_id, r.indicator, r.year, r.source,
        r.value * coalesce(c.multiplier, 1) as value
    from resolved r
    left join {{ ref('value_corrections') }} c
      on r.reg_id = c.reg_id and r.indicator = c.indicator
)

select * from corrected