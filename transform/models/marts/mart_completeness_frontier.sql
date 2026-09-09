{{ config(materialized='table') }}

/*
  The completeness frontier.

  The analysis dataset requires every indicator to be observed for every region,
  so a single region missing a single indicator removes that indicator for
  everyone. Dropping the worst-covered regions therefore buys back indicators.

  This model sweeps that trade-off: for each number of regions dropped, how many
  regions and indicators remain. Regions are dropped worst-first, by how many
  indicators they lack.

  Equivalent to build_appendix() in the original pandas pipeline, which wrote one
  dataset per step. Here the sweep is a single queryable table with an n_dropped
  column, so the whole frontier can be inspected at once rather than as 15
  separate files.

  Ties in missingness are broken by reg_id. pandas' drop_duplicates(keep="first")
  relied on row order, which is not reproducible; an explicit tiebreak is.
*/

with base as (
    select * from {{ ref('mart_indicators_resolved') }}
),

totals as (
    select
        count(distinct indicator) as n_indicators,
        count(distinct reg_id)    as n_regions
    from base
),

region_missing as (
    select
        b.reg_id,
        (select n_indicators from totals) - count(distinct b.indicator) as n_missing
    from base b
    group by b.reg_id
),

ranked as (
    select
        reg_id,
        n_missing,
        row_number() over (order by n_missing desc, reg_id) as drop_rank
    from region_missing
),

steps as (
    select unnest(generate_series(0, {{ var('frontier_max_dropped') }},
                                  {{ var('frontier_step') }})) as n_dropped
),

kept as (
    select s.n_dropped, r.reg_id
    from steps s
    join ranked r
      on r.drop_rank > s.n_dropped
),

region_counts as (
    select n_dropped, count(*) as n_regions
    from kept
    group by n_dropped
),

indicator_coverage as (
    select
        k.n_dropped,
        b.indicator,
        count(distinct b.reg_id) as n_covered
    from kept k
    join base b on b.reg_id = k.reg_id
    group by k.n_dropped, b.indicator
),

complete_counts as (
    select
        c.n_dropped,
        count(*) as n_indicators
    from indicator_coverage c
    join region_counts rc on rc.n_dropped = c.n_dropped
    where c.n_covered = rc.n_regions
    group by c.n_dropped
),

still_missing as (
    select
        k.n_dropped,
        count(distinct r.reg_id) as n_incomplete_regions
    from kept k
    join ranked r on r.reg_id = k.reg_id
    where r.n_missing > 0
    group by k.n_dropped
)

select
    rc.n_dropped,
    rc.n_regions,
    coalesce(cc.n_indicators, 0) as n_indicators,
    coalesce(sm.n_incomplete_regions, 0) as n_incomplete_regions,
    rc.n_regions * coalesce(cc.n_indicators, 0) as n_cells
from region_counts rc
left join complete_counts cc on cc.n_dropped = rc.n_dropped
left join still_missing sm on sm.n_dropped = rc.n_dropped
order by rc.n_dropped