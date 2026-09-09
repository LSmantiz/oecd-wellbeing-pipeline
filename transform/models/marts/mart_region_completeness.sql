{{ config(materialized='table') }}

/*
  How many indicators each region is missing, and the order in which regions
  are dropped when trading regions for indicator coverage.

  Ties are broken by reg_id. The pandas original ranked with sort_values, whose
  tie order depended on row order and was not reproducible.
*/

with base as (
    select * from {{ ref('mart_indicators_resolved') }}
),

totals as (
    select count(distinct indicator) as n_indicators from base
)

select
    b.reg_id,
    count(distinct b.indicator) as n_present,
    (select n_indicators from totals) - count(distinct b.indicator) as n_missing,
    row_number() over (
        order by (select n_indicators from totals) - count(distinct b.indicator) desc,
                 b.reg_id
    ) as drop_rank
from base b
group by b.reg_id