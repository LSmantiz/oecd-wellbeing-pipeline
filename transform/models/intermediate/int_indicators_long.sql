select reg_id, region, indicator, year, value, source from {{ ref('stg_wellbeing') }}
union all
select reg_id, region, indicator, year, value, source from {{ ref('stg_demographic') }}
union all
select reg_id, region, indicator, year, value, source from {{ ref('stg_labour') }}
union all
select reg_id, region, indicator, year, value, source from {{ ref('stg_innovation') }}
union all
select reg_id, region, indicator, year, value, source from {{ ref('stg_social') }}
union all
select reg_id, region, indicator, year, value, source from {{ ref('stg_economic') }}