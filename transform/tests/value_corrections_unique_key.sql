select reg_id, indicator, count(*) as n
from {{ ref('value_corrections') }}
group by reg_id, indicator
having count(*) > 1