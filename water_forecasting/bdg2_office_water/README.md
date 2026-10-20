# Water dataset — PUBLIC (Building Data Genome Project 2)

## Source

**Building Data Genome Project 2 (BDG2)** — an open dataset of hourly smart-meter readings
from 1,600+ non-residential buildings across 19 sites, recorded in **2016-2017**.

- GitHub: <https://github.com/buds-lab/building-data-genome-project-2>

Check the repository for the current license/citation terms before redistributing.

## What we use

One building's water meter series: **site `Panther`, building `Larry` (office)**, hourly,
2016-01-01 → 2017-12-31 (17,544 rows). Extracted from BDG2's water-meter file
(`water.csv`, column `Panther_office_Larry`).

## Files expected in this folder

| file | description |
|---|---|
| `Panther_office_Larry_water.csv` | the extracted series (`datetime`, `flow_rate_l_per_h`, `consumption_l`) |
| `water_consumption_office.csv` | working copy produced by `../shift_water_dataset.py` (dates shifted forward to line up with the energy series; **litres**) |

Columns: `datetime` (hourly), `flow_rate_l_per_h` (L/h), `consumption_l` (litres per hour).

## How to rebuild

1. Download `water.csv` from the BDG2 repo link above (`data/meters/cleaned/water_cleaned.csv`
   in the GitHub repo).
2. Keep the `datetime` column and the `Panther_office_Larry` column; rename it to the
   columns above (BDG2 values are litres per hour — the project uses litres everywhere, never m³).
3. Save as `Panther_office_Larry_water.csv` here and run
   `python water_forecasting/shift_water_dataset.py` to create `water_consumption_office.csv`.
