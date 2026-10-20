from pathlib import Path
import pandas as pd
import holidays

input_file = str(Path(__file__).resolve().parent / "bdg2_office_water" / "Panther_office_Larry_water.csv")  # MODIFY if your raw file lives elsewhere

output_file = str(Path(__file__).resolve().parent / "bdg2_office_water" / "water_consumption_office.csv")


# Load
df = pd.read_csv(input_file)

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])


# US federal holidays that fall on a weekday currently read as building-closed
# (near-zero, same as a weekend) because the source meter is a real US office.
# Once shifted, that closure would land on an arbitrary weekday with no holiday
# significance in the new calendar and look like unexplained noise. Fix: before
# shifting, replace each such day's raw values with the nearest earlier (or, at
# the series start, nearest later) non-holiday same-weekday's real readings, so
# the raw series looks like a normal office year with no holiday signal baked
# in. No "is_holiday" feature is added anywhere — this only corrects raw values.
raw_years = range(df["datetime"].min().year, df["datetime"].max().year + 1)
us_holidays = holidays.US(years=list(raw_years))
weekday_holidays = sorted(pd.Timestamp(d) for d in us_holidays.keys() if d.weekday() < 5)
holiday_dates = set(d.date() for d in weekday_holidays)

df["_date"] = df["datetime"].dt.date


def _valid_template(date_):
    day_rows = df[df["_date"] == date_.date()]
    return (date_.date() not in holiday_dates
            and len(day_rows) == 24
            and day_rows["consumption_l"].notna().all())


print(f"\nNormalizing {len(weekday_holidays)} weekday US holidays to look like normal days:")
for hol in weekday_holidays:
    candidate = hol - pd.Timedelta(days=7)
    while candidate >= df["datetime"].min() and not _valid_template(candidate):
        candidate -= pd.Timedelta(days=7)
    if candidate < df["datetime"].min():
        candidate = hol + pd.Timedelta(days=7)
        while candidate <= df["datetime"].max() and not _valid_template(candidate):
            candidate += pd.Timedelta(days=7)
        if candidate > df["datetime"].max():
            raise ValueError(f"Could not find a normal-day template for holiday {hol}")

    template = df[df["_date"] == candidate.date()].copy().sort_values("datetime")
    target_mask = df["_date"] == hol.date()
    df.loc[target_mask, "consumption_l"] = template["consumption_l"].values
    df.loc[target_mask, "flow_rate_l_per_h"] = template["flow_rate_l_per_h"].values
    print(f"  {hol.date()} ({hol.strftime('%A')}, {us_holidays[hol.date()]}) "
          f"<- template {candidate.date()}")

df = df.drop(columns=["_date"])


# The raw dataset length isn't a multiple of 7 days, so shifting it straight to
# target_end rotates the day-of-week (e.g. old Sunday lands on new Monday).
# Fix: pad with one extra day (copied from the previous same-weekday) so the
# shift becomes an exact multiple of 7 days and weekday alignment is preserved.
original_end = df["datetime"].max()
next_day = original_end.normalize() + pd.Timedelta(days=1)
prev_same_weekday = next_day - pd.Timedelta(days=7)

padding_day = df[df["datetime"].dt.date == prev_same_weekday.date()].copy()
padding_day["datetime"] = padding_day["datetime"] + pd.Timedelta(days=7)

df = pd.concat([df, padding_day], ignore_index=True).sort_values("datetime").reset_index(drop=True)


# Calculate shift to end at 31/08/2026 23:00
original_end = df["datetime"].max()
target_end = pd.Timestamp("2026-08-31 23:00:00")

shift = target_end - original_end

print("Original end:", original_end)
print("Shift applied:", shift)
print("Shift mod 7 days (hours, should be 0):", shift.total_seconds() / 3600 % (24 * 7))


# Apply shift
df["datetime"] = df["datetime"] + shift


# Simulate office building behavior
hour = df["datetime"].dt.hour
day_of_week = df["datetime"].dt.dayofweek

working_hour = (
    (day_of_week < 5) &
    (hour >= 8) &
    (hour < 18)
)

# Reduce consumption when building is closed
df.loc[~working_hour, "consumption_l"] *= 0.15
df.loc[~working_hour, "flow_rate_l_per_h"] *= 0.15


# Save with SAME columns
df.to_csv(
    output_file,
    index=False
)


print("Finished!")
print("Rows:", len(df))
print("New start:", df["datetime"].min())
print("New end:", df["datetime"].max())
print(df.head())