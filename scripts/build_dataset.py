"""Aggregate the raw 2022 Citi Bike trip files into the small tables the notebook uses.

Raw data: https://s3.amazonaws.com/tripdata/index.html (2022-citibike-tripdata.zip, ~6 GB unzipped).
Usage:  uv run python scripts/build_dataset.py /path/to/2022-citibike-tripdata
"""
import json
import sys
from pathlib import Path

import duckdb

RAW = Path(sys.argv[1]).expanduser()
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# One folder per month; skip duplicate copies such as "202202-citibike-tripdata 2".
files = sorted(
    str(f) for d in RAW.glob("2022??-citibike-tripdata") if d.is_dir() for f in d.glob("*.csv")
)
print(f"{len(files)} files")

con = duckdb.connect()
con.execute(f"""
CREATE TABLE raw AS
SELECT *, epoch(ended_at - started_at) AS duration_s
FROM read_csv({files}, header = true, all_varchar = false,
              types = {{'start_station_id': 'VARCHAR', 'end_station_id': 'VARCHAR'}})
WHERE started_at >= '2022-01-01' AND started_at < '2023-01-01'
""")

total = con.execute("SELECT count(*) FROM raw").fetchone()[0]
log = {"raw_trips": total}
for name, cond in [
    ("negative_duration", "duration_s < 0"),
    ("under_60s", "duration_s BETWEEN 0 AND 59.999"),
    ("over_24h", "duration_s > 86400"),
    ("missing_station", "duration_s BETWEEN 60 AND 86400 AND (start_station_id IS NULL OR end_station_id IS NULL)"),
]:
    log[name] = con.execute(f"SELECT count(*) FROM raw WHERE {cond}").fetchone()[0]

con.execute("""
CREATE TABLE trips AS SELECT * FROM raw
WHERE duration_s BETWEEN 60 AND 86400
  AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
""")
log["clean_trips"] = con.execute("SELECT count(*) FROM trips").fetchone()[0]
q = con.execute("SELECT quantile_cont(duration_s, [0.01,0.05,0.25,0.5,0.75,0.95,0.99]) FROM raw WHERE duration_s >= 0").fetchone()[0]
log["duration_percentiles_s"] = dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], [round(x, 1) for x in q]))
(OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2))
print(json.dumps(log, indent=2))

con.execute(f"""
COPY (
  SELECT CAST(started_at AS DATE) AS date,
         count(*) AS trips,
         count(*) FILTER (member_casual = 'member') AS member_trips,
         count(*) FILTER (member_casual = 'casual') AS casual_trips,
         count(*) FILTER (rideable_type = 'electric_bike') AS electric_trips,
         round(median(duration_s) / 60, 2) AS median_duration_min
  FROM trips GROUP BY 1 ORDER BY 1
) TO '{OUT}/daily_trips.csv' (HEADER)
""")

con.execute(f"""
COPY (
  SELECT hour(started_at) AS hour,
         dayofweek(started_at) IN (0, 6) AS is_weekend,
         member_casual,
         count(*) AS trips
  FROM trips GROUP BY ALL ORDER BY ALL
) TO '{OUT}/hourly_trips.csv' (HEADER)
""")

con.execute(f"""
COPY (
  WITH starts AS (
    SELECT start_station_name AS name, median(start_lat) AS latitude, median(start_lng) AS longitude, count(*) AS starts
    FROM trips GROUP BY 1
  ), ends AS (
    SELECT end_station_name AS name, count(*) AS ends FROM trips GROUP BY 1
  )
  SELECT s.name, s.latitude, s.longitude, s.starts, coalesce(e.ends, 0) AS ends
  FROM starts s LEFT JOIN ends e USING (name)
  WHERE s.latitude BETWEEN 40.4 AND 41.0 AND s.longitude BETWEEN -74.3 AND -73.6
  ORDER BY s.starts DESC
) TO '{OUT}/stations.csv' (HEADER)
""")

# Origin-destination pairs (round trips excluded); pairs under 5 trips dropped to keep the file small.
con.execute(f"""
COPY (
  SELECT start_station_name AS origin, end_station_name AS destination, count(*) AS trips
  FROM trips WHERE start_station_name <> end_station_name
  GROUP BY 1, 2 HAVING count(*) >= 5 ORDER BY trips DESC
) TO '{OUT}/od_pairs.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
print("done")
