# Datetimes are naive-UTC everywhere

## TL;DR:

> All datetimes in anemoi-datasets are naive and mean UTC.
>
> Any timezone-aware input is converted to UTC through its own offset and stripped to naive as early as possible.

## Context

anemoi-datasets manipulates datetimes at every stage: recipe dates entered by users, valid/base datetimes attached to data as it is ingested, the date axis stored on disk, and the dates used to index a dataset when it is read back. These datetimes arrive in inconsistent forms — some are naive (`tzinfo is None`), some carry an explicit UTC marker (`+00:00` / `Z`), and some carry a non-UTC offset (`+02:00`).

Three distinct forms must be distinguished:

- **naive** — no offset attached. Its meaning depends entirely on a convention.
- **UTC (tz-aware)** — offset `+00:00`. Same instant as the naive form under a UTC convention.
- **non-UTC (tz-aware)** — an explicit offset such as `+02:00`. This is the only form that carries
  extra information beyond the wall-clock reading.

> Regarding the separator (space vs T, '2022-01-01 00:00:00' vs '2022-01-01T00:00:00') : T is the ISO 8601 canonical date/time separator; a space is the common relaxed form. In python, datetime.fromisoformat accepts both and produces the identical object. Purely cosmetic — no semantic difference, ever.

Several forces constrain how we may treat them:

1. **Reproducibility.** The same recipe must produce the same dataset on any machine. If a naive datetime were interpreted as *local* time, the output would depend on the machine's `$TZ`, which is unacceptable. Converting a non-UTC datetime to UTC, by contrast, uses only the offset embedded in the value and never the local clock, so it is machine-independent.

2. **The storage and compute layers cannot represent a timezone.** Dates are stored and computed as numpy `datetime64`, which has no timezone concept — it is an integer offset from the Unix epoch.  pytorch, at the model boundary, has no datetime type at all; dates become plain numbers. (Only pandas can *optionally* carry a timezone, and only as opt-in metadata.) Any timezone information would therefore be lost at these boundaries regardless of what we do upstream.

3. **Comparability.** Mixing naive and tz-aware datetimes raises `TypeError` in Python, and two "equal" instants expressed with different offsets do not compare or hash equal in sets. Date-set operations that match one block of dates against another silently fail when the two sides use different representations.




## Decision

**All datetimes in anemoi-datasets are naive and mean UTC. Any timezone-aware input is converted to UTC through its own offset and stripped to naive as early as possible — at the outermost edge where it enters the system.**

Precisely:

- A **naive** datetime is taken to mean UTC and is left unchanged. It is never interpreted as local time.
- A **UTC** datetime has its (redundant) marker removed, leaving the instant unchanged.
- A **non-UTC** datetime is converted to UTC using the offset embedded in the value, then stripped.

Non-UTC datetimes are tolerated only at the input edge and eliminated immediately; nothing downstream ever sees a timezone. This applies uniformly to recipe dates, to datetimes attached to data during ingestion, to the date axis written to disk, and to dates supplied when reading a dataset back.

The two conventions in `anemoi-datasets` in practice:

- **create** — every datetime is normalised to naive-UTC at the point it enters the build: recipe dates as they are parsed, and as data is ingested. The date axis is then written to disk as timezone-less, so by construction the stored values are naive-UTC.

- **usage** — timezone-aware datetime are not supported.

## Consequences

- The same recipe produces the same dataset on any machine, independent of `$TZ`.
- Date-set operations (`concat` / `join` / missing-date matching) compare like with like, so the silent-empty-result class of bug is eliminated.
- Downstream code — storage, arithmetic, and the model boundary — never has to reason about timezones, matching what numpy and pytorch can actually represent.
