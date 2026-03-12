---
data_stage: source_data
product: irregular_observations
version: 0.1.0
status: draft
source: adr-1.md
---

## 1. Introduction

This document specifies requirements for irregular observations datasets stored in Zarr for ML workflows.
The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in RFC 2119.

## 2. Scope

This specification applies to tabular, irregular observations where the number of observations varies by time window.
This specification does not define gridded array storage. Combining gridded arrays and observations is out of scope for the
storage format and is handled by the data loader at training time.

## 3. Store-Level Requirements

### 3.1 General Architecture

- The observations archive MUST be stored as a Zarr store.
> From ADR-1:
>
> “Each dataset will contain only one observation type.”
>
> “Similar observation types should be combined into a single dataset, padding with NaNs if needed, as long as the padding remains small.”
>
> These statements are ambiguous together: one implies strict single-type-per-dataset, while the other implies multiple similar types may coexist in one dataset.
>
> *TODO: clarify this ambiguity in the ADR and update the spec accordingly.*
- The store MUST represent exactly one observation type.
- Implementations SHOULD keep the total number of observation stores small by combining similar observation types when practical.
- When combining similar observation types, missing feature values MAY be represented with `NaN` padding.
- Runtime transformations (for example thinning or sub-area extraction) SHOULD be applied at dataset-open time rather than requiring store recreation.

### 3.2 Required Root Objects

- The Zarr root MUST contain a 2D array named `data`.
- The Zarr root MUST contain a 2D array named `index`.
- The Zarr root SHOULD contain a `metadata` group for statistics and descriptive metadata.

## 4. `data` Array Requirements

### 4.1 Tabular Data Structure Model

- The `data` array MUST be interpreted as a 2D tabular data structure.
- Axis 0 MUST represent rows (one logical observation record per row).
- Axis 1 MUST represent columns (observation attributes).
- The first four columns (in this exact order) MUST be: `date`, `time`, `latitude`, `longitude`.
- Additional columns MAY appear after the four core columns and are observation-type specific.
> Comment:
>
> “It is currently unclear where column names are stored. In particular, this specification does not yet define whether a dedicated 1D array of column names is REQUIRED/OPTIONAL, or what that array SHOULD be called.”
>
> *TODO: define column-name storage (including object name, dtype/encoding, and ordering constraints).*

### 4.2 Value Conventions

- `longitude` values MUST be normalized to the range `[0, 360)`.
- Observation rows MUST be sorted in lexicographic order by (`date`, `time`, `latitude`, `longitude`).
- Observation date-times MUST be rounded to the nearest second before storage.

### 4.3 Date/Time Encoding

- All data values (including therefore `date` and `time`) MUST be stored as `float32` and so only the integer part of `date` and `time` values is used (see rounding to seconds above); any fractional part MUST be ignored by readers.
- `date` and `time` MUST be stored as separate columns in the `data` array.
- `date` MUST be encoded as days since Unix epoch.
- `time` MUST be encoded as seconds within day.

### 4.4 Chunking and I/O

- Chunking MUST be selected primarily for I/O throughput rather than fixed sample geometry.
- Chunking MUST be applied only along rows (axis 0).
- Columns (axis 1) MUST NOT be split across multiple chunks.
- Chunk sizes SHOULD target approximately 64 MB to 256 MB per chunk (or the platform’s optimal block-size equivalent).
- Implementations SHOULD use chunk caching to reduce repeated reads.

## 5. `index` Array Requirements

### 5.1 Tabular Data Structure Model

- The `index` array MUST be interpreted as a 2D tabular data structure.
- Axis 0 MUST represent rows (one logical index entry per row).
- Axis 1 MUST represent columns (index attributes).
- The three columns (in this exact order) MUST be: `epoch`, `start`, `length`.
- `epoch` MUST be Unix epoch seconds.
- The index table MUST be created for one specific fixed time resolution (for example, `1h`).
- The `epoch` column MUST be regularly spaced at that exact resolution.
- The `epoch` sequence MUST cover the full observation-time span of the dataset.
- For each row `i`, `start` and `length` MUST describe observations in the half-open interval `[epoch_i, epoch_{i+1})`.
- `start` MUST be the row index in `data` of the first observation whose timestamp is within `[epoch_i, epoch_{i+1})`.
- `length` MUST be the number of rows in `data` whose timestamps are within `[epoch_i, epoch_{i+1})`.
- If no observations fall in `[epoch_i, epoch_{i+1})`, `length` MUST be `0`.
> Comment:
>
> “The chunking policy for the `index` array is currently unclear. It is not yet specified whether `index` SHOULD be chunked, MAY be unchunked, or MUST follow the same row-only chunking rule as `data`.”
>
> *TODO: clarify `index` chunking requirements and update this specification.*

## 7. Metadata and Provenance Requirements

- The store MUST include metadata required to reproduce training/inference data source selection.
- Provenance metadata MUST be propagated into checkpoint metadata for downstream inference.
- The `metadata` group SHOULD include summary statistics needed for training-time normalization or quality checks.

## 8. Interoperability and Evolution

- Readers MUST ignore unknown optional metadata keys.
- Producers SHOULD version the format and record the version in root or `metadata` attributes.

## 9. Non-Normative Notes

- Empirical index performance is implementation-dependent and not normative for conformance.
