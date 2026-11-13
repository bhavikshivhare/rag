# Onboarding and Migration

## Standard onboarding

A Growth customer typically completes standard onboarding in 2 weeks. The process covers workspace configuration, connector setup, first dashboard, and a team handover session.

Enterprise onboarding typically takes 4 to 6 weeks and includes architecture review, security review, and a staged rollout plan.

## Migration

Historical data can be imported through CSV upload or the bulk import API. A single CSV import file supports up to 1,000,000 rows. Larger datasets are split into multiple files or streamed through the bulk import job API.

Northwind supports backfilling up to 24 months of historical data for supported sources.

## Validation

After an import, Northwind produces a data validation report within 3 business days. The report compares row counts, schema mapping, and rejected records.

## Training

Enterprise contracts include 2 live training sessions for up to 15 participants each, plus access to recorded training material. Additional sessions can be purchased as professional services.

## Dedicated resources

Enterprise customers receive a designated solutions engineer during onboarding and for the first 90 days after go-live.

## Cutover

Northwind recommends running the legacy system in parallel for at least 2 weeks before retiring it. There is no automatic cutover; the customer decides the go-live date.
