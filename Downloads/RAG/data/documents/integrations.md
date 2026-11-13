# Integrations and Connectors

## Native connectors

Northwind Analytics provides more than 40 native connectors. Core data warehouse connectors include Snowflake, BigQuery, Redshift, and PostgreSQL.

Business application connectors include Salesforce, HubSpot, and Shopify. A Zapier connector is available in beta and is not covered by the uptime commitment.

## Warehouse sync schedule

Warehouse connectors run incremental syncs every 15 minutes. A full refresh runs nightly. Customers can trigger a manual sync from the connector settings page, subject to a limit of 4 manual syncs per hour.

## Webhooks

Outbound webhooks are delivered with at least once semantics. Failed deliveries are retried 5 times using exponential backoff over a period of up to 30 minutes. After the final retry the event is marked as failed and can be replayed manually for 7 days.

## Custom sources

Customers with unsupported sources can use the generic REST connector or the SDK. Custom connector development is a professional services engagement and is billed separately.

## Connector permissions

Connectors use scoped credentials. Northwind recommends a read-only service account for warehouse connectors. Credentials are stored encrypted and are never displayed after they are saved.
