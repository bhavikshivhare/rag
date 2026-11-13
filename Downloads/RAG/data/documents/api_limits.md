# API Rate Limits and Usage Rules

## Request limits

The default rate limit is 600 requests per minute per API key. Growth workspaces receive 1,200 requests per minute. Enterprise workspaces receive a custom limit defined in the contract.

Burst traffic is allowed up to 100 requests per second for short intervals.

## Rate limit errors

Requests that exceed the limit receive HTTP status 429 with a Retry-After header indicating the number of seconds to wait. Clients should implement exponential backoff and honor the Retry-After value.

Every response includes X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset headers.

## Pagination

List endpoints use cursor-based pagination. The maximum page size is 200 records. A default page size of 50 records applies when the page size parameter is omitted.

## Batch endpoints

The batch ingestion endpoint accepts up to 500 records per request. Larger imports must be split into multiple batch calls or submitted through the bulk import job API.

## Payload and timeout limits

The maximum request payload size is 10 MB. Synchronous requests time out after 30 seconds. Long-running exports must use the asynchronous job endpoint.

## API versioning

API v1 is deprecated and will sunset on 2026-12-31. API v2 is the current stable version. The deprecated /v1/search endpoint is removed from v2 and replaced by /v2/query.
