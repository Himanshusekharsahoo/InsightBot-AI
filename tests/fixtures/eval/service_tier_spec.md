# InsightBot Enterprise Service Tier Specifications

## Reliability & SLA Guarantees
The Enterprise service tier guarantees a 99.99% monthly availability SLA with automated multi-region failover. In the event of service degradation exceeding 0.01% in any billing cycle, affected tenants receive a 25% service credit.

## Hardware Security Modules
Dedicated Hardware Security Modules (HSM) run in FIPS 140-2 Level 3 compliance mode. Tenant encryption keys are generated directly inside physical HSM partitions and are never stored in memory or persistent logs.

## Rate Limits
The rate limit for the Enterprise tier is 50,000 requests per minute per tenant, backed by a distributed token bucket limiter.
