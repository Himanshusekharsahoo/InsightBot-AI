# Internal Network Mesh & Ingress Routing Architecture

## Ingress Edge Gateway
External ingress traffic reaches the public load balancers on port 443 with TLS 1.3 termination. The edge gateway validates mutual TLS client certificates before routing packets into the private virtual network.

## Service Mesh & Private Worker Pods
All ingress traffic is forwarded from the edge gateways into the private worker cluster hosted on subnet 10.240.0.0/16. Ingress proxies dispatch requests directly to internal worker execution pods listening on port 8443 using mTLS backplane communication. Worker pods never expose public IP addresses.
