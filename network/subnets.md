# Task 5 — Subnet Allocation (10.20.0.0/16)

## Method

For each subnet, the smallest prefix that fits is the smallest `/n` such that
`2^(32-n) - 2 >= hosts_needed` (the `-2` removes the network and broadcast
addresses). Subnets are allocated **largest requirement first**, each one
starting right after the previous one ends, beginning at `10.20.0.0`.

| Requirement | Hosts needed | Smallest block that fits | Prefix |
|---|---|---|---|
| app | 1,000 | 1,024 addresses (2^10) | /22 |
| public | 250 | 256 addresses (2^8) | /24 |
| data | 100 | 128 addresses (2^7) | /25 |
| management | 20 | 32 addresses (2^5) | /27 |

## Allocation, in order

1. **app** — needs 1,000 hosts → 2^10 − 2 = 1022 usable → `/22`
   Starts at `10.20.0.0`.
2. **public** — needs 250 hosts → 2^8 − 2 = 254 usable → `/24`
   Starts right after app's block ends (`10.20.3.255`), i.e. at `10.20.4.0`.
3. **data** — needs 100 hosts → 2^7 − 2 = 126 usable → `/25`
   Starts right after public's block ends (`10.20.4.255`), i.e. at `10.20.5.0`.
4. **management** — needs 20 hosts → 2^5 − 2 = 30 usable → `/27`
   Starts right after data's block ends (`10.20.5.127`), i.e. at `10.20.5.128`.

## Final table

| Subnet | Network address | Prefix | Usable host range | Broadcast | Usable hosts |
|---|---|---|---|---|---|
| app | 10.20.0.0 | /22 | 10.20.0.1 – 10.20.3.254 | 10.20.3.255 | 1,022 |
| public | 10.20.4.0 | /24 | 10.20.4.1 – 10.20.4.254 | 10.20.4.255 | 254 |
| data | 10.20.5.0 | /25 | 10.20.5.1 – 10.20.5.126 | 10.20.5.127 | 126 |
| management | 10.20.5.128 | /27 | 10.20.5.129 – 10.20.5.158 | 10.20.5.159 | 30 |

All four blocks fit inside `10.20.0.0/16` and do not overlap; the next free
block after management is `10.20.5.160/27` onward, unused.

## Why a stateless NACL needs separate rules for replies

A NACL has no connection table: it evaluates every single packet in
isolation, in both directions, against its rule list. It does not remember
that an inbound SYN it allowed should let the matching reply back out (or
vice versa). So a reply — a packet travelling in the *opposite* direction of
the request that caused it — needs its **own, explicit** rule, matching the
reply's own direction, source/destination, and port range (here, the
ephemeral port range 1024–65535 on the client/caller side). Without that
rule, the request would be allowed in but its reply would be silently
dropped on the way out (or the reverse), and the connection would simply
hang.

## Why rule order matters for the 198.51.100.0/24 block

Evaluation stops at the **first matching rule**, lowest rule number first. If
the "allow TCP 80/443 from anywhere" rule had a lower number than the
"deny 198.51.100.0/24" rule, a packet from that abusive range on port 443
would match the broad allow rule first and never reach the deny rule — the
block would have no effect. Giving the deny rule the lowest rule number (100)
guarantees it is checked, and therefore applies, before any of the broader
allow rules below it.
