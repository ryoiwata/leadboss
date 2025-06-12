# LeadBoss

## Low-latency Event Acquisition Data-lake for Buffer-Online Spike Sorter

A reproducible, laptop-scale pipeline that pulls open-access electrophysiology, runs a software clone of Neuralink’s BOSS sorter, and lands compressed spike events in a query-ready lakehouse.

## Purpose
Neural-data platforms (e.g. Neuralink’s Neuralake) must:

1. Ingest 30 kHz extracellular streams from hundreds of channels.
2. Sort spikes on the fly under millisecond latency / <200 mW.
3. Serve sub-second queries & dashboards to scientists and firmware teams.