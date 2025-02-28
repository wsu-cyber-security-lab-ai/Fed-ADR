from enum import Enum

class AggregationTypes(Enum):
    MudHog = "mudhog"
    FedAvg = "fedavg"
    Median = "median"
    GM = "gm"
    Trimmed = "trimmed"
    Krum = "krum"
    MKrum = "mkrum"
    FoolsGold = "foolsgold"
