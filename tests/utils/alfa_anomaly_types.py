"""ALFA anomaly-type normalization shared by preprocessing and evaluation."""

PAPER_TABLE_VIII_TYPES = (
    "Engine full power loss",
    "Elevator stuck at zero",
    "Rudder stuck to left",
    "Rudder stuck to right",
    "Both aileron stuck at zero",
    "Left aileron stuck at zero",
)

EXTRA_FINE_GRAINED_TYPES = ("Right aileron stuck at zero",)

FINE_GRAINED_ANOMALY_TYPES = PAPER_TABLE_VIII_TYPES + EXTRA_FINE_GRAINED_TYPES


def canonical_anomaly_type(source_file="", flight_id="", fault_type="unknown"):
    """Map ALFA filenames to paper-style fine-grained anomaly labels."""
    source = str(source_file or flight_id or "").lower().replace("-", "_")
    fallback = str(fault_type or "unknown")

    if "engine_failure" in source:
        return "Engine full power loss"
    if "elevator_failure" in source:
        return "Elevator stuck at zero"
    if "rudder_left_failure" in source:
        return "Rudder stuck to left"
    if "rudder_right_failure" in source:
        return "Rudder stuck to right"
    if (
        "both_ailerons_failure" in source
        or "left_aileron__right_aileron__failure" in source
    ):
        return "Both aileron stuck at zero"
    if "left_aileron_failure" in source:
        return "Left aileron stuck at zero"
    if "right_aileron_failure" in source:
        return "Right aileron stuck at zero"
    if "no_failure" in source:
        return "normal"
    return fallback


def canonical_anomaly_type_from_row(row):
    return canonical_anomaly_type(
        source_file=row.get("source_file", ""),
        flight_id=row.get("flight_id", ""),
        fault_type=row.get("fault_type", "unknown"),
    )


def add_fine_anomaly_type(data):
    data = data.copy()
    data["fine_anomaly_type"] = data.apply(canonical_anomaly_type_from_row, axis=1)
    return data


def anomaly_type_sort_key(name):
    order = {value: index for index, value in enumerate(FINE_GRAINED_ANOMALY_TYPES)}
    return order.get(name, len(order)), name
