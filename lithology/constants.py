"""Project-wide constants.

Centralized here so every module (parser, alignment, quality report,
dataset export) agrees on the same sentinel values.
"""

# LAS NULL sentinel as documented in the well-log spec. The actual value is
# still re-read from each LAS file's own ``~Well Information / NULL`` line
# (see io.las_parser) because not every vendor uses -9999; this is only the
# fallback when a file omits the NULL line.
DEFAULT_LAS_NULL_VALUE = -9999.0

# Curves every downstream component is normalized to internally.
CANONICAL_CURVES = ("DEPT", "GK", "KS", "PS")

# Label used for "no valid ground truth here" positions (missing interval
# coverage, or a lithology point that is not core/lab verified and therefore
# must not be used to supervise the lithology head). Follows the common
# PyTorch convention so it can be passed directly as
# ``nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)``.
IGNORE_INDEX = -100

# Sentinel class id meaning "no boundary" is never used; boundary is a
# binary head (0/1) with IGNORE_INDEX for points with no coverage at all.
BOUNDARY_NEGATIVE = 0
BOUNDARY_POSITIVE = 1

# Interval semantics: whether an interval [top, bottom] includes its bottom
# depth. Configurable, see config.AlignmentConfig.interval_semantics.
INTERVAL_CLOSED = "closed"        # [top, bottom]
INTERVAL_HALF_OPEN = "half_open"  # [top, bottom)
