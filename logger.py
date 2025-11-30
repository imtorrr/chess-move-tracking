import logging
from rich.logging import RichHandler

# Set the default logging level
# This can be overridden by setting the LOG_LEVEL environment variable
# For example, to see debug messages: export LOG_LEVEL=DEBUG
# Possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "DEBUG" 

# Configure logging with RichHandler
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True,
                           keywords=[
                               "error", "warning", "info", "debug", "critical",
                               "fail", "success", "exception", "traceback"
                           ])]
)

# Get a logger instance
log = logging.getLogger("chess-comp")
