try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed: pip install lerobot==0.7.0") from err

from .configuration_lewam_jepa import LeWAMJEPAConfig
from .modeling_lewam_jepa import LeWAMJEPAPolicy
from .processor_lewam_jepa import make_lewam_jepa_pre_post_processors

__version__ = "0.1.0"

__all__ = ["LeWAMJEPAConfig", "LeWAMJEPAPolicy", "make_lewam_jepa_pre_post_processors"]
