"""Rate limiter for API calls with RPM and RPD tracking."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for LLM API calls.
    
    Enforces:
    - RPM (Requests Per Minute) limit with artificial delays
    - RPD (Requests Per Day) tracking with warnings and stops
    """

    def __init__(
        self,
        rpm_limit: int = 10,
        rpd_limit: int = 500,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            rpm_limit: Requests per minute limit (default: 10 for Gemini free tier)
            rpd_limit: Requests per day limit (default: 500 for Gemini free tier)
            cache_dir: Directory to store rate limit state
        """
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.cache_dir = cache_dir or Path("./cache")
        self.state_file = self.cache_dir / "rate_limit_state.json"
        
        # Thread safety
        self.lock = Lock()

        # RPM tracking
        self.request_times: list[float] = []

        # RPD tracking
        self.daily_count = 0
        self.current_date = datetime.now().date()
        
        # Load state from disk
        self._load_state()

    def _load_state(self):
        """Load rate limit state from disk."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Check if same day
            saved_date = datetime.fromisoformat(state.get('date', '')).date()
            if saved_date == self.current_date:
                self.daily_count = state.get('daily_count', 0)
                logger.info(f"[RATE LIMIT] Loaded state: {self.daily_count}/{self.rpd_limit} requests today")
            else:
                logger.info(f"[RATE LIMIT] New day - resetting daily counter")
                self.daily_count = 0
                
        except Exception as e:
            logger.warning(f"Failed to load rate limit state: {e}")

    def _save_state(self):
        """Save rate limit state to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'date': self.current_date.isoformat(),
                'daily_count': self.daily_count,
                'timestamp': datetime.now().isoformat(),
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save rate limit state: {e}")

    def wait_if_needed(self):
        """
        Wait if needed to stay under RPM limit.
        
        Also checks RPD limit and raises exception if exceeded.
        """
        with self.lock:
            now = time.time()
            
            # Check if new day
            current_date = datetime.now().date()
            if current_date != self.current_date:
                logger.info(f"[RATE LIMIT] New day - resetting daily counter (was {self.daily_count})")
                self.current_date = current_date
                self.daily_count = 0
                self._save_state()
            
            # Check RPD limit
            if self.daily_count >= self.rpd_limit:
                self._handle_rpd_limit_reached()
            
            # Check RPD warnings
            if self.daily_count == self.rpd_limit // 2:  # 50%
                logger.warning("")
                logger.warning("=" * 80)
                logger.warning(f"⚠️  RATE LIMIT WARNING: {self.daily_count}/{self.rpd_limit} requests (50%)")
                logger.warning("=" * 80)
                logger.warning("")
            elif self.daily_count == int(self.rpd_limit * 0.75):  # 75%
                logger.warning("")
                logger.warning("=" * 80)
                logger.warning(f"⚠️  RATE LIMIT WARNING: {self.daily_count}/{self.rpd_limit} requests (75%)")
                logger.warning("Approaching daily limit! Consider:")
                logger.warning("  1. Reduce --workers to 1")
                logger.warning("  2. Process in batches with --resume")
                logger.warning("  3. Switch to --llm-provider ollama (local)")
                logger.warning("=" * 80)
                logger.warning("")
            
            # Remove requests older than 1 minute
            cutoff = now - 60
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Check if we need to wait
            if len(self.request_times) >= self.rpm_limit:
                # Need to wait until oldest request is > 1 minute old
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest) + 0.1  # Add 0.1s buffer
                
                if wait_time > 0:
                    logger.debug(f"[RATE LIMIT] {len(self.request_times)}/{self.rpm_limit} requests in last minute")
                    logger.debug(f"[RATE LIMIT] Sleeping {wait_time:.1f}s to stay under {self.rpm_limit} RPM")
                    time.sleep(wait_time)
                    now = time.time()
            
            # Record this request
            self.request_times.append(now)
            self.daily_count += 1
            
            # Save state periodically (every 10 requests)
            if self.daily_count % 10 == 0:
                self._save_state()
            
            # Log progress
            rpm_count = len(self.request_times)
            logger.info(
                f"[RATE LIMIT] Request #{self.daily_count} today "
                f"({rpm_count}/{self.rpm_limit} this minute, "
                f"{self.daily_count}/{self.rpd_limit} today, "
                f"{(self.daily_count/self.rpd_limit)*100:.1f}%)"
            )

    def _handle_rpd_limit_reached(self):
        """Handle reaching the daily request limit."""
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"🛑 RATE LIMIT REACHED: {self.rpd_limit} requests for today")
        logger.error("=" * 80)
        logger.error("")
        logger.error("You have reached the Gemini free tier daily limit (500 RPD).")
        logger.error("")
        logger.error("Options:")
        logger.error("")
        logger.error("1. PAUSE AND RESUME TOMORROW:")
        logger.error("   - Stop the pipeline now (Ctrl+C)")
        logger.error("   - Resume tomorrow with: --resume flag")
        logger.error("   - Progress is saved in index.jsonl and cache")
        logger.error("")
        logger.error("2. SWITCH TO LOCAL OLLAMA:")
        logger.error("   - Install Ollama: https://ollama.com/download")
        logger.error("   - Pull model: ollama pull deepseek-r1:8b")
        logger.error("   - Restart pipeline with: --llm-provider ollama --resume")
        logger.error("")
        logger.error("3. UPGRADE TO GEMINI PRO:")
        logger.error("   - Higher rate limits")
        logger.error("   - Visit: https://ai.google.dev/pricing")
        logger.error("")
        
        # Interactive prompt
        try:
            print("\nWhat would you like to do?")
            print("  [p] Pause and exit (resume tomorrow)")
            print("  [o] Switch to Ollama (requires local setup)")
            print("  [c] Continue anyway (may fail with API errors)")
            print()
            choice = input("Choice [p/o/c]: ").strip().lower()
            
            if choice == 'p' or choice == '':
                logger.info("User chose to pause. Exiting...")
                import sys
                sys.exit(0)
            elif choice == 'o':
                logger.info("User chose to switch to Ollama.")
                logger.info("Please restart the pipeline with: --llm-provider ollama --resume")
                import sys
                sys.exit(0)
            elif choice == 'c':
                logger.warning("User chose to continue despite rate limit!")
                logger.warning("API calls may fail with 429 errors.")
                return
            else:
                logger.info("Invalid choice. Exiting...")
                import sys
                sys.exit(0)
                
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting due to user interrupt...")
            import sys
            sys.exit(0)

    def reset_daily_count(self):
        """Reset daily counter (for testing or manual override)."""
        with self.lock:
            logger.info(f"[RATE LIMIT] Manually resetting daily counter (was {self.daily_count})")
            self.daily_count = 0
            self._save_state()

    def get_stats(self) -> dict:
        """Get current rate limit statistics."""
        with self.lock:
            now = time.time()
            cutoff = now - 60
            recent_requests = [t for t in self.request_times if t > cutoff]
            
            return {
                'rpm_current': len(recent_requests),
                'rpm_limit': self.rpm_limit,
                'rpd_current': self.daily_count,
                'rpd_limit': self.rpd_limit,
                'rpd_percentage': (self.daily_count / self.rpd_limit) * 100,
                'date': self.current_date.isoformat(),
            }


# Global rate limiter instance (will be initialized by config)
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config=None) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        if config:
            rpm = getattr(config.rate_limit, 'rpm_limit', 10) if hasattr(config, 'rate_limit') else 10
            rpd = getattr(config.rate_limit, 'rpd_limit', 500) if hasattr(config, 'rate_limit') else 500
            cache_dir = getattr(config, 'cache_dir', Path("./cache"))
        else:
            rpm = 10
            rpd = 500
            cache_dir = Path("./cache")
        
        _rate_limiter = RateLimiter(
            rpm_limit=rpm,
            rpd_limit=rpd,
            cache_dir=cache_dir,
        )
        
        logger.info(f"[RATE LIMIT] Initialized: {rpm} RPM, {rpd} RPD")
    
    return _rate_limiter
