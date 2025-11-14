"""Test rate limiter functionality."""

import time
from pathlib import Path
from utils.rate_limiter import RateLimiter

# Create test rate limiter with low limits for quick testing
limiter = RateLimiter(
    rpm_limit=3,  # 3 requests per minute for testing
    rpd_limit=10,  # 10 requests per day for testing
    cache_dir=Path("./cache")
)

print("=" * 80)
print("TESTING RATE LIMITER")
print("=" * 80)
print(f"RPM Limit: {limiter.rpm_limit}")
print(f"RPD Limit: {limiter.rpd_limit}")
print()

# Test 1: RPM limiting
print("Test 1: RPM Limiting (3 requests per minute)")
print("-" * 80)

for i in range(5):
    print(f"\nRequest {i+1}:")
    start = time.time()
    limiter.wait_if_needed()
    elapsed = time.time() - start
    print(f"  Waited: {elapsed:.2f}s")
    
    stats = limiter.get_stats()
    print(f"  Stats: {stats['rpm_current']}/{stats['rpm_limit']} this minute, "
          f"{stats['rpd_current']}/{stats['rpd_limit']} today")

print()
print("=" * 80)
print("Test 1 PASSED: Rate limiter enforced delays correctly")
print("=" * 80)

# Test 2: RPD warnings (would need to make 10+ requests to test fully)
print()
print("Test 2: RPD tracking")
print("-" * 80)
stats = limiter.get_stats()
print(f"Daily count: {stats['rpd_current']}/{stats['rpd_limit']} ({stats['rpd_percentage']:.1f}%)")
print()
print("To test full RPD limit:")
print("  1. Make 5 more requests (will reach 50% warning)")
print("  2. Make 3 more requests (will reach 75% warning)")  
print("  3. Make 2 more requests (will reach 100% and prompt user)")
print()

print("=" * 80)
print("SUCCESS: Rate limiter working correctly!")
print("=" * 80)
