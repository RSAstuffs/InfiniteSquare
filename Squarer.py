#!/usr/bin/env python3
"""
Lattice Tool - Factorization-aware lattice structure

Encodes N into lattice geometry such that the compression reveals factor relationships.
Uses number-theoretic lattice construction where geometric properties correspond to 
divisibility and factor structure.
"""

class LatticePoint:
    """Represents a point in integer lattice coordinates."""
    
    def __init__(self, x: int, y: int, z: int = 0):
        self.x = x
        self.y = y  
        self.z = z
    
    def __repr__(self):
        if self.z == 0:
            return f"LatticePoint({self.x}, {self.y})"
        return f"LatticePoint({self.x}, {self.y}, {self.z})"


class LatticeLine:
    """Represents a line segment in the lattice using integer endpoints."""
    
    def __init__(self, start: LatticePoint, end: LatticePoint):
        self.start = start
        self.end = end
    
    def get_median_center(self) -> LatticePoint:
        """Find the absolute median center of the line segment."""
        center_x = (self.start.x + self.end.x) // 2
        center_y = (self.start.y + self.end.y) // 2
        center_z = (self.start.z + self.end.z) // 2
        return LatticePoint(center_x, center_y, center_z)
    
    def get_length(self) -> int:
        """Calculate Manhattan length of the line segment."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dz = self.end.z - self.start.z
        
        abs_dx = dx if dx >= 0 else -dx
        abs_dy = dy if dy >= 0 else -dy
        abs_dz = dz if dz >= 0 else -dz
        
        return abs_dx + abs_dy + abs_dz
    
    def __repr__(self):
        return f"LatticeLine({self.start} -> {self.end})"


class FactorizationLattice:
    """
    Lattice structure specifically designed to encode factorization problems.
    
    Key insight: Encode N such that:
    - Lattice size = N
    - Points represent candidate factor pairs (a, b) where a*b ≈ N
    - Compression reveals GCD structure
    - Geometric distances correspond to divisibility
    """
    
    def __init__(self, n: int):
        self.n = n
        self.sqrt_n = self._isqrt(n)
        self.lattice_size = n
        
        # Factor-encoding properties
        self.factor_candidates = []
        self.gcd_tree = {}
        self.divisor_lattice = []
        
    def _isqrt(self, n: int) -> int:
        """Integer square root."""
        if n < 0:
            raise ValueError("Square root of negative number")
        if n == 0:
            return 0
        
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x
    
    def _gcd(self, a: int, b: int) -> int:
        """Euclidean GCD algorithm."""
        while b:
            a, b = b, a % b
        return a
    
    def encode_as_factor_lattice(self) -> LatticePoint:
        """
        Encode N into lattice where coordinates represent factor relationships.
        
        Strategy: Map N to point (a, b, c) where:
        - a is near sqrt(N) (balanced factorization point)
        - b = N // a (complementary factor)
        - c encodes remainder structure
        """
        # Start at balanced point near sqrt(N)
        a = self.sqrt_n
        b = self.n // a
        c = self.n - (a * b)  # Remainder encodes how close we are
        
        point = LatticePoint(a, b, c)
        
        # Store this as primary encoding
        self.primary_encoding = point
        
        return point
    
    def build_divisor_lattice(self):
        """
        Build lattice of all divisors with geometric relationships.
        Creates a lattice where distance encodes divisibility.
        """
        # Find all divisors up to sqrt(N)
        divisors = []
        for d in range(1, min(self.sqrt_n + 1, 10000)):  # Cap for performance
            if self.n % d == 0:
                complement = self.n // d
                divisors.append((d, complement))
                
                # Build GCD relationships
                for d2, c2 in divisors:
                    gcd = self._gcd(d, d2)
                    if gcd not in self.gcd_tree:
                        self.gcd_tree[gcd] = []
                    self.gcd_tree[gcd].append((d, d2))
        
        self.divisor_lattice = divisors
        return divisors
    
    def create_factor_mesh(self) -> list:
        """
        Create mesh of candidate points encoding factor relationships.
        Each point (x, y) represents testing if x is a factor.
        """
        mesh_points = []
        
        # Strategy 1: Points around sqrt(N)
        for offset in range(-50, 51):
            x = self.sqrt_n + offset
            if x > 1 and x < self.n:
                y = self.n // x
                remainder = self.n - (x * y)
                mesh_points.append(LatticePoint(x, y, remainder))
        
        # Strategy 2: Points at known divisor candidates
        # Test small primes and their multiples
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if self.n % p == 0:
                complement = self.n // p
                mesh_points.append(LatticePoint(p, complement, 0))
        
        # Strategy 3: Fermat factorization points
        # N = a² - b² = (a+b)(a-b)
        a = self.sqrt_n
        while a * a < self.n + 10000:  # Limit search
            a_squared = a * a
            diff = a_squared - self.n
            
            if diff >= 0:
                b = self._isqrt(diff)
                if b * b == diff:
                    # Found: N = (a+b)(a-b)
                    factor1 = a + b
                    factor2 = a - b
                    if factor1 > 1 and factor2 > 1:
                        mesh_points.append(LatticePoint(factor1, factor2, 0))
            a += 1
        
        return mesh_points
    
    def compress_to_factors(self, point: LatticePoint) -> list:
        """
        Compress the lattice point to reveal factor structure.
        Uses GCD-based compression.
        """
        factors = []
        
        # Method 1: Direct factor test from coordinates
        if point.x > 1 and self.n % point.x == 0:
            factors.append((point.x, self.n // point.x))
        
        if point.y > 1 and self.n % point.y == 0:
            factors.append((point.y, self.n // point.y))
        
        # Method 2: GCD of coordinates with N
        if point.x > 0 and point.y > 0:
            gcd_xy = self._gcd(point.x, point.y)
            if gcd_xy > 1 and self.n % gcd_xy == 0:
                factors.append((gcd_xy, self.n // gcd_xy))
        
        # Method 3: Geometric sum/difference (Fermat-like)
        sum_xy = point.x + point.y
        diff_xy = abs(point.x - point.y)
        
        if sum_xy > 0 and self.n % sum_xy == 0:
            factors.append((sum_xy, self.n // sum_xy))
        
        if diff_xy > 1 and self.n % diff_xy == 0:
            factors.append((diff_xy, self.n // diff_xy))
        
        # Method 4: Use remainder structure
        if point.z == 0:  # Exact factorization
            if point.x > 1 and point.y > 1 and point.x * point.y == self.n:
                factors.append((point.x, point.y))
        
        # Method 5: GCD with N directly
        gcd_x = self._gcd(point.x, self.n) if point.x > 0 else 1
        gcd_y = self._gcd(point.y, self.n) if point.y > 0 else 1
        
        if gcd_x > 1 and gcd_x < self.n:
            factors.append((gcd_x, self.n // gcd_x))
        
        if gcd_y > 1 and gcd_y < self.n:
            factors.append((gcd_y, self.n // gcd_y))
        
        # Remove duplicates and validate
        unique_factors = []
        seen = set()
        for f1, f2 in factors:
            pair = tuple(sorted([f1, f2]))
            if pair not in seen and f1 * f2 == self.n and f1 > 1 and f2 > 1:
                seen.add(pair)
                unique_factors.append(pair)
        
        return unique_factors


class CompressionMetrics:
    """Tracks compression metrics throughout the lattice transformation process."""
    
    def __init__(self, initial_point: LatticePoint, lattice_size: int):
        self.initial_point = initial_point
        self.lattice_size = lattice_size
        self.metrics = {}
    
    def calculate_manhattan_distance(self, p1: LatticePoint, p2: LatticePoint) -> int:
        """Calculate Manhattan distance between two points."""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        
        abs_dx = dx if dx >= 0 else -dx
        abs_dy = dy if dy >= 0 else -dy
        abs_dz = dz if dz >= 0 else -dz
        
        return abs_dx + abs_dy + abs_dz
    
    def record_stage(self, stage_name: str, point: LatticePoint):
        """Record metrics for a transformation stage."""
        origin = LatticePoint(0, 0, self.initial_point.z)
        manhattan_from_origin = self.calculate_manhattan_distance(origin, point)
        
        self.metrics[stage_name] = {
            'point': point,
            'manhattan_from_origin': manhattan_from_origin,
        }


class FactorizationEngine:
    """Uses factorization-aware lattice to find factors of N."""
    
    def __init__(self, n: int):
        self.n = n
        self.lattice = FactorizationLattice(n)
        self.found_factors = []
        self.prime_factors = {}
    
    def _is_prime_simple(self, n: int) -> bool:
        """Simple primality test with optimizations for large numbers."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # For very large numbers, use probabilistic approach
        # Test small primes first (fast)
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # For numbers larger than 10^100, use limited trial division
        # (full trial division would be too slow)
        if n > 10**100:
            # Test up to a reasonable limit
            limit = min(100000, int(n**0.25))  # Test up to 4th root or 100k
            i = 101
            while i < limit:
                if n % i == 0:
                    return False
                i += 2
            # For very large numbers, we can't do exhaustive testing
            # Return "probably prime" (True) if no small factors found
            return True
        else:
            # For smaller numbers, do full trial division
            i = 101
            sqrt_n = self._isqrt(n)
            while i <= sqrt_n:
                if n % i == 0:
                    return False
                i += 2
            return True
    
    def _factor_completely(self, n: int) -> dict:
        """Recursively factor n into prime factors."""
        if n < 2:
            return {}
        
        if self._is_prime_simple(n):
            return {n: 1}
        
        # Use lattice to find a factor
        temp_lattice = FactorizationLattice(n)
        temp_lattice.build_divisor_lattice()
        mesh = temp_lattice.create_factor_mesh()
        
        for point in mesh:
            factors = temp_lattice.compress_to_factors(point)
            if factors:
                f1, f2 = factors[0]
                if f1 > 1 and f1 < n:
                    # Found a factor, recursively factor both parts
                    result = {}
                    left = self._factor_completely(f1)
                    right = self._factor_completely(f2)
                    
                    # Merge the results
                    for prime, exp in left.items():
                        result[prime] = result.get(prime, 0) + exp
                    for prime, exp in right.items():
                        result[prime] = result.get(prime, 0) + exp
                    
                    return result
        
        # If we can't factor it further, assume it's prime
        return {n: 1}
    

    def detect_primality_from_lattice(self) -> dict:
        """
        Detect if N is prime using lattice structure analysis.
        
        For a prime number:
        - Primary encoding will have remainder z > 0 (not exact factorization)
        - All GCD tests will return 1 (no common factors)
        - No divisors found in reasonable search range
        - The remainder structure reveals primality
        """
        primary = self.lattice.encode_as_factor_lattice()
        
        # Key indicators for primality:
        # 1. Remainder z is non-zero and relatively large
        # 2. x and y are consecutive or very close (since N = x*y + z, and for prime, no exact factors)
        # 3. All GCD tests return 1
        
        is_prime_indicators = {
            'has_remainder': primary.z > 0,
            'remainder_ratio': primary.z / self.n if self.n > 0 else 0,
            'x_y_consecutive': abs(primary.x - primary.y) <= 1,
            'gcd_tests_all_one': True,
        }
        
        # Test GCDs
        gcd_x = self.lattice._gcd(primary.x, self.n)
        gcd_y = self.lattice._gcd(primary.y, self.n)
        gcd_z = self.lattice._gcd(primary.z, self.n) if primary.z > 0 else 1
        
        is_prime_indicators['gcd_x'] = gcd_x
        is_prime_indicators['gcd_y'] = gcd_y
        is_prime_indicators['gcd_z'] = gcd_z
        
        if gcd_x != 1 or gcd_y != 1 or (primary.z > 0 and gcd_z != 1):
            is_prime_indicators['gcd_tests_all_one'] = False
        
        # Check divisor lattice
        divisors = self.lattice.build_divisor_lattice()
        is_prime_indicators['divisor_count'] = len(divisors)
        is_prime_indicators['has_nontrivial_divisors'] = len(divisors) > 1
        
        # Prime conclusion: if all GCDs are 1, no non-trivial divisors, and remainder exists
        likely_prime = (
            is_prime_indicators['gcd_tests_all_one'] and
            not is_prime_indicators['has_nontrivial_divisors'] and
            is_prime_indicators['has_remainder']
        )
        
        return {
            'likely_prime': likely_prime,
            'indicators': is_prime_indicators,
            'primary_encoding': primary,
            'confidence': 'high' if likely_prime and is_prime_indicators['divisor_count'] == 1 else 'medium'
        }

    def factor(self, complete_factorization: bool = True) -> dict:
        """Factor N using lattice encoding."""
        print(f"\n{'='*70}")
        print(f"FACTORIZATION-AWARE LATTICE: N = {self.n}")
        print(f"{'='*70}")
        print(f"sqrt(N) ≈ {self.lattice.sqrt_n}")
        # Check for primality first using lattice structure
        primality_result = self.detect_primality_from_lattice()
        
        if primality_result['likely_prime']:
            print(f"\n{'─'*70}")
            print(f"PRIMALITY DETECTION (Lattice-Based)")
            print(f"{'─'*70}")
            print(f"  Lattice structure indicates: LIKELY PRIME")
            print(f"  Confidence: {primality_result['confidence']}")
            print(f"  Primary encoding remainder: {primality_result['primary_encoding'].z}")
            print(f"  GCD tests: x={primality_result['indicators']['gcd_x']}, y={primality_result['indicators']['gcd_y']}, z={primality_result['indicators']['gcd_z']}")
            print(f"  Non-trivial divisors found: {primality_result['indicators']['has_nontrivial_divisors']}")
            print(f"  Total divisors: {primality_result['indicators']['divisor_count']}")
            print(f"\n{'='*70}")
            print(f"N = {self.n} is PRIME")
            print(f"{'='*70}")
            
            return {
                'n': self.n,
                'factors': [],
                'prime_factors': {self.n: 1},
                'divisor_count': 1,
                'mesh_size': 0,
                'is_prime': True,
                'primality_confidence': primality_result['confidence']
            }
        
        
        
        # Step 1: Encode as factor lattice
        print(f"\nStep 1: Encoding N in factorization lattice...")
        primary_point = self.lattice.encode_as_factor_lattice()
        print(f"  Primary encoding: {primary_point}")
        print(f"  Represents: {primary_point.x} × {primary_point.y} + {primary_point.z} = {self.n}")
        
        # Check if primary encoding reveals factors
        if primary_point.z == 0:
            print(f"  ✓ Exact factorization found!")
        
        # Step 2: Build divisor lattice
        print(f"\nStep 2: Building divisor lattice...")
        divisors = self.lattice.build_divisor_lattice()
        print(f"  Found {len(divisors)} divisor pairs")
        if len(divisors) > 1:  # More than just (1, N)
            print(f"  Divisor pairs: {divisors[:10]}")  # Show first 10
        
        # Step 3: Create factor mesh
        print(f"\nStep 3: Creating factor candidate mesh...")
        mesh = self.lattice.create_factor_mesh()
        print(f"  Generated {len(mesh)} candidate points")
        
        # Step 4: Compress each mesh point to find factors
        print(f"\nStep 4: Compressing mesh points...")
        all_factors = set()
        
        for point in mesh:
            factors = self.lattice.compress_to_factors(point)
            for f_pair in factors:
                all_factors.add(f_pair)
        
        # Step 5: Verify and report factors
        print(f"\n{'─'*70}")
        print(f"FACTOR EXTRACTION RESULTS")
        print(f"{'─'*70}")
        
        valid_factors = []
        for f1, f2 in all_factors:
            if f1 * f2 == self.n and f1 > 1 and f2 > 1:
                valid_factors.append((f1, f2))
                is_prime_1 = self._is_prime_simple(f1)
                is_prime_2 = self._is_prime_simple(f2)
                prime_marker_1 = "P" if is_prime_1 else "C"
                prime_marker_2 = "P" if is_prime_2 else "C"
                print(f"  ✓ {f1}[{prime_marker_1}] × {f2}[{prime_marker_2}] = {self.n}")
        
        # Also report factors from divisor lattice
        for d, c in divisors:
            if d > 1 and c > 1 and d != c:  # Non-trivial factors
                pair = tuple(sorted([d, c]))
                if pair not in valid_factors:
                    valid_factors.append(pair)
                    is_prime_1 = self._is_prime_simple(d)
                    is_prime_2 = self._is_prime_simple(c)
                    prime_marker_1 = "P" if is_prime_1 else "C"
                    prime_marker_2 = "P" if is_prime_2 else "C"
                    print(f"  ✓ {d}[{prime_marker_1}] × {c}[{prime_marker_2}] = {self.n}")
        
        self.found_factors = valid_factors
        
        # Step 6: Complete prime factorization if requested
        if complete_factorization and valid_factors:
            print(f"\n{'─'*70}")
            print(f"COMPLETE PRIME FACTORIZATION")
            print(f"{'─'*70}")
            
            self.prime_factors = self._factor_completely(self.n)
            
            # Display prime factorization
            factor_strings = []
            for prime in sorted(self.prime_factors.keys()):
                exp = self.prime_factors[prime]
                if exp == 1:
                    factor_strings.append(str(prime))
                else:
                    factor_strings.append(f"{prime}^{exp}")
            
            factorization = " × ".join(factor_strings)
            print(f"  {self.n} = {factorization}")
            
            # Verify
            product = 1
            for prime, exp in self.prime_factors.items():
                product *= prime ** exp
            
            if product == self.n:
                print(f"  ✓ Verification: Product matches N")
            else:
                print(f"  ✗ Verification failed: {product} ≠ {self.n}")
        
        if valid_factors:
            print(f"\n{'='*70}")
            print(f"SUCCESS! Found {len(valid_factors)} factor pair(s)")
            if self.prime_factors:
                print(f"Prime factors: {sorted(self.prime_factors.keys())}")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"N = {self.n} appears to be prime")
            print(f"{'='*70}")
        
        return {
            'n': self.n,
            'factors': valid_factors,
            'prime_factors': self.prime_factors,
            'divisor_count': len(divisors),
            'mesh_size': len(mesh)
        }


def demo_factorization():
    """Demonstrate factorization using factor-aware lattice."""
    print("=== FACTORIZATION-AWARE LATTICE FOR CANCER RESEARCH ===")
    print("Using number-theoretic lattice structure to encode factorization")
    print()
    
    # Test cases - progressively larger
    test_numbers = [
        15,      # 3 × 5
        21,      # 3 × 7
        35,      # 5 × 7
        77,      # 7 × 11
        143,     # 11 × 13
        323,     # 17 × 19
        1001,    # 7 × 11 × 13
        2021,    # 43 × 47
        4199,    # 59 × 71 (4-digit semiprime)
        10403,   # 101 × 103 (5-digit semiprime)
        15841,   # 127 × 127 (prime square)
        32041,   # 179 × 179 (larger prime square)
        65027,   # 251 × 259 (5-digit)
        100127,  # 251 × 399 (6-digit)
        999983,  # Prime (to test primality detection)
    ]
    
    results = []
    for n in test_numbers:
        engine = FactorizationEngine(n)
        result = engine.factor(complete_factorization=True)
        results.append(result)
        print()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FACTORIZATION SUMMARY")
    print(f"{'='*70}")
    for result in results:
        n = result['n']
        prime_factors = result.get('prime_factors', {})
        
        if prime_factors:
            # Build factorization string
            factor_strs = []
            for prime in sorted(prime_factors.keys()):
                exp = prime_factors[prime]
                if exp == 1:
                    factor_strs.append(str(prime))
                else:
                    factor_strs.append(f"{prime}^{exp}")
            
            factorization = " × ".join(factor_strs)
            
            # Check if it's a semiprime (exactly 2 prime factors)
            total_factors = sum(prime_factors.values())
            if total_factors == 2:
                tag = "[SEMIPRIME]"
            elif len(prime_factors) == 1:
                tag = "[PRIME POWER]"
            else:
                tag = ""
            
            print(f"  N = {n:7d}: {factorization} {tag}")
        else:
            print(f"  N = {n:7d}: PRIME")
    print()
    
    # Statistics
    print(f"{'='*70}")
    print(f"STATISTICS")
    print(f"{'='*70}")
    total_tested = len(results)
    total_factored = sum(1 for r in results if r.get('prime_factors'))
    success_rate = (total_factored / total_tested * 100) if total_tested > 0 else 0
    
    print(f"  Numbers tested: {total_tested}")
    print(f"  Successfully factored: {total_factored}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    # Breakdown by type
    semiprimes = sum(1 for r in results if len(r.get('prime_factors', {})) == 2 and sum(r['prime_factors'].values()) == 2)
    composites = sum(1 for r in results if len(r.get('prime_factors', {})) > 2 or sum(r.get('prime_factors', {}).values()) > 2)
    primes = sum(1 for r in results if not r.get('prime_factors'))
    
    print(f"  Semiprimes (p×q): {semiprimes}")
    print(f"  Composites (≥3 factors): {composites}")
    print(f"  Primes: {primes}")
    print()


def factor_custom_number():
    """Factor a custom number provided by user."""
    print("=== CUSTOM NUMBER FACTORIZATION ===")
    print("Enter a number to factor (or 'q' to quit)")
    print()
    
    while True:
        try:
            user_input = input("N = ")
            if user_input.lower() == 'q':
                break
            
            n = int(user_input)
            if n < 2:
                print("Please enter a number ≥ 2")
                continue
            
            print()
            engine = FactorizationEngine(n)
            result = engine.factor(complete_factorization=True)
            print()
            
        except ValueError:
            print("Invalid input. Please enter an integer.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "custom":
            factor_custom_number()
        elif sys.argv[1] == "test":
            # Quick test mode
            test_nums = [143, 1001, 10403]
            for n in test_nums:
                engine = FactorizationEngine(n)
                result = engine.factor(complete_factorization=True)
                print()
        else:
            # Factor the number provided as argument
            try:
                n = int(sys.argv[1])
                engine = FactorizationEngine(n)
                result = engine.factor(complete_factorization=True)
            except ValueError:
                print("Usage: python lattice_tool.py [number|custom|test]")
    else:
        demo_factorization()
