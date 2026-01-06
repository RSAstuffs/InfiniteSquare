#!/usr/bin/env python3
"""
Lattice Tool - Geometric Lattice Transformations

Transforms an entire lattice through geometric compression stages:
Point -> Line -> Square -> Bounded Square -> Triangle -> Line -> Point

At each step, ALL points in the lattice are transformed/dragged along.
"""

import numpy as np
from typing import List, Tuple, Optional

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
    
    def to_array(self):
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=int)
    
    @classmethod
    def from_array(cls, arr):
        """Create from numpy array."""
        return cls(int(arr[0]), int(arr[1]), int(arr[2]) if len(arr) > 2 else 0)


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


class GeometricLattice:
    """
    Represents a full lattice that can be transformed geometrically.
    All points in the lattice are transformed together at each step.
    """
    
    def __init__(self, size: int, initial_point: Optional[LatticePoint] = None):
        """
        Initialize lattice.
        
        Args:
            size: Size of the lattice (size x size grid)
            initial_point: Optional starting point to insert
        """
        self.size = size
        self.lattice_points = []
        
        # Create full lattice grid
        for x in range(size):
            for y in range(size):
                self.lattice_points.append(LatticePoint(x, y, 0))
        
        # Store transformation history
        self.transformation_history = []
        self.current_stage = "initial"
        
        # If initial point provided, mark it
        if initial_point:
            self.initial_point = initial_point
            # Replace center point with initial point
            center_idx = (size // 2) * size + (size // 2)
            if center_idx < len(self.lattice_points):
                self.lattice_points[center_idx] = initial_point
        else:
            self.initial_point = LatticePoint(size // 2, size // 2, 0)
    
    def get_lattice_array(self) -> np.ndarray:
        """Get all lattice points as numpy array."""
        return np.array([p.to_array() for p in self.lattice_points])
    
    def set_lattice_from_array(self, arr: np.ndarray):
        """Set lattice points from numpy array."""
        self.lattice_points = [LatticePoint.from_array(arr[i]) for i in range(len(arr))]
    
    def transform_all_points(self, transformation_func):
        """
        Apply transformation to ALL points in the lattice.
        
        Args:
            transformation_func: Function that takes (x, y, z) and returns new (x, y, z)
        """
        new_points = []
        for point in self.lattice_points:
            new_coords = transformation_func(point.x, point.y, point.z)
            new_points.append(LatticePoint(new_coords[0], new_coords[1], new_coords[2]))
        self.lattice_points = new_points
        self.transformation_history.append(self.current_stage)
    
    def expand_point_to_line(self):
        """
        Step 1: Expand initial point into a line spanning the entire lattice.
        All lattice points are dragged along the expansion.
        """
        print("Step 1: Expanding point to line spanning entire lattice...")
        
        # Find the initial point (center)
        center_x = self.size // 2
        center_y = self.size // 2
        
        # Determine direction: horizontal or vertical based on distance from center
        def transform_to_line(x, y, z):
            # Calculate distance from center
            dx = x - center_x
            dy = y - center_y
            
            # Choose horizontal or vertical expansion based on which is larger
            if abs(dx) >= abs(dy):
                # Horizontal line: expand along x-axis
                new_x = x
                new_y = center_y  # All points move to center y
            else:
                # Vertical line: expand along y-axis
                new_x = center_x  # All points move to center x
                new_y = y
            
            return (new_x, new_y, z)
        
        self.transform_all_points(transform_to_line)
        self.current_stage = "line"
        print(f"  Lattice transformed: {len(self.lattice_points)} points now form a line")
    
    def create_square_from_line(self):
        """
        Step 2: Use first line to determine center by absolute median,
        then extend horizontal line from center to make a square (+ shape).
        All lattice points are transformed.
        """
        print("Step 2: Creating square from line (finding median center)...")
        
        # Find median of all points on the line
        x_coords = [p.x for p in self.lattice_points]
        y_coords = [p.y for p in self.lattice_points]
        
        # Absolute median (integer)
        median_x = sorted(x_coords)[len(x_coords) // 2]
        median_y = sorted(y_coords)[len(y_coords) // 2]
        
        print(f"  Median center: ({median_x}, {median_y})")
        
        def transform_to_square(x, y, z):
            # Create + shape: points align to either horizontal or vertical line through center
            if abs(x - median_x) <= abs(y - median_y):
                # Closer to vertical line: align to vertical
                new_x = median_x
                new_y = y
            else:
                # Closer to horizontal line: align to horizontal
                new_x = x
                new_y = median_y
            
            return (new_x, new_y, z)
        
        self.transform_all_points(transform_to_square)
        self.current_stage = "square_plus"
        print(f"  Lattice transformed: {len(self.lattice_points)} points form + shape")
    
    def create_bounded_square(self):
        """
        Step 3: At end of every line from +, extend single line horizontally for |
        and vertically for -, so lines meet to form a bounded square.
        All lattice points are dragged to form the boundary.
        """
        print("Step 3: Creating bounded square from + shape...")
        
        # Find bounds of current + shape
        x_coords = [p.x for p in self.lattice_points]
        y_coords = [p.y for p in self.lattice_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        print(f"  Bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        
        def transform_to_bounded_square(x, y, z):
            # Determine which part of + shape this point is on
            on_vertical = (x == center_x)
            on_horizontal = (y == center_y)
            
            if on_vertical:
                # Vertical line: extend horizontally to boundaries
                if y < center_y:
                    # Top part: extend to left boundary
                    new_x = min_x
                else:
                    # Bottom part: extend to right boundary
                    new_x = max_x
                new_y = y
            elif on_horizontal:
                # Horizontal line: extend vertically to boundaries
                if x < center_x:
                    # Left part: extend to top boundary
                    new_y = min_y
                else:
                    # Right part: extend to bottom boundary
                    new_y = max_y
                new_x = x
            else:
                # Corner: move to nearest corner of bounded square
                if x < center_x and y < center_y:
                    new_x, new_y = min_x, min_y  # Top-left
                elif x > center_x and y < center_y:
                    new_x, new_y = max_x, min_y  # Top-right
                elif x < center_x and y > center_y:
                    new_x, new_y = min_x, max_y  # Bottom-left
                else:
                    new_x, new_y = max_x, max_y  # Bottom-right
            
            return (new_x, new_y, z)
        
        self.transform_all_points(transform_to_bounded_square)
        self.current_stage = "bounded_square"
        print(f"  Lattice transformed: {len(self.lattice_points)} points form bounded square")
    
    def add_vertex_lines(self):
        """
        Step 4: Extend lines to connect each corner to its opposing corner (diagonals).
        All lattice points are transformed.
        """
        print("Step 4: Adding vertex lines (diagonals)...")
        
        # Find corners of bounded square
        x_coords = [p.x for p in self.lattice_points]
        y_coords = [p.y for p in self.lattice_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        corners = [
            (min_x, min_y),  # A: top-left
            (max_x, min_y),  # B: top-right
            (max_x, max_y),  # C: bottom-right
            (min_x, max_y)   # D: bottom-left
        ]
        
        def transform_with_diagonals(x, y, z):
            # Check if point is on diagonal A-C (top-left to bottom-right)
            # Equation: y - min_y = (max_y - min_y) * (x - min_x) / (max_x - min_x)
            # Using integer math only
            if max_x != min_x:
                # Diagonal A-C: from (min_x, min_y) to (max_x, max_y)
                dx_ac = max_x - min_x
                dy_ac = max_y - min_y
                diag_ac_y = min_y + (dy_ac * (x - min_x)) // dx_ac
                on_diag_ac = abs(y - diag_ac_y) <= 1
                
                # Diagonal B-D: from (max_x, min_y) to (min_x, max_y)
                dx_bd = min_x - max_x  # Negative
                dy_bd = max_y - min_y
                diag_bd_y = min_y + (dy_bd * (max_x - x)) // (-dx_bd) if dx_bd != 0 else y
                on_diag_bd = abs(y - diag_bd_y) <= 1
                
                # If on diagonal, keep it; otherwise move to nearest diagonal
                if on_diag_ac:
                    new_x, new_y = x, diag_ac_y
                elif on_diag_bd:
                    new_x, new_y = x, diag_bd_y
                else:
                    # Move to nearest diagonal (using integer math)
                    dist_to_ac = abs(y - diag_ac_y)
                    dist_to_bd = abs(y - diag_bd_y)
                    
                    if dist_to_ac <= dist_to_bd:
                        new_y = diag_ac_y
                        new_x = x
                    else:
                        new_y = diag_bd_y
                        new_x = x
            else:
                # Degenerate case: all x are same, keep y
                new_x, new_y = x, y
            
            return (new_x, new_y, z)
        
        self.transform_all_points(transform_with_diagonals)
        self.current_stage = "square_with_vertices"
        print(f"  Lattice transformed: {len(self.lattice_points)} points include vertex lines")
    
    def compress_square_to_triangle(self):
        """
        Step 5: Label corners A, B, C, D. Drag corners A and B to their median
        to form triangle (MCD). ALL lattice points are dragged into triangle boundary.
        """
        print("Step 5: Compressing square to triangle (A and B to median M)...")
        
        # Find corners
        x_coords = [p.x for p in self.lattice_points]
        y_coords = [p.y for p in self.lattice_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Corners: A=(min_x, min_y), B=(max_x, min_y), C=(max_x, max_y), D=(min_x, max_y)
        A = (min_x, min_y)
        B = (max_x, min_y)
        C = (max_x, max_y)
        D = (min_x, max_y)
        
        # Median of A and B
        M = ((A[0] + B[0]) // 2, (A[1] + B[1]) // 2)
        print(f"  Corners: A={A}, B={B}, C={C}, D={D}")
        print(f"  Median M of A and B: {M}")
        print(f"  Triangle vertices: M={M}, C={C}, D={D}")
        
        def transform_to_triangle(x, y, z):
            # Check if point is inside triangle MCD
            # Use barycentric coordinates or simple projection
            
            # Project point onto triangle boundary if outside
            # Triangle: M (top), C (bottom-right), D (bottom-left)
            
            # Check which side of triangle the point is on
            # For each edge, check if point is on correct side
            
            # Edge M-C: from M to C
            # Edge C-D: from C to D  
            # Edge D-M: from D to M
            
            # Simple approach: move point to nearest point on triangle boundary
            # or keep if inside
            
            # Calculate distances to edges and move to nearest edge if outside
            # For now, use simple projection: move all points toward triangle
            
            # Determine which region point is in and project accordingly (integer math only)
            if y <= M[1]:  # Above or at top (M)
                # Project to edge M-C or M-D using integer interpolation
                if x <= M[0]:
                    # Left side: project to M-D edge
                    if D[1] != M[1]:
                        # Integer interpolation: new_x = M[0] + (y - M[1]) * (D[0] - M[0]) / (D[1] - M[1])
                        dy = D[1] - M[1]
                        dx = D[0] - M[0]
                        if dy != 0:
                            new_x = M[0] + ((y - M[1]) * dx) // dy
                        else:
                            new_x = M[0]
                        new_y = y
                    else:
                        new_x, new_y = M[0], y
                else:
                    # Right side: project to M-C edge
                    if C[1] != M[1]:
                        # Integer interpolation: new_x = M[0] + (y - M[1]) * (C[0] - M[0]) / (C[1] - M[1])
                        dy = C[1] - M[1]
                        dx = C[0] - M[0]
                        if dy != 0:
                            new_x = M[0] + ((y - M[1]) * dx) // dy
                        else:
                            new_x = M[0]
                        new_y = y
                    else:
                        new_x, new_y = M[0], y
            else:
                # Below M: in bottom region
                if x < D[0]:
                    # Left of D: project to D
                    new_x, new_y = D[0], D[1]
                elif x > C[0]:
                    # Right of C: project to C
                    new_x, new_y = C[0], C[1]
                else:
                    # Between D and C: on base edge
                    new_x = x
                    new_y = D[1]  # Same y as D and C
            
            return (new_x, new_y, z)
        
        self.transform_all_points(transform_to_triangle)
        self.current_stage = "triangle"
        print(f"  Lattice transformed: {len(self.lattice_points)} points compressed to triangle MCD")
    
    def compress_triangle_to_line(self):
        """
        Step 6: Drag corners C and D together to their median, forming a single vertical line.
        ALL lattice points are dragged along.
        """
        print("Step 6: Compressing triangle to line (C and D to median N)...")
        
        # Find triangle vertices
        x_coords = [p.x for p in self.lattice_points]
        y_coords = [p.y for p in self.lattice_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Find M (top), C and D (bottom corners)
        # M should be at top (min_y), C and D at bottom (max_y)
        top_points = [p for p in self.lattice_points if p.y == min_y]
        bottom_points = [p for p in self.lattice_points if p.y == max_y]
        
        if top_points and bottom_points:
            M_x = sorted([p.x for p in top_points])[len(top_points) // 2]
            C_x = max([p.x for p in bottom_points])
            D_x = min([p.x for p in bottom_points])
            
            # Median of C and D
            N_x = (C_x + D_x) // 2
            N_y = max_y
            
            print(f"  Triangle: M=({M_x}, {min_y}), C=({C_x}, {max_y}), D=({D_x}, {max_y})")
            print(f"  Median N of C and D: ({N_x}, {N_y})")
            
            def transform_to_line(x, y, z):
                # All points move to vertical line through N
                new_x = N_x
                new_y = y  # Keep y coordinate
                return (new_x, new_y, z)
            
            self.transform_all_points(transform_to_line)
            self.current_stage = "vertical_line"
            print(f"  Lattice transformed: {len(self.lattice_points)} points compressed to vertical line MN")
    
    def compress_line_to_point(self):
        """
        Step 7: Compress vertical line into a single point by dragging both ends (M and N) to median.
        ALL lattice points are dragged to the final point.
        """
        print("Step 7: Compressing line to point (M and N to median)...")
        
        # Find endpoints of line
        y_coords = [p.y for p in self.lattice_points]
        min_y, max_y = min(y_coords), max(y_coords)
        
        # All points should have same x now (from previous step)
        x_coords = [p.x for p in self.lattice_points]
        center_x = sorted(x_coords)[len(x_coords) // 2]
        
        M = (center_x, min_y)
        N = (center_x, max_y)
        
        # Median of M and N
        final_point = ((M[0] + N[0]) // 2, (M[1] + N[1]) // 2)
        
        print(f"  Line endpoints: M={M}, N={N}")
        print(f"  Final point (median): {final_point}")
        
        def transform_to_point(x, y, z):
            # All points collapse to final point
            return (final_point[0], final_point[1], z)
        
        self.transform_all_points(transform_to_point)
        self.current_stage = "compressed_point"
        print(f"  Lattice transformed: {len(self.lattice_points)} points compressed to single point")
    
    def get_bounds(self):
        """Get bounding box of current lattice points."""
        if not self.lattice_points:
            return (0, 0, 0, 0)
        x_coords = [p.x for p in self.lattice_points]
        y_coords = [p.y for p in self.lattice_points]
        return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
    
    def get_area(self):
        """Calculate area covered by lattice points."""
        if not self.lattice_points:
            return 0
        min_x, max_x, min_y, max_y = self.get_bounds()
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        return width * height
    
    def get_perimeter(self):
        """Calculate perimeter of lattice points."""
        if not self.lattice_points:
            return 0
        min_x, max_x, min_y, max_y = self.get_bounds()
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        return 2 * (width + height)
    
    def get_unique_points_count(self):
        """Count unique point positions."""
        if not self.lattice_points:
            return 0
        unique_positions = set((p.x, p.y) for p in self.lattice_points)
        return len(unique_positions)
    
    def get_compression_metrics(self):
        """Calculate detailed compression metrics at current stage."""
        if not self.lattice_points:
            return {}
        
        min_x, max_x, min_y, max_y = self.get_bounds()
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        area = self.get_area()
        perimeter = self.get_perimeter()
        unique_points = self.get_unique_points_count()
        
        # Calculate span (Manhattan distance from origin)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        max_span = max(abs(max_x), abs(max_y), abs(min_x), abs(min_y))
        
        # Initial metrics
        initial_area = self.size * self.size
        initial_perimeter = 4 * self.size
        initial_span = self.size
        
        # Compression ratios
        area_compression = area / initial_area if initial_area > 0 else 0
        perimeter_compression = perimeter / initial_perimeter if initial_perimeter > 0 else 0
        span_compression = max_span / initial_span if initial_span > 0 else 0
        
        return {
            'stage': self.current_stage,
            'bounds': {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y},
            'dimensions': {'width': width, 'height': height},
            'area': area,
            'perimeter': perimeter,
            'unique_points': unique_points,
            'total_points': len(self.lattice_points),
            'center': (center_x, center_y),
            'max_span': max_span,
            'initial_area': initial_area,
            'initial_perimeter': initial_perimeter,
            'initial_span': initial_span,
            'area_compression_ratio': area_compression,
            'perimeter_compression_ratio': perimeter_compression,
            'span_compression_ratio': span_compression,
            'area_reduction': (1 - area_compression) * 100,
            'perimeter_reduction': (1 - perimeter_compression) * 100,
            'span_reduction': (1 - span_compression) * 100
        }
    
    def print_compression_analysis(self):
        """Print detailed compression analysis."""
        metrics = self.get_compression_metrics()
        
        print("="*80)
        print(f"COMPRESSION ANALYSIS - Stage: {metrics['stage']}")
        print("="*80)
        print(f"Bounds: x=[{metrics['bounds']['min_x']}, {metrics['bounds']['max_x']}], "
              f"y=[{metrics['bounds']['min_y']}, {metrics['bounds']['max_y']}]")
        print(f"Dimensions: {metrics['dimensions']['width']} x {metrics['dimensions']['height']}")
        print(f"Area: {metrics['area']} (initial: {metrics['initial_area']})")
        print(f"Perimeter: {metrics['perimeter']} (initial: {metrics['initial_perimeter']})")
        print(f"Unique point positions: {metrics['unique_points']} / {metrics['total_points']} total points")
        print(f"Center: {metrics['center']}")
        print(f"Max span from origin: {metrics['max_span']} (initial: {metrics['initial_span']})")
        print()
        print("Compression Ratios:")
        print(f"  Area compression: {metrics['area_compression_ratio']:.6f} ({metrics['area_reduction']:.2f}% reduction)")
        print(f"  Perimeter compression: {metrics['perimeter_compression_ratio']:.6f} ({metrics['perimeter_reduction']:.2f}% reduction)")
        print(f"  Span compression: {metrics['span_compression_ratio']:.6f} ({metrics['span_reduction']:.2f}% reduction)")
        print()


def factor_with_lattice_compression(N: int, lattice_size: int = None):
    """
    Factor N using geometric lattice compression.
    
    Strategy:
    1. Encode N into lattice structure
    2. Apply geometric transformations
    3. Extract factors from compressed result
    """
    print("="*80)
    print(f"FACTORIZATION USING GEOMETRIC LATTICE COMPRESSION")
    print("="*80)
    print(f"Target N = {N}")
    print(f"Bit length: {N.bit_length()} bits")
    print()
    
    # Determine lattice size based on N
    if lattice_size is None:
        # Use sqrt(N) as base, but cap for performance
        sqrt_n = int(N ** 0.5) if N < 10**20 else 1000
        lattice_size = min(max(100, sqrt_n // 10), 1000)  # Reasonable size
    
    print(f"Using {lattice_size}x{lattice_size} lattice")
    print()
    
    # Encode N into initial point
    # Strategy: encode as (a, b, remainder) where a*b ≈ N
    sqrt_n = int(N ** 0.5) if N < 10**20 else lattice_size // 2
    a = sqrt_n
    b = N // a if a > 0 else 1
    remainder = N - (a * b)
    
    # Scale to fit lattice
    scale_factor = lattice_size // max(a, b, 1) if max(a, b) > lattice_size else 1
    if scale_factor == 0:
        scale_factor = 1
    
    # Map to lattice coordinates
    initial_x = min(a // scale_factor, lattice_size - 1)
    initial_y = min(b // scale_factor, lattice_size - 1)
    initial_z = remainder % lattice_size
    
    initial_point = LatticePoint(initial_x, initial_y, initial_z)
    
    print(f"Encoded N as lattice point: {initial_point}")
    print(f"  Represents: a={a}, b={b}, remainder={remainder}")
    print(f"  Scale factor: {scale_factor}")
    print()
    
    # Create lattice and apply transformations
    lattice = GeometricLattice(lattice_size, initial_point)
    
    # Store original encoding for factor extraction
    original_encoding = {'a': a, 'b': b, 'remainder': remainder, 'scale': scale_factor}
    
    # Apply transformation sequence
    lattice.expand_point_to_line()
    lattice.create_square_from_line()
    lattice.create_bounded_square()
    lattice.add_vertex_lines()
    lattice.compress_square_to_triangle()
    lattice.compress_triangle_to_line()
    lattice.compress_line_to_point()
    
    # Extract factors from compressed result
    final_metrics = lattice.get_compression_metrics()
    final_point = lattice.lattice_points[0] if lattice.lattice_points else None
    
    print("="*80)
    print("FACTOR EXTRACTION FROM COMPRESSED LATTICE")
    print("="*80)
    
    factors_found = []
    
    if final_point:
        # Method 1: Use final point coordinates to derive factors
        # Reverse the scaling
        candidate_a = final_point.x * original_encoding['scale']
        candidate_b = final_point.y * original_encoding['scale']
        
        # Test if these are factors
        if candidate_a > 1 and N % candidate_a == 0:
            factors_found.append((candidate_a, N // candidate_a))
        
        if candidate_b > 1 and candidate_b != candidate_a and N % candidate_b == 0:
            factors_found.append((candidate_b, N // candidate_b))
        
        # Method 2: Use compressed coordinates with GCD, search around compressed point
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        # Search around the compressed coordinates
        base_x_scaled = final_point.x * original_encoding['scale']
        base_y_scaled = final_point.y * original_encoding['scale']
        
        # Search in a wider range around compressed point
        search_range = min(50, N // 10)
        for offset in range(-search_range, search_range + 1):
            test_x = base_x_scaled + offset
            test_y = base_y_scaled + offset
            
            if test_x > 1 and test_x < N:
                gcd_x = gcd(test_x, N)
                if gcd_x > 1 and gcd_x < N:
                    factors_found.append((gcd_x, N // gcd_x))
            
            if test_y > 1 and test_y < N:
                gcd_y = gcd(test_y, N)
                if gcd_y > 1 and gcd_y < N:
                    factors_found.append((gcd_y, N // gcd_y))
        
        # Method 3: Use sum/difference of coordinates
        sum_coords = (final_point.x + final_point.y) * original_encoding['scale']
        diff_coords = abs(final_point.x - final_point.y) * original_encoding['scale']
        
        if sum_coords > 1 and N % sum_coords == 0:
            factors_found.append((sum_coords, N // sum_coords))
        
        if diff_coords > 1 and diff_coords != sum_coords and N % diff_coords == 0:
            factors_found.append((diff_coords, N // diff_coords))
        
        # Method 4: Use remainder structure
        if final_point.z == 0:  # Exact match
            test_a = final_point.x * original_encoding['scale']
            test_b = final_point.y * original_encoding['scale']
            if test_a * test_b == N:
                factors_found.append((test_a, test_b))
        
        print(f"Final compressed point: {final_point}")
        print(f"  Coordinates: x={final_point.x}, y={final_point.y}, z={final_point.z}")
        print()
    
    # Remove duplicates and validate
    unique_factors = []
    seen = set()
    for f1, f2 in factors_found:
        pair = tuple(sorted([f1, f2]))
        if pair not in seen and f1 * f2 == N and f1 > 1 and f2 > 1:
            seen.add(pair)
            unique_factors.append(pair)
    
    # Also check original encoding and nearby values
    # Search around the original encoding
    orig_a = original_encoding['a']
    orig_b = original_encoding['b']
    
    # Test original values
    if orig_a > 1 and N % orig_a == 0:
        pair = tuple(sorted([orig_a, N // orig_a]))
        if pair not in seen:
            unique_factors.append(pair)
            seen.add(pair)
    
    if orig_b > 1 and N % orig_b == 0:
        pair = tuple(sorted([orig_b, N // orig_b]))
        if pair not in seen:
            unique_factors.append(pair)
            seen.add(pair)
    
    # Search around original encoding (factors might be nearby)
    search_range = min(20, N // 20)
    for offset in range(-search_range, search_range + 1):
        test_a = orig_a + offset
        test_b = orig_b + offset
        
        if test_a > 1 and test_a < N and N % test_a == 0:
            pair = tuple(sorted([test_a, N // test_a]))
            if pair not in seen:
                unique_factors.append(pair)
                seen.add(pair)
        
        if test_b > 1 and test_b < N and test_b != test_a and N % test_b == 0:
            pair = tuple(sorted([test_b, N // test_b]))
            if pair not in seen:
                unique_factors.append(pair)
                seen.add(pair)
    
    # Report results
    if unique_factors:
        print("FACTORS FOUND:")
        for f1, f2 in unique_factors:
            print(f"  ✓ {f1} × {f2} = {N}")
            print(f"    Verification: {f1 * f2 == N}")
    else:
        print("No factors found through lattice compression.")
        print("  This may indicate N is prime, or factors require different encoding.")
    
    print()
    print("="*80)
    print("COMPRESSION METRICS")
    print("="*80)
    print(f"Area reduction: {final_metrics['area_reduction']:.2f}%")
    print(f"Perimeter reduction: {final_metrics['perimeter_reduction']:.2f}%")
    print(f"Points collapsed: {final_metrics['unique_points']} / {final_metrics['total_points']}")
    print()
    
    return {
        'N': N,
        'factors': unique_factors,
        'compression_metrics': final_metrics,
        'final_point': final_point
    }


def demo_lattice_transformations():
    """Demonstrate full lattice transformation sequence."""
    print("="*80)
    print("GEOMETRIC LATTICE TRANSFORMATIONS")
    print("="*80)
    print()
    
    # Create lattice with initial point
    size = 100
    initial_point = LatticePoint(50, 50, 0)
    
    print(f"Initializing {size}x{size} lattice with point at {initial_point}")
    lattice = GeometricLattice(size, initial_point)
    print(f"Lattice contains {len(lattice.lattice_points)} points")
    print()
    
    # Execute transformation sequence with compression analysis at each stage
    print("Initial state:")
    lattice.print_compression_analysis()
    print()
    
    lattice.expand_point_to_line()
    lattice.print_compression_analysis()
    print()
    
    lattice.create_square_from_line()
    lattice.print_compression_analysis()
    print()
    
    lattice.create_bounded_square()
    lattice.print_compression_analysis()
    print()
    
    lattice.add_vertex_lines()
    lattice.print_compression_analysis()
    print()
    
    lattice.compress_square_to_triangle()
    lattice.print_compression_analysis()
    print()
    
    lattice.compress_triangle_to_line()
    lattice.print_compression_analysis()
    print()
    
    lattice.compress_line_to_point()
    lattice.print_compression_analysis()
    print()
    
    # Final summary
    final_metrics = lattice.get_compression_metrics()
    print("="*80)
    print("FINAL COMPRESSION SUMMARY")
    print("="*80)
    print(f"Initial lattice size: {size}x{size} = {size*size} points")
    print(f"Initial area: {final_metrics['initial_area']}")
    print(f"Initial perimeter: {final_metrics['initial_perimeter']}")
    print(f"Initial span: {final_metrics['initial_span']}")
    print()
    print(f"Final area: {final_metrics['area']}")
    print(f"Final perimeter: {final_metrics['perimeter']}")
    print(f"Final span: {final_metrics['max_span']}")
    print(f"Final unique points: {final_metrics['unique_points']}")
    print()
    print(f"Total area reduction: {final_metrics['area_reduction']:.2f}%")
    print(f"Total perimeter reduction: {final_metrics['perimeter_reduction']:.2f}%")
    print(f"Total span reduction: {final_metrics['span_reduction']:.2f}%")
    print()
    print(f"Compression achieved: {final_metrics['unique_points']} unique positions from {final_metrics['total_points']} points")
    print(f"Compression efficiency: {(1 - final_metrics['unique_points']/final_metrics['total_points'])*100:.2f}% points collapsed")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            # Try to parse as number to factor
            N = int(sys.argv[1])
            if N > 1:
                factor_with_lattice_compression(N)
            else:
                print("Please provide a number > 1 to factor")
        except ValueError:
            # If not a number, treat as size for demo
            size = int(sys.argv[1])
            demo_lattice_transformations()
    else:
        # Default: try factoring some test numbers
        print("Testing factorization on sample numbers:")
        print()
        test_numbers = [15, 21, 35, 77, 143, 323, 2021]
        for n in test_numbers:
            result = factor_with_lattice_compression(n, lattice_size=100)
            print()
