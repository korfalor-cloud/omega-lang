/// Computational geometry: points, lines, polygons, convex hull, intersection.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn distance_to(&self, other: &Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn manhattan_distance(&self, other: &Self) -> f64 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    pub fn midpoint(&self, other: &Self) -> Self {
        Self { x: (self.x + other.x) / 2.0, y: (self.y + other.y) / 2.0 }
    }

    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        Self { x: self.x + dx, y: self.y + dy }
    }

    pub fn rotate(&self, angle: f64, origin: &Self) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        let dx = self.x - origin.x;
        let dy = self.y - origin.y;
        Self {
            x: origin.x + dx * cos - dy * sin,
            y: origin.y + dx * sin + dy * cos,
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(&self, other: &Self) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

#[derive(Debug, Clone)]
pub struct Line2D {
    pub start: Point2D,
    pub end: Point2D,
}

impl Line2D {
    pub fn new(start: Point2D, end: Point2D) -> Self { Self { start, end } }

    pub fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    pub fn direction(&self) -> Point2D {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-10 { Point2D::new(0.0, 0.0) } else { Point2D::new(dx / len, dy / len) }
    }

    pub fn midpoint(&self) -> Point2D {
        self.start.midpoint(&self.end)
    }

    pub fn point_at(&self, t: f64) -> Point2D {
        Point2D {
            x: self.start.x + t * (self.end.x - self.start.x),
            y: self.start.y + t * (self.end.y - self.start.y),
        }
    }

    /// Distance from point to line segment.
    pub fn distance_to_point(&self, p: &Point2D) -> f64 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < 1e-10 {
            return self.start.distance_to(p);
        }
        let t = ((p.x - self.start.x) * dx + (p.y - self.start.y) * dy / len_sq).clamp(0.0, 1.0);
        let proj = self.point_at(t);
        p.distance_to(&proj)
    }

    /// Do two line segments intersect?
    pub fn intersects(&self, other: &Self) -> bool {
        self.intersection_point(other).is_some()
    }

    /// Intersection point of two line segments.
    pub fn intersection_point(&self, other: &Self) -> Option<Point2D> {
        let p = &self.start;
        let r = Point2D::new(self.end.x - self.start.x, self.end.y - self.start.y);
        let q = &other.start;
        let s = Point2D::new(other.end.x - other.start.x, other.end.y - other.start.y);

        let rxs = r.cross(&s);
        let qpxr = Point2D::new(q.x - p.x, q.y - p.y).cross(&r);

        if rxs.abs() < 1e-10 {
            return None; // Parallel or collinear
        }

        let t = Point2D::new(q.x - p.x, q.y - p.y).cross(&s) / rxs;
        let u = qpxr / rxs;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            Some(Point2D::new(p.x + t * r.x, p.y + t * r.y))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Point2D>,
}

impl Polygon {
    pub fn new(vertices: Vec<Point2D>) -> Self {
        Self { vertices }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Signed area of the polygon.
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }
        area / 2.0
    }

    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        if n < 2 {
            return 0.0;
        }
        let mut perimeter = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            perimeter += self.vertices[i].distance_to(&self.vertices[j]);
        }
        perimeter
    }

    /// Centroid of the polygon.
    pub fn centroid(&self) -> Point2D {
        let n = self.vertices.len();
        if n == 0 {
            return Point2D::new(0.0, 0.0);
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for v in &self.vertices {
            cx += v.x;
            cy += v.y;
        }
        Point2D::new(cx / n as f64, cy / n as f64)
    }

    /// Is the polygon convex?
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut sign = 0i32;
        for i in 0..n {
            let j = (i + 1) % n;
            let k = (i + 2) % n;
            let dx1 = self.vertices[j].x - self.vertices[i].x;
            let dy1 = self.vertices[j].y - self.vertices[i].y;
            let dx2 = self.vertices[k].x - self.vertices[j].x;
            let dy2 = self.vertices[k].y - self.vertices[j].y;
            let cross = dx1 * dy2 - dy1 * dx2;
            if cross.abs() > 1e-10 {
                let current_sign = if cross > 0.0 { 1 } else { -1 };
                if sign == 0 {
                    sign = current_sign;
                } else if sign != current_sign {
                    return false;
                }
            }
        }
        true
    }

    /// Is a point inside the polygon (ray casting)?
    pub fn contains_point(&self, point: &Point2D) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];
            if (vi.y > point.y) != (vj.y > point.y)
                && point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Bounding box of the polygon.
    pub fn bounding_box(&self) -> (Point2D, Point2D) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for v in &self.vertices {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
        }
        (Point2D::new(min_x, min_y), Point2D::new(max_x, max_y))
    }

    /// Winding number of the polygon around a point.
    pub fn winding_number(&self, point: &Point2D) -> i32 {
        let mut wn = 0i32;
        let n = self.vertices.len();
        for i in 0..n {
            let j = (i + 1) % n;
            if self.vertices[i].y <= point.y {
                if self.vertices[j].y > point.y {
                    let dx = self.vertices[j].x - self.vertices[i].x;
                    let dy = self.vertices[j].y - self.vertices[i].y;
                    if dx * (point.y - self.vertices[i].y) - dy * (point.x - self.vertices[i].x) > 0.0 {
                        wn += 1;
                    }
                }
            } else {
                if self.vertices[j].y <= point.y {
                    let dx = self.vertices[j].x - self.vertices[i].x;
                    let dy = self.vertices[j].y - self.vertices[i].y;
                    if dx * (point.y - self.vertices[i].y) - dy * (point.x - self.vertices[i].x) < 0.0 {
                        wn -= 1;
                    }
                }
            }
        }
        wn
    }
}

/// Convex hull using Graham scan.
pub fn convex_hull(points: &[Point2D]) -> Vec<Point2D> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find lowest point (and leftmost if tie)
    let mut lowest = 0;
    for i in 1..points.len() {
        if points[i].y < points[lowest].y
            || (points[i].y == points[lowest].y && points[i].x < points[lowest].x)
        {
            lowest = i;
        }
    }

    let pivot = points[lowest];
    let mut sorted: Vec<Point2D> = points.to_vec();
    sorted.swap(0, lowest);

    sorted[1..].sort_by(|a, b| {
        let cross = (a.x - pivot.x) * (b.y - pivot.y) - (a.y - pivot.y) * (b.x - pivot.x);
        if cross.abs() < 1e-10 {
            let da = (a.x - pivot.x).powi(2) + (a.y - pivot.y).powi(2);
            let db = (b.x - pivot.x).powi(2) + (b.y - pivot.y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        } else if cross > 0.0 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });

    let mut hull = Vec::new();
    for point in sorted {
        while hull.len() > 1 {
            let n = hull.len();
            let cross = (hull[n - 1].x - hull[n - 2].x) * (point.y - hull[n - 2].y)
                - (hull[n - 1].y - hull[n - 2].y) * (point.x - hull[n - 2].x);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(point);
    }

    hull
}

/// Closest pair of points (divide and conquer).
pub fn closest_pair(points: &[Point2D]) -> (Point2D, Point2D, f64) {
    if points.len() < 2 {
        return (Point2D::zero(), Point2D::zero(), f64::INFINITY);
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

    let (p1, p2, dist) = closest_pair_rec(&sorted);
    (p1, p2, dist)
}

fn closest_pair_rec(points: &[Point2D]) -> (Point2D, Point2D, f64) {
    let n = points.len();
    if n <= 3 {
        let mut min_dist = f64::INFINITY;
        let mut best = (points[0], points[1]);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = points[i].distance_to(&points[j]);
                if d < min_dist {
                    min_dist = d;
                    best = (points[i], points[j]);
                }
            }
        }
        return (best.0, best.1, min_dist);
    }

    let mid = n / 2;
    let mid_x = points[mid].x;

    let (left_p1, left_p2, left_dist) = closest_pair_rec(&points[..mid]);
    let (right_p1, right_p2, right_dist) = closest_pair_rec(&points[mid..]);

    let (mut best_p1, mut best_p2, mut best_dist) = if left_dist < right_dist {
        (left_p1, left_p2, left_dist)
    } else {
        (right_p1, right_p2, right_dist)
    };

    // Check strip
    let strip: Vec<Point2D> = points.iter()
        .filter(|p| (p.x - mid_x).abs() < best_dist)
        .copied()
        .collect();

    for i in 0..strip.len() {
        for j in (i + 1)..strip.len().min(i + 7) {
            let d = strip[i].distance_to(&strip[j]);
            if d < best_dist {
                best_dist = d;
                best_p1 = strip[i];
                best_p2 = strip[j];
            }
        }
    }

    (best_p1, best_p2, best_dist)
}

impl Point2D {
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
}

/// Circle through three points.
pub fn circumcircle(a: &Point2D, b: &Point2D, c: &Point2D) -> Option<(Point2D, f64)> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-10 {
        return None;
    }

    let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
        + (b.x * b.x + b.y * b.y) * (c.y - a.y)
        + (c.x * c.x + c.y * c.y) * (a.y - b.y))
        / d;
    let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
        + (b.x * b.x + b.y * b.y) * (a.x - c.x)
        + (c.x * c.x + c.y * c.y) * (b.x - a.x))
        / d;

    let center = Point2D::new(ux, uy);
    let radius = center.distance_to(a);
    Some((center, radius))
}

/// Point-in-circle test.
pub fn point_in_circle(point: &Point2D, center: &Point2D, radius: f64) -> bool {
    point.distance_to(center) <= radius
}

/// Circle-circle intersection.
pub fn circle_intersection(c1: &Point2D, r1: f64, c2: &Point2D, r2: f64) -> Vec<Point2D> {
    let d = c1.distance_to(c2);
    if d > r1 + r2 + 1e-10 || d < (r1 - r2).abs() - 1e-10 || d < 1e-10 && (r1 - r2).abs() < 1e-10 {
        return Vec::new();
    }

    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h_sq = r1 * r1 - a * a;
    if h_sq < -1e-10 {
        return Vec::new();
    }
    let h = h_sq.max(0.0).sqrt();

    let px = c1.x + a * (c2.x - c1.x) / d;
    let py = c1.y + a * (c2.y - c1.y) / d;

    if h < 1e-10 {
        return vec![Point2D::new(px, py)];
    }

    vec![
        Point2D::new(px + h * (c2.y - c1.y) / d, py - h * (c2.x - c1.x) / d),
        Point2D::new(px - h * (c2.y - c1.y) / d, py + h * (c2.x - c1.x) / d),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_area() {
        let square = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ]);
        assert!((square.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contains_point() {
        let square = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(0.0, 2.0),
        ]);
        assert!(square.contains_point(&Point2D::new(1.0, 1.0)));
        assert!(!square.contains_point(&Point2D::new(3.0, 3.0)));
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.5),
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(1.0, 1.0),
        ];
        let hull = convex_hull(&points);
        assert!(hull.len() <= points.len());
    }

    #[test]
    fn test_line_intersection() {
        let l1 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(2.0, 2.0));
        let l2 = Line2D::new(Point2D::new(0.0, 2.0), Point2D::new(2.0, 0.0));
        let intersection = l1.intersection_point(&l2);
        assert!(intersection.is_some());
        let p = intersection.unwrap();
        assert!((p.x - 1.0).abs() < 1e-10);
        assert!((p.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_intersection() {
        let c1 = Point2D::new(0.0, 0.0);
        let c2 = Point2D::new(1.0, 0.0);
        let points = circle_intersection(&c1, 1.0, &c2, 1.0);
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_closest_pair() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(0.1, 0.1),
            Point2D::new(5.0, 5.0),
        ];
        let (_, _, dist) = closest_pair(&points);
        assert!(dist < 0.2);
    }
}
