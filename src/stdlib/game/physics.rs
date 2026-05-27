/// 2D physics engine for game development.

#[derive(Debug, Clone)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn add(&self, other: &Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    pub fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }

    pub fn scale(&self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }

    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(&self) -> Vec2 {
        let len = self.length();
        if len < 1e-10 {
            Vec2::zero()
        } else {
            self.scale(1.0 / len)
        }
    }

    pub fn distance(&self, other: &Vec2) -> f64 {
        self.sub(other).length()
    }

    pub fn reflect(&self, normal: &Vec2) -> Vec2 {
        let d = self.dot(normal) * 2.0;
        Vec2::new(self.x - d * normal.x, self.y - d * normal.y)
    }
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub mass: f64,
    pub restitution: f64,
    pub friction: f64,
    pub is_static: bool,
    pub gravity_scale: f64,
}

impl RigidBody {
    pub fn new(mass: f64) -> Self {
        Self {
            position: Vec2::zero(),
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            mass,
            restitution: 0.5,
            friction: 0.1,
            is_static: false,
            gravity_scale: 1.0,
        }
    }

    pub fn static_body() -> Self {
        Self {
            is_static: true,
            mass: f64::INFINITY,
            ..Self::new(1.0)
        }
    }

    pub fn apply_force(&mut self, force: &Vec2) {
        if !self.is_static {
            self.acceleration = self.acceleration.add(&force.scale(1.0 / self.mass));
        }
    }

    pub fn apply_impulse(&mut self, impulse: &Vec2) {
        if !self.is_static {
            self.velocity = self.velocity.add(&impulse.scale(1.0 / self.mass));
        }
    }

    pub fn update(&mut self, dt: f64, gravity: &Vec2) {
        if self.is_static {
            return;
        }

        // Apply gravity
        let grav = gravity.scale(self.gravity_scale);
        self.acceleration = self.acceleration.add(&grav);

        // Update velocity
        self.velocity = self.velocity.add(&self.acceleration.scale(dt));

        // Apply friction
        self.velocity = self.velocity.scale(1.0 - self.friction * dt);

        // Update position
        self.position = self.position.add(&self.velocity.scale(dt));

        // Reset acceleration
        self.acceleration = Vec2::zero();
    }
}

#[derive(Debug, Clone)]
pub enum Collider {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Point,
}

#[derive(Debug, Clone)]
pub struct Collision {
    pub entity_a: usize,
    pub entity_b: usize,
    pub normal: Vec2,
    pub depth: f64,
    pub contact_point: Vec2,
}

#[derive(Debug)]
pub struct PhysicsEngine {
    gravity: Vec2,
    bodies: Vec<RigidBody>,
    colliders: Vec<Collider>,
    collisions: Vec<Collision>,
}

impl PhysicsEngine {
    pub fn new() -> Self {
        Self {
            gravity: Vec2::new(0.0, 9.81),
            bodies: Vec::new(),
            colliders: Vec::new(),
            collisions: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, x: f64, y: f64) {
        self.gravity = Vec2::new(x, y);
    }

    pub fn add_body(&mut self, body: RigidBody, collider: Collider) -> usize {
        let id = self.bodies.len();
        self.bodies.push(body);
        self.colliders.push(collider);
        id
    }

    pub fn get_body(&self, id: usize) -> Option<&RigidBody> {
        self.bodies.get(id)
    }

    pub fn get_body_mut(&mut self, id: usize) -> Option<&mut RigidBody> {
        self.bodies.get_mut(id)
    }

    pub fn step(&mut self, dt: f64) {
        // Update bodies
        for body in &mut self.bodies {
            body.update(dt, &self.gravity);
        }

        // Detect collisions
        self.collisions.clear();
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                if let Some(collision) = self.check_collision(i, j) {
                    self.collisions.push(collision);
                }
            }
        }

        // Resolve collisions
        let collisions = self.collisions.clone();
        for collision in &collisions {
            self.resolve_collision(collision);
        }
    }

    fn check_collision(&self, a: usize, b: usize) -> Option<Collision> {
        let body_a = &self.bodies[a];
        let body_b = &self.bodies[b];
        let collider_a = &self.colliders[a];
        let collider_b = &self.colliders[b];

        match (collider_a, collider_b) {
            (Collider::Circle { radius: r1 }, Collider::Circle { radius: r2 }) => {
                let dist = body_a.position.distance(&body_b.position);
                let min_dist = r1 + r2;
                if dist < min_dist {
                    let normal = body_b.position.sub(&body_a.position).normalize();
                    Some(Collision {
                        entity_a: a,
                        entity_b: b,
                        normal,
                        depth: min_dist - dist,
                        contact_point: body_a.position.add(&normal.scale(*r1)),
                    })
                } else {
                    None
                }
            }
            (Collider::Rectangle { width: w1, height: h1 }, Collider::Rectangle { width: w2, height: h2 }) => {
                let dx = (body_a.position.x - body_b.position.x).abs();
                let dy = (body_a.position.y - body_b.position.y).abs();
                let overlap_x = (w1 + w2) / 2.0 - dx;
                let overlap_y = (h1 + h2) / 2.0 - dy;

                if overlap_x > 0.0 && overlap_y > 0.0 {
                    let (normal, depth) = if overlap_x < overlap_y {
                        let nx = if body_a.position.x < body_b.position.x { -1.0 } else { 1.0 };
                        (Vec2::new(nx, 0.0), overlap_x)
                    } else {
                        let ny = if body_a.position.y < body_b.position.y { -1.0 } else { 1.0 };
                        (Vec2::new(0.0, ny), overlap_y)
                    };
                    Some(Collision {
                        entity_a: a,
                        entity_b: b,
                        normal,
                        depth,
                        contact_point: body_a.position.add(&body_b.position).scale(0.5),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn resolve_collision(&mut self, collision: &Collision) {
        let a = collision.entity_a;
        let b = collision.entity_b;

        if self.bodies[a].is_static && self.bodies[b].is_static {
            return;
        }

        // Separate bodies
        let total_mass = self.bodies[a].mass + self.bodies[b].mass;
        if !self.bodies[a].is_static {
            let ratio = self.bodies[b].mass / total_mass;
            self.bodies[a].position = self.bodies[a].position.sub(
                &collision.normal.scale(collision.depth * ratio)
            );
        }
        if !self.bodies[b].is_static {
            let ratio = self.bodies[a].mass / total_mass;
            self.bodies[b].position = self.bodies[b].position.add(
                &collision.normal.scale(collision.depth * ratio)
            );
        }

        // Calculate relative velocity
        let rel_vel = self.bodies[a].velocity.sub(&self.bodies[b].velocity);
        let vel_along_normal = rel_vel.dot(&collision.normal);

        // Don't resolve if velocities are separating
        if vel_along_normal > 0.0 {
            return;
        }

        // Calculate restitution
        let e = self.bodies[a].restitution.min(self.bodies[b].restitution);
        let j = -(1.0 + e) * vel_along_normal / (1.0 / self.bodies[a].mass + 1.0 / self.bodies[b].mass);

        let impulse = collision.normal.scale(j);
        if !self.bodies[a].is_static {
            self.bodies[a].velocity = self.bodies[a].velocity.add(&impulse.scale(1.0 / self.bodies[a].mass));
        }
        if !self.bodies[b].is_static {
            self.bodies[b].velocity = self.bodies[b].velocity.sub(&impulse.scale(1.0 / self.bodies[b].mass));
        }
    }

    pub fn collisions(&self) -> &[Collision] {
        &self.collisions
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length(), 5.0);

        let b = a.normalize();
        assert!((b.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rigidbody_update() {
        let mut body = RigidBody::new(1.0);
        body.velocity = Vec2::new(10.0, 0.0);
        body.update(1.0, &Vec2::zero());
        assert_eq!(body.position.x, 10.0);
    }

    #[test]
    fn test_circle_collision() {
        let mut engine = PhysicsEngine::new();
        engine.set_gravity(0.0, 0.0);

        let mut body1 = RigidBody::new(1.0);
        body1.position = Vec2::new(0.0, 0.0);
        let mut body2 = RigidBody::new(1.0);
        body2.position = Vec2::new(1.5, 0.0);

        engine.add_body(body1, Collider::Circle { radius: 1.0 });
        engine.add_body(body2, Collider::Circle { radius: 1.0 });

        engine.step(1.0);
        assert!(!engine.collisions.is_empty());
    }

    #[test]
    fn test_static_body() {
        let body = RigidBody::static_body();
        assert!(body.is_static);
        assert!(body.mass.is_infinite());
    }

    #[test]
    fn test_reflect() {
        let vel = Vec2::new(1.0, -1.0);
        let normal = Vec2::new(0.0, 1.0);
        let reflected = vel.reflect(&normal);
        assert!((reflected.x - 1.0).abs() < 1e-10);
        assert!((reflected.y - 1.0).abs() < 1e-10);
    }
}
