/// 2D physics engine: rigid bodies, collision detection, constraints, joints.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn dot(self, other: Self) -> f64 { self.x * other.x + self.y * other.y }
    pub fn cross(self, other: Self) -> f64 { self.x * other.y - self.y * other.x }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_squared(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn perp(self) -> Self { Self { x: -self.y, y: self.x } }
    pub fn rotate(self, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self { x: self.x * c - self.y * y: self.x * s + self.y * c }
    }
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self { x: self.x + (other.x - self.x) * t, y: self.y + (other.y - self.y) * t }
    }
    pub fn distance(self, other: Self) -> f64 { (self - other).length() }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y } }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y } }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self { x: self.x * rhs, y: self.y * rhs } }
}

impl std::ops::Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y } }
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub angle: f64,
    pub angular_velocity: f64,
    pub mass: f64,
    pub inertia: f64,
    pub restitution: f64,
    pub friction: f64,
    pub is_static: bool,
    pub shape: Shape,
    pub force: Vec2,
    pub torque: f64,
}

#[derive(Debug, Clone)]
pub enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Polygon { vertices: Vec<Vec2> },
}

#[derive(Debug, Clone)]
pub struct Contact {
    pub body_a: usize,
    pub body_b: usize,
    pub normal: Vec2,
    pub depth: f64,
    pub point: Vec2,
}

#[derive(Debug, Clone)]
pub struct Joint {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_a: Vec2,
    pub anchor_b: Vec2,
    pub joint_type: JointType,
}

#[derive(Debug, Clone)]
pub enum JointType {
    Distance { length: f64, stiffness: f64 },
    Revolute { min_angle: f64, max_angle: f64 },
    Weld { angle: f64 },
}

pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    joints: Vec<Joint>,
    gravity: Vec2,
    dt: f64,
    damping: f64,
    iterations: usize,
}

impl PhysicsWorld {
    pub fn new(gravity: Vec2, dt: f64) -> Self {
        Self {
            bodies: Vec::new(),
            joints: Vec::new(),
            gravity,
            dt,
            damping: 0.999,
            iterations: 10,
        }
    }

    pub fn add_body(&mut self, mut body: RigidBody) -> usize {
        let id = self.bodies.len();
        body.id = id;
        if body.mass > 0.0 {
            body.inertia = match &body.shape {
                Shape::Circle { radius } => 0.5 * body.mass * radius * radius,
                Shape::Rectangle { width, height } => {
                    body.mass * (width * width + height * height) / 12.0
                }
                Shape::Polygon { vertices } => {
                    let mut i = 0.0;
                    for v in vertices {
                        i += v.length_squared();
                    }
                    body.mass * i / vertices.len() as f64
                }
            };
        }
        self.bodies.push(body);
        id
    }

    pub fn add_joint(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    pub fn body(&self, id: usize) -> &RigidBody { &self.bodies[id] }
    pub fn body_mut(&mut self, id: usize) -> &mut RigidBody { &mut self.bodies[id] }
    pub fn body_count(&self) -> usize { self.bodies.len() }

    pub fn apply_force(&mut self, id: usize, force: Vec2) {
        self.bodies[id].force = self.bodies[id].force + force;
    }

    pub fn apply_impulse(&mut self, id: usize, impulse: Vec2, point: Vec2) {
        let body = &mut self.bodies[id];
        if body.is_static { return; }
        body.velocity = body.velocity + impulse * (1.0 / body.mass);
        let r = point - body.position;
        body.angular_velocity += r.cross(impulse) / body.inertia;
    }

    pub fn step(&mut self) {
        // Apply gravity
        for body in &mut self.bodies {
            if !body.is_static {
                body.force = body.force + self.gravity * body.mass;
            }
        }

        // Integrate velocities
        for body in &mut self.bodies {
            if body.is_static { continue; }
            body.velocity = body.velocity + body.force * (self.dt / body.mass);
            body.angular_velocity += body.torque * self.dt / body.inertia;
            body.velocity = body.velocity * self.damping;
            body.angular_velocity *= self.damping;
        }

        // Detect collisions
        let contacts = self.detect_collisions();

        // Solve constraints
        for _ in 0..self.iterations {
            for contact in &contacts {
                self.resolve_contact(contact);
            }
            for joint in &self.joints {
                self.solve_joint(joint);
            }
        }

        // Integrate positions
        for body in &mut self.bodies {
            if body.is_static { continue; }
            body.position = body.position + body.velocity * self.dt;
            body.angle += body.angular_velocity * self.dt;
            body.force = Vec2::zero();
            body.torque = 0.0;
        }
    }

    fn detect_collisions(&self) -> Vec<Contact> {
        let mut contacts = Vec::new();
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                if self.bodies[i].is_static && self.bodies[j].is_static {
                    continue;
                }
                if let Some(contact) = self.check_collision(i, j) {
                    contacts.push(contact);
                }
            }
        }
        contacts
    }

    fn check_collision(&self, a: usize, b: usize) -> Option<Contact> {
        let body_a = &self.bodies[a];
        let body_b = &self.bodies[b];

        match (&body_a.shape, &body_b.shape) {
            (Shape::Circle { radius: ra }, Shape::Circle { radius: rb }) => {
                let diff = body_b.position - body_a.position;
                let dist = diff.length();
                let depth = ra + rb - dist;
                if depth > 0.0 {
                    let normal = if dist < 1e-10 { Vec2::new(1.0, 0.0) } else { diff.normalize() };
                    Some(Contact {
                        body_a: a,
                        body_b: b,
                        normal,
                        depth,
                        point: body_a.position + normal * *ra,
                    })
                } else {
                    None
                }
            }
            (Shape::Rectangle { width: wa, height: ha }, Shape::Rectangle { width: wb, height: hb }) => {
                let aabb_a = AABB::from_body(body_a, *wa, *ha);
                let aabb_b = AABB::from_body(body_b, *wb, *hb);
                aabb_a.intersect(&aabb_b).map(|(normal, depth, point)| Contact {
                    body_a: a, body_b: b, normal, depth, point,
                })
            }
            (Shape::Circle { radius }, Shape::Rectangle { width, height }) => {
                self.circle_rect_collision(body_a, *radius, body_b, *width, *height, a, b)
            }
            (Shape::Rectangle { width, height }, Shape::Circle { radius }) => {
                self.circle_rect_collision(body_b, *radius, body_a, *width, *height, b, a)
                    .map(|mut c| { std::mem::swap(&mut c.body_a, &mut c.body_b); c.normal = -c.normal; c })
            }
            _ => None, // Polygon collisions omitted for brevity
        }
    }

    fn circle_rect_collision(
        &self, circle: &RigidBody, radius: f64,
        rect: &RigidBody, width: f64, height: f64,
        circle_id: usize, rect_id: usize,
    ) -> Option<Contact> {
        let local = (circle.position - rect.position).rotate(-rect.angle);
        let half_w = width / 2.0;
        let half_h = height / 2.0;

        let closest = Vec2::new(
            local.x.clamp(-half_w, half_w),
            local.y.clamp(-half_h, half_h),
        );

        let diff = local - closest;
        let dist = diff.length();

        if dist < radius {
            let normal = if dist < 1e-10 {
                Vec2::new(1.0, 0.0)
            } else {
                diff.normalize().rotate(rect.angle)
            };
            Some(Contact {
                body_a: circle_id,
                body_b: rect_id,
                normal,
                depth: radius - dist,
                point: circle.position - normal * radius,
            })
        } else {
            None
        }
    }

    fn resolve_contact(&mut self, contact: &Contact) {
        let body_a = &self.bodies[contact.body_a];
        let body_b = &self.bodies[contact.body_b];

        let inv_mass_a = if body_a.is_static { 0.0 } else { 1.0 / body_a.mass };
        let inv_mass_b = if body_b.is_static { 0.0 } else { 1.0 / body_b.mass };
        let inv_mass_sum = inv_mass_a + inv_mass_b;
        if inv_mass_sum < 1e-10 { return; }

        // Position correction (Baumgarte stabilization)
        let correction = contact.normal * (contact.depth / inv_mass_sum * 0.8);
        if !body_a.is_static {
            self.bodies[contact.body_a].position = self.bodies[contact.body_a].position - correction * inv_mass_a;
        }
        if !body_b.is_static {
            self.bodies[contact.body_b].position = self.bodies[contact.body_b].position + correction * inv_mass_b;
        }

        // Relative velocity at contact point
        let ra = contact.point - self.bodies[contact.body_a].position;
        let rb = contact.point - self.bodies[contact.body_b].position;
        let vel_a = self.bodies[contact.body_a].velocity + ra.perp() * self.bodies[contact.body_a].angular_velocity;
        let vel_b = self.bodies[contact.body_b].velocity + rb.perp() * self.bodies[contact.body_b].angular_velocity;
        let relative_vel = vel_b - vel_a;
        let vel_along_normal = relative_vel.dot(contact.normal);

        if vel_along_normal > 0.0 { return; } // Separating

        let restitution = self.bodies[contact.body_a].restitution.min(self.bodies[contact.body_b].restitution);
        let ra_cross = ra.cross(contact.normal);
        let rb_cross = rb.cross(contact.normal);
        let inv_inertia_a = if self.bodies[contact.body_a].is_static { 0.0 } else { 1.0 / self.bodies[contact.body_a].inertia };
        let inv_inertia_b = if self.bodies[contact.body_b].is_static { 0.0 } else { 1.0 / self.bodies[contact.body_b].inertia };

        let angular_term = ra_cross * ra_cross * inv_inertia_a + rb_cross * rb_cross * inv_inertia_b;
        let j = -(1.0 + restitution) * vel_along_normal / (inv_mass_sum + angular_term);
        let impulse = contact.normal * j;

        if !self.bodies[contact.body_a].is_static {
            self.bodies[contact.body_a].velocity = self.bodies[contact.body_a].velocity - impulse * inv_mass_a;
            self.bodies[contact.body_a].angular_velocity -= ra_cross * j * inv_inertia_a;
        }
        if !self.bodies[contact.body_b].is_static {
            self.bodies[contact.body_b].velocity = self.bodies[contact.body_b].velocity + impulse * inv_mass_b;
            self.bodies[contact.body_b].angular_velocity += rb_cross * j * inv_inertia_b;
        }

        // Friction
        let mut tangent = relative_vel - contact.normal * vel_along_normal;
        let tl = tangent.length();
        if tl > 1e-10 {
            tangent = tangent * (1.0 / tl);
        } else {
            return;
        }

        let friction_coeff = (self.bodies[contact.body_a].friction * self.bodies[contact.body_b].friction).sqrt();
        let jt = -relative_vel.dot(tangent) / (inv_mass_sum + ra.cross(tangent).powi(2) * inv_inertia_a + rb.cross(tangent).powi(2) * inv_inertia_b);

        let friction_impulse = if jt.abs() < j * friction_coeff {
            tangent * jt
        } else {
            tangent * (-j * friction_coeff)
        };

        if !self.bodies[contact.body_a].is_static {
            self.bodies[contact.body_a].velocity = self.bodies[contact.body_a].velocity - friction_impulse * inv_mass_a;
        }
        if !self.bodies[contact.body_b].is_static {
            self.bodies[contact.body_b].velocity = self.bodies[contact.body_b].velocity + friction_impulse * inv_mass_b;
        }
    }

    fn solve_joint(&mut self, joint: &Joint) {
        let anchor_a_world = self.bodies[joint.body_a].position + joint.anchor_a.rotate(self.bodies[joint.body_a].angle);
        let anchor_b_world = self.bodies[joint.body_b].position + joint.anchor_b.rotate(self.bodies[joint.body_b].angle);
        let diff = anchor_b_world - anchor_a_world;

        match &joint.joint_type {
            JointType::Distance { length, stiffness } => {
                let current_len = diff.length();
                if current_len < 1e-10 { return; }
                let error = current_len - length;
                let normal = diff.normalize();
                let force = normal * (error * stiffness);

                if !self.bodies[joint.body_a].is_static {
                    self.bodies[joint.body_a].velocity = self.bodies[joint.body_a].velocity + force * (self.dt / self.bodies[joint.body_a].mass);
                }
                if !self.bodies[joint.body_b].is_static {
                    self.bodies[joint.body_b].velocity = self.bodies[joint.body_b].velocity - force * (self.dt / self.bodies[joint.body_b].mass);
                }
            }
            JointType::Revolute { .. } => {
                // Position correction for revolute joint
                let inv_a = if self.bodies[joint.body_a].is_static { 0.0 } else { 1.0 / self.bodies[joint.body_a].mass };
                let inv_b = if self.bodies[joint.body_b].is_static { 0.0 } else { 1.0 / self.bodies[joint.body_b].mass };
                let total = inv_a + inv_b;
                if total < 1e-10 { return; }

                let correction = diff * (1.0 / total);
                if !self.bodies[joint.body_a].is_static {
                    self.bodies[joint.body_a].position = self.bodies[joint.body_a].position + correction * inv_a;
                }
                if !self.bodies[joint.body_b].is_static {
                    self.bodies[joint.body_b].position = self.bodies[joint.body_b].position - correction * inv_b;
                }
            }
            JointType::Weld { angle: target_angle } => {
                let angle_diff = self.bodies[joint.body_b].angle - self.bodies[joint.body_a].angle - target_angle;
                let inv_a = if self.bodies[joint.body_a].is_static { 0.0 } else { 1.0 / self.bodies[joint.body_a].inertia };
                let inv_b = if self.bodies[joint.body_b].is_static { 0.0 } else { 1.0 / self.bodies[joint.body_b].inertia };
                let total = inv_a + inv_b;
                if total < 1e-10 { return; }

                let correction = angle_diff / total;
                if !self.bodies[joint.body_a].is_static {
                    self.bodies[joint.body_a].angular_velocity += correction * inv_a * 10.0;
                }
                if !self.bodies[joint.body_b].is_static {
                    self.bodies[joint.body_b].angular_velocity -= correction * inv_b * 10.0;
                }
            }
        }
    }

    pub fn raycast(&self, origin: Vec2, direction: Vec2, max_dist: f64) -> Option<(usize, f64, Vec2)> {
        let dir = direction.normalize();
        let mut closest = None;
        let mut closest_t = max_dist;

        for body in &self.bodies {
            match &body.shape {
                Shape::Circle { radius } => {
                    let oc = origin - body.position;
                    let a = dir.dot(dir);
                    let b = 2.0 * oc.dot(dir);
                    let c = oc.dot(oc) - radius * radius;
                    let disc = b * b - 4.0 * a * c;
                    if disc >= 0.0 {
                        let t = (-b - disc.sqrt()) / (2.0 * a);
                        if t > 0.0 && t < closest_t {
                            closest_t = t;
                            closest = Some((body.id, t, origin + dir * t));
                        }
                    }
                }
                _ => {} // Omitted for brevity
            }
        }
        closest
    }
}

#[derive(Debug, Clone)]
struct AABB {
    min: Vec2,
    max: Vec2,
}

impl AABB {
    fn from_body(body: &RigidBody, width: f64, height: f64) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;
        // Simplified: ignore rotation for AABB
        Self {
            min: Vec2::new(body.position.x - hw, body.position.y - hh),
            max: Vec2::new(body.position.x + hw, body.position.y + hh),
        }
    }

    fn intersect(&self, other: &AABB) -> Option<(Vec2, f64, Vec2)> {
        let overlap_x = (self.max.x.min(other.max.x)) - (self.min.x.max(other.min.x));
        let overlap_y = (self.max.y.min(other.max.y)) - (self.min.y.max(other.min.y));

        if overlap_x > 0.0 && overlap_y > 0.0 {
            if overlap_x < overlap_y {
                let normal = if self.min.x < other.min.x { Vec2::new(1.0, 0.0) } else { Vec2::new(-1.0, 0.0) };
                let point = Vec2::new(
                    if self.min.x < other.min.x { self.max.x } else { self.min.x },
                    (self.min.y.max(other.min.y) + self.max.y.min(other.max.y)) / 2.0,
                );
                Some((normal, overlap_x, point))
            } else {
                let normal = if self.min.y < other.min.y { Vec2::new(0.0, 1.0) } else { Vec2::new(0.0, -1.0) };
                let point = Vec2::new(
                    (self.min.x.max(other.min.x) + self.max.x.min(other.max.x)) / 2.0,
                    if self.min.y < other.min.y { self.max.y } else { self.min.y },
                );
                Some((normal, overlap_y, point))
            }
        } else {
            None
        }
    }
}

/// Spring-damper system.
pub struct Spring {
    pub body_a: usize,
    pub body_b: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
}

impl Spring {
    pub fn force(&self, body_a: &RigidBody, body_b: &RigidBody) -> (Vec2, Vec2) {
        let diff = body_b.position - body_a.position;
        let dist = diff.length();
        if dist < 1e-10 { return (Vec2::zero(), Vec2::zero()); }

        let dir = diff * (1.0 / dist);
        let stretch = dist - self.rest_length;
        let relative_vel = body_b.velocity - body_a.velocity;
        let vel_along = relative_vel.dot(dir);

        let force_mag = self.stiffness * stretch + self.damping * vel_along;
        let force = dir * force_mag;
        (force, -force)
    }
}

/// Particle system for soft bodies / fluids.
pub struct ParticleSystem {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub forces: Vec<Vec2>,
    pub masses: Vec<f64>,
    springs: Vec<(usize, usize, f64, f64)>, // (a, b, rest, stiffness)
    pub gravity: Vec2,
    pub damping: f64,
}

impl ParticleSystem {
    pub fn new(gravity: Vec2) -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            forces: Vec::new(),
            masses: Vec::new(),
            springs: Vec::new(),
            gravity,
            damping: 0.99,
        }
    }

    pub fn add_particle(&mut self, pos: Vec2, mass: f64) -> usize {
        let id = self.positions.len();
        self.positions.push(pos);
        self.velocities.push(Vec2::zero());
        self.forces.push(Vec2::zero());
        self.masses.push(mass);
        id
    }

    pub fn add_spring(&mut self, a: usize, b: usize, rest_length: f64, stiffness: f64) {
        self.springs.push((a, b, rest_length, stiffness));
    }

    pub fn step(&mut self, dt: f64) {
        // Reset forces
        for f in &mut self.forces {
            *f = Vec2::zero();
        }

        // Apply gravity
        for i in 0..self.positions.len() {
            self.forces[i] = self.forces[i] + self.gravity * self.masses[i];
        }

        // Apply spring forces
        for &(a, b, rest, stiffness) in &self.springs {
            let diff = self.positions[b] - self.positions[a];
            let dist = diff.length();
            if dist < 1e-10 { continue; }
            let dir = diff * (1.0 / dist);
            let force = dir * (stiffness * (dist - rest));
            self.forces[a] = self.forces[a] + force;
            self.forces[b] = self.forces[b] - force;
        }

        // Integrate
        for i in 0..self.positions.len() {
            let acc = self.forces[i] * (1.0 / self.masses[i]);
            self.velocities[i] = (self.velocities[i] + acc * dt) * self.damping;
            self.positions[i] = self.positions[i] + self.velocities[i] * dt;
        }
    }
}

/// Verlet integration for position-based dynamics.
pub struct VerletSystem {
    pub positions: Vec<Vec2>,
    pub old_positions: Vec<Vec2>,
    pub masses: Vec<f64>,
    pub pinned: Vec<bool>,
    constraints: Vec<(usize, usize, f64)>,
    pub gravity: Vec2,
    pub damping: f64,
}

impl VerletSystem {
    pub fn new(gravity: Vec2) -> Self {
        Self {
            positions: Vec::new(),
            old_positions: Vec::new(),
            masses: Vec::new(),
            pinned: Vec::new(),
            constraints: Vec::new(),
            gravity,
            damping: 0.99,
        }
    }

    pub fn add_particle(&mut self, pos: Vec2, mass: f64, pinned: bool) -> usize {
        let id = self.positions.len();
        self.positions.push(pos);
        self.old_positions.push(pos);
        self.masses.push(mass);
        self.pinned.push(pinned);
        id
    }

    pub fn add_constraint(&mut self, a: usize, b: usize, length: f64) {
        self.constraints.push((a, b, length));
    }

    pub fn step(&mut self, dt: f64) {
        // Verlet integration
        for i in 0..self.positions.len() {
            if self.pinned[i] { continue; }
            let vel = (self.positions[i] - self.old_positions[i]) * self.damping;
            self.old_positions[i] = self.positions[i];
            self.positions[i] = self.positions[i] + vel + self.gravity * (dt * dt);
        }

        // Solve constraints
        for _ in 0..5 {
            for &(a, b, length) in &self.constraints {
                let diff = self.positions[b] - self.positions[a];
                let dist = diff.length();
                if dist < 1e-10 { continue; }
                let correction = diff * ((dist - length) / dist * 0.5);

                if !self.pinned[a] {
                    self.positions[a] = self.positions[a] + correction;
                }
                if !self.pinned[b] {
                    self.positions[b] = self.positions[b] - correction;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_collision() {
        let mut world = PhysicsWorld::new(Vec2::new(0.0, -9.81), 0.016);
        let a = world.add_body(RigidBody {
            id: 0,
            position: Vec2::new(0.0, 0.0),
            velocity: Vec2::new(1.0, 0.0),
            acceleration: Vec2::zero(),
            angle: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            inertia: 0.0,
            restitution: 0.8,
            friction: 0.5,
            is_static: false,
            shape: Shape::Circle { radius: 1.0 },
            force: Vec2::zero(),
            torque: 0.0,
        });
        let b = world.add_body(RigidBody {
            id: 0,
            position: Vec2::new(2.5, 0.0),
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            angle: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            inertia: 0.0,
            restitution: 0.8,
            friction: 0.5,
            is_static: false,
            shape: Shape::Circle { radius: 1.0 },
            force: Vec2::zero(),
            torque: 0.0,
        });

        for _ in 0..100 {
            world.step();
        }

        // Bodies should have separated after collision
        let diff = world.body(b).position - world.body(a).position;
        assert!(diff.length() > 1.5);
    }

    #[test]
    fn test_spring() {
        let mut ps = ParticleSystem::new(Vec2::new(0.0, 0.0));
        let a = ps.add_particle(Vec2::new(0.0, 0.0), 1.0);
        let b = ps.add_particle(Vec2::new(2.0, 0.0), 1.0);
        ps.add_spring(a, b, 1.0, 10.0);

        for _ in 0..1000 {
            ps.step(0.001);
        }

        // Particles should settle near rest length
        let dist = (ps.positions[b] - ps.positions[a]).length();
        assert!((dist - 1.0).abs() < 0.1);
    }
}
