/// 2D/3D particle system with forces and emitters.

#[derive(Debug, Clone)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn length(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-10 { *self } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn dot(&self, other: &Self) -> f64 { self.x * other.x + self.y * other.y }
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

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub mass: f64,
    pub lifetime: f64,
    pub age: f64,
    pub color: (f64, f64, f64, f64), // RGBA
    pub size: f64,
    pub active: bool,
}

impl Particle {
    pub fn new(position: Vec2) -> Self {
        Self {
            position,
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            mass: 1.0,
            lifetime: f64::INFINITY,
            age: 0.0,
            color: (1.0, 1.0, 1.0, 1.0),
            size: 1.0,
            active: true,
        }
    }

    pub fn with_velocity(mut self, v: Vec2) -> Self { self.velocity = v; self }
    pub fn with_mass(mut self, m: f64) -> Self { self.mass = m; self }
    pub fn with_lifetime(mut self, t: f64) -> Self { self.lifetime = t; self }
    pub fn with_color(mut self, r: f64, g: f64, b: f64, a: f64) -> Self { self.color = (r, g, b, a); self }
    pub fn with_size(mut self, s: f64) -> Self { self.size = s; self }

    pub fn apply_force(&mut self, force: Vec2) {
        self.acceleration = self.acceleration + force * (1.0 / self.mass);
    }

    pub fn update(&mut self, dt: f64) {
        if !self.active {
            return;
        }
        self.velocity = self.velocity + self.acceleration * dt;
        self.position = self.position + self.velocity * dt;
        self.acceleration = Vec2::zero();
        self.age += dt;
        if self.age >= self.lifetime {
            self.active = false;
        }
    }

    pub fn alpha(&self) -> f64 {
        if self.lifetime == f64::INFINITY {
            return self.color.3;
        }
        let ratio = (self.age / self.lifetime).clamp(0.0, 1.0);
        self.color.3 * (1.0 - ratio)
    }
}

#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    pub position: Vec2,
    pub rate: f64,
    pub spread: f64,
    pub initial_speed: f64,
    pub initial_speed_variance: f64,
    pub lifetime: f64,
    pub lifetime_variance: f64,
    pub mass: f64,
    pub color: (f64, f64, f64, f64),
    pub size: f64,
    pub gravity: Vec2,
    accumulator: f64,
    pub max_particles: usize,
}

impl ParticleEmitter {
    pub fn new(position: Vec2, rate: f64) -> Self {
        Self {
            position,
            rate,
            spread: std::f64::consts::PI * 2.0,
            initial_speed: 100.0,
            initial_speed_variance: 20.0,
            lifetime: 2.0,
            lifetime_variance: 0.5,
            mass: 1.0,
            color: (1.0, 0.8, 0.2, 1.0),
            size: 3.0,
            gravity: Vec2::new(0.0, -200.0),
            accumulator: 0.0,
            max_particles: 1000,
        }
    }

    pub fn with_spread(mut self, spread: f64) -> Self { self.spread = spread; self }
    pub fn with_speed(mut self, speed: f64) -> Self { self.initial_speed = speed; self }
    pub fn with_lifetime(mut self, lt: f64) -> Self { self.lifetime = lt; self }
    pub fn with_gravity(mut self, g: Vec2) -> Self { self.gravity = g; self }
    pub fn with_color(mut self, r: f64, g: f64, b: f64, a: f64) -> Self { self.color = (r, g, b, a); self }

    fn pseudo_random(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

#[derive(Debug)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub emitters: Vec<ParticleEmitter>,
    pub forces: Vec<Vec2>,
    seed: u64,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            emitters: Vec::new(),
            forces: Vec::new(),
            seed: 12345,
        }
    }

    pub fn add_emitter(&mut self, emitter: ParticleEmitter) {
        self.emitters.push(emitter);
    }

    pub fn add_global_force(&mut self, force: Vec2) {
        self.forces.push(force);
    }

    pub fn update(&mut self, dt: f64) {
        // Emit new particles
        for emitter in &mut self.emitters {
            emitter.accumulator += emitter.rate * dt;
            let seed = &mut self.seed;
            while emitter.accumulator >= 1.0 && self.particles.len() < emitter.max_particles {
                emitter.accumulator -= 1.0;
                let angle = ParticleEmitter::pseudo_random(seed) * emitter.spread - emitter.spread / 2.0;
                let speed = emitter.initial_speed + (ParticleEmitter::pseudo_random(seed) - 0.5) * emitter.initial_speed_variance * 2.0;
                let lt = emitter.lifetime + (ParticleEmitter::pseudo_random(seed) - 0.5) * emitter.lifetime_variance * 2.0;

                let mut p = Particle::new(emitter.position)
                    .with_velocity(Vec2::new(angle.cos() * speed, angle.sin() * speed))
                    .with_mass(emitter.mass)
                    .with_lifetime(lt.max(0.1))
                    .with_color(emitter.color.0, emitter.color.1, emitter.color.2, emitter.color.3)
                    .with_size(emitter.size);
                self.particles.push(p);
            }
        }

        // Update particles
        for particle in &mut self.particles {
            // Apply global forces
            for &force in &self.forces {
                particle.apply_force(force);
            }
            // Apply gravity from emitters (simplified: use first emitter's gravity)
            if let Some(emitter) = self.emitters.first() {
                particle.apply_force(emitter.gravity);
            }
            particle.update(dt);
        }

        // Remove dead particles
        self.particles.retain(|p| p.active);
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Spring force between two points.
pub fn spring_force(position: Vec2, anchor: Vec2, rest_length: f64, stiffness: f64, damping: f64, velocity: Vec2) -> Vec2 {
    let delta = position - anchor;
    let distance = delta.length();
    if distance < 1e-10 {
        return Vec2::zero();
    }
    let direction = delta.normalize();
    let displacement = distance - rest_length;
    let spring = direction * (-stiffness * displacement);
    let damp = velocity * (-damping);
    spring + damp
}

/// Gravitational attraction between two particles.
pub fn gravitational_force(p1: &Particle, p2: &Particle, g_constant: f64) -> Vec2 {
    let delta = p2.position - p1.position;
    let dist_sq = delta.dot(&delta).max(1.0); // prevent singularity
    let force_mag = g_constant * p1.mass * p2.mass / dist_sq;
    delta.normalize() * force_mag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_update() {
        let mut p = Particle::new(Vec2::zero());
        p.apply_force(Vec2::new(100.0, 0.0));
        p.update(1.0);
        assert!(p.position.x > 0.0);
    }

    #[test]
    fn test_particle_lifetime() {
        let mut p = Particle::new(Vec2::zero()).with_lifetime(1.0);
        p.update(0.5);
        assert!(p.active);
        p.update(0.6);
        assert!(!p.active);
    }

    #[test]
    fn test_emitter() {
        let mut system = ParticleSystem::new();
        let emitter = ParticleEmitter::new(Vec2::zero(), 100.0);
        system.add_emitter(emitter);
        system.update(0.1);
        assert!(system.particle_count() > 0);
    }

    #[test]
    fn test_spring_force() {
        let f = spring_force(Vec2::new(10.0, 0.0), Vec2::zero(), 5.0, 1.0, 0.1, Vec2::zero());
        assert!(f.x < 0.0); // Pulls back toward anchor
    }
}
