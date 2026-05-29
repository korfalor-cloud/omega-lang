/// Advanced game engine: ECS, collision detection, physics, sprite animation, tilemap.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec2 helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < 1e-10 { Self::ZERO } else { self.scale(1.0 / l) }
    }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
}

// ---------------------------------------------------------------------------
// ECS World
// ---------------------------------------------------------------------------

pub type Entity = u64;

#[derive(Debug)]
pub struct World {
    next_id: Entity,
    alive: Vec<Entity>,
    positions: HashMap<Entity, Vec2>,
    velocities: HashMap<Entity, Vec2>,
    colliders: HashMap<Entity, Collider>,
    bodies: HashMap<Entity, PhysicsBody>,
    animations: HashMap<Entity, SpriteAnimation>,
}

impl World {
    pub fn new() -> Self {
        Self { next_id: 0, alive: Vec::new(), positions: HashMap::new(),
               velocities: HashMap::new(), colliders: HashMap::new(),
               bodies: HashMap::new(), animations: HashMap::new() }
    }

    pub fn spawn(&mut self) -> Entity {
        let id = self.next_id; self.next_id += 1; self.alive.push(id); id
    }
    pub fn despawn(&mut self, e: Entity) {
        self.alive.retain(|&i| i != e);
        self.positions.remove(&e); self.velocities.remove(&e);
        self.colliders.remove(&e); self.bodies.remove(&e);
        self.animations.remove(&e);
    }
    pub fn is_alive(&self, e: Entity) -> bool { self.alive.contains(&e) }
    pub fn entity_count(&self) -> usize { self.alive.len() }

    pub fn set_position(&mut self, e: Entity, p: Vec2) { self.positions.insert(e, p); }
    pub fn get_position(&self, e: Entity) -> Option<&Vec2> { self.positions.get(&e) }
    pub fn get_position_mut(&mut self, e: Entity) -> Option<&mut Vec2> { self.positions.get_mut(&e) }
    pub fn set_velocity(&mut self, e: Entity, v: Vec2) { self.velocities.insert(e, v); }
    pub fn get_velocity(&self, e: Entity) -> Option<&Vec2> { self.velocities.get(&e) }
    pub fn get_velocity_mut(&mut self, e: Entity) -> Option<&mut Vec2> { self.velocities.get_mut(&e) }
    pub fn set_collider(&mut self, e: Entity, c: Collider) { self.colliders.insert(e, c); }
    pub fn get_collider(&self, e: Entity) -> Option<&Collider> { self.colliders.get(&e) }
    pub fn set_body(&mut self, e: Entity, b: PhysicsBody) { self.bodies.insert(e, b); }
    pub fn get_body(&self, e: Entity) -> Option<&PhysicsBody> { self.bodies.get(&e) }
    pub fn set_animation(&mut self, e: Entity, a: SpriteAnimation) { self.animations.insert(e, a); }
    pub fn get_animation_mut(&mut self, e: Entity) -> Option<&mut SpriteAnimation> { self.animations.get_mut(&e) }
    pub fn alive(&self) -> &[Entity] { &self.alive }
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PhysicsBody {
    pub mass: f64, pub restitution: f64, pub friction: f64,
    pub is_static: bool, pub gravity_scale: f64,
}

impl PhysicsBody {
    pub fn dynamic(mass: f64) -> Self {
        Self { mass, restitution: 0.5, friction: 0.1, is_static: false, gravity_scale: 1.0 }
    }
    pub fn static_body() -> Self {
        Self { mass: f64::INFINITY, restitution: 0.0, friction: 0.0,
               is_static: true, gravity_scale: 0.0 }
    }
}

pub struct PhysicsSystem { pub gravity: Vec2 }

impl PhysicsSystem {
    pub fn new(gravity: Vec2) -> Self { Self { gravity } }

    pub fn update(&self, world: &mut World, dt: f64) {
        let entities: Vec<Entity> = world.alive().to_vec();
        for e in entities {
            if world.get_body(e).map_or(false, |b| b.is_static) { continue; }
            let gs = world.get_body(e).map_or(1.0, |b| b.gravity_scale);
            let friction = world.get_body(e).map_or(0.1, |b| b.friction);

            if let Some(vel) = world.get_velocity_mut(e) {
                vel.x += self.gravity.x * gs * dt;
                vel.y += self.gravity.y * gs * dt;
                let d = 1.0 - friction * dt;
                vel.x *= d; vel.y *= d;
            }
            let vel = world.get_velocity(e).copied().unwrap_or(Vec2::ZERO);
            if let Some(pos) = world.get_position_mut(e) {
                pos.x += vel.x * dt; pos.y += vel.y * dt;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Collision detection (AABB, Circle)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Collider { Aabb { hw: f64, hh: f64 }, Circle { radius: f64 } }

#[derive(Debug, Clone)]
pub struct Hit { pub a: Entity, pub b: Entity, pub normal: Vec2, pub depth: f64 }

fn aabb_vs_aabb(pa: Vec2, hw1: f64, hh1: f64, pb: Vec2, hw2: f64, hh2: f64)
    -> Option<(Vec2, f64)>
{
    let ox = hw1 + hw2 - (pa.x - pb.x).abs();
    let oy = hh1 + hh2 - (pa.y - pb.y).abs();
    if ox > 0.0 && oy > 0.0 {
        if ox < oy { Some((Vec2::new(if pa.x < pb.x { -1.0 } else { 1.0 }, 0.0), ox)) }
        else       { Some((Vec2::new(0.0, if pa.y < pb.y { -1.0 } else { 1.0 }), oy)) }
    } else { None }
}

fn circle_vs_circle(pa: Vec2, ra: f64, pb: Vec2, rb: f64) -> Option<(Vec2, f64)> {
    let d = pa.distance(pb);
    if d < ra + rb { Some((pb.sub(pa).normalize(), ra + rb - d)) } else { None }
}

fn aabb_vs_circle(pa: Vec2, hw: f64, hh: f64, pc: Vec2, r: f64) -> Option<(Vec2, f64)> {
    let cx = pc.x.clamp(pa.x - hw, pa.x + hw);
    let cy = pc.y.clamp(pa.y - hh, pa.y + hh);
    let diff = pc.sub(Vec2::new(cx, cy));
    let d = diff.length();
    if d < r {
        if d < 1e-10 { Some((Vec2::new(0.0, -1.0), r)) }
        else { Some((diff.normalize(), r - d)) }
    } else { None }
}

pub fn detect_collisions(world: &World) -> Vec<Hit> {
    let mut hits = Vec::new();
    let ents = world.alive();
    for i in 0..ents.len() {
        for j in (i + 1)..ents.len() {
            let (a, b) = (ents[i], ents[j]);
            let pa = match world.get_position(a) { Some(&p) => p, None => continue };
            let pb = match world.get_position(b) { Some(&p) => p, None => continue };
            let ca = match world.get_collider(a) { Some(c) => c.clone(), None => continue };
            let cb = match world.get_collider(b) { Some(c) => c.clone(), None => continue };
            let res = match (&ca, &cb) {
                (Collider::Aabb{hw:w1,hh:h1}, Collider::Aabb{hw:w2,hh:h2}) =>
                    aabb_vs_aabb(pa,*w1,*h1,pb,*w2,*h2),
                (Collider::Circle{radius:r1}, Collider::Circle{radius:r2}) =>
                    circle_vs_circle(pa,*r1,pb,*r2),
                (Collider::Aabb{hw,hh}, Collider::Circle{radius:r}) =>
                    aabb_vs_circle(pa,*hw,*hh,pb,*r),
                (Collider::Circle{radius:r}, Collider::Aabb{hw,hh}) =>
                    aabb_vs_circle(pb,*hw,*hh,pa,*r).map(|(n,d)| (Vec2::new(-n.x,-n.y), d)),
            };
            if let Some((normal, depth)) = res {
                hits.push(Hit { a, b, normal, depth });
            }
        }
    }
    hits
}

pub fn resolve_collisions(world: &mut World, hits: &[Hit]) {
    for h in hits {
        let sa = world.get_body(h.a).map_or(false, |b| b.is_static);
        let sb = world.get_body(h.b).map_or(false, |b| b.is_static);
        if sa && sb { continue; }
        let ma = world.get_body(h.a).map_or(1.0, |b| b.mass);
        let mb = world.get_body(h.b).map_or(1.0, |b| b.mass);
        let total = ma + mb;
        if !sa { let r = mb / total; if let Some(p) = world.get_position_mut(h.a) { p.x -= h.normal.x * h.depth * r; p.y -= h.normal.y * h.depth * r; } }
        if !sb { let r = ma / total; if let Some(p) = world.get_position_mut(h.b) { p.x += h.normal.x * h.depth * r; p.y += h.normal.y * h.depth * r; } }

        let va = world.get_velocity(h.a).copied().unwrap_or(Vec2::ZERO);
        let vb = world.get_velocity(h.b).copied().unwrap_or(Vec2::ZERO);
        let vn = va.sub(vb).dot(h.normal);
        if vn > 0.0 { continue; }
        let j = -(1.0 + 0.5) * vn / (1.0 / ma + 1.0 / mb);
        let imp = h.normal.scale(j);
        if !sa { if let Some(v) = world.get_velocity_mut(h.a) { v.x += imp.x / ma; v.y += imp.y / ma; } }
        if !sb { if let Some(v) = world.get_velocity_mut(h.b) { v.x -= imp.x / mb; v.y -= imp.y / mb; } }
    }
}

// ---------------------------------------------------------------------------
// Sprite animation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpriteAnimation {
    pub frames: Vec<usize>, pub duration: f64, pub elapsed: f64,
    pub current: usize, pub looping: bool, pub finished: bool,
}

impl SpriteAnimation {
    pub fn new(frames: Vec<usize>, duration: f64, looping: bool) -> Self {
        Self { frames, duration, elapsed: 0.0, current: 0, looping, finished: false }
    }
    pub fn update(&mut self, dt: f64) {
        if self.finished { return; }
        self.elapsed += dt;
        while self.elapsed >= self.duration {
            self.elapsed -= self.duration;
            self.current += 1;
            if self.current >= self.frames.len() {
                if self.looping { self.current = 0; }
                else { self.current = self.frames.len() - 1; self.finished = true; return; }
            }
        }
    }
    pub fn frame(&self) -> usize { self.frames[self.current] }
    pub fn reset(&mut self) { self.elapsed = 0.0; self.current = 0; self.finished = false; }
}

pub fn animation_system(world: &mut World, dt: f64) {
    for &e in world.alive().to_vec().iter() {
        if let Some(a) = world.get_animation_mut(e) { a.update(dt); }
    }
}

// ---------------------------------------------------------------------------
// Tilemap
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TileMap {
    pub tiles: Vec<Vec<usize>>, pub tw: f64, pub th: f64,
}

impl TileMap {
    pub fn new(rows: usize, cols: usize, tw: f64, th: f64) -> Self {
        Self { tiles: vec![vec![0; cols]; rows], tw, th }
    }
    pub fn rows(&self) -> usize { self.tiles.len() }
    pub fn cols(&self) -> usize { if self.tiles.is_empty() { 0 } else { self.tiles[0].len() } }
    pub fn set(&mut self, r: usize, c: usize, v: usize) {
        if r < self.rows() && c < self.cols() { self.tiles[r][c] = v; }
    }
    pub fn get(&self, r: usize, c: usize) -> Option<usize> {
        if r < self.rows() && c < self.cols() { Some(self.tiles[r][c]) } else { None }
    }
    pub fn tile_pos(&self, r: usize, c: usize) -> Vec2 {
        Vec2::new(c as f64 * self.tw, r as f64 * self.th)
    }
    pub fn world_to_tile(&self, p: Vec2) -> (usize, usize) {
        ((p.y / self.th).floor().max(0.0) as usize, (p.x / self.tw).floor().max(0.0) as usize)
    }
    pub fn visible(&self, cam_x: f64, cam_y: f64, cam_w: f64, cam_h: f64) -> Vec<(usize, Vec2)> {
        let (r0, c0) = self.world_to_tile(Vec2::new(cam_x, cam_y));
        let (r1, c1) = self.world_to_tile(Vec2::new(cam_x + cam_w, cam_y + cam_h));
        let mut out = Vec::new();
        for r in r0..=r1.min(self.rows().saturating_sub(1)) {
            for c in c0..=c1.min(self.cols().saturating_sub(1)) {
                if self.tiles[r][c] != 0 { out.push((self.tiles[r][c], self.tile_pos(r, c))); }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_basic() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
        assert!((v.normalize().length() - 1.0).abs() < 1e-10);
        assert_eq!(Vec2::new(1.0, 2.0).add(Vec2::new(3.0, 4.0)), Vec2::new(4.0, 6.0));
        assert!((Vec2::new(3.0, 0.0).distance(Vec2::new(0.0, 4.0)) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_spawn_despawn() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.is_alive(e));
        assert_eq!(w.entity_count(), 1);
        w.despawn(e);
        assert!(!w.is_alive(e));
    }

    #[test]
    fn test_components() {
        let mut w = World::new();
        let e = w.spawn();
        w.set_position(e, Vec2::new(10.0, 20.0));
        assert_eq!(w.get_position(e), Some(&Vec2::new(10.0, 20.0)));
    }

    #[test]
    fn test_physics_gravity() {
        let mut w = World::new();
        let e = w.spawn();
        w.set_position(e, Vec2::ZERO);
        w.set_velocity(e, Vec2::ZERO);
        w.set_body(e, PhysicsBody::dynamic(1.0));
        PhysicsSystem::new(Vec2::new(0.0, 100.0)).update(&mut w, 0.016);
        assert!(w.get_position(e).unwrap().y > 0.0);
    }

    #[test]
    fn test_static_body() {
        let mut w = World::new();
        let e = w.spawn();
        w.set_position(e, Vec2::new(5.0, 5.0));
        w.set_velocity(e, Vec2::new(10.0, 10.0));
        w.set_body(e, PhysicsBody::static_body());
        PhysicsSystem::new(Vec2::new(0.0, 100.0)).update(&mut w, 1.0);
        assert_eq!(w.get_position(e).unwrap(), &Vec2::new(5.0, 5.0));
    }

    #[test]
    fn test_aabb_collision() {
        let mut w = World::new();
        let a = w.spawn(); w.set_position(a, Vec2::ZERO); w.set_collider(a, Collider::Aabb { hw: 5.0, hh: 5.0 });
        let b = w.spawn(); w.set_position(b, Vec2::new(8.0, 0.0)); w.set_collider(b, Collider::Aabb { hw: 5.0, hh: 5.0 });
        let hits = detect_collisions(&w);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].depth - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_collision() {
        let mut w = World::new();
        let a = w.spawn(); w.set_position(a, Vec2::ZERO); w.set_collider(a, Collider::Circle { radius: 3.0 });
        let b = w.spawn(); w.set_position(b, Vec2::new(5.0, 0.0)); w.set_collider(b, Collider::Circle { radius: 3.0 });
        let hits = detect_collisions(&w);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].depth - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_vs_circle() {
        let mut w = World::new();
        let a = w.spawn(); w.set_position(a, Vec2::ZERO); w.set_collider(a, Collider::Aabb { hw: 2.0, hh: 2.0 });
        let b = w.spawn(); w.set_position(b, Vec2::new(4.0, 0.0)); w.set_collider(b, Collider::Circle { radius: 3.0 });
        assert_eq!(detect_collisions(&w).len(), 1);
    }

    #[test]
    fn test_no_collision() {
        let mut w = World::new();
        let a = w.spawn(); w.set_position(a, Vec2::ZERO); w.set_collider(a, Collider::Circle { radius: 1.0 });
        let b = w.spawn(); w.set_position(b, Vec2::new(10.0, 0.0)); w.set_collider(b, Collider::Circle { radius: 1.0 });
        assert!(detect_collisions(&w).is_empty());
    }

    #[test]
    fn test_resolve() {
        let mut w = World::new();
        let a = w.spawn(); w.set_position(a, Vec2::ZERO); w.set_velocity(a, Vec2::new(1.0, 0.0));
        w.set_body(a, PhysicsBody::dynamic(1.0)); w.set_collider(a, Collider::Aabb { hw: 5.0, hh: 5.0 });
        let b = w.spawn(); w.set_position(b, Vec2::new(8.0, 0.0)); w.set_velocity(b, Vec2::new(-1.0, 0.0));
        w.set_body(b, PhysicsBody::dynamic(1.0)); w.set_collider(b, Collider::Aabb { hw: 5.0, hh: 5.0 });
        let hits = detect_collisions(&w);
        resolve_collisions(&mut w, &hits);
        assert!(w.get_position(a).unwrap().x < w.get_position(b).unwrap().x);
    }

    #[test]
    fn test_animation_loop() {
        let mut a = SpriteAnimation::new(vec![0, 1, 2], 0.1, true);
        a.update(0.1); assert_eq!(a.frame(), 1);
        a.update(0.1); assert_eq!(a.frame(), 2);
        a.update(0.1); assert_eq!(a.frame(), 0); // loops
        assert!(!a.finished);
    }

    #[test]
    fn test_animation_onshot() {
        let mut a = SpriteAnimation::new(vec![0, 1], 0.1, false);
        a.update(0.1); a.update(0.1);
        assert!(a.finished);
    }

    #[test]
    fn test_animation_reset() {
        let mut a = SpriteAnimation::new(vec![0, 1, 2], 0.1, false);
        a.update(0.25); a.reset();
        assert_eq!(a.current, 0);
        assert!(!a.finished);
    }

    #[test]
    fn test_animation_system() {
        let mut w = World::new();
        let e = w.spawn();
        w.set_animation(e, SpriteAnimation::new(vec![0, 1, 2], 0.1, true));
        animation_system(&mut w, 0.1);
        assert_eq!(w.get_animation_mut(e).unwrap().frame(), 1);
    }

    #[test]
    fn test_tilemap_basics() {
        let mut m = TileMap::new(4, 8, 32.0, 32.0);
        assert_eq!(m.rows(), 4); assert_eq!(m.cols(), 8);
        m.set(1, 2, 5);
        assert_eq!(m.get(1, 2), Some(5));
        assert_eq!(m.get(99, 99), None);
    }

    #[test]
    fn test_tilemap_world_to_tile() {
        let m = TileMap::new(10, 10, 32.0, 32.0);
        let (r, c) = m.world_to_tile(Vec2::new(70.0, 35.0));
        assert_eq!(c, 2); assert_eq!(r, 1);
    }

    #[test]
    fn test_tilemap_culling() {
        let mut m = TileMap::new(10, 10, 32.0, 32.0);
        m.set(1, 1, 3); m.set(5, 5, 7);
        let vis = m.visible(0.0, 0.0, 64.0, 64.0);
        let ids: Vec<usize> = vis.iter().map(|(t,_)| *t).collect();
        assert!(ids.contains(&3));
        assert!(!ids.contains(&7));
    }
}
