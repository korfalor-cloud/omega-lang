/// Entity Component System for game development.

use std::collections::HashMap;
use std::any::{Any, TypeId};

pub type Entity = u64;

#[derive(Debug)]
pub struct World {
    entities: Vec<Entity>,
    next_entity: Entity,
    components: HashMap<TypeId, HashMap<Entity, Box<dyn Any>>>,
    tags: HashMap<Entity, Vec<String>>,
    to_destroy: Vec<Entity>,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            next_entity: 0,
            components: HashMap::new(),
            tags: HashMap::new(),
            to_destroy: Vec::new(),
        }
    }

    pub fn spawn(&mut self) -> Entity {
        let entity = self.next_entity;
        self.next_entity += 1;
        self.entities.push(entity);
        entity
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.to_destroy.push(entity);
    }

    pub fn flush(&mut self) {
        for entity in self.to_destroy.drain(..) {
            self.entities.retain(|&e| e != entity);
            self.tags.remove(&entity);
            for component_map in self.components.values_mut() {
                component_map.remove(&entity);
            }
        }
    }

    pub fn add_component<T: 'static>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();
        self.components
            .entry(type_id)
            .or_insert_with(HashMap::new)
            .insert(entity, Box::new(component));
    }

    pub fn get_component<T: 'static>(&self, entity: Entity) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.components.get(&type_id)?
            .get(&entity)?
            .downcast_ref()
    }

    pub fn get_component_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.components.get_mut(&type_id)?
            .get_mut(&entity)?
            .downcast_mut()
    }

    pub fn remove_component<T: 'static>(&mut self, entity: Entity) -> Option<Box<dyn Any>> {
        let type_id = TypeId::of::<T>();
        self.components.get_mut(&type_id)?.remove(&entity)
    }

    pub fn has_component<T: 'static>(&self, entity: Entity) -> bool {
        let type_id = TypeId::of::<T>();
        self.components.get(&type_id)
            .map_or(false, |map| map.contains_key(&entity))
    }

    pub fn add_tag(&mut self, entity: Entity, tag: &str) {
        self.tags.entry(entity)
            .or_insert_with(Vec::new)
            .push(tag.to_string());
    }

    pub fn has_tag(&self, entity: Entity, tag: &str) -> bool {
        self.tags.get(&entity)
            .map_or(false, |tags| tags.iter().any(|t| t == tag))
    }

    pub fn remove_tag(&mut self, entity: Entity, tag: &str) {
        if let Some(tags) = self.tags.get_mut(&entity) {
            tags.retain(|t| t != tag);
        }
    }

    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn query<T: 'static>(&self) -> Vec<(Entity, &T)> {
        let type_id = TypeId::of::<T>();
        match self.components.get(&type_id) {
            Some(map) => map.iter()
                .filter(|(e, _)| self.entities.contains(e))
                .filter_map(|(e, c)| c.downcast_ref::<T>().map(|c| (*e, c)))
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn query_mut<T: 'static>(&mut self) -> Vec<(Entity, &mut T)> {
        let type_id = TypeId::of::<T>();
        match self.components.get_mut(&type_id) {
            Some(map) => map.iter_mut()
                .filter(|(e, _)| self.entities.contains(e))
                .filter_map(|(e, c)| c.downcast_mut::<T>().map(|c| (*e, c)))
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn query_with_tag<T: 'static>(&self, tag: &str) -> Vec<(Entity, &T)> {
        self.query::<T>().into_iter()
            .filter(|(e, _)| self.has_tag(*e, tag))
            .collect()
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.contains(&entity) && !self.to_destroy.contains(&entity)
    }
}

/// System trait for ECS
pub trait System {
    fn update(&self, world: &mut World, dt: f64);
}

/// Transform component
#[derive(Debug, Clone)]
pub struct Transform {
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn with_rotation(mut self, rotation: f64) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn with_scale(mut self, sx: f64, sy: f64) -> Self {
        self.scale_x = sx;
        self.scale_y = sy;
        self
    }

    pub fn distance_to(&self, other: &Transform) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Velocity component
#[derive(Debug, Clone)]
pub struct Velocity {
    pub dx: f64,
    pub dy: f64,
    pub angular: f64,
}

impl Velocity {
    pub fn new() -> Self {
        Self { dx: 0.0, dy: 0.0, angular: 0.0 }
    }

    pub fn with_velocity(mut self, dx: f64, dy: f64) -> Self {
        self.dx = dx;
        self.dy = dy;
        self
    }

    pub fn speed(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Sprite component
#[derive(Debug, Clone)]
pub struct Sprite {
    pub width: f64,
    pub height: f64,
    pub color: String,
    pub visible: bool,
    pub layer: i32,
}

impl Sprite {
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            color: "white".to_string(),
            visible: true,
            layer: 0,
        }
    }

    pub fn with_color(mut self, color: &str) -> Self {
        self.color = color.to_string();
        self
    }

    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }
}

/// Health component
#[derive(Debug, Clone)]
pub struct Health {
    pub current: f64,
    pub max: f64,
}

impl Health {
    pub fn new(max: f64) -> Self {
        Self { current: max, max }
    }

    pub fn damage(&mut self, amount: f64) {
        self.current = (self.current - amount).max(0.0);
    }

    pub fn heal(&mut self, amount: f64) {
        self.current = (self.current + amount).min(self.max);
    }

    pub fn is_alive(&self) -> bool {
        self.current > 0.0
    }

    pub fn ratio(&self) -> f64 {
        self.current / self.max
    }
}

/// Movement system
pub struct MovementSystem;

impl System for MovementSystem {
    fn update(&self, world: &mut World, dt: f64) {
        let entities: Vec<Entity> = world.entities().to_vec();
        for entity in entities {
            let dx = world.get_component::<Velocity>(entity).map(|v| v.dx * dt);
            let dy = world.get_component::<Velocity>(entity).map(|v| v.dy * dt);
            if let (Some(dx), Some(dy)) = (dx, dy) {
                if let Some(transform) = world.get_component_mut::<Transform>(entity) {
                    transform.x += dx;
                    transform.y += dy;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_entity() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.is_alive(e));
    }

    #[test]
    fn test_components() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_component(e, Transform::new().with_position(10.0, 20.0));

        let transform = world.get_component::<Transform>(e).unwrap();
        assert_eq!(transform.x, 10.0);
    }

    #[test]
    fn test_query() {
        let mut world = World::new();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.add_component(e1, Transform::new());
        world.add_component(e2, Transform::new());

        let results = world.query::<Transform>();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tags() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_tag(e, "player");
        assert!(world.has_tag(e, "player"));
    }

    #[test]
    fn test_despawn() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_component(e, Transform::new());
        world.despawn(e);
        world.flush();
        assert!(!world.is_alive(e));
    }

    #[test]
    fn test_health() {
        let mut health = Health::new(100.0);
        health.damage(30.0);
        assert_eq!(health.current, 70.0);
        health.heal(10.0);
        assert_eq!(health.current, 80.0);
        assert!(health.is_alive());
    }

    #[test]
    fn test_movement_system() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_component(e, Transform::new());
        world.add_component(e, Velocity::new().with_velocity(10.0, 20.0));

        MovementSystem.update(&mut world, 1.0);

        let transform = world.get_component::<Transform>(e).unwrap();
        assert_eq!(transform.x, 10.0);
        assert_eq!(transform.y, 20.0);
    }
}
