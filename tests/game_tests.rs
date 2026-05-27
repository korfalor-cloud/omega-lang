use omega_lang::stdlib::game::ecs::{World, Transform, Velocity, Health, MovementSystem};
use omega_lang::stdlib::game::physics::{Vec2, RigidBody, Collider, PhysicsEngine};
use omega_lang::stdlib::game::audio::{AudioClip, AudioEngine};
use omega_lang::stdlib::game::input::{InputManager, Key, MouseButton};

#[test]
fn test_world_spawn() {
    let mut world = World::new();
    let e = world.spawn();
    assert!(world.is_alive(e));
    assert_eq!(world.entity_count(), 1);
}

#[test]
fn test_world_components() {
    let mut world = World::new();
    let e = world.spawn();
    world.add_component(e, Transform::new().with_position(10.0, 20.0));

    let transform = world.get_component::<Transform>(e).unwrap();
    assert_eq!(transform.x, 10.0);
    assert_eq!(transform.y, 20.0);
}

#[test]
fn test_world_query() {
    let mut world = World::new();
    let e1 = world.spawn();
    let e2 = world.spawn();
    world.add_component(e1, Transform::new());
    world.add_component(e2, Transform::new());

    let results = world.query::<Transform>();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_world_tags() {
    let mut world = World::new();
    let e = world.spawn();
    world.add_tag(e, "player");
    assert!(world.has_tag(e, "player"));
    assert!(!world.has_tag(e, "enemy"));
}

#[test]
fn test_world_despawn() {
    let mut world = World::new();
    let e = world.spawn();
    world.add_component(e, Transform::new());
    world.despawn(e);
    world.flush();
    assert!(!world.is_alive(e));
}

#[test]
fn test_health_component() {
    let mut health = Health::new(100.0);
    assert_eq!(health.current, 100.0);
    health.damage(30.0);
    assert_eq!(health.current, 70.0);
    health.heal(10.0);
    assert_eq!(health.current, 80.0);
    assert!(health.is_alive());
    assert!((health.ratio() - 0.8).abs() < 1e-10);
}

#[test]
fn test_velocity() {
    let vel = Velocity::new().with_velocity(10.0, 20.0);
    assert_eq!(vel.dx, 10.0);
    assert_eq!(vel.dy, 20.0);
    assert!((vel.speed() - 22.36).abs() < 0.1);
}

#[test]
fn test_transform_distance() {
    let a = Transform::new().with_position(0.0, 0.0);
    let b = Transform::new().with_position(3.0, 4.0);
    assert_eq!(a.distance_to(&b), 5.0);
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

#[test]
fn test_vec2_operations() {
    let a = Vec2::new(3.0, 4.0);
    assert_eq!(a.length(), 5.0);

    let b = a.normalize();
    assert!((b.length() - 1.0).abs() < 1e-10);

    let c = Vec2::new(1.0, 0.0);
    assert_eq!(a.dot(&c), 3.0);

    let d = a.scale(2.0);
    assert_eq!(d.x, 6.0);
    assert_eq!(d.y, 8.0);
}

#[test]
fn test_rigidbody() {
    let mut body = RigidBody::new(1.0);
    body.velocity = Vec2::new(10.0, 0.0);
    body.update(1.0, &Vec2::zero());
    assert_eq!(body.position.x, 10.0);
}

#[test]
fn test_rigidbody_gravity() {
    let mut body = RigidBody::new(1.0);
    body.update(1.0, &Vec2::new(0.0, 10.0));
    assert!(body.velocity.y > 0.0);
}

#[test]
fn test_static_body() {
    let mut body = RigidBody::static_body();
    body.update(1.0, &Vec2::new(0.0, 10.0));
    assert_eq!(body.velocity.y, 0.0);
    assert_eq!(body.position.x, 0.0);
}

#[test]
fn test_physics_circle_collision() {
    let mut engine = PhysicsEngine::new();
    engine.set_gravity(0.0, 0.0);

    let mut body1 = RigidBody::new(1.0);
    body1.position = Vec2::new(0.0, 0.0);
    let mut body2 = RigidBody::new(1.0);
    body2.position = Vec2::new(1.5, 0.0);

    engine.add_body(body1, Collider::Circle { radius: 1.0 });
    engine.add_body(body2, Collider::Circle { radius: 1.0 });

    engine.step(1.0);
    assert!(!engine.collisions().is_empty());
}

#[test]
fn test_audio_clip() {
    let clip = AudioClip::sine_wave("test", 440.0, 1.0, 44100);
    assert_eq!(clip.data.len(), 44100);
    assert!((clip.duration - 1.0).abs() < 0.01);
}

#[test]
fn test_audio_engine() {
    let mut engine = AudioEngine::new();
    let clip = AudioClip::sine_wave("beep", 440.0, 0.5, 44100);
    engine.load_clip(clip);

    let source = engine.create_source("beep");
    engine.play(source);

    assert!(engine.playing_sources().contains(&source));
}

#[test]
fn test_input_key_press() {
    let mut input = InputManager::new();
    input.key_down(Key::W);
    input.update();
    assert!(input.is_key_down(Key::W));
    assert!(input.is_key_pressed(Key::W));
}

#[test]
fn test_input_key_release() {
    let mut input = InputManager::new();
    input.key_down(Key::W);
    input.update();
    input.key_up(Key::W);
    input.update();
    assert!(!input.is_key_down(Key::W));
    assert!(input.is_key_released(Key::W));
}

#[test]
fn test_input_mouse() {
    let mut input = InputManager::new();
    input.set_mouse_position(100.0, 200.0);
    assert_eq!(input.mouse_position(), (100.0, 200.0));
}

#[test]
fn test_input_axis() {
    let mut input = InputManager::new();
    input.register_axis("horizontal", vec![Key::D], vec![Key::A]);

    input.key_down(Key::D);
    input.update();
    assert!(input.get_axis("horizontal") > 0.0);
}
