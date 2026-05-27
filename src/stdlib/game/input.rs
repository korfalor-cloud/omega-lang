/// Input management system for games.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Space, Enter, Escape, Tab, Backspace, Delete,
    Up, Down, Left, Right,
    LShift, RShift, LCtrl, RCtrl, LAlt, RAlt,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone)]
pub struct InputState {
    pub keys_down: Vec<Key>,
    pub keys_pressed: Vec<Key>,
    pub keys_released: Vec<Key>,
    pub mouse_position: (f64, f64),
    pub mouse_buttons_down: Vec<MouseButton>,
    pub mouse_buttons_pressed: Vec<MouseButton>,
    pub mouse_delta: (f64, f64),
    pub scroll_delta: f64,
}

#[derive(Debug)]
pub struct InputManager {
    current: InputState,
    previous_keys: Vec<Key>,
    previous_mouse_buttons: Vec<MouseButton>,
    axis_mappings: HashMap<String, AxisMapping>,
    action_mappings: HashMap<String, ActionMapping>,
}

#[derive(Debug, Clone)]
pub struct AxisMapping {
    pub positive: Vec<Key>,
    pub negative: Vec<Key>,
    pub sensitivity: f64,
    pub gravity: f64,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct ActionMapping {
    pub keys: Vec<Key>,
    pub pressed: bool,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            current: InputState {
                keys_down: Vec::new(),
                keys_pressed: Vec::new(),
                keys_released: Vec::new(),
                mouse_position: (0.0, 0.0),
                mouse_buttons_down: Vec::new(),
                mouse_buttons_pressed: Vec::new(),
                mouse_delta: (0.0, 0.0),
                scroll_delta: 0.0,
            },
            previous_keys: Vec::new(),
            previous_mouse_buttons: Vec::new(),
            axis_mappings: HashMap::new(),
            action_mappings: HashMap::new(),
        }
    }

    pub fn update(&mut self) {
        // Calculate pressed and released keys
        self.current.keys_pressed = self.current.keys_down.iter()
            .filter(|k| !self.previous_keys.contains(k))
            .cloned()
            .collect();
        self.current.keys_released = self.previous_keys.iter()
            .filter(|k| !self.current.keys_down.contains(k))
            .cloned()
            .collect();

        self.current.mouse_buttons_pressed = self.current.mouse_buttons_down.iter()
            .filter(|b| !self.previous_mouse_buttons.contains(b))
            .cloned()
            .collect();

        // Update axis mappings
        for mapping in self.axis_mappings.values_mut() {
            let positive = mapping.positive.iter().any(|k| self.current.keys_down.contains(k));
            let negative = mapping.negative.iter().any(|k| self.current.keys_down.contains(k));

            let target = if positive { 1.0 } else if negative { -1.0 } else { 0.0 };
            let diff = target - mapping.value;

            if diff.abs() > 0.001 {
                let rate = if target == 0.0 { mapping.gravity } else { mapping.sensitivity };
                mapping.value += diff.signum() * rate * 0.016; // Assume ~60fps
                mapping.value = mapping.value.clamp(-1.0, 1.0);
            } else {
                mapping.value = target;
            }
        }

        self.previous_keys = self.current.keys_down.clone();
        self.previous_mouse_buttons = self.current.mouse_buttons_down.clone();
    }

    pub fn key_down(&mut self, key: Key) {
        if !self.current.keys_down.contains(&key) {
            self.current.keys_down.push(key);
        }
    }

    pub fn key_up(&mut self, key: Key) {
        self.current.keys_down.retain(|&k| k != key);
    }

    pub fn mouse_button_down(&mut self, button: MouseButton) {
        if !self.current.mouse_buttons_down.contains(&button) {
            self.current.mouse_buttons_down.push(button);
        }
    }

    pub fn mouse_button_up(&mut self, button: MouseButton) {
        self.current.mouse_buttons_down.retain(|&b| b != button);
    }

    pub fn set_mouse_position(&mut self, x: f64, y: f64) {
        self.current.mouse_delta = (
            x - self.current.mouse_position.0,
            y - self.current.mouse_position.1,
        );
        self.current.mouse_position = (x, y);
    }

    pub fn set_scroll_delta(&mut self, delta: f64) {
        self.current.scroll_delta = delta;
    }

    pub fn is_key_down(&self, key: Key) -> bool {
        self.current.keys_down.contains(&key)
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.current.keys_pressed.contains(&key)
    }

    pub fn is_key_released(&self, key: Key) -> bool {
        self.current.keys_released.contains(&key)
    }

    pub fn is_mouse_button_down(&self, button: MouseButton) -> bool {
        self.current.mouse_buttons_down.contains(&button)
    }

    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.current.mouse_buttons_pressed.contains(&button)
    }

    pub fn mouse_position(&self) -> (f64, f64) {
        self.current.mouse_position
    }

    pub fn mouse_delta(&self) -> (f64, f64) {
        self.current.mouse_delta
    }

    pub fn scroll_delta(&self) -> f64 {
        self.current.scroll_delta
    }

    pub fn register_axis(&mut self, name: &str, positive: Vec<Key>, negative: Vec<Key>) {
        self.axis_mappings.insert(name.to_string(), AxisMapping {
            positive,
            negative,
            sensitivity: 3.0,
            gravity: 3.0,
            value: 0.0,
        });
    }

    pub fn get_axis(&self, name: &str) -> f64 {
        self.axis_mappings.get(name).map_or(0.0, |m| m.value)
    }

    pub fn register_action(&mut self, name: &str, keys: Vec<Key>) {
        self.action_mappings.insert(name.to_string(), ActionMapping {
            keys,
            pressed: false,
        });
    }

    pub fn is_action_active(&self, name: &str) -> bool {
        self.action_mappings.get(name).map_or(false, |m| {
            m.keys.iter().any(|k| self.current.keys_down.contains(k))
        })
    }

    pub fn is_action_pressed(&self, name: &str) -> bool {
        self.action_mappings.get(name).map_or(false, |m| {
            m.keys.iter().any(|k| self.current.keys_pressed.contains(k))
        })
    }

    pub fn clear(&mut self) {
        self.current.keys_down.clear();
        self.current.mouse_buttons_down.clear();
    }
}

/// Virtual joystick for touch/mobile input
pub struct VirtualJoystick {
    position: (f64, f64),
    delta: (f64, f64),
    radius: f64,
    active: bool,
    dead_zone: f64,
}

impl VirtualJoystick {
    pub fn new(x: f64, y: f64, radius: f64) -> Self {
        Self {
            position: (x, y),
            delta: (0.0, 0.0),
            radius,
            active: false,
            dead_zone: 0.1,
        }
    }

    pub fn touch_down(&mut self, x: f64, y: f64) {
        let dx = x - self.position.0;
        let dy = y - self.position.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= self.radius {
            self.active = true;
            self.update_delta(x, y);
        }
    }

    pub fn touch_move(&mut self, x: f64, y: f64) {
        if self.active {
            self.update_delta(x, y);
        }
    }

    pub fn touch_up(&mut self) {
        self.active = false;
        self.delta = (0.0, 0.0);
    }

    fn update_delta(&mut self, x: f64, y: f64) {
        let dx = (x - self.position.0) / self.radius;
        let dy = (y - self.position.1) / self.radius;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist > 1.0 {
            self.delta = (dx / dist, dy / dist);
        } else if dist < self.dead_zone {
            self.delta = (0.0, 0.0);
        } else {
            self.delta = (dx, dy);
        }
    }

    pub fn delta(&self) -> (f64, f64) {
        self.delta
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn angle(&self) -> f64 {
        self.delta.1.atan2(self.delta.0)
    }

    pub fn magnitude(&self) -> f64 {
        (self.delta.0 * self.delta.0 + self.delta.1 * self.delta.1).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_press() {
        let mut input = InputManager::new();
        input.key_down(Key::W);
        input.update();
        assert!(input.is_key_down(Key::W));
        assert!(input.is_key_pressed(Key::W));
    }

    #[test]
    fn test_key_release() {
        let mut input = InputManager::new();
        input.key_down(Key::W);
        input.update();
        input.key_up(Key::W);
        input.update();
        assert!(!input.is_key_down(Key::W));
        assert!(input.is_key_released(Key::W));
    }

    #[test]
    fn test_axis_mapping() {
        let mut input = InputManager::new();
        input.register_axis("horizontal", vec![Key::D], vec![Key::A]);

        input.key_down(Key::D);
        input.update();
        let val = input.get_axis("horizontal");
        assert!(val > 0.0);
    }

    #[test]
    fn test_mouse_position() {
        let mut input = InputManager::new();
        input.set_mouse_position(100.0, 200.0);
        assert_eq!(input.mouse_position(), (100.0, 200.0));
    }

    #[test]
    fn test_virtual_joystick() {
        let mut joystick = VirtualJoystick::new(100.0, 100.0, 50.0);
        joystick.touch_down(130.0, 100.0);
        assert!(joystick.is_active());
        assert!(joystick.delta().0 > 0.0);
    }
}
