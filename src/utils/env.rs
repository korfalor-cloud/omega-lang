use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};

pub fn get_var(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

pub fn get_var_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

pub fn set_var(key: &str, value: &str) {
    std::env::set_var(key, value);
}

pub fn remove_var(key: &str) {
    std::env::remove_var(key);
}

pub fn vars() -> HashMap<String, String> {
    std::env::vars().collect()
}

pub fn current_dir() -> OmegaResult<String> {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn set_current_dir(path: &str) -> OmegaResult<()> {
    std::env::set_current_dir(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn home_dir() -> Option<String> {
    dirs::home_dir().map(|p| p.to_string_lossy().to_string())
}

pub fn temp_dir() -> String {
    std::env::temp_dir().to_string_lossy().to_string()
}

pub fn args() -> Vec<String> {
    std::env::args().collect()
}

pub fn os() -> &'static str {
    std::env::consts::OS
}

pub fn arch() -> &'static str {
    std::env::consts::ARCH
}

pub fn family() -> &'static str {
    std::env::consts::FAMILY
}

pub fn hostname() -> Option<String> {
    hostname::get().ok().map(|h| h.to_string_lossy().to_string())
}

pub fn num_cpus() -> usize {
    num_cpus::get()
}

pub fn pid() -> u32 {
    std::process::id()
}

pub fn exit(code: i32) -> ! {
    std::process::exit(code)
}
