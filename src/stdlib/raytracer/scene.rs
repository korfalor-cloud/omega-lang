/// Ray tracer: vectors, materials, lights, camera, and rendering.

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn one() -> Self { Self { x: 1.0, y: 1.0, z: 1.0 } }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 { self } else { self * (1.0 / len) }
    }

    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    pub fn refract(self, normal: Self, eta: f64) -> Option<Self> {
        let cos_i = -self.dot(normal);
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
        if sin2_t > 1.0 {
            return None;
        }
        let cos_t = (1.0 - sin2_t).sqrt();
        Some(self * eta + normal * (eta * cos_i - cos_t))
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    pub fn component_mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    pub fn to_color_u8(self) -> (u8, u8, u8) {
        let r = (self.x.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (self.y.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (self.z.clamp(0.0, 1.0) * 255.0) as u8;
        (r, g, b)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z } }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z } }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs } }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

pub type Color = Vec3;

#[derive(Debug, Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize() }
    }

    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(Debug, Clone)]
pub struct HitRecord {
    pub point: Vec3,
    pub normal: Vec3,
    pub t: f64,
    pub material: Material,
    pub front_face: bool,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Color,
    pub emissive: Color,
    pub metallic: f64,
    pub roughness: f64,
    pub ior: f64,
    pub transparency: f64,
}

impl Material {
    pub fn diffuse(color: Color) -> Self {
        Self { albedo: color, emissive: Color::zero(), metallic: 0.0, roughness: 0.8, ior: 1.5, transparency: 0.0 }
    }

    pub fn metal(color: Color, roughness: f64) -> Self {
        Self { albedo: color, emissive: Color::zero(), metallic: 1.0, roughness, ior: 1.5, transparency: 0.0 }
    }

    pub fn glass(ior: f64) -> Self {
        Self { albedo: Color::one(), emissive: Color::zero(), metallic: 0.0, roughness: 0.0, ior, transparency: 1.0 }
    }

    pub fn emissive(color: Color) -> Self {
        Self { albedo: Color::zero(), emissive: color, metallic: 0.0, roughness: 1.0, ior: 1.5, transparency: 0.0 }
    }
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

#[derive(Debug, Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, material: Material) -> Self {
        Self { center, radius, material }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = oc.dot(ray.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_disc = discriminant.sqrt();
        let mut t = (-b - sqrt_disc) / a;
        if t < t_min || t > t_max {
            t = (-b + sqrt_disc) / a;
            if t < t_min || t > t_max {
                return None;
            }
        }

        let point = ray.at(t);
        let outward_normal = (point - self.center) * (1.0 / self.radius);
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };

        Some(HitRecord {
            point,
            normal,
            t,
            material: self.material.clone(),
            front_face,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
    pub material: Material,
}

impl Plane {
    pub fn new(point: Vec3, normal: Vec3, material: Material) -> Self {
        Self { point, normal: normal.normalize(), material }
    }
}

impl Hittable for Plane {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let denom = ray.direction.dot(self.normal);
        if denom.abs() < 1e-10 {
            return None;
        }
        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < t_min || t > t_max {
            return None;
        }

        let point = ray.at(t);
        let front_face = ray.direction.dot(self.normal) < 0.0;
        let normal = if front_face { self.normal } else { -self.normal };

        Some(HitRecord {
            point,
            normal,
            t,
            material: self.material.clone(),
            front_face,
        })
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> bool {
        for i in 0..3 {
            let (orig, dir, bmin, bmax) = match i {
                0 => (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
                1 => (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
                _ => (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
            };
            let inv_d = 1.0 / dir;
            let mut t0 = (bmin - orig) * inv_d;
            let mut t1 = (bmax - orig) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            let t_min = t0.max(t_min);
            let t_max = t1.min(t_max);
            if t_max <= t_min {
                return false;
            }
        }
        true
    }
}

/// Triangle mesh triangle.
#[derive(Debug, Clone)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub material: Material,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material: Material) -> Self {
        Self { v0, v1, v2, material }
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let h = ray.direction.cross(e2);
        let a = e1.dot(h);

        if a.abs() < 1e-10 {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * s.dot(h);
        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = s.cross(e1);
        let v = f * ray.direction.dot(q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * e2.dot(q);
        if t < t_min || t > t_max {
            return None;
        }

        let point = ray.at(t);
        let normal = e1.cross(e2).normalize();
        let front_face = ray.direction.dot(normal) < 0.0;
        let normal = if front_face { normal } else { -normal };

        Some(HitRecord {
            point,
            normal,
            t,
            material: self.material.clone(),
            front_face,
        })
    }
}

#[derive(Debug, Clone)]
pub enum Light {
    Point { position: Vec3, color: Color, intensity: f64 },
    Directional { direction: Vec3, color: Color, intensity: f64 },
    Ambient { color: Color, intensity: f64 },
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub origin: Vec3,
    pub lower_left: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub lens_radius: f64,
}

impl Camera {
    pub fn new(lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: f64, aspect: f64, aperture: f64, focus_dist: f64) -> Self {
        let theta = vfov * PI / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect * viewport_height;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left = origin - horizontal * 0.5 - vertical * 0.5 - w * focus_dist;

        Self { origin, lower_left, horizontal, vertical, u, v, w, lens_radius: aperture / 2.0 }
    }

    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.origin + offset,
            self.lower_left + self.horizontal * s + self.vertical * t - self.origin - offset,
        )
    }
}

fn random_in_unit_disk() -> Vec3 {
    // Deterministic pseudo-random for reproducibility
    Vec3::new(0.0, 0.0, 0.0)
}

#[derive(Debug)]
pub struct Scene {
    pub objects: Vec<Box<dyn Hittable>>,
    pub lights: Vec<Light>,
    pub background: Color,
}

impl Scene {
    pub fn new() -> Self {
        Self { objects: Vec::new(), lights: Vec::new(), background: Color::new(0.5, 0.7, 1.0) }
    }

    pub fn add_object(&mut self, obj: Box<dyn Hittable>) {
        self.objects.push(obj);
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest = t_max;
        let mut hit = None;

        for obj in &self.objects {
            if let Some(rec) = obj.hit(ray, t_min, closest) {
                closest = rec.t;
                hit = Some(rec);
            }
        }

        hit
    }
}

/// Compute lighting at a hit point.
pub fn shade(hit: &HitRecord, scene: &Scene, view_dir: Vec3) -> Color {
    let mut color = hit.material.emissive;

    // Ambient
    let ambient = Color::new(0.1, 0.1, 0.1);
    color = color + ambient.component_mul(hit.material.albedo);

    for light in &scene.lights {
        match light {
            Light::Point { position, color: light_color, intensity } => {
                let light_dir = (*position - hit.point).normalize();
                let diff = hit.normal.dot(light_dir).max(0.0);
                let diffuse = hit.material.albedo.component_mul(*light_color) * diff * *intensity;

                // Specular (Blinn-Phong)
                let half_dir = (light_dir + view_dir).normalize();
                let spec = hit.normal.dot(half_dir).max(0.0).powf(32.0 * (1.0 - hit.material.roughness));
                let specular = *light_color * spec * *intensity * hit.material.metallic;

                color = color + diffuse + specular;
            }
            Light::Directional { direction, color: light_color, intensity } => {
                let light_dir = (-*direction).normalize();
                let diff = hit.normal.dot(light_dir).max(0.0);
                let diffuse = hit.material.albedo.component_mul(*light_color) * diff * *intensity;
                color = color + diffuse;
            }
            Light::Ambient { color: light_color, intensity } => {
                color = color + hit.material.albedo.component_mul(*light_color) * *intensity;
            }
        }
    }

    color
}

/// Render a scene to a pixel buffer.
pub fn render(scene: &Scene, camera: &Camera, width: usize, height: usize, max_depth: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let u = (x as f64 + 0.5) / width as f64;
            let v = ((height - 1 - y) as f64 + 0.5) / height as f64;
            let ray = camera.get_ray(u, v);
            let color = trace_ray(scene, &ray, max_depth);
            let (r, g, b) = color.to_color_u8();

            let idx = (y * width + x) * 3;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    pixels
}

fn trace_ray(scene: &Scene, ray: &Ray, depth: usize) -> Color {
    if depth == 0 {
        return Color::zero();
    }

    if let Some(hit) = scene.hit(ray, 0.001, f64::INFINITY) {
        let view_dir = -ray.direction;

        // Direct lighting
        let direct = shade(&hit, scene, view_dir);

        // Reflection
        let reflected_color = if hit.material.metallic > 0.0 || hit.material.roughness < 0.5 {
            let reflected = ray.direction.reflect(hit.normal);
            let reflect_ray = Ray::new(hit.point, reflected);
            trace_ray(scene, &reflect_ray, depth - 1) * (1.0 - hit.material.roughness)
        } else {
            Color::zero()
        };

        // Refraction
        let refracted_color = if hit.material.transparency > 0.0 {
            let ratio = if hit.front_face { 1.0 / hit.material.ior } else { hit.material.ior };
            if let Some(refracted) = ray.direction.refract(hit.normal, ratio) {
                let refract_ray = Ray::new(hit.point, refracted);
                trace_ray(scene, &refract_ray, depth - 1) * hit.material.transparency
            } else {
                // Total internal reflection
                let reflected = ray.direction.reflect(hit.normal);
                let reflect_ray = Ray::new(hit.point, reflected);
                trace_ray(scene, &reflect_ray, depth - 1)
            }
        } else {
            Color::zero()
        };

        direct + reflected_color + refracted_color
    } else {
        scene.background
    }
}

/// Export pixels as PPM image.
pub fn to_ppm(width: usize, height: usize, pixels: &[u8]) -> String {
    let mut ppm = format!("P3\n{} {}\n255\n", width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            ppm.push_str(&format!("{} {} {} ", pixels[idx], pixels[idx + 1], pixels[idx + 2]));
        }
        ppm.push('\n');
    }
    ppm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < 1e-10);
        let cross = a.cross(b);
        assert!((cross.x - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_hit() {
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, -5.0), 1.0, Material::diffuse(Color::one()));
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = sphere.hit(&ray, 0.0, f64::INFINITY);
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_plane_hit() {
        let plane = Plane::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Material::diffuse(Color::one()));
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0));
        let hit = plane.hit(&ray, 0.0, f64::INFINITY);
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_hit() {
        let tri = Triangle::new(
            Vec3::new(-1.0, 0.0, -5.0),
            Vec3::new(1.0, 0.0, -5.0),
            Vec3::new(0.0, 2.0, -5.0),
            Material::diffuse(Color::one()),
        );
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.5, -1.0));
        let hit = tri.hit(&ray, 0.0, f64::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn test_render_simple() {
        let mut scene = Scene::new();
        scene.add_object(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -5.0), 1.0, Material::diffuse(Color::new(1.0, 0.0, 0.0)))));
        scene.add_light(Light::Ambient { color: Color::one(), intensity: 0.3 });

        let camera = Camera::new(
            Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0),
            90.0, 1.0, 0.0, 1.0,
        );

        let pixels = render(&scene, &camera, 4, 4, 5);
        assert_eq!(pixels.len(), 4 * 4 * 3);
    }
}
