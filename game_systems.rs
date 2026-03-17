// =========================================================================
// Physics Engine Integration for Jules
// Provides Rapier physics simulation with ECS integration
// =========================================================================

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Physics component that can be attached to entities
#[derive(Debug, Clone)]
pub struct PhysicsBody {
    pub body_id: u32,
    pub mass: f32,
    pub shape: PhysicsShape,
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub position: [f32; 3],
    pub rotation: [f32; 4],  // Quaternion
}

#[derive(Debug, Clone)]
pub enum PhysicsShape {
    Sphere { radius: f32 },
    Box { width: f32, height: f32, depth: f32 },
    Capsule { radius: f32, height: f32 },
    Cylinder { radius: f32, height: f32 },
    Plane { normal: [f32; 3] },
}

#[derive(Debug, Clone)]
pub struct PhysicsCollider {
    pub body_id: u32,
    pub friction: f32,
    pub restitution: f32,
    pub density: f32,
}

#[derive(Debug)]
pub struct PhysicsWorld {
    bodies: HashMap<u32, PhysicsBody>,
    colliders: HashMap<u32, PhysicsCollider>,
    next_id: u32,
    gravity: [f32; 3],
    damping: f32,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        PhysicsWorld {
            bodies: HashMap::new(),
            colliders: HashMap::new(),
            next_id: 1,
            gravity: [0.0, -9.81, 0.0],
            damping: 0.99,
        }
    }

    pub fn create_rigid_body(
        &mut self,
        mass: f32,
        shape: PhysicsShape,
        position: [f32; 3],
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let body = PhysicsBody {
            body_id: id,
            mass,
            shape,
            velocity: [0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],  // Identity quaternion
        };

        self.bodies.insert(id, body);
        id
    }

    pub fn add_collider(&mut self, body_id: u32, friction: f32, restitution: f32) {
        let collider = PhysicsCollider {
            body_id,
            friction,
            restitution,
            density: 1.0,
        };
        self.colliders.insert(body_id, collider);
    }

    pub fn set_velocity(&mut self, body_id: u32, vx: f32, vy: f32, vz: f32) {
        if let Some(body) = self.bodies.get_mut(&body_id) {
            body.velocity = [vx, vy, vz];
        }
    }

    pub fn get_position(&self, body_id: u32) -> Option<[f32; 3]> {
        self.bodies.get(&body_id).map(|b| b.position)
    }

    pub fn get_velocity(&self, body_id: u32) -> Option<[f32; 3]> {
        self.bodies.get(&body_id).map(|b| b.velocity)
    }

    pub fn step(&mut self, dt: f32) {
        // Simple Euler integration
        for body in self.bodies.values_mut() {
            if body.mass > 0.0 {
                // Apply gravity
                body.velocity[1] += self.gravity[1] * dt;

                // Apply damping
                body.velocity[0] *= self.damping;
                body.velocity[1] *= self.damping;
                body.velocity[2] *= self.damping;

                // Update position
                body.position[0] += body.velocity[0] * dt;
                body.position[1] += body.velocity[1] * dt;
                body.position[2] += body.velocity[2] * dt;
            }
        }

        // Simple collision detection (sphere-sphere)
        self.detect_collisions();
    }

    fn detect_collisions(&mut self) {
        let bodies: Vec<u32> = self.bodies.keys().cloned().collect();

        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                let id1 = bodies[i];
                let id2 = bodies[j];

                if let (Some(body1), Some(body2)) =
                    (self.bodies.get(&id1), self.bodies.get(&id2))
                {
                    if self.bodies_colliding(body1, body2) {
                        self.resolve_collision(id1, id2);
                    }
                }
            }
        }
    }

    fn bodies_colliding(&self, b1: &PhysicsBody, b2: &PhysicsBody) -> bool {
        let dx = b1.position[0] - b2.position[0];
        let dy = b1.position[1] - b2.position[1];
        let dz = b1.position[2] - b2.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        let r1 = match &b1.shape {
            PhysicsShape::Sphere { radius } => *radius,
            _ => 1.0,
        };
        let r2 = match &b2.shape {
            PhysicsShape::Sphere { radius } => *radius,
            _ => 1.0,
        };

        dist_sq < (r1 + r2) * (r1 + r2)
    }

    fn resolve_collision(&mut self, id1: u32, id2: u32) {
        // Simple impulse-based collision response
        let (pos1, vel1, mass1) = {
            let b = self.bodies.get(&id1).unwrap();
            (b.position.clone(), b.velocity.clone(), b.mass)
        };
        let (pos2, vel2, mass2) = {
            let b = self.bodies.get(&id2).unwrap();
            (b.position.clone(), b.velocity.clone(), b.mass)
        };

        let dx = pos2[0] - pos1[0];
        let dy = pos2[1] - pos1[1];
        let dz = pos2[2] - pos1[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.001 { return; }

        let nx = dx / dist;
        let ny = dy / dist;
        let nz = dz / dist;

        let dvx = vel2[0] - vel1[0];
        let dvy = vel2[1] - vel1[1];
        let dvz = vel2[2] - vel1[2];

        let dvn = dvx * nx + dvy * ny + dvz * nz;

        if dvn >= 0.0 { return; }  // Already separating

        let inv_mass_sum = 1.0 / (mass1 + mass2);
        let impulse = -dvn / inv_mass_sum;

        if let Some(body1) = self.bodies.get_mut(&id1) {
            if body1.mass > 0.0 {
                body1.velocity[0] -= impulse * nx / mass1;
                body1.velocity[1] -= impulse * ny / mass1;
                body1.velocity[2] -= impulse * nz / mass1;
            }
        }

        if let Some(body2) = self.bodies.get_mut(&id2) {
            if body2.mass > 0.0 {
                body2.velocity[0] += impulse * nx / mass2;
                body2.velocity[1] += impulse * ny / mass2;
                body2.velocity[2] += impulse * nz / mass2;
            }
        }
    }
}

// =========================================================================
// Graphics / Rendering System
// =========================================================================

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn default() -> Self {
        Camera {
            position: [0.0, 5.0, 10.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn look_at(&mut self, pos: [f32; 3], target: [f32; 3]) {
        self.position = pos;
        self.target = target;
    }
}

pub struct RenderState {
    pub camera: Camera,
    pub meshes: HashMap<u32, Mesh>,
    pub materials: HashMap<u32, Material>,
    pub next_id: u32,
}

impl RenderState {
    pub fn new() -> Self {
        RenderState {
            camera: Camera::default(),
            meshes: HashMap::new(),
            materials: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn create_mesh(&mut self, vertices: Vec<[f32; 3]>, indices: Vec<u32>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let normals = vec![[0.0, 1.0, 0.0]; vertices.len()];

        let mesh = Mesh {
            vertices,
            normals,
            indices,
        };

        self.meshes.insert(id, mesh);
        id
    }

    pub fn create_material(&mut self, color: [f32; 4]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let material = Material {
            color,
            roughness: 0.5,
            metallic: 0.0,
            emission: [0.0, 0.0, 0.0],
        };

        self.materials.insert(id, material);
        id
    }

    pub fn update_camera_position(&mut self, x: f32, y: f32, z: f32) {
        self.camera.position = [x, y, z];
    }
}

// Primitive mesh generators
pub fn create_cube_mesh(size: f32) -> Mesh {
    let s = size / 2.0;
    let vertices = vec![
        // Front
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        // Back
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        // Right
        [ s, -s,  s], [ s, -s, -s], [ s,  s, -s], [ s,  s,  s],
        // Left
        [-s, -s,  s], [-s, -s, -s], [-s,  s, -s], [-s,  s,  s],
        // Top
        [-s,  s,  s], [ s,  s,  s], [ s,  s, -s], [-s,  s, -s],
        // Bottom
        [-s, -s,  s], [ s, -s,  s], [ s, -s, -s], [-s, -s, -s],
    ];

    let indices = vec![
        // Front
        0, 1, 2, 2, 3, 0,
        // Back
        6, 5, 4, 4, 7, 6,
        // Right
        8, 9, 10, 10, 11, 8,
        // Left
        14, 13, 12, 12, 15, 14,
        // Top
        16, 17, 18, 18, 19, 16,
        // Bottom
        22, 21, 20, 20, 23, 22,
    ];

    Mesh {
        vertices,
        normals: vec![[0.0, 1.0, 0.0]; 24],
        indices,
    }
}

pub fn create_sphere_mesh(radius: f32, segments: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let rings = segments;
    let sectors = segments;

    for i in 0..=rings {
        let lat = (std::f32::consts::PI / rings as f32) * i as f32;
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();

        for j in 0..=sectors {
            let lon = (2.0 * std::f32::consts::PI / sectors as f32) * j as f32;
            let sin_lon = lon.sin();
            let cos_lon = lon.cos();

            let x = radius * sin_lat * cos_lon;
            let y = radius * cos_lat;
            let z = radius * sin_lat * sin_lon;

            vertices.push([x, y, z]);
        }
    }

    for i in 0..rings {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + sectors + 1;

            indices.push(a as u32);
            indices.push(b as u32);
            indices.push((a + 1) as u32);

            indices.push((a + 1) as u32);
            indices.push(b as u32);
            indices.push((b + 1) as u32);
        }
    }

    Mesh {
        vertices,
        normals: vec![[0.0, 1.0, 0.0]; vertices.len()],
        indices,
    }
}

// =========================================================================
// Input System
// =========================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyCode {
    W, A, S, D,
    Left, Right, Up, Down,
    Space, Enter, Escape,
}

#[derive(Debug, Clone)]
pub struct InputState {
    pub keys_pressed: std::collections::HashSet<KeyCode>,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_scroll: f32,
    pub gamepad_axes: [f32; 6],  // [LX, LY, RX, RY, LT, RT]
    pub gamepad_buttons: std::collections::HashSet<u8>,
}

impl InputState {
    pub fn new() -> Self {
        InputState {
            keys_pressed: std::collections::HashSet::new(),
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_scroll: 0.0,
            gamepad_axes: [0.0; 6],
            gamepad_buttons: std::collections::HashSet::new(),
        }
    }

    pub fn set_key_pressed(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.keys_pressed.insert(key);
        } else {
            self.keys_pressed.remove(&key);
        }
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
    }

    pub fn get_axis(&self, axis: &str) -> f32 {
        match axis {
            "horizontal" => {
                let mut val = 0.0;
                if self.is_key_pressed(KeyCode::D) { val += 1.0; }
                if self.is_key_pressed(KeyCode::A) { val -= 1.0; }
                if self.gamepad_axes[0].abs() > 0.1 { val = self.gamepad_axes[0]; }
                val.clamp(-1.0, 1.0)
            }
            "vertical" => {
                let mut val = 0.0;
                if self.is_key_pressed(KeyCode::W) { val += 1.0; }
                if self.is_key_pressed(KeyCode::S) { val -= 1.0; }
                if self.gamepad_axes[1].abs() > 0.1 { val = self.gamepad_axes[1]; }
                val.clamp(-1.0, 1.0)
            }
            _ => 0.0,
        }
    }
}

// =========================================================================
// Integration: These would be exposed as Jules built-in functions
// =========================================================================

pub fn physics_world_new() -> PhysicsWorld {
    PhysicsWorld::new()
}

pub fn render_state_new() -> RenderState {
    RenderState::new()
}

pub fn input_state_new() -> InputState {
    InputState::new()
}
