// Lightweight ECS microbenchmark harness.
// Run with: `cargo run --release --bin bench-ecs -- <entities> <steps> <dt>`

use std::time::Instant;

use jules::interp::{EcsWorld, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let dt: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.016);

    println!("bench-ecs: entities={}, steps={}, dt={}", n, steps, dt);

    let mut world = EcsWorld::default();

    for _ in 0..n {
        let id = world.spawn();
        world.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
    }

    // Warmup
    for _ in 0..3 {
        run_step(&mut world, dt);
    }

    // Baseline measurement
    let t0 = Instant::now();
    for _ in 0..steps {
        run_step(&mut world, dt);
    }
    let elapsed = t0.elapsed();
    println!("baseline elapsed: {:.3}s", elapsed.as_secs_f64());

    // If SIMD feature enabled, run the gather/process/scatter path too
    #[cfg(feature = "phase6-simd")]
    {
        println!("running SIMD-path comparison (feature=phase6-simd)");
        let mut world2 = EcsWorld::default();
        for _ in 0..n {
            let id = world2.spawn();
            world2.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world2.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
        }
        let t1 = Instant::now();
        for _ in 0..steps {
            run_step_simd(&mut world2, dt);
        }
        let elapsed_simd = t1.elapsed();
        println!("simd elapsed: {:.3}s", elapsed_simd.as_secs_f64());
    }
}

fn run_step(world: &mut EcsWorld, dt: f32) {
    let ids = world.query(&vec!["pos".to_string(), "vel".to_string()], &vec![]);
    for id in ids {
        let pos_val = world.get_component(id, "pos").unwrap().clone();
        let vel_val = world.get_component(id, "vel").unwrap().clone();
        let new_pos = match (pos_val, vel_val) {
            (Value::Vec3(p), Value::Vec3(v)) => Value::Vec3([p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt]),
            _ => continue,
        };
        world.insert_component(id, "pos", new_pos);
    }
}

#[cfg(feature = "phase6-simd")]
fn run_step_simd(world: &mut EcsWorld, dt: f32) {
    // gather
    let ids = world.query(&vec!["pos".to_string(), "vel".to_string()], &vec![]);
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(ids.len());
    let mut velocities: Vec<[f32; 3]> = Vec::with_capacity(ids.len());
    for id in &ids {
        let p = world.get_component(*id, "pos").cloned();
        let v = world.get_component(*id, "vel").cloned();
        match (p, v) {
            (Some(Value::Vec3(pv)), Some(Value::Vec3(vv))) => {
                positions.push(pv);
                velocities.push(vv);
            }
            _ => {
                positions.push([0.0, 0.0, 0.0]);
                velocities.push([0.0, 0.0, 0.0]);
            }
        }
    }

    // call centralized SIMD helper
    jules::phase6_simd::update_positions(&mut positions, &velocities, dt);

    // scatter
    for (i, id) in ids.iter().enumerate() {
        world.insert_component(*id, "pos", Value::Vec3(positions[i]));
    }
}
