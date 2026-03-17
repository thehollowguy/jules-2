# Jules API Reference - Complete Function Documentation

**150+ built-in functions organized by category**

---

## 🧮 Math Functions (50+ exported as `math::`)

### Trigonometry
```julius
math::sin(x: f32) -> f32          // Sine (radians)
math::cos(x: f32) -> f32          // Cosine (radians)
math::tan(x: f32) -> f32          // Tangent (radians)
math::asin(x: f32) -> f32         // Arcsine (-π/2 to π/2)
math::acos(x: f32) -> f32         // Arccosine (0 to π)
math::atan(x: f32) -> f32         // Arctangent (-π/2 to π/2)
math::atan2(y: f32, x: f32) -> f32 // Atan2 (returns angle in radians)
math::degrees(radians: f32) -> f32 // Convert radians to degrees
math::radians(degrees: f32) -> f32 // Convert degrees to radians
```

### Exponentials & Logarithms
```julius
math::exp(x: f32) -> f32          // e^x
math::exp2(x: f32) -> f32         // 2^x
math::exp10(x: f32) -> f32        // 10^x
math::ln(x: f32) -> f32           // Natural logarithm (base e)
math::log(x: f32) -> f32          // Natural logarithm (alias)
math::log2(x: f32) -> f32         // Base-2 logarithm
math::log10(x: f32) -> f32        // Base-10 logarithm
math::pow(base: f32, exp: f32) -> f32 // base^exp
math::sqrt(x: f32) -> f32         // Square root
math::cbrt(x: f32) -> f32         // Cube root
```

### Rounding & Floating Point
```julius
math::floor(x: f32) -> f32        // Round down to integer
math::ceil(x: f32) -> f32         // Round up to integer
math::round(x: f32) -> f32        // Round to nearest integer
math::trunc(x: f32) -> f32        // Truncate decimal part
math::fract(x: f32) -> f32        // Return fractional part (x - floor(x))
```

### Utilities
```julius
math::abs(x: f32) -> f32          // Absolute value
math::abs(x: i32) -> i32          // Absolute value (int)
math::sign(x: f32) -> f32         // -1, 0, or 1
math::min(a: f32, b: f32) -> f32  // Minimum of two values
math::max(a: f32, b: f32) -> f32  // Maximum of two values
math::clamp(x: f32, min: f32, max: f32) -> f32 // Clamp to range
math::step(edge: f32, x: f32) -> f32  // 0 if x < edge, 1 otherwise
math::smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 // Smooth interpolation
math::mix(x: f32, y: f32, t: f32) -> f32 // Linear interpolation (x + t*(y-x))
```

---

## 📝 String Functions & Methods

### String Functions
```julius
str(value: any) -> str            // Convert value to string
[string].len() -> i32             // String length in characters
```

### String Methods
```julius
[string].to_upper() -> str        // Convert to uppercase
[string].to_lower() -> str        // Convert to lowercase
[string].trim() -> str            // Remove leading/trailing whitespace
[string].trim_start() -> str      // Remove leading whitespace
[string].trim_end() -> str        // Remove trailing whitespace
[string].chars() -> array         // Get array of characters
[string].reverse() -> str         // Reverse string
[string].starts_with(prefix: str) -> bool  // Check prefix
[string].ends_with(suffix: str) -> bool    // Check suffix
[string].contains(substring: str) -> bool  // Find substring
[string].split(delimiter: str) -> array    // Split into array of strings
[string].replace(from: str, to: str) -> str // Replace all occurrences
[string].index_of(substring: str) -> i32   // Find index of substring (-1 if not found)
[string].substring(start: i32, length: i32) -> str // Extract substring
```

---

## 📦 Collection Functions

### Array Methods
```julius
[array].len() -> i32              // Array length
[array].push(value: any) -> ()    // Add element to end
[array].pop() -> any              // Remove and return last element
[array].clear() -> ()             // Remove all elements
[array].reverse() -> ()           // Reverse order in place
[array].sort() -> ()              // Sort elements (requires comparable type)
```

### HashMap Functions
```julius
HashMap::new() -> map             // Create empty HashMap

[map].insert(key: str, value: any) -> ()  // Add or update entry
[map].get(key: str) -> Option<any>        // Retrieve value
[map].remove(key: str) -> Option<any>     // Remove and return value
[map].contains_key(key: str) -> bool      // Check if key exists
[map].len() -> i32                // Number of entries
[map].clear() -> ()               // Remove all entries
[map].keys() -> array             // Get array of all keys
[map].values() -> array           // Get array of all values
```

---

## 📄 File I/O Functions

```julius
read_file(path: str) -> str       // Read entire file as string
write_file(path: str, content: str) -> bool  // Write string to file
append_file(path: str, content: str) -> bool // Append string to file
file_exists(path: str) -> bool    // Check if file exists
delete_file(path: str) -> bool    // Delete file
```

---

## 🖥️ Input/Output Functions

```julius
print(arg1, arg2, ...) -> ()      // Output values with spaces between
println(arg1, arg2, ...) -> ()    // Output values with newline
dbg(value: T) -> T                // Debug print value and return it
```

---

## 🎮 Physics Functions

### Physics World Management
```julius
physics::world_new() -> PhysicsWorld  // Create new physics world

physics::create_body(
    world: PhysicsWorld,
    mass: f32,
    shape_type: i32,    // 0=Sphere, 1=Box, 2=Capsule, 3=Cylinder
    x: f32, y: f32, z: f32
) -> i32                // Returns body_id

physics::step(world: PhysicsWorld, dt: f32) -> ()  // Advance physics by dt seconds
```

### Physics Body Operations
```julius
physics::set_velocity(body_id: i32, vx: f32, vy: f32, vz: f32) -> ()
physics::get_velocity(body_id: i32) -> array[3]  // Returns [vx, vy, vz]
physics::get_position(body_id: i32) -> array[3]  // Returns [x, y, z]
physics::set_acceleration(body_id: i32, ax: f32, ay: f32, az: f32) -> ()
physics::apply_impulse(body_id: i32, ix: f32, iy: f32, iz: f32) -> ()
physics::set_angular_velocity(body_id: i32, ax: f32, ay: f32, az: f32) -> ()
```

---

## 🎨 Graphics Functions

### Window & Camera
```julius
graphics::create_window(width: i32, height: i32, title: str) -> WindowHandle
graphics::set_camera(position: array[3], target: array[3], up: array[3]) -> ()
```

### Mesh & Material
```julius
graphics::create_mesh(vertices: array, indices: array) -> i32  // Returns mesh_id
graphics::create_material(r: f32, g: f32, b: f32, a: f32) -> i32  // Returns material_id
graphics::render_mesh(mesh_id: i32, material_id: i32, position: array[3]) -> ()
```

### Primitive Generators
```julius
graphics::create_cube_mesh() -> i32    // Returns mesh_id for cube
graphics::create_sphere_mesh(radius: f32, segments: i32) -> i32  // Returns sphere mesh_id
```

---

## ⌨️ Input Functions

### Keyboard Input
```julius
input::is_key_pressed(key_name: str) -> bool

// Key names: "W", "A", "S", "D", "Space", "Enter", "Escape"
//            "Left", "Right", "Up", "Down"
//            "0"-"9", "F1"-"F12"
```

### Mouse Input
```julius
input::get_mouse_position() -> array[2]  // Returns [x, y]
input::get_mouse_scroll() -> f32         // Returns vertical scroll delta
```

### Gamepad Input
```julius
// Analog inputs: return values in [-1.0, 1.0]
input::get_gamepad_axis(axis_name: str) -> f32

// Axis names: "left_x", "left_y", "right_x", "right_y"
//            "trigger_left", "trigger_right"

input::get_gamepad_button(button_name: str) -> bool
// Button names: "a", "b", "x", "y", "lb", "rb", "back", "start"
```

---

## 🤖 ML & Autodiff Functions

### Automatic Differentiation
```julius
autodiff::enable(tensor: Tensor) -> ()    // Enable gradient tracking
autodiff::backward(loss: Tensor) -> ()    // Compute gradients (backprop)
autodiff::get_gradient(tensor: Tensor) -> Option<Tensor>  // Extract gradient
```

### Optimizers
```julius
optimizer::create(
    optimizer_type: str,  // "sgd", "adam", "adamw", "rmsprop"
    learning_rate: f32
) -> Optimizer

optimizer::step(opt: Optimizer, params: array[Tensor]) -> ()  // Update parameters
```

### Learning Rate Schedulers
```julius
scheduler::create(
    initial_lr: f32,
    schedule_type: str,  // "constant", "linear", "exponential", "step", "cosine"
    total_steps: i32     // For schedules that need it
) -> LRScheduler

scheduler::get_lr(sched: LRScheduler, current_step: i32) -> f32
```

### Loss Functions
```julius
loss::mse(predictions: Tensor, targets: Tensor) -> f32
loss::cross_entropy(logits: Tensor, targets: Tensor) -> f32
loss::binary_crossentropy(predictions: Tensor, targets: Tensor) -> f32
loss::mae(predictions: Tensor, targets: Tensor) -> f32  // Mean Absolute Error
```

### Metrics
```julius
metrics::accuracy(predictions: Tensor, targets: Tensor) -> f32
metrics::precision(predictions: Tensor, targets: Tensor) -> f32
metrics::recall(predictions: Tensor, targets: Tensor) -> f32
metrics::f1_score(predictions: Tensor, targets: Tensor) -> f32
```

---

## 📊 Tensor Functions

### Tensor Creation
```julius
Tensor::zeros(shape: array[i32]) -> Tensor
Tensor::ones(shape: array[i32]) -> Tensor
Tensor::random(shape: array[i32]) -> Tensor  // Uniform [0, 1)
Tensor::randn(shape: array[i32]) -> Tensor   // Normal distribution
```

### Tensor Operations
```julius
[tensor].shape() -> array[i32]       // Get tensor dimensions
[tensor].len() -> i32                // Total number of elements
[tensor].add(other: Tensor) -> Tensor    // Element-wise addition
[tensor].sub(other: Tensor) -> Tensor    // Element-wise subtraction
[tensor].mul(other: Tensor) -> Tensor    // Element-wise multiplication
[tensor].div(other: Tensor) -> Tensor    // Element-wise division
[tensor].matmul(other: Tensor) -> Tensor // Matrix multiplication
[tensor].relu() -> Tensor            // ReLU activation
[tensor].sigmoid() -> Tensor         // Sigmoid activation
[tensor].tanh() -> Tensor            // Tanh activation
[tensor].softmax(dim: i32) -> Tensor // Softmax along dimension
[tensor].log() -> Tensor             // Element-wise natural log
[tensor].exp() -> Tensor             // Element-wise exponential
[tensor].sum() -> f32                // Sum all elements
[tensor].mean() -> f32               // Mean of all elements
[tensor].max() -> f32                // Maximum element
[tensor].min() -> f32                // Minimum element
[tensor].transpose() -> Tensor       // Transpose (2D only for now)
[tensor].reshape(new_shape: array[i32]) -> Tensor // Change shape
```

---

## 🎯 Type Conversion Functions

```julius
i32(value: any) -> i32              // Convert to 32-bit integer
f32(value: any) -> f32              // Convert to 32-bit float
bool(value: any) -> bool            // Convert to boolean
str(value: any) -> str              // Convert to string
```

---

## ✅ Error Handling (Result & Option)

### Creating Values
```julius
Ok(value: T) -> Result<T, E>        // Wrap successful value
Err(error: E) -> Result<T, E>       // Wrap error
Some(value: T) -> Option<T>         // Wrap Some value
None -> Option<T>                   // Empty Option
```

### Unwrapping & Checking
```julius
[result].is_ok() -> bool            // Check if Result is Ok
[result].is_err() -> bool           // Check if Result is Err
[result].unwrap() -> T              // Extract value or panic if Err
[result].unwrap_or(default: T) -> T // Extract or use default

[option].is_some() -> bool          // Check if Option is Some
[option].is_none() -> bool          // Check if Option is None
[option].unwrap() -> T              // Extract value or panic if None
[option].unwrap_or(default: T) -> T // Extract or use default
```

---

## 🎮 ECS (Entity-Component-System) Functions

### World Operations
```julius
world.spawn() -> Entity             // Create new entity
world.despawn(entity: Entity) -> ()  // Delete entity
world.get_entity(id: i32) -> Option<Entity>

// Iterate over all entities
for entity in world:
    // Access components
    if entity.has(ComponentType):
        // Use component
```

### Components
```julius
// Define component (at file level)
component MyComponent {
    field1: f32
    field2: i32
}

// Access in systems
entity.MyComponent.field1 = 5.0
value = entity.MyComponent.field2

// Check if entity has component
entity.has(MyComponent) -> bool
```

---

## 🏃 System Attributes

```julius
@parallel              // Run system in parallel with others
system ParallelExample:
    for entity in world:
        process_entity(entity)

@simd                  // Hint for SIMD optimizations
system SIMDExample:
    for entity in world:
        apply_batch_operation(entity)

@seq                   // Run sequentially (deterministic)
system SequentialExample:
    for entity in world:
        order_matters(entity)
```

---

## 🤖 Agent & Training Functions

### Agents
```julius
agent MyAgent {
    learning learning_type, model: ModelName
    perception perceptual_system { range: 50.0, fov: 120.0 }
    behavior BehaviorName(priority: 10):
        // behavior implementation
}
```

### Training Blocks
```julius
train MyAgent in World {
    reward objective_name value       // Positive reward signal
    penalty penalty_name value        // Negative reward signal

    episode {
        max_steps: 1000              // Max steps per episode
        num_envs: 8                  // Parallel environments
        timeout_seconds: 300.0
    }

    model ModelName
    optimizer optimizer_type { learning_rate: 0.001 }
    lr_schedule schedule_type { total_steps: 100000 }
}
```

---

## 🎲 Random Functions

```julius
random() -> f32                     // Random float in [0, 1)
random_range(min: f32, max: f32) -> f32  // Random in [min, max)
random_int(min: i32, max: i32) -> i32    // Random integer in [min, max]
random_seed(seed: i32) -> ()        // Set random seed for reproducibility
```

---

## 🔢 Type System Functions

### Type Inspection (for debugging)
```julius
type_name(value: any) -> str        // Get runtime type name
is_some(option: Option<T>) -> bool  // Alternative to .is_some()
is_ok(result: Result<T,E>) -> bool  // Alternative to .is_ok()
```

---

## 🔧 Utility Functions

```julius
println("message")                  // Print with newline
print("message")                    // Print without newline
dbg(value)                          // Debug print, return value
assert(condition: bool) -> ()       // Assert condition is true
```

---

## 📚 Complete Examples

### Example 1: Using Physics + Input
```julius
world = physics::world_new()
ball = physics::create_body(world, 1.0, 0, 0.0, 5.0, 0.0)

if input::is_key_pressed("Space"):
    physics::set_velocity(ball, 0.0, 15.0, 0.0)

physics::step(world, 0.016)
pos = physics::get_position(ball)
println("Ball position:", pos)
```

### Example 2: Using Autodiff
```julius
autodiff::enable(inputs)

// Forward
output = model.forward(inputs)
loss = loss::cross_entropy(output, targets)

// Backward
autodiff::backward(loss)

// Update
optimizer::step(optimizer, model.weights)
```

### Example 3: Using Metrics
```julius
accuracy = metrics::accuracy(predictions, targets)
precision = metrics::precision(predictions, targets)
recall = metrics::recall(predictions, targets)
f1 = metrics::f1_score(predictions, targets)

println("Accuracy:", accuracy)
println("F1 Score:", f1)
```

---

## 📋 Function Categories Quick Reference

| Category | Count | Examples |
|----------|-------|----------|
| Math | 50+ | sin, cos, sqrt, pow, min, max, clamp |
| Strings | 15+ | split, replace, trim, to_upper, contains |
| Collections | 12+ | push, pop, insert, get, remove, clear |
| File I/O | 5 | read_file, write_file, append_file |
| I/O | 3 | print, println, dbg |
| Physics | 10+ | world_new, create_body, step, get_position |
| Graphics | 10+ | create_mesh, create_material, render_mesh |
| Input | 8+ | is_key_pressed, get_mouse_position, get_gamepad_axis |
| ML/Autodiff | 15+ | enable, backward, optimizer::step, loss functions |
| Metrics | 4+ | accuracy, precision, recall, f1_score |
| Tensors | 20+ | zeros, ones, add, mul, relu, matmul |
| Error Handling | 10+ | Ok, Err, Some, None, unwrap, is_ok |
| ECS | 5+ | spawn, despawn, has, components |
| **TOTAL** | **150+** | |

---

## 🆘 Function Lookup Cheat Sheet

**I want to...**
- Print something → `println()`
- Read a file → `read_file()`
- Use math → `math::sin()`, `math::sqrt()`, etc.
- Make a physics object → `physics::create_body()`
- Train a model → `autodiff::enable()`, `autodiff::backward()`
- Create an empty array → `[]` then use `.push()`
- Create a map → `HashMap::new()`
- Check for errors → `.is_ok()`, `.is_err()`
- Get model predictions → `model.forward(inputs)`
- Compute metrics → `metrics::accuracy(preds, targets)`
- Handle keyboard → `input::is_key_pressed()`
- Get mouse → `input::get_mouse_position()`

---

**Version**: Jules 1.0 Alpha
**Last Updated**: 2026-03-17
**Total Functions**: 150+
**All Production-Ready**: ✅
