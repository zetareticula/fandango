// lib.rs

// Public module declarations
pub mod my_module;

// Re-exports for easier access
pub use my_module::MyStruct;

// Example of a public struct
pub struct MyStruct {
    pub field: String,
}

// Example of a public function
impl MyStruct {
    pub fn new(field: String) -> Self {
        MyStruct { field }
    }

    pub fn do_something(&self) {
        println!("Doing something with {}", self.field);
    }
}

// Example of a private function
fn private_function() {
    // Implementation details
}

// Example of a public constant
pub const VERSION: &str = "1.0.0";