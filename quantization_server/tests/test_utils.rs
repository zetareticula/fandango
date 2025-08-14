use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use std::thread;

pub struct TestServer {
    process: Child,
    pub base_url: String,
}

impl TestServer {
    pub fn new() -> Self {
        let port = port_check::free_local_port().expect("No free ports available");
        
        let process = Command::new("cargo")
            .args(["run", "--bin", "quantization_server"])
            .env("PORT", port.to_string())
            .env("RUST_LOG", "debug")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to start test server");
        
        let base_url = format!("http://localhost:{}", port);
        let server = TestServer { process, base_url };
        
        // Wait for server to be ready
        server.wait_for_ready();
        server
    }
    
    fn wait_for_ready(&self) {
        let client = reqwest::blocking::Client::new();
        let start = Instant::now();
        let timeout = Duration::from_secs(10);
        
        loop {
            if start.elapsed() > timeout {
                panic!("Server did not become ready in time");
            }
            
            match client.get(format!("{}/health", self.base_url)).send() {
                Ok(resp) if resp.status().is_success() => break,
                _ => thread::sleep(Duration::from_millis(100)),
            }
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.process.kill();
    }
}

pub fn setup_logging() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    
    ONCE.call_once(|| {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    });
}

pub fn create_test_model(server: &TestServer, model_name: &str) -> String {
    let client = reqwest::blocking::Client::new();
    
    let request = serde_json::json!({
        "model_path": "test/models/small_model.bin",
        "model_name": model_name,
        "bits": 8,
        "dims": [128, 128]
    });
    
    let response = client
        .post(format!("{}/api/v1/quantize", server.base_url))
        .json(&request)
        .send()
        .expect("Failed to create test model");
    
    assert!(response.status().is_success(), "Failed to create test model");
    
    let json: serde_json::Value = response.json().expect("Failed to parse response");
    json["model_name"].as_str().unwrap().to_string()
}
