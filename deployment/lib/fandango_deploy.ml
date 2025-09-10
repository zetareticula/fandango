open Lwt.Syntax

(* Configuration types *)
type environment = Dev | Staging | Production

type deployment_config = {
  environment : environment;
  replicas : int;
  cpu_limit : string;
  memory_limit : string;
  image_tag : string;
  namespace : string;
  domain : string option;
}

type deployment_target = 
  | Docker of { registry : string; tag : string }
  | Kubernetes of { cluster : string; config_path : string }
  | Local of { port : int }

(* Deployment strategy factorization *)
module DeploymentStrategy = struct
  type t = {
    name : string;
    validate : deployment_config -> (unit, string) result;
    deploy : deployment_config -> deployment_target -> (unit, string) result Lwt.t;
    rollback : deployment_config -> deployment_target -> (unit, string) result Lwt.t;
  }

  let create_strategy name validate deploy rollback = 
    { name; validate; deploy; rollback }
end

(* Environment-specific configurations *)
let default_config env = match env with
  | Dev -> {
      environment = Dev;
      replicas = 1;
      cpu_limit = "250m";
      memory_limit = "256Mi";
      image_tag = "dev";
      namespace = "fandango-dev";
      domain = None;
    }
  | Staging -> {
      environment = Staging;
      replicas = 2;
      cpu_limit = "500m";
      memory_limit = "512Mi";
      image_tag = "staging";
      namespace = "fandango-staging";
      domain = Some "staging.fandango.example.com";
    }
  | Production -> {
      environment = Production;
      replicas = 3;
      cpu_limit = "1000m";
      memory_limit = "1Gi";
      image_tag = "latest";
      namespace = "fandango-prod";
      domain = Some "fandango.example.com";
    }

(* Kubernetes deployment strategy *)
let kubernetes_strategy = 
  let validate config = 
    if config.replicas > 0 then Ok () 
    else Error "Replicas must be greater than 0"
  in
  
  let deploy config target = 
    match target with
    | Kubernetes { cluster; config_path } ->
        let* () = Lwt_io.printf "Deploying to Kubernetes cluster: %s\n" cluster in
        let* () = Lwt_io.printf "Using config: %s\n" config_path in
        let* () = Lwt_io.printf "Environment: %s, Replicas: %d\n" 
          (match config.environment with 
           | Dev -> "dev" 
           | Staging -> "staging" 
           | Production -> "production") 
          config.replicas in
        
        (* Generate Kubernetes manifests *)
        let deployment_yaml = Printf.sprintf {|
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fandango-quantization-server
  namespace: %s
spec:
  replicas: %d
  selector:
    matchLabels:
      app: fandango
  template:
    metadata:
      labels:
        app: fandango
    spec:
      containers:
      - name: fandango
        image: fandango:%s
        resources:
          limits:
            cpu: %s
            memory: %s
|} config.namespace config.replicas config.image_tag config.cpu_limit config.memory_limit in
        
        let* () = Lwt_io.with_file ~mode:Output "/tmp/fandango-deployment.yaml" 
          (fun oc -> Lwt_io.write oc deployment_yaml) in
        
        (* Apply the deployment *)
        let cmd = Printf.sprintf "kubectl apply -f /tmp/fandango-deployment.yaml --kubeconfig=%s" config_path in
        let* result = Lwt_process.exec (Lwt_process.shell cmd) in
        (match result with
         | Unix.WEXITED 0 -> Lwt.return (Ok ())
         | _ -> Lwt.return (Error "Failed to apply Kubernetes deployment"))
    | _ -> Lwt.return (Error "Invalid target for Kubernetes strategy")
  in
  
  let rollback config target =
    match target with
    | Kubernetes { cluster; config_path } ->
        let* () = Lwt_io.printf "Rolling back deployment in cluster: %s\n" cluster in
        let cmd = Printf.sprintf "kubectl rollout undo deployment/fandango-quantization-server -n %s --kubeconfig=%s" 
          config.namespace config_path in
        let* result = Lwt_process.exec (Lwt_process.shell cmd) in
        (match result with
         | Unix.WEXITED 0 -> Lwt.return (Ok ())
         | _ -> Lwt.return (Error "Failed to rollback deployment"))
    | _ -> Lwt.return (Error "Invalid target for Kubernetes rollback")
  in
  
  DeploymentStrategy.create_strategy "kubernetes" validate deploy rollback

(* Docker deployment strategy *)
let docker_strategy = 
  let validate config = Ok () in
  
  let deploy config target = 
    match target with
    | Docker { registry; tag } ->
        let* () = Lwt_io.printf "Building and pushing Docker image: %s:%s\n" registry tag in
        
        (* Build the image *)
        let build_cmd = Printf.sprintf "docker build -t %s:%s ." registry tag in
        let* build_result = Lwt_process.exec (Lwt_process.shell build_cmd) in
        
        (match build_result with
         | Unix.WEXITED 0 ->
             (* Push the image *)
             let push_cmd = Printf.sprintf "docker push %s:%s" registry tag in
             let* push_result = Lwt_process.exec (Lwt_process.shell push_cmd) in
             (match push_result with
              | Unix.WEXITED 0 -> Lwt.return (Ok ())
              | _ -> Lwt.return (Error "Failed to push Docker image"))
         | _ -> Lwt.return (Error "Failed to build Docker image"))
    | _ -> Lwt.return (Error "Invalid target for Docker strategy")
  in
  
  let rollback config target = 
    Lwt.return (Error "Docker rollback not implemented")
  in
  
  DeploymentStrategy.create_strategy "docker" validate deploy rollback

(* Local deployment strategy *)
let local_strategy = 
  let validate config = Ok () in
  
  let deploy config target = 
    match target with
    | Local { port } ->
        let* () = Lwt_io.printf "Starting local deployment on port %d\n" port in
        let cmd = Printf.sprintf "cd ../quantization_server && PORT=%d cargo run --release" port in
        let* () = Lwt_io.printf "Running: %s\n" cmd in
        Lwt.return (Ok ())
    | _ -> Lwt.return (Error "Invalid target for local strategy")
  in
  
  let rollback config target = 
    Lwt.return (Error "Local rollback not implemented")
  in
  
  DeploymentStrategy.create_strategy "local" validate deploy rollback

(* Main deployment orchestrator *)
let deploy_fandango strategy config target = 
  let* () = Lwt_io.printf "Starting deployment with strategy: %s\n" strategy.DeploymentStrategy.name in
  
  match strategy.validate config with
  | Error msg -> 
      let* () = Lwt_io.printf "Validation failed: %s\n" msg in
      Lwt.return (Error msg)
  | Ok () ->
      let* result = strategy.deploy config target in
      (match result with
       | Ok () -> 
           let* () = Lwt_io.printf "Deployment completed successfully!\n" in
           Lwt.return (Ok ())
       | Error msg ->
           let* () = Lwt_io.printf "Deployment failed: %s\n" msg in
           Lwt.return (Error msg))

(* Deployment pipeline composition *)
let create_pipeline strategies = 
  fun config target ->
    let rec execute_strategies = function
      | [] -> Lwt.return (Ok ())
      | strategy :: rest ->
          let* result = deploy_fandango strategy config target in
          (match result with
           | Ok () -> execute_strategies rest
           | Error _ as err -> Lwt.return err)
    in
    execute_strategies strategies
