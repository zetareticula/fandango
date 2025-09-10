open Lwt.Syntax
open Cmdliner
open Fandango_deploy

let deploy_cmd =
  let env_arg = 
    let doc = "Target environment (dev, staging, production)" in
    Arg.(required & opt (some string) None & info ["e"; "env"] ~doc)
  in
  
  let strategy_arg = 
    let doc = "Deployment strategy (kubernetes, docker, local)" in
    Arg.(required & opt (some string) None & info ["s"; "strategy"] ~doc)
  in
  
  let target_arg = 
    let doc = "Deployment target configuration" in
    Arg.(required & opt (some string) None & info ["t"; "target"] ~doc)
  in
  
  let deploy_func env_str strategy_str target_str =
    let env = match env_str with
      | "dev" -> Dev
      | "staging" -> Staging  
      | "production" -> Production
      | _ -> failwith "Invalid environment"
    in
    
    let config = default_config env in
    
    let strategy = match strategy_str with
      | "kubernetes" -> kubernetes_strategy
      | "docker" -> docker_strategy
      | "local" -> local_strategy
      | _ -> failwith "Invalid strategy"
    in
    
    let target = match strategy_str with
      | "kubernetes" -> Kubernetes { cluster = "default"; config_path = "~/.kube/config" }
      | "docker" -> Docker { registry = "fandango"; tag = config.image_tag }
      | "local" -> Local { port = 8080 }
      | _ -> failwith "Invalid target"
    in
    
    Lwt_main.run (
      let* result = deploy_fandango strategy config target in
      match result with
      | Ok () -> 
          let* () = Lwt_io.printf "✅ Deployment successful!\n" in
          Lwt.return (`Ok ())
      | Error msg ->
          let* () = Lwt_io.printf "❌ Deployment failed: %s\n" msg in
          Lwt.return (`Error (false, msg))
    )
  in
  
  let doc = "Deploy Fandango to target environment" in
  let info = Cmd.info "deploy" ~doc in
  Cmd.v info Term.(const deploy_func $ env_arg $ strategy_arg $ target_arg)

let pipeline_cmd =
  let config_file_arg = 
    let doc = "Pipeline configuration file" in
    Arg.(required & opt (some file) None & info ["c"; "config"] ~doc)
  in
  
  let pipeline_func config_file =
    let* () = Lwt_io.printf "🚀 Running deployment pipeline from: %s\n" config_file in
    
    (* Example multi-stage pipeline *)
    let dev_config = default_config Dev in
    let staging_config = default_config Staging in
    
    let pipeline = create_pipeline [docker_strategy; kubernetes_strategy] in
    
    let* dev_result = pipeline dev_config (Local { port = 8080 }) in
    match dev_result with
    | Ok () ->
        let* () = Lwt_io.printf "✅ Dev deployment successful, proceeding to staging\n" in
        let* staging_result = pipeline staging_config (Kubernetes { cluster = "staging"; config_path = "~/.kube/config" }) in
        (match staging_result with
         | Ok () -> 
             let* () = Lwt_io.printf "✅ Pipeline completed successfully!\n" in
             Lwt.return (`Ok ())
         | Error msg ->
             let* () = Lwt_io.printf "❌ Staging deployment failed: %s\n" msg in
             Lwt.return (`Error (false, msg)))
    | Error msg ->
        let* () = Lwt_io.printf "❌ Dev deployment failed: %s\n" msg in
        Lwt.return (`Error (false, msg))
  in
  
  let doc = "Run deployment pipeline" in
  let info = Cmd.info "pipeline" ~doc in
  Cmd.v info Term.(const (fun config -> Lwt_main.run (pipeline_func config)) $ config_file_arg)

let status_cmd =
  let status_func () =
    let* () = Lwt_io.printf "🔍 Checking Fandango deployment status...\n" in
    
    (* Check local server *)
    let* () = Lwt_io.printf "Local server: " in
    let* local_result = Lwt_process.exec (Lwt_process.shell "curl -s http://localhost:8080/health") in
    let* () = match local_result with
      | Unix.WEXITED 0 -> Lwt_io.printf "✅ Running\n"
      | _ -> Lwt_io.printf "❌ Not running\n"
    in
    
    (* Check Kubernetes deployment *)
    let* () = Lwt_io.printf "Kubernetes deployment: " in
    let* k8s_result = Lwt_process.exec (Lwt_process.shell "kubectl get deployment fandango-quantization-server 2>/dev/null") in
    let* () = match k8s_result with
      | Unix.WEXITED 0 -> Lwt_io.printf "✅ Deployed\n"
      | _ -> Lwt_io.printf "❌ Not deployed\n"
    in
    
    Lwt.return (`Ok ())
  in
  
  let doc = "Check deployment status" in
  let info = Cmd.info "status" ~doc in
  Cmd.v info Term.(const (fun () -> Lwt_main.run (status_func ())))

let main_cmd =
  let doc = "Fandango deployment orchestration tool" in
  let info = Cmd.info "fandango-deploy" ~doc ~version:"1.0.0" in
  Cmd.group info [deploy_cmd; pipeline_cmd; status_cmd]

let () = exit (Cmd.eval main_cmd)
