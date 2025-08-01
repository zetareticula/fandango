use actix_web::{get, App, HttpResponse, HttpServer, Responder, web};
use log::info;

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Fandango Server is running!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let addr = "0.0.0.0:8080";
    info!("Starting Fandango HTTP server at http://{}", addr);

    HttpServer::new(|| {
        App::new()
            .service(hello)
    })
    .bind(addr)?
    .run()
    .await
}
