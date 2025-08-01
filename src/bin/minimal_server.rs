
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

use actix_web::{get, web, App, Error, HttpRequest, HttpResponse, HttpServer, Responder};
use actix_web_actors::ws;
use actix_web::middleware::Logger;
use log::info;
use std::time::Instant;

/// WebSocket connection state
struct MyWebSocket {
    hb: Instant,
}

impl MyWebSocket {
    fn new() -> Self {
        Self { hb: Instant::now() }
    }
}

impl actix::Actor for MyWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket connection started");
        self.hb = Instant::now();
        ctx.text(r#"{"type":"system","message":"Connected to Fandango WebSocket"}"#);
    }
}

/// Handle messages from client
impl actix::StreamHandler<Result<ws::Message, ws::ProtocolError>> for MyWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.hb = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                info!("Received message: {}", text);
                ctx.text(text);
            }
            Ok(ws::Message::Binary(bin)) => {
                info!("Received binary message: {} bytes", bin.len());
                ctx.binary(bin);
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => {
                ctx.stop();
            }
        }
    }
}

/// WebSocket handshake and start `MyWebSocket` actor
async fn websocket(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    ws::start(MyWebSocket::new(), &req, stream)
}

#[get("/")]
async fn hello() -> impl Responder {
    "Fandango WebSocket Server"
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let addr = "0.0.0.0:8080";
    info!("Starting Fandango WebSocket server at ws://{}/ws", addr);

    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .service(hello)
            .route("/ws", web::get().to(websocket))
    })
    .bind(addr)?
    .run()
    .await
}
