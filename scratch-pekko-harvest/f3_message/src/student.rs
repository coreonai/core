use super::{Handler, Message, Response};

#[derive(Default)]
pub struct CounterActor {
    n: u64,
}

impl Handler for CounterActor {
    fn handle(&mut self, msg: Message) -> Response {
        todo!("Ping->Pong, Inc bumps, Get returns count")
    }
}
