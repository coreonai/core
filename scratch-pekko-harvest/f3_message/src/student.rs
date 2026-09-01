use super::{Handler, Message, Response};

#[derive(Default)]
pub struct CounterActor {
    n: u64,
}

impl Handler for CounterActor {
    fn handle(&mut self, msg: Message) -> Response {
        let _ = (msg, &mut self.n);
        todo!("Ping->Pong, Inc bumps, Get returns count")
    }
}
