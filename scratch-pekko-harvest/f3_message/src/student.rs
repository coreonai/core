use super::{Handler, Message, Response};

#[derive(Default)]
pub struct CounterActor {
    n: u64,
}

impl Handler for CounterActor {
    fn handle(&mut self, msg: Message) -> Response {
        match msg {
            Message::Ping => todo!("Ping arm"),
            Message::Inc => todo!("Inc arm"),
            Message::Get => todo!("Get arm"),
        }
    }
}
