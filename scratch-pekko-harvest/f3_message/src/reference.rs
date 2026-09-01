use super::{Handler, Message, Response};

#[derive(Default)]
pub struct CounterActor {
    n: u64,
}

impl Handler for CounterActor {
    fn handle(&mut self, msg: Message) -> Response {
        match msg {
            Message::Ping => Response::Pong,
            Message::Inc => {
                self.n += 1;
                Response::Count(self.n)
            }
            Message::Get => Response::Count(self.n),
        }
    }
}
