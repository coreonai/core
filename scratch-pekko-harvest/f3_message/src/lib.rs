//! F3: Pekko-shaped message handling without pulling pekko-actor into the scratch.

pub mod reference;
#[cfg(feature = "student")]
pub mod student;

#[cfg(feature = "student")]
pub use student as impls;
#[cfg(not(feature = "student"))]
pub use reference as impls;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    Ping,
    Inc,
    Get,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Response {
    Pong,
    Count(u64),
    Ignored,
}

pub trait Handler {
    fn handle(&mut self, msg: Message) -> Response;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_pong() {
        let mut a = impls::CounterActor::default();
        assert_eq!(a.handle(Message::Ping), Response::Pong);
    }

    #[test]
    fn inc_and_get() {
        let mut a = impls::CounterActor::default();
        assert_eq!(a.handle(Message::Inc), Response::Count(1));
        assert_eq!(a.handle(Message::Inc), Response::Count(2));
        assert_eq!(a.handle(Message::Get), Response::Count(2));
    }
}
