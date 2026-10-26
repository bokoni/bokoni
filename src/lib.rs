//! Bokoni : langage de script intégré à Rust, syntaxe à la Ruby, typage graduel.
//!
//! ```
//! use bokoni::Engine;
//!
//! let mut engine = Engine::new();
//! engine.register_fn("double", |x: i64| x * 2);
//! let v: i64 = engine.eval_as("def add(a: int, b: int): int = a + b\nadd(1, double(20))").unwrap();
//! assert_eq!(v, 41);
//! ```

pub mod ast;
pub mod diagnostic;
pub mod engine;
pub mod intern;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod printer;
pub mod span;
pub mod value;

pub use diagnostic::{Error, ErrorKind};
pub use engine::{Engine, FromValue, IntoValue};
pub use value::Value;
